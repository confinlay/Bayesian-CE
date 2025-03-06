import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models.FlexibleAutoencoder import FlexibleAutoencoder
from models.BNN_VI import BayesianNeuralNetworkVI
from train import train_BNN_VI_classification, cprint

class EncoderBackbone(nn.Module):
    """Wrapper to use autoencoder's encoder as a backbone"""
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder
        
    def forward(self, x):
        """Extract features using autoencoder's encoder"""
        features = self.autoencoder.encode(x)
        return features

class IdentityBackbone(nn.Module):
    """Simple identity backbone for raw input processing"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        """Returns flattened input"""
        if len(x.shape) > 2:
            return x.view(x.size(0), -1)
        return x

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def load_mnist_data(batch_size=64):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader, train_dataset, test_dataset

def train_multiple_models(
    dims=[8, 16, 32, 64, 128], 
    batch_size=64, 
    ae_epochs=20, 
    bnn_epochs=50, 
    include_raw=False,
    base_save_dir="../model_saves/multi_dim"
):
    """
    Train multiple autoencoders with different hidden dimensions and BNNs with these autoencoders as backbones.
    
    Args:
        dims: List of hidden dimensions for autoencoders
        batch_size: Batch size for training
        ae_epochs: Number of epochs for autoencoder training
        bnn_epochs: Number of epochs for BNN training
        include_raw: Whether to also train a BNN on raw input
        base_save_dir: Base directory for saving models
        
    Returns:
        dict: Dictionary containing all trained models
            - 'autoencoders': Dict mapping dimensions to autoencoder models
            - 'bnn_models': Dict mapping dimensions to BNN models
            - 'raw_bnn': BNN trained on raw input (if include_raw=True)
    """
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Create save directories
    ae_dir = f"{base_save_dir}/autoencoders"
    bnn_dir = f"{base_save_dir}/BNNs"
    
    os.makedirs(ae_dir, exist_ok=True)
    os.makedirs(bnn_dir, exist_ok=True)
    
    # Load data
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data(batch_size)
    
    # Dictionary to store trained models
    result = {
        'autoencoders': {},
        'bnn_models': {},
        'raw_bnn': None
    }
    
    # 1. Train autoencoders with different hidden dimensions
    for hidden_dim in dims:
        print(f"\n{'='*50}")
        print(f"Training autoencoder with hidden dimension: {hidden_dim}")
        print(f"{'='*50}")
        
        # Create autoencoder with this dimension
        autoencoder = FlexibleAutoencoder(
            hidden_dim=hidden_dim,
            input_shape=(1, 28, 28),
            device=device
        )
        
        # Train model using existing train_model method
        model_name = f"autoencoder_dim{hidden_dim}"
        train_losses, val_losses = autoencoder.train_model(
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=ae_epochs,
            lr=1e-3,
            early_stopping_patience=5,
            verbose=True,
            save_path=ae_dir,
            model_name=model_name
        )
        
        # Store trained model
        result['autoencoders'][hidden_dim] = autoencoder
        
        # Plot training losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, 'b-', label='Train')
        plt.plot(val_losses, 'r-', label='Validation')
        plt.title(f'Autoencoder (dim={hidden_dim}) Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{ae_dir}/{model_name}_loss.png')
        plt.close()
        
        # Show sample reconstructions
        with torch.no_grad():
            images, _ = next(iter(test_loader))
            images = images[:8].to(device)
            reconstructions, _ = autoencoder(images)
            
            fig, axes = plt.subplots(2, 8, figsize=(16, 4))
            for i in range(8):
                # Original
                axes[0, i].imshow(images[i].cpu().squeeze(0), cmap='gray')
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # Reconstruction
                axes[1, i].imshow(reconstructions[i].cpu().squeeze(0), cmap='gray')
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{ae_dir}/{model_name}_samples.png')
            plt.close()
    
    # 2. Train BNN_VI models using each autoencoder as backbone
    for hidden_dim, autoencoder in result['autoencoders'].items():
        print(f"\n{'='*50}")
        print(f"Training BNN_VI with autoencoder backbone (dim={hidden_dim})")
        print(f"{'='*50}")
        
        # Create encoder backbone
        encoder_backbone = EncoderBackbone(autoencoder)
        
        # Create BNN model
        bnn_model = BayesianNeuralNetworkVI(
            backbone=encoder_backbone,
            input_dim=hidden_dim,  # Input to BNN is the latent dimension of autoencoder
            output_dim=10,  # MNIST has 10 classes
            hidden_dim=128,
            prior_mu=0.0,
            prior_sigma=0.1,
            kl_weight=0.1,
            device=device
        )
        
        # Train BNN using existing training function
        model_name = f"BNN_dim{hidden_dim}"
        metrics = train_BNN_VI_classification(
            net=bnn_model,
            name=model_name,
            batch_size=batch_size,
            nb_epochs=bnn_epochs,
            trainset=train_dataset,
            valset=test_dataset,
            device=device,
            lr=1e-3,
            patience=5,
            model_saves_dir=bnn_dir
        )
        
        # Store trained model
        result['bnn_models'][hidden_dim] = bnn_model
    
    # Optionally train a BNN_VI on raw inputs
    if include_raw:
        print(f"\n{'='*50}")
        print(f"Training BNN_VI with raw input (dim=784)")
        print(f"{'='*50}")
        
        # Create backbone and BNN
        raw_backbone = IdentityBackbone()
        raw_bnn = BayesianNeuralNetworkVI(
            backbone=raw_backbone,
            input_dim=784,  # Flattened MNIST image
            output_dim=10,
            hidden_dim=128,
            prior_mu=0.0,
            prior_sigma=0.1,
            kl_weight=0.1,
            device=device
        )
        
        # Train the raw input BNN
        metrics = train_BNN_VI_classification(
            net=raw_bnn,
            name="BNN_raw_input",
            batch_size=batch_size,
            nb_epochs=bnn_epochs,
            trainset=train_dataset,
            valset=test_dataset,
            device=device,
            lr=1e-3,
            patience=5,
            model_saves_dir=bnn_dir
        )
        
        # Store raw BNN
        result['raw_bnn'] = raw_bnn
    
    print("\nTraining complete! All models saved to:")
    print(f"Autoencoders: {ae_dir}")
    print(f"BNN models: {bnn_dir}")
    
    return result