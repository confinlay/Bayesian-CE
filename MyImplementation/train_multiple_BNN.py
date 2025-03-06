import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models.FlexibleAutoencoder import FlexibleAutoencoder
from models.BNN_VI import BayesianNeuralNetworkVI
from train import train_BNN_VI_classification, cprint
from clue import NewCLUE
import torch.nn.functional as F

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

def run_multi_model_clue(
    models, 
    model_names,
    input_data,
    decoder=None,
    uncertainty_weight=1.0,
    distance_weight=0.005,
    steps=200,
    lr=0.1,
    device=None,
    is_bayesian=None,
    visualize=True,
    n_cols=5
):
    """
    Run CLUE optimization on the same input data for multiple models.
    
    Args:
        models: List of models to run CLUE on. Each model should be able to extract features
                and make predictions.
        model_names: List of names for each model (for visualization labels)
        input_data: Input data tensor of shape [1, channels, height, width]
        decoder: Optional decoder to visualize reconstructions
        uncertainty_weight: Weight for uncertainty term in CLUE loss
        distance_weight: Weight for distance term in CLUE loss
        steps: Number of steps for CLUE optimization
        lr: Learning rate for CLUE optimization
        device: Device to run optimization on ('cpu', 'cuda', or 'mps')
        is_bayesian: List of booleans indicating which models are Bayesian
                    (if None, attempts to detect automatically)
        visualize: Whether to create visualization plots
        n_cols: Number of columns in visualization grid
        
    Returns:
        dict: Dictionary containing:
            - 'original_data': The original input data
            - 'original_latents': List of original latent representations
            - 'optimized_latents': List of CLUE-optimized latent representations
            - 'original_entropies': List of original prediction entropies
            - 'optimized_entropies': List of optimized prediction entropies
            - 'reconstructions': List of decoded optimized latents (if decoder provided)
            - 'fig': Matplotlib figure (if visualize=True)
    """
    # Set device if not provided
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    # Auto-detect Bayesian models if not specified
    if is_bayesian is None:
        is_bayesian = [hasattr(model, 'sample_predict') or hasattr(model, 'sample_predict_z') for model in models]
    
    # Ensure input is on the correct device
    input_data = input_data.to(device)
    
    # Results dictionary
    results = {
        'original_data': input_data,
        'original_latents': [],
        'optimized_latents': [],
        'original_entropies': [],
        'optimized_entropies': [],
        'reconstructions': []
    }
    
    # Extract latent representations and run CLUE for each model
    for i, (model, model_name, bayesian) in enumerate(zip(models, model_names, is_bayesian)):
        model = model.to(device)
        model.eval()
        
        # Extract original latent
        with torch.no_grad():
            # Handle different model types and extract features
            if hasattr(model, 'extract_features'):
                # Models with explicit feature extraction method
                z0 = model.extract_features(input_data)
            elif hasattr(model, 'encode'):
                # Autoencoder-type models
                z0 = model.encode(input_data)
            else:
                # Models that return (features, output) tuple
                try:
                    z0, _ = model(input_data)
                except:
                    # Models that just return features
                    z0 = model(input_data)
            
            # Calculate original entropy
            if bayesian:
                if hasattr(model, 'sample_predict_z'):
                    probs = model.sample_predict_z(z0).mean(0)
                else:
                    probs = model.sample_predict(input_data).mean(0)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(1)
            else:
                if hasattr(model, 'forward'):
                    logits = model(z0)
                else:
                    # Handle classifier with separate prediction head
                    try:
                        logits = model.classifier(z0)
                    except:
                        _, logits = model(input_data)
                        
                probs = F.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(1)
        
        # Store original latent and entropy
        results['original_latents'].append(z0.detach())
        results['original_entropies'].append(entropy.item())
        
        # Determine the classifier component for CLUE
        if hasattr(model, 'classifier'):
            classifier = model.classifier
        else:
            classifier = model
        
        # Run CLUE optimization
        clue_optimizer = NewCLUE(
            classifier=classifier,
            z0=z0,
            uncertainty_weight=uncertainty_weight,
            distance_weight=distance_weight,
            lr=lr,
            device=device,
            bayesian=bayesian,
            verbose=False
        )
        
        z_optimized = clue_optimizer.optimize(steps=steps)
        results['optimized_latents'].append(z_optimized)
        
        # Calculate optimized entropy
        with torch.no_grad():
            if bayesian:
                if hasattr(model, 'sample_predict_z'):
                    probs = model.sample_predict_z(z_optimized).mean(0)
                else:
                    # Create reconstruction first
                    recon = decoder(z_optimized)
                    probs = model.sample_predict(recon).mean(0)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(1)
            else:
                if hasattr(model, 'forward'):
                    logits = model(z_optimized)
                else:
                    logits = model.classifier(z_optimized)
                probs = F.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(1)
        
        results['optimized_entropies'].append(entropy.item())
        
        # Generate reconstructions if decoder provided
        if decoder is not None:
            decoder = decoder.to(device)
            with torch.no_grad():
                reconstruction = decoder(z_optimized)
                results['reconstructions'].append(reconstruction.detach())
    
    # Visualization
    if visualize:
        # Calculate rows needed
        n_models = len(models)
        n_rows = (n_models * 2 + 1) if decoder is not None else n_models
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        
        # Add row labels along the left side
        for i, model_name in enumerate(model_names):
            row_index = i * 2 + 1 if decoder is not None else i
            fig.text(0.02, 1 - (row_index + 0.5) / n_rows, 
                     f"{model_name} CLUE", 
                     ha='left', va='center', fontsize=12, fontweight='bold')
            
            if decoder is not None:
                fig.text(0.02, 1 - (row_index + 1.5) / n_rows, 
                         f"{model_name} Diff", 
                         ha='left', va='center', fontsize=12, fontweight='bold')
        
        # Original image in the first row
        for col in range(min(n_cols, 1)):  # Just need one column for the original image
            axes[0, col].imshow(input_data[0, 0].cpu().detach(), cmap='gray')
            axes[0, col].axis('off')
            
            # Create title with all original entropies
            entropy_str = " | ".join([f"{name}: {ent:.3f}" 
                                     for name, ent in zip(model_names, results['original_entropies'])])
            axes[0, col].set_title(f"Original\n{entropy_str}")
            
            # Hide unused column cells in the first row
            for c in range(1, n_cols):
                axes[0, c].axis('off')
        
        # If we have reconstructions, show the optimized images and diffs
        if decoder is not None:
            original_recon = None
            if len(results['reconstructions']) > 0:
                original_recon = decoder(results['original_latents'][0])
            
            for i, (reconstruction, model_name) in enumerate(zip(results['reconstructions'], model_names)):
                # Reconstructed image
                axes[i*2+1, 0].imshow(reconstruction[0, 0].cpu().detach(), cmap='gray')
                axes[i*2+1, 0].axis('off')
                axes[i*2+1, 0].set_title(f"Optimized\nEntropy: {results['optimized_entropies'][i]:.3f}")
                
                # Difference image (if original_recon is available)
                if original_recon is not None:
                    diff = reconstruction[0, 0].cpu().detach() - original_recon[0, 0].cpu().detach()
                    axes[i*2+2, 0].imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
                    axes[i*2+2, 0].axis('off')
                    axes[i*2+2, 0].set_title('Difference (Red- Blue+)')
                
                # Hide unused column cells in the model rows
                for c in range(1, n_cols):
                    axes[i*2+1, c].axis('off')
                    axes[i*2+2, c].axis('off')
        
        plt.tight_layout()
        results['fig'] = fig
    
    return results