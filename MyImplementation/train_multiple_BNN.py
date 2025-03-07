import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import glob

from models.FlexibleAutoencoder import FlexibleAutoencoder
from models.BNN_VI import BayesianNeuralNetworkVI
from train import train_BNN_VI_classification, cprint
from clue.new_CLUE import NewCLUE
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
            early_stopping_patience=10,
            verbose=True,
            save_path=ae_dir,
            model_name=model_name,
            lr_scheduler='plateau'
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
    result,  # This expects the dictionary returned by train_multiple_models
    input_data,
    uncertainty_weight=1.0,
    distance_weight=0.005,
    steps=200,
    lr=0.1,
    device=None,
    visualize=True,
    n_cols=5
):
    """
    Run CLUE optimization on BNN models trained with different autoencoder backbones.
    
    Args:
        result: Dictionary returned by train_multiple_models containing autoencoders and BNN models
        input_data: Input data tensor of shape [1, channels, height, width]
        uncertainty_weight: Weight for uncertainty term in CLUE loss
        distance_weight: Weight for distance term in CLUE loss
        steps: Number of steps for CLUE optimization
        lr: Learning rate for CLUE optimization
        device: Device to run optimization on ('cpu', 'cuda', or 'mps')
        visualize: Whether to create visualization plots
        n_cols: Number of columns in visualization grid
        
    Returns:
        dict: Dictionary containing optimization results and visualizations
    """
    # Set device if not provided
    if device is None:
        device = get_device()
    
    # Ensure input is on the correct device
    input_data = input_data.to(device)
    
    # Results dictionary
    results = {
        'original_data': input_data,
        'original_latents': {},
        'optimized_latents': {},
        'original_entropies': {},
        'optimized_entropies': {},
        'original_predictions': {},
        'optimized_predictions': {},
        'reconstructions': {},
        'original_reconstructions': {}
    }
    
    # Process each BNN model with its corresponding autoencoder
    for hidden_dim, bnn_model in result['bnn_models'].items():
        # Get the corresponding autoencoder
        autoencoder = result['autoencoders'][hidden_dim]
        
        # Move models to the correct device
        bnn_model = bnn_model.to(device)
        autoencoder = autoencoder.to(device)
        
        # Set models to evaluation mode
        bnn_model.eval()
        autoencoder.eval()
        
        # Extract original latent using the corresponding autoencoder
        with torch.no_grad():
            # Get the latent representation
            z0 = autoencoder.encode(input_data)
            
            # Get original prediction and entropy
            probs = bnn_model.sample_predict_z(z0).mean(0)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(1)
            pred = probs.argmax(dim=1)
            
            # Get original reconstruction
            original_reconstruction = autoencoder.decode(z0)
            results['original_reconstructions'][hidden_dim] = original_reconstruction.detach()
        
        # Store original latent, entropy and prediction
        results['original_latents'][hidden_dim] = z0.detach()
        results['original_entropies'][hidden_dim] = entropy.item()
        results['original_predictions'][hidden_dim] = pred.item()
        
        # Run CLUE optimization
        clue_optimizer = NewCLUE(
            classifier=bnn_model,
            z0=z0,
            uncertainty_weight=uncertainty_weight,
            distance_weight=distance_weight,
            lr=lr,
            device=device,
            bayesian=True,
            verbose=False
        )
        
        z_optimized = clue_optimizer.optimize(steps=steps)
        results['optimized_latents'][hidden_dim] = z_optimized
        
        # Calculate optimized entropy and prediction
        with torch.no_grad():
            probs = bnn_model.sample_predict_z(z_optimized).mean(0)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(1)
            pred = probs.argmax(dim=1)
        
        results['optimized_entropies'][hidden_dim] = entropy.item()
        results['optimized_predictions'][hidden_dim] = pred.item()
        
        # Generate reconstruction
        with torch.no_grad():
            reconstruction = autoencoder.decode(z_optimized)
            results['reconstructions'][hidden_dim] = reconstruction.detach()
    
    # Handle raw BNN if it exists
    if result.get('raw_bnn') is not None:
        raw_bnn = result['raw_bnn'].to(device)
        raw_bnn.eval()
        
        with torch.no_grad():
            # For raw BNN, the "latent" is the flattened input
            flattened_input = input_data.view(input_data.size(0), -1)
            probs = raw_bnn.sample_predict(input_data).mean(0)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(1)
            pred = probs.argmax(dim=1)
        
        results['original_latents']['raw'] = flattened_input.detach()
        results['original_entropies']['raw'] = entropy.item()
        results['original_predictions']['raw'] = pred.item()
        
        # Raw input CLUE is handled differently - we use the GenericCLUE or similar approach
        # This is typically handled separately since we're optimizing the actual image
        # not a latent representation
    
    # Visualization
    if visualize:
        # Create figure
        fig, axes = plt.subplots(len(result['bnn_models']) + 1, n_cols, 
                                 figsize=(4*n_cols, 3*(len(result['bnn_models'])+1)))
        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        
        # Make axes 2D if there's only one row
        if len(result['bnn_models']) == 1:
            axes = axes.reshape(2, -1)
        
        # Original image in the first row
        axes[0, 0].imshow(input_data[0, 0].cpu().detach(), cmap='gray')
        axes[0, 0].axis('off')
        axes[0, 0].set_title("Original")
        
        # Add entropy values and predictions to the title
        entropy_str = " | ".join([f"dim={dim}: {ent:.3f} (pred={results['original_predictions'][dim]})" 
                                 for dim, ent in results['original_entropies'].items()])
        axes[0, 1].text(0.5, 0.5, f"Original Entropies & Predictions:\n{entropy_str}", 
                      ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].axis('off')
        
        # Hide unused column cells in the first row
        for c in range(2, n_cols):
            axes[0, c].axis('off')
        
        # Show reconstructions for each hidden dimension
        for i, (hidden_dim, reconstruction) in enumerate(results['reconstructions'].items()):
            # Row index (add 1 to skip the original image row)
            row = i + 1
            
            # Original reconstruction
            axes[row, 0].imshow(results['original_reconstructions'][hidden_dim][0, 0].cpu().detach(), cmap='gray')
            axes[row, 0].axis('off')
            axes[row, 0].set_title(f"dim={hidden_dim} Original")
            
            # Optimized reconstruction
            axes[row, 1].imshow(reconstruction[0, 0].cpu().detach(), cmap='gray')
            axes[row, 1].axis('off')
            axes[row, 1].set_title(f"dim={hidden_dim} Optimized")
            
            # Original and optimized entropy values and predictions
            orig_ent = results['original_entropies'][hidden_dim]
            opt_ent = results['optimized_entropies'][hidden_dim]
            orig_pred = results['original_predictions'][hidden_dim]
            opt_pred = results['optimized_predictions'][hidden_dim]
            reduction = orig_ent - opt_ent
            reduction_pct = (reduction / orig_ent) * 100 if orig_ent > 0 else 0
            
            entropy_text = (f"Original: {orig_ent:.3f} (pred={orig_pred})\n"
                          f"Optimized: {opt_ent:.3f} (pred={opt_pred})\n"
                          f"Reduction: {reduction:.3f} ({reduction_pct:.1f}%)")
            
            axes[row, 2].text(0.5, 0.5, entropy_text, ha='center', va='center', 
                            transform=axes[row, 2].transAxes)
            axes[row, 2].axis('off')
            
            # Difference image 
            with torch.no_grad():
                diff = reconstruction[0, 0].cpu().detach() - results['original_reconstructions'][hidden_dim][0, 0].cpu().detach()
                axes[row, 3].imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
                axes[row, 3].axis('off')
                axes[row, 3].set_title('Difference (Red- Blue+)')
            
            # Hide unused column cells
            for c in range(4, n_cols):
                axes[row, c].axis('off')
        
        plt.tight_layout()
        results['fig'] = fig
    
    return results

def load_pretrained_models(
    dims=[8, 16, 32, 64, 128],
    include_raw=True,
    base_save_dir="../model_saves/multi_dim",
    device=None
):
    """
    Load pretrained models from saved files into the same data structure that
    train_multiple_models returns and that run_multi_model_clue expects.
    
    Args:
        dims: List of hidden dimensions to load
        include_raw: Whether to try loading a raw BNN model
        base_save_dir: Base directory where models were saved
        device: Device to load models to ('cpu', 'cuda', or 'mps')
        
    Returns:
        dict: Dictionary containing loaded models:
            - 'autoencoders': Dict mapping dimensions to autoencoder models
            - 'bnn_models': Dict mapping dimensions to BNN models
            - 'raw_bnn': BNN trained on raw input (if include_raw=True and file exists)
    """
    # Set up device if not provided
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    print(f"Loading models to device: {device}")
    
    # Create directory paths
    ae_dir = f"{base_save_dir}/autoencoders"
    bnn_dir = f"{base_save_dir}/BNNs"
    
    # Check if directories exist
    if not os.path.exists(ae_dir):
        raise FileNotFoundError(f"Autoencoder directory not found: {ae_dir}")
    if not os.path.exists(bnn_dir):
        raise FileNotFoundError(f"BNN directory not found: {bnn_dir}")
    
    # Initialize result dictionary with same structure as train_multiple_models
    result = {
        'autoencoders': {},
        'bnn_models': {},
        'raw_bnn': None
    }
    
    # Load autoencoder models
    for hidden_dim in dims:
        # Look for best model first, then final, then latest
        ae_best_path = f"{ae_dir}/autoencoder_dim{hidden_dim}_best_model.pth"
        ae_final_path = f"{ae_dir}/autoencoder_dim{hidden_dim}_final_model.pth"
        ae_latest_path = f"{ae_dir}/autoencoder_dim{hidden_dim}_latest_model.pth"
        
        # Try loading in order of preference
        if os.path.exists(ae_best_path):
            ae_path = ae_best_path
        elif os.path.exists(ae_final_path):
            ae_path = ae_final_path
        elif os.path.exists(ae_latest_path):
            ae_path = ae_latest_path
        else:
            print(f"Warning: No autoencoder model found for dimension {hidden_dim}")
            continue
        
        try:
            print(f"Loading autoencoder with dimension {hidden_dim} from {ae_path}")
            autoencoder = FlexibleAutoencoder.load(ae_path, device=device)
            result['autoencoders'][hidden_dim] = autoencoder
        except Exception as e:
            print(f"Error loading autoencoder dim={hidden_dim}: {e}")
    
    # Load BNN models
    for hidden_dim in dims:
        if hidden_dim not in result['autoencoders']:
            print(f"Skipping BNN for dimension {hidden_dim} since autoencoder wasn't loaded")
            continue
        
        # Get the corresponding autoencoder
        autoencoder = result['autoencoders'][hidden_dim]
        
        # Create directory path where BNN model would be saved
        bnn_model_dir = f"{bnn_dir}/BNN_dim{hidden_dim}_models"
        
        if not os.path.exists(bnn_model_dir):
            print(f"Warning: BNN model directory not found for dimension {hidden_dim}: {bnn_model_dir}")
            continue
        
        # Find the latest BNN checkpoint
        bnn_checkpoints = glob.glob(f"{bnn_model_dir}/BNN_VI_best_*.pt")
        if not bnn_checkpoints:
            print(f"Warning: No BNN checkpoint found for dimension {hidden_dim}")
            continue
        
        # Get the latest checkpoint
        bnn_checkpoint = sorted(bnn_checkpoints)[-1]
        
        try:
            # Create encoder backbone
            class EncoderBackbone(torch.nn.Module):
                def __init__(self, autoencoder):
                    super().__init__()
                    self.autoencoder = autoencoder
                    
                def forward(self, x):
                    return self.autoencoder.encode(x)
            
            encoder_backbone = EncoderBackbone(autoencoder)
            
            # Create BNN model with the right structure
            bnn_model = BayesianNeuralNetworkVI(
                backbone=encoder_backbone,
                input_dim=hidden_dim,
                output_dim=10,  # MNIST has 10 classes
                hidden_dim=128,
                prior_mu=0.0,
                prior_sigma=0.1,
                kl_weight=0.1,
                device=device
            )
            
            # Load checkpoint
            print(f"Loading BNN with dimension {hidden_dim} from {bnn_checkpoint}")
            bnn_model.load_checkpoint(bnn_checkpoint)
            result['bnn_models'][hidden_dim] = bnn_model
        except Exception as e:
            print(f"Error loading BNN dim={hidden_dim}: {e}")
    
    # Optionally load raw BNN
    if include_raw:
        raw_bnn_model_dir = f"{bnn_dir}/BNN_raw_input_models"
        
        if os.path.exists(raw_bnn_model_dir):
            # Find the latest raw BNN checkpoint
            raw_bnn_checkpoints = glob.glob(f"{raw_bnn_model_dir}/BNN_VI_best_*.pt")
            
            if raw_bnn_checkpoints:
                # Get the latest checkpoint
                raw_bnn_checkpoint = sorted(raw_bnn_checkpoints)[-1]
                
                try:
                    # Create identity backbone
                    class IdentityBackbone(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            
                        def forward(self, x):
                            if len(x.shape) > 2:
                                return x.view(x.size(0), -1)
                            return x
                    
                    raw_backbone = IdentityBackbone()
                    
                    # Create raw BNN model
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
                    
                    # Load checkpoint
                    print(f"Loading raw BNN from {raw_bnn_checkpoint}")
                    raw_bnn.load_checkpoint(raw_bnn_checkpoint)
                    result['raw_bnn'] = raw_bnn
                except Exception as e:
                    print(f"Error loading raw BNN: {e}")
    
    # Summary
    print("\nLoaded models summary:")
    print(f"Autoencoders: {list(result['autoencoders'].keys())}")
    print(f"BNN models: {list(result['bnn_models'].keys())}")
    print(f"Raw BNN: {'Loaded' if result['raw_bnn'] is not None else 'Not loaded'}")
    
    return result