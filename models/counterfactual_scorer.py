import os
import torch
from torch.utils.data import DataLoader, Subset, random_split
from .mnist_conv_autoencoder import MNISTConvAutoencoder

class MNISTClassAutoencoders:
    def __init__(self, latent_dim=32, device='mps', models_dir='models/class_autoencoders'):
        """
        Initialize 10 class-specific autoencoders for MNIST digits.
        
        Args:
            latent_dim: Dimension of the latent space in the autoencoders
            device: Device to run the models on ('cuda', 'cpu', or 'mps')
            models_dir: Directory to save the trained models
        """
        self.latent_dim = latent_dim
        self.device = device
        self.models_dir = models_dir
        self.autoencoders = {}  # Dictionary to store the 10 autoencoders (one per digit)
        
        # Create directory for models if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize 10 autoencoders (one for each digit class)
        for digit in range(10):
            model_path = os.path.join(models_dir, f'mnist_autoencoder_class_{digit}.pth')
            self.autoencoders[digit] = MNISTConvAutoencoder(
                latent_dim=latent_dim,
                device=device,
                model_save_path=model_path
            )
    
    def train_all(self, mnist_dataset, val_split=0.1, batch_size=64, num_epochs=20, **train_kwargs):
        """
        Train all 10 class-specific autoencoders.
        
        Args:
            mnist_dataset: The full MNIST dataset
            val_split: Proportion of data to use for validation
            batch_size: Batch size for training
            num_epochs: Number of epochs to train for
            **train_kwargs: Additional arguments for the train_model method
        """
        # Create DataLoader for each digit class
        for digit in range(10):
            print(f"\n===== Training autoencoder for digit {digit} =====")
            
            # Filter dataset to only include images of this digit
            digit_indices = [i for i, (_, label) in enumerate(mnist_dataset) if label == digit]
            digit_dataset = Subset(mnist_dataset, digit_indices)
            
            # Split into train and validation
            val_size = int(len(digit_dataset) * val_split)
            train_size = len(digit_dataset) - val_size
            train_dataset, val_dataset = random_split(digit_dataset, [train_size, val_size])
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
            
            # Train the autoencoder for this digit
            self.autoencoders[digit].train_model(train_loader, val_loader, num_epochs=num_epochs, **train_kwargs)
    
    def train_single_class(self, mnist_dataset, digit_class, val_split=0.1, batch_size=64, num_epochs=20, test_samples=5, **train_kwargs):
        """
        Train a single class-specific autoencoder and test it with a few reconstructions.
        
        Args:
            mnist_dataset: The full MNIST dataset
            digit_class: The digit class to train (0-9)
            val_split: Proportion of data to use for validation
            batch_size: Batch size for training
            num_epochs: Number of epochs to train for
            test_samples: Number of samples to test reconstruction on
            **train_kwargs: Additional arguments for the train_model method
            
        Returns:
            The trained autoencoder
        """
        print(f"\n===== Training autoencoder for digit {digit_class} =====")
        
        # Filter dataset to only include images of this digit
        digit_indices = [i for i, (_, label) in enumerate(mnist_dataset) if label == digit_class]
        digit_dataset = Subset(mnist_dataset, digit_indices)
        
        # Split into train and validation
        val_size = int(len(digit_dataset) * val_split)
        train_size = len(digit_dataset) - val_size
        train_dataset, val_dataset = random_split(digit_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
        
        # Train the autoencoder for this digit
        self.autoencoders[digit_class].train_model(train_loader, val_loader, num_epochs=num_epochs, **train_kwargs)
        
        # Test the autoencoder with a few reconstructions
        if test_samples > 0:
            # Get a few test samples
            test_loader = DataLoader(val_dataset, batch_size=test_samples)
            test_data, test_labels = next(iter(test_loader))
            
            # Create labels array with the same digit class
            test_labels = torch.tensor([digit_class] * len(test_data))
            
            # Visualize reconstructions
            print(f"Testing reconstructions for digit {digit_class}...")
            self.visualize_reconstructions(test_data, test_labels, num_samples=test_samples)
        
        return self.autoencoders[digit_class]
    
    def save_models(self):
        """Save all autoencoder models"""
        for digit, autoencoder in self.autoencoders.items():
            print(f"Saving autoencoder for digit {digit}...")
            autoencoder.save_model()
    
    def load_models(self):
        """Load all autoencoder models"""
        for digit, autoencoder in self.autoencoders.items():
            try:
                print(f"Loading autoencoder for digit {digit}...")
                autoencoder.load_model()
            except FileNotFoundError:
                print(f"Model for digit {digit} not found! You need to train it first.")
    
    def calculate_im1(self, x_prime, original_class, counterfactual_class, epsilon=1e-6):
        """
        Calculate the IM1 score for a counterfactual explanation.
        
        Args:
            x_prime: The counterfactual image (tensor of shape [1, 1, 28, 28])
            original_class: The original class/digit
            counterfactual_class: The counterfactual class/digit
            epsilon: Small constant to prevent division by zero
            
        Returns:
            The IM1 score (lower is better, indicates a more realistic counterfactual)
        """
        # Ensure model evaluation mode
        self.autoencoders[original_class].eval()
        self.autoencoders[counterfactual_class].eval()

        x_prime = x_prime.to(self.device)
        
        with torch.no_grad():
            # Get reconstructions from both autoencoders
            recon_original, _ = self.autoencoders[original_class](x_prime)
            recon_counterfactual, _ = self.autoencoders[counterfactual_class](x_prime)
            
            # Calculate reconstruction losses (squared L2 norms)
            loss_original = torch.sum((x_prime - recon_original) ** 2).item()
            loss_counterfactual = torch.sum((x_prime - recon_counterfactual) ** 2).item()
            
            # Calculate IM1 score
            im1 = loss_counterfactual / (loss_original + epsilon)
            
        return im1
    
    def calculate_im1_batch(self, counterfactuals, original_classes, counterfactual_classes, epsilon=1e-6):
        """
        Calculate IM1 scores for a batch of counterfactual explanations.
        
        Args:
            counterfactuals: Batch of counterfactual images (tensor of shape [N, 1, 28, 28])
            original_classes: List of original classes (length N)
            counterfactual_classes: List of counterfactual classes (length N)
            epsilon: Small constant to prevent division by zero
            
        Returns:
            List of IM1 scores
        """
        im1_scores = []
        for i in range(len(counterfactuals)):
            x_prime = counterfactuals[i:i+1]  # Get a single image with batch dimension
            orig_class = original_classes[i]
            cf_class = counterfactual_classes[i]
            im1 = self.calculate_im1(x_prime, orig_class, cf_class, epsilon)
            im1_scores.append(im1)
        return im1_scores
    
    def get_reconstruction(self, x, digit_class):
        """
        Get the reconstruction of an image using the class-specific autoencoder.
        
        Args:
            x: The input image (tensor of shape [1, 1, 28, 28])
            digit_class: The digit class (0-9)
            
        Returns:
            The reconstructed image
        """
        self.autoencoders[digit_class].eval()
        with torch.no_grad():
            recon, _ = self.autoencoders[digit_class](x)
        return recon
    
    def visualize_reconstructions(self, data, labels, num_samples=5, save_path=None):
        """
        Visualize original images and their reconstructions using class-specific autoencoders.
        
        Args:
            data: Batch of images (tensor of shape [N, 1, 28, 28])
            labels: List or tensor of class labels
            num_samples: Number of samples to visualize (default: 5)
            save_path: Path to save the visualization (if None, will display instead)
            
        Returns:
            None (displays or saves the visualization)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Ensure we don't try to visualize more samples than we have
        num_samples = min(num_samples, len(data))
        
        # Create a figure with rows for each sample and columns for original and reconstruction
        fig, axes = plt.subplots(num_samples, 2, figsize=(6, 2 * num_samples))
        
        # If only one sample, make sure axes is 2D
        if num_samples == 1:
            axes = axes.reshape(1, 2)
        
        for i in range(num_samples):
            # Get a single image and its label
            img = data[i:i+1]
            label = labels[i].item() if torch.is_tensor(labels) else labels[i]
            
            # Get reconstruction using the appropriate autoencoder
            recon = self.get_reconstruction(img, label)
            
            # Convert tensors to numpy arrays for plotting
            img_np = img.squeeze().cpu().numpy()
            recon_np = recon.squeeze().cpu().numpy()
            
            # Plot original image
            axes[i, 0].imshow(img_np, cmap='gray')
            axes[i, 0].set_title(f"Original (Class {label})")
            axes[i, 0].axis('off')
            
            # Plot reconstruction
            axes[i, 1].imshow(recon_np, cmap='gray')
            axes[i, 1].set_title(f"Reconstruction")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def visualize_cross_reconstructions(self, data, labels, save_path=None):
        """
        Visualize original images and their reconstructions using different class autoencoders.
        Shows how each class-specific autoencoder reconstructs the same image.
        
        Args:
            data: Batch of images (tensor of shape [N, 1, 28, 28])
            labels: List or tensor of class labels
            save_path: Path to save the visualization (if None, will display instead)
            
        Returns:
            None (displays or saves the visualization)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import torch.nn.functional as F
        
        num_classes = 10  # MNIST has 10 classes
        
        # Create a figure with rows for each class (0-9) and columns for original + each class reconstruction
        fig, axes = plt.subplots(num_classes, num_classes + 1, figsize=(2 * (num_classes + 1), 2 * num_classes))
        
        # Find one example of each class
        class_examples = {}
        for i in range(len(data)):
            label = labels[i].item() if torch.is_tensor(labels) else labels[i]
            if label not in class_examples and len(class_examples) < num_classes:
                class_examples[label] = i
            if len(class_examples) == num_classes:
                break
        
        # If we couldn't find examples of all classes, print a warning
        if len(class_examples) < num_classes:
            print(f"Warning: Could only find examples for {len(class_examples)} classes in the provided data")
        
        # For each class (row)
        for row, (label, idx) in enumerate(sorted(class_examples.items())):
            # Get a single image of this class
            img = data[idx:idx+1].to(self.device)
            
            # Convert tensor to numpy array for plotting
            img_np = img.squeeze().cpu().numpy()
            
            # Plot original image
            axes[row, 0].imshow(img_np, cmap='gray')
            axes[row, 0].set_title(f"Original\n(Class {label})")
            axes[row, 0].axis('off')
            
            # Get reconstructions using each class-specific autoencoder (columns)
            for col in range(num_classes):
                recon = self.get_reconstruction(img, col)
                recon_np = recon.squeeze().cpu().numpy()
                
                # Calculate MSE between original and reconstruction
                mse = F.mse_loss(img.squeeze().cpu(), recon.squeeze().cpu()).item()
                mse_rounded = round(mse, 4)
                
                # Plot reconstruction
                axes[row, col+1].imshow(recon_np, cmap='gray')
                title = f"Class {col} AE\nMSE: {mse_rounded}"
                if col == label:
                    title += "\n(correct class)"
                axes[row, col+1].set_title(title)
                axes[row, col+1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Cross-reconstruction visualization saved to {save_path}")
        else:
            plt.show()