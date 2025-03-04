import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from ddim_mnist import DiffusionModel, SimpleAutoencoder

# Create directories for outputs
os.makedirs("samples", exist_ok=True)


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def plot_images(images, title=""):
    """Plot a grid of images."""
    # Convert from torch tensors if needed
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 10))
    num_images = min(16, len(images))
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    return plt


def train_autoencoder(device, batch_size=128, epochs=5):
    """Train a simple autoencoder on MNIST."""
    print("Training autoencoder...")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4 if device.type != "mps" else 0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4 if device.type != "mps" else 0
    )
    
    # Create and train autoencoder
    autoencoder = SimpleAutoencoder(latent_dim=256, device=device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        # Training
        autoencoder.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            output = autoencoder(data)
            loss = criterion(output, data)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        # Validation
        autoencoder.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                output = autoencoder(data)
                test_loss += criterion(output, data).item()
        
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    # Save the model
    autoencoder.save_checkpoint("autoencoder_weights.pt")
    
    # Visualize some reconstructions
    with torch.no_grad():
        test_data = next(iter(test_loader))[0][:16].to(device)
        reconstructions = autoencoder(test_data)
        
        # Plot reconstructions
        plot = plot_images(reconstructions, "Autoencoder Reconstructions")
        plt.savefig("autoencoder_reconstructions.png")
        plt.close()
    
    return autoencoder


def prepare_dataset(autoencoder, device, batch_size=64, image_size=28):
    """Prepare MNIST dataset with feature vectors for DDIM training."""
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Ensure consistent image size
        transforms.Resize((image_size, image_size), antialias=True) 
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Extract feature vectors for training and testing data
    print("Generating feature vectors...")
    
    train_images = []
    train_features = []
    test_images = []
    test_features = []
    
    # Determine number of workers (MPS doesn't work well with multiple workers)
    num_workers = 0 if device.type == "mps" else 4
    
    # Process training data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            features = autoencoder.encode(images)
            
            train_images.append(images.cpu())
            train_features.append(features.cpu())
    
    # Process test data
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            features = autoencoder.encode(images)
            
            test_images.append(images.cpu())
            test_features.append(features.cpu())
    
    # Concatenate batches
    train_images = torch.cat(train_images)
    train_features = torch.cat(train_features)
    test_images = torch.cat(test_images)
    test_features = torch.cat(test_features)
    
    # Create datasets
    train_dataset = TensorDataset(train_images, train_features)
    test_dataset = TensorDataset(test_images, test_features)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader, test_features[:16]  # Keep first 16 test features for visualization


def train_diffusion_model(train_loader, test_loader, test_features, device, epochs=30, existing_model=None):
    """Train the DDIM model.
    
    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        test_features: Feature vectors for test samples
        device: Device to use for training
        epochs: Number of epochs to train for
        existing_model: Optional existing DiffusionModel to continue training
        
    Returns:
        Trained DiffusionModel
    """
    # Create or use existing diffusion model
    if existing_model is not None:
        print("Using existing diffusion model...")
        diffusion_model = existing_model
    else:
        # Create a new diffusion model
        print("Creating diffusion model...")
        diffusion_model = DiffusionModel(
            time_embedding_dim=32,
            feature_embedding_dim=64,
            image_size=28,
            widths=[32, 64, 128],
            block_depth=2,
            device=device
        )
        
        # Compute dataset statistics for normalization
        diffusion_model.adapt_input_data(train_loader)
    
    # Create optimizer
    optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-4)
    
    # Training loop
    print("Training diffusion model...")
    for epoch in range(epochs):
        # Training
        diffusion_model.train()
        train_loss = 0
        start_time = time.time()
        
        for batch_idx, (images, features) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            features = features.to(device)
            
            # Training step
            loss = diffusion_model.training_step((images, features), optimizer)
            train_loss += loss
            
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss:.6f}")
        
        # Calculate average loss
        train_loss /= len(train_loader)
        
        # Validation
        diffusion_model.eval()
        with torch.no_grad():
            val_loss = 0
            for images, features in test_loader:
                images = images.to(device)
                features = features.to(device)
                
                # Normalize images
                images = diffusion_model.normalize(images)
                
                # Sample random diffusion times
                batch_size = images.shape[0]
                diffusion_times = torch.rand(size=(batch_size, 1), device=device)
                
                # Get alphas for diffusion times
                alphas = diffusion_model._diffusion_schedule(diffusion_times)
                
                # Reshape alphas for proper broadcasting
                alphas_expanded = alphas.view(-1, 1, 1, 1)
                
                # Add noise to images
                noise = torch.randn_like(images)
                noisy_images = torch.sqrt(alphas_expanded) * images + torch.sqrt(1 - alphas_expanded) * noise
                
                # Predict noise
                pred_noise = diffusion_model(noisy_images, diffusion_times, features)
                val_loss += nn.functional.mse_loss(noise, pred_noise).item()
        
        val_loss /= len(test_loader)
        
        # Print progress
        elapsed = time.time() - start_time
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Time: {elapsed:.2f}s")
        
        # Generate images every 5 epochs
        if (epoch + 1) % 5 == 0:
            # Generate images using test features
            generate_samples(diffusion_model, test_features, device, f"samples/epoch_{epoch+1:03d}.png")
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            diffusion_model.save_checkpoint(f"diffusion_model_epoch_{epoch+1}.pt")
    
    # Save final model
    diffusion_model.save_checkpoint("diffusion_model.pt")
    
    return diffusion_model


def demonstrate_diffusion_autoencoder(diffusion_model, autoencoder, device, num_samples=5):
    """Demonstrate the diffusion autoencoder by manipulating latent vectors."""
    print("Demonstrating diffusion autoencoder capabilities...")
    
    # Load some test images
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Sample test images
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    test_images = []
    for idx in indices:
        image, _ = test_dataset[idx]
        test_images.append(image)
    
    # Stack into tensor
    test_images = torch.stack(test_images).to(device)
    
    # Get latent vectors
    with torch.no_grad():
        latent_vectors = autoencoder.encode(test_images)
    
    # Add noise to images at different diffusion times
    diffusion_times = [0.1, 0.3, 0.5, 0.7, 0.9]
    noisy_images_list = []
    
    with torch.no_grad():
        for t in diffusion_times:
            noisy_images, _ = diffusion_model.add_noise(test_images, t)
            noisy_images_list.append(noisy_images)
    
    # Perturb latent vectors
    # For demonstration, we'll create 3 perturbed versions of each latent vector
    perturbed_vectors = []
    perturbation_strengths = [0.0, 0.5, 1.0, 2.0]  # No perturbation, mild, medium, strong
    
    for strength in perturbation_strengths:
        if strength == 0.0:
            perturbed_vectors.append(latent_vectors)  # Original vectors
        else:
            # Add random noise to latent vectors
            noise = torch.randn_like(latent_vectors) * strength
            perturbed_vectors.append(latent_vectors + noise)
    
    # Denoise with different latent vectors
    results = []
    
    # Use highest noise level for clearest demonstration
    noisy_images = noisy_images_list[-1]  # Most noisy images
    
    with torch.no_grad():
        for i, vectors in enumerate(perturbed_vectors):
            denoised = diffusion_model.denoise_with_latent(
                noisy_images, vectors, diffusion_times[-1], num_steps=30
            )
            results.append(denoised)
    
    # Create visualization
    fig = plt.figure(figsize=(2 + len(perturbation_strengths) * 2, 2 + num_samples * 2))
    
    # Original images
    for i in range(num_samples):
        plt.subplot(2 + num_samples, len(perturbation_strengths) + 1, i * (len(perturbation_strengths) + 1) + 1)
        plt.imshow(test_images[i].cpu().squeeze(), cmap="gray")
        if i == 0:
            plt.title("Original")
        plt.axis("off")
    
    # Noisy images
    for i in range(num_samples):
        plt.subplot(2 + num_samples, len(perturbation_strengths) + 1, (i + 1) * (len(perturbation_strengths) + 1))
        plt.imshow(noisy_images[i].cpu().squeeze(), cmap="gray")
        if i == 0:
            plt.title(f"Noisy (t={diffusion_times[-1]:.1f})")
        plt.axis("off")
    
    # Results with different perturbation strengths
    for p, strength in enumerate(perturbation_strengths):
        for i in range(num_samples):
            plt.subplot(2 + num_samples, len(perturbation_strengths) + 1, 
                       (i * (len(perturbation_strengths) + 1)) + p + 2)
            plt.imshow(results[p][i].cpu().squeeze(), cmap="gray")
            if i == 0:
                plt.title(f"Perturb: {strength}")
            plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("diffusion_autoencoder_demo.png", dpi=150)
    plt.close()
    print("Diffusion autoencoder demonstration saved to diffusion_autoencoder_demo.png")
    
    return test_images, noisy_images, results


def generate_samples(diffusion_model, feature_vectors, device, save_path=None, num_steps=20):
    """Generate and optionally save samples using the diffusion model."""
    diffusion_model.eval()
    
    with torch.no_grad():
        # Move feature vectors to device
        if isinstance(feature_vectors, np.ndarray):
            feature_vectors = torch.from_numpy(feature_vectors).float()
        feature_vectors = feature_vectors.to(device)
        
        # Generate images
        generated_images = diffusion_model.generate(
            feature_vectors=feature_vectors,
            batch_size=feature_vectors.size(0),
            num_steps=num_steps
        )
        
        # Plot images
        plt = plot_images(generated_images, "Generated Samples")
        
        # Save the figure if path is provided
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Samples saved to {save_path}")
        else:
            plt.show()
        
        return generated_images


def generate_conditioned_samples(diffusion_model, autoencoder, device, num_pairs=5):
    """Generate conditioned samples using different test images."""
    # Load test data
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Sample some test images
    indices = np.random.choice(len(test_dataset), num_pairs, replace=False)
    test_images = []
    for idx in indices:
        image, _ = test_dataset[idx]
        test_images.append(image)
    
    # Convert to tensor
    test_images = torch.stack(test_images).to(device)
    
    # Get feature vectors
    with torch.no_grad():
        feature_vectors = autoencoder.encode(test_images)
    
    # Generate images from these feature vectors
    with torch.no_grad():
        generated_images = diffusion_model.generate(
            feature_vectors=feature_vectors,
            batch_size=num_pairs,
            num_steps=50
        )
    
    # Convert tensors to numpy for plotting
    test_images_np = test_images.cpu().numpy()
    generated_images_np = generated_images.cpu().numpy()
    
    # Plot originals and generations side by side
    plt.figure(figsize=(num_pairs * 2, 4))
    for i in range(num_pairs):
        # Original image
        plt.subplot(2, num_pairs, i + 1)
        plt.imshow(test_images_np[i].squeeze(), cmap="gray")
        plt.title("Original")
        plt.axis("off")
        
        # Generated image
        plt.subplot(2, num_pairs, i + 1 + num_pairs)
        plt.imshow(generated_images_np[i].squeeze(), cmap="gray")
        plt.title("Generated")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("conditioned_samples.png")
    plt.close()
    print("Conditioned samples saved to conditioned_samples.png")


if __name__ == "__main__":
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Train or load autoencoder
    if os.path.exists("autoencoder_weights.pt"):
        print("Loading existing autoencoder...")
        autoencoder = SimpleAutoencoder(device=device)
        autoencoder.load_checkpoint("autoencoder_weights.pt")
    else:
        autoencoder = train_autoencoder(device)
    
    # Prepare dataset with feature vectors
    train_loader, test_loader, test_features = prepare_dataset(autoencoder, device)
    
    # Train or load diffusion model
    if os.path.exists("diffusion_model.pt"):
        print("Loading existing diffusion model...")
        diffusion_model = DiffusionModel(device=device)
        diffusion_model.load_checkpoint("diffusion_model.pt")
    else:
        diffusion_model = train_diffusion_model(train_loader, test_loader, test_features, device)
    
    # Generate conditioned samples
    generate_conditioned_samples(diffusion_model, autoencoder, device)
    
    # Demonstrate diffusion autoencoder capabilities
    demonstrate_diffusion_autoencoder(diffusion_model, autoencoder, device)
    
    print("Done! Check the 'samples' directory, 'conditioned_samples.png', and 'diffusion_autoencoder_demo.png' for results.") 