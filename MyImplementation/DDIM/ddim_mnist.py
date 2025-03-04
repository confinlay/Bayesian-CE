import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):
    """Residual block for the U-Net architecture."""
    def __init__(self, in_channels, out_channels, time_emb_dim, feature_emb_dim):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time and feature projections
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.feature_proj = nn.Linear(feature_emb_dim, out_channels)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection if input channels don't match output channels
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, t_emb, f_emb):
        # First convolution block
        h = self.norm1(x)
        h = F.silu(h)  # SiLU activation (aka Swish): x * sigmoid(x)
        h = self.conv1(h)
        
        # Add time and feature conditioning
        time_bias = self.time_proj(t_emb)[:, :, None, None]  # Add spatial dimensions
        feature_bias = self.feature_proj(f_emb)[:, :, None, None]  # Add spatial dimensions
        
        # Add time and feature embeddings
        h = h + time_bias + feature_bias
        
        # Second convolution block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        # Skip connection
        x = self.skip(x)
            
        return x + h


class DownSample(nn.Module):
    """Downsampling layer with convolution."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    """Upsampling layer with convolution."""
    def __init__(self, channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class DiffusionModel(nn.Module):
    """
    Minimal DDIM (Denoising Diffusion Implicit Model) for MNIST dataset
    that conditions on feature vectors from an autoencoder.
    
    This implements a diffusion autoencoder approach where:
    1. An image is encoded to get a latent vector
    2. The original image has noise added at various timesteps
    3. The model denoises the image conditioned on the latent vector
    
    This allows for latent space manipulation - you can perturb the latent
    vector and observe how it affects the denoising process.
    """
    def __init__(
        self,
        time_embedding_dim=32,
        feature_embedding_dim=64,
        image_size=28,
        widths=[32, 64, 128],
        block_depth=2,
        device=None
    ):
        super().__init__()
        
        self.image_size = image_size
        
        # Handle device setup with MPS support
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        print(f"DiffusionModel using device: {self.device}")
        
        # Register buffers for normalization statistics
        self.register_buffer('img_mean', torch.tensor([0.0]))
        self.register_buffer('img_std', torch.tensor([1.0]))
        
        # Time embedding layers (for diffusion timestep)
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
        )
        
        # Feature embedding layers (for conditioning)
        self.feature_embed = nn.Sequential(
            nn.Linear(256, feature_embedding_dim),  # Assuming 256-dim features
            nn.SiLU(),
            nn.Linear(feature_embedding_dim, feature_embedding_dim),
            nn.SiLU(),
        )
        
        # U-Net encoder (downsampling)
        self.encoder_blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        # Initial convolution to get to the first width
        self.initial_conv = nn.Conv2d(1, widths[0], kernel_size=1)
        
        # Encoder path
        in_channels = widths[0]
        for i, width in enumerate(widths):
            # Add residual blocks
            blocks = nn.ModuleList()
            for _ in range(block_depth):
                blocks.append(ResidualBlock(in_channels, width, time_embedding_dim, feature_embedding_dim))
                in_channels = width
            self.encoder_blocks.append(blocks)
            
            # Add skip connections
            self.skips.append(nn.Conv2d(width, width, kernel_size=1))
            
            # Add downsampling except for the last block
            if i < len(widths) - 1:
                self.downsamples.append(DownSample(width))
                
        # U-Net bottleneck
        self.bottleneck_blocks = nn.ModuleList([
            ResidualBlock(widths[-1], widths[-1], time_embedding_dim, feature_embedding_dim),
            ResidualBlock(widths[-1], widths[-1], time_embedding_dim, feature_embedding_dim),
        ])
        
        # U-Net decoder (upsampling)
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # Decoder path
        for i, width in enumerate(reversed(widths)):
            # Add upsampling except for the first block
            if i > 0:
                self.upsamples.append(UpSample(width))
            
            # Add residual blocks with skip connections from encoder
            blocks = nn.ModuleList()
            in_channels = width * 2 if i > 0 else width  # Double channels due to skip connection
            for j in range(block_depth + 1):
                if j == 0 and i > 0:
                    # First block after skip connection has double input channels
                    blocks.append(ResidualBlock(in_channels, width, time_embedding_dim, feature_embedding_dim))
                else:
                    blocks.append(ResidualBlock(width, width, time_embedding_dim, feature_embedding_dim))
            self.decoder_blocks.append(blocks)
                
        # Output layer
        self.final_conv = nn.Conv2d(widths[0], 1, kernel_size=1)
        
        # Move model to device
        self.to(self.device)
    
    def _diffusion_schedule(self, diffusion_times):
        """Returns noise schedules for training and sampling."""
        # Cosine schedule as proposed in the improved DDPM paper
        start_angle = torch.tensor(0.999, device=self.device).acos()
        end_angle = torch.tensor(0.01, device=self.device).acos()
        
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        alphas = diffusion_angles.cos() ** 2
        alphas_prev = torch.cat([torch.ones_like(alphas[:, :1]), alphas[:, :-1]], dim=1)
        
        return alphas, alphas_prev
    
    def normalize(self, images):
        """Normalize images to mean 0 and std 1."""
        return (images - self.img_mean) / self.img_std
    
    def denormalize(self, images):
        """Convert normalized images back to pixel values."""
        images = images * self.img_std + self.img_mean
        return torch.clamp(images, 0.0, 1.0)
    
    def adapt_input_data(self, data_loader):
        """Compute mean and std from a data loader."""
        print("Computing dataset statistics...")
        
        # Get a batch of images
        images, _ = next(iter(data_loader))  # Assuming (images, features) tuple
        
        # Compute statistics
        self.img_mean = images.mean().to(self.device)
        self.img_std = images.std().to(self.device)
        
        print(f"Dataset statistics - Mean: {self.img_mean.item():.4f}, Std: {self.img_std.item():.4f}")
    
    def forward(self, noisy_images, diffusion_times, feature_vectors):
        """Forward pass through the U-Net denoising model."""
        # Embed diffusion timesteps
        t_emb = self.time_embed(diffusion_times)
        
        # Embed feature vectors for conditioning
        f_emb = self.feature_embed(feature_vectors)
        
        # Initial convolution
        x = self.initial_conv(noisy_images)
        
        # U-Net encoder (downsampling path)
        skip_outputs = []
        for i, blocks in enumerate(self.encoder_blocks):
            # Process through residual blocks
            for block in blocks:
                x = block(x, t_emb, f_emb)
            
            # Store skip connection
            skip_outputs.append(self.skips[i](x))
            
            # Downsample if not the last block
            if i < len(self.encoder_blocks) - 1:
                x = self.downsamples[i](x)
        
        # U-Net bottleneck
        for block in self.bottleneck_blocks:
            x = block(x, t_emb, f_emb)
        
        # U-Net decoder (upsampling path) with skip connections
        for i, blocks in enumerate(self.decoder_blocks):
            # Upsample if not the first block
            if i > 0:
                x = self.upsamples[i-1](x)
                # Add skip connection
                skip_index = len(skip_outputs) - i
                x = torch.cat([x, skip_outputs[skip_index]], dim=1)
            
            # Process through residual blocks
            for block in blocks:
                x = block(x, t_emb, f_emb)
        
        # Output layer
        x = self.final_conv(x)
        return x
    
    def training_step(self, batch, optimizer):
        """Training step for the diffusion model."""
        images, feature_vectors = batch
        images = images.to(self.device)
        feature_vectors = feature_vectors.to(self.device)
        
        # Normalize images
        images = self.normalize(images)
        
        # Sample uniform random diffusion times
        batch_size = images.shape[0]
        diffusion_times = torch.rand(size=(batch_size, 1), device=self.device)
        
        # Get alphas for chosen diffusion times
        alphas, _ = self._diffusion_schedule(diffusion_times)
        
        # Generate random noise and add to images
        noise = torch.randn_like(images)
        noisy_images = torch.sqrt(alphas) * images + torch.sqrt(1 - alphas) * noise
        
        # Train the network to predict the noise
        pred_noise = self(noisy_images, diffusion_times, feature_vectors)
        loss = F.mse_loss(noise, pred_noise)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def add_noise(self, images, diffusion_time):
        """Add noise to images at a specific diffusion time.
        
        Args:
            images: Clean images to add noise to
            diffusion_time: Value between 0.0 and 1.0 indicating noise level
            
        Returns:
            Noisy images
        """
        # Ensure inputs are on the correct device
        images = images.to(self.device)
        if isinstance(diffusion_time, float):
            diffusion_time = torch.tensor([[diffusion_time]], device=self.device)
        else:
            diffusion_time = diffusion_time.to(self.device)
            
        # Normalize images
        images = self.normalize(images)
        
        # Get alpha values for the diffusion time
        alphas, _ = self._diffusion_schedule(diffusion_time)
        
        # Add noise
        noise = torch.randn_like(images)
        return torch.sqrt(alphas) * images + torch.sqrt(1 - alphas) * noise, noise
    
    @torch.no_grad()
    def generate(
        self,
        feature_vectors,
        batch_size=16,
        num_steps=50,
        progress_callback=None,
    ):
        """Generate images using DDIM sampling.
        
        Args:
            feature_vectors: Latent vectors from autoencoder to condition on
            batch_size: Number of images to generate
            num_steps: Number of denoising steps to perform
            progress_callback: Optional callback function to report progress
            
        Returns:
            Generated images
        """
        # Move feature vectors to device
        feature_vectors = feature_vectors.to(self.device)
        
        # Start with random noise
        generated_images = torch.randn(
            (batch_size, 1, self.image_size, self.image_size), 
            device=self.device
        )
        
        # Setup timesteps for DDIM sampling
        diffusion_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        
        # DDIM sampling loop
        for step, t in enumerate(diffusion_steps[:-1]):
            # Current and next diffusion time
            diffusion_time = torch.ones((batch_size, 1), device=self.device) * t
            next_diffusion_time = torch.ones((batch_size, 1), device=self.device) * diffusion_steps[step + 1]
            
            # Get schedule values
            alphas, _ = self._diffusion_schedule(diffusion_time)
            alphas_next, _ = self._diffusion_schedule(next_diffusion_time)
            
            # Predict noise for current step
            pred_noise = self(generated_images, diffusion_time, feature_vectors)
            
            # DDIM formula for implicit sampling
            x_0_pred = (generated_images - torch.sqrt(1 - alphas) * pred_noise) / torch.sqrt(alphas)
            
            # Clamp predicted clean images for stability
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
            
            # Get the sample for the next step using DDIM formulation
            coeff1 = torch.sqrt(alphas_next)
            coeff2 = torch.sqrt(1 - alphas_next)
            
            generated_images = coeff1 * x_0_pred + coeff2 * pred_noise
            
            # Report progress
            if progress_callback is not None and step % (num_steps // 10) == 0:
                progress_callback(step, self.denormalize(x_0_pred))
        
        return self.denormalize(generated_images)
    
    @torch.no_grad()
    def denoise_with_latent(self, noisy_images, latent_vectors, diffusion_time, num_steps=20):
        """Denoise images at a specific noise level conditioned on latent vectors.
        
        This is the key function for the diffusion autoencoder approach, letting you:
        1. Start with a noisy image 
        2. Control the denoising process with a latent vector
        3. Potentially perturb the latent vector to see how it affects the output
        
        Args:
            noisy_images: Images with noise added
            latent_vectors: Latent vectors to condition on (can be perturbed)
            diffusion_time: Value between 0.0 and 1.0 indicating starting noise level
            num_steps: Number of denoising steps to perform
            
        Returns:
            Denoised images
        """
        # Ensure inputs are on the right device
        noisy_images = noisy_images.to(self.device)
        latent_vectors = latent_vectors.to(self.device)
        
        if isinstance(diffusion_time, float):
            diffusion_time = torch.tensor(diffusion_time, device=self.device)
        else:
            diffusion_time = diffusion_time.to(self.device)
            
        # Start with the noisy images
        batch_size = noisy_images.shape[0]
        denoised_images = noisy_images.clone()
        
        # Setup timesteps from diffusion_time to 0
        diffusion_steps = torch.linspace(diffusion_time.item(), 0.0, num_steps + 1, device=self.device)
        
        # DDIM sampling loop
        for step, t in enumerate(diffusion_steps[:-1]):
            # Current and next diffusion time
            current_time = torch.ones((batch_size, 1), device=self.device) * t
            next_time = torch.ones((batch_size, 1), device=self.device) * diffusion_steps[step + 1]
            
            # Get schedule values
            alphas, _ = self._diffusion_schedule(current_time)
            alphas_next, _ = self._diffusion_schedule(next_time)
            
            # Predict noise for current step
            pred_noise = self(denoised_images, current_time, latent_vectors)
            
            # DDIM formula for implicit sampling
            x_0_pred = (denoised_images - torch.sqrt(1 - alphas) * pred_noise) / torch.sqrt(alphas)
            
            # Clamp predicted clean images for stability
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
            
            # Get the sample for the next step using DDIM formulation
            coeff1 = torch.sqrt(alphas_next)
            coeff2 = torch.sqrt(1 - alphas_next)
            
            denoised_images = coeff1 * x_0_pred + coeff2 * pred_noise
            
        return self.denormalize(denoised_images)
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'img_mean': self.img_mean,
            'img_std': self.img_std,
        }, path)
        print(f"Model saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.img_mean = checkpoint.get('img_mean', torch.tensor([0.0], device=self.device))
        self.img_std = checkpoint.get('img_std', torch.tensor([1.0], device=self.device))
        self.eval()
        print(f"Model loaded from {path}")


class SimpleAutoencoder(nn.Module):
    """
    Simple autoencoder for MNIST to generate feature vectors.
    """
    def __init__(self, latent_dim=256, device=None):
        super().__init__()
        
        # Handle device setup with MPS support
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        print(f"Autoencoder using device: {self.device}")
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        self.to(self.device)
    
    def forward(self, x):
        x = x.to(self.device)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        x = x.to(self.device)
        return self.encoder(x)
    
    def decode(self, z):
        z = z.to(self.device)
        return self.decoder(z)
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save(self.state_dict(), path)
        print(f"Autoencoder saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
        print(f"Autoencoder loaded from {path}") 