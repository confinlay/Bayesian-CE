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
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        # If out_channels is not specified, use the same as in_channels
        out_channels = out_channels or in_channels
        
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # Now explicitly support different input and output channel counts
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class DiffusionModel(nn.Module):
    """Diffusion model with U-Net architecture conditioned on feature vectors."""
    
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
        
        # Set device
        self.device = device if device else (
            torch.device("cuda") if torch.cuda.is_available() else 
            torch.device("mps") if torch.backends.mps.is_available() else 
            torch.device("cpu")
        )
        
        # Store hyperparameters
        self.image_size = image_size
        self.widths = widths
        self.time_embedding_dim = time_embedding_dim
        self.feature_embedding_dim = feature_embedding_dim
        
        # Time step embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        
        # Feature vector embedding for conditioning
        self.feature_embed = nn.Sequential(
            nn.Linear(256, feature_embedding_dim),  # Assuming feature vector dim is 256
            nn.SiLU(),
            nn.Linear(feature_embedding_dim, feature_embedding_dim),
        )
        
        # Initial convolution that maps image to the first feature width
        self.initial_conv = nn.Conv2d(1, widths[0], kernel_size=3, padding=1)
        
        # U-Net encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList([])
        self.downsamples = nn.ModuleList([])
        self.skips = nn.ModuleList([])
        
        # Input width for encoder blocks starts with the first width
        input_width = widths[0]
        
        # For each width in the encoder path
        for i, width in enumerate(widths):
            # Add a series of ResidualBlocks for this width
            blocks = []
            for _ in range(block_depth):
                blocks.append(
                    ResidualBlock(
                        in_channels=input_width,
                        out_channels=width,
                        time_emb_dim=time_embedding_dim,
                        feature_emb_dim=feature_embedding_dim
                    )
                )
                input_width = width
            
            # Add the blocks to encoder_blocks
            self.encoder_blocks.append(nn.ModuleList(blocks))
            
            # Add skip connection (identity if no change in channels needed)
            self.skips.append(
                nn.Conv2d(width, width, kernel_size=1) if i < len(widths) - 1 else nn.Identity()
            )
            
            # Add a downsampling layer except for the last width
            if i < len(widths) - 1:
                self.downsamples.append(DownSample(width))
        
        # U-Net bottleneck
        self.bottleneck_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=widths[-1],
                out_channels=widths[-1],
                time_emb_dim=time_embedding_dim,
                feature_emb_dim=feature_embedding_dim
            )
            for _ in range(block_depth)
        ])
        
        # U-Net decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])
        
        # Create reversed width list for decoder
        decoder_widths = list(reversed(widths))
        
        # For each width in the decoder path (excluding the first/bottleneck)
        for i in range(len(decoder_widths)):
            decoder_blocks = []
            
            # First block in decoder might be the bottleneck
            if i == 0:
                # No skip connection for bottleneck, input channels = output channels 
                in_channels = decoder_widths[i]
            else:
                # After the first block, we need to handle skip connections
                # Input is the output from previous layer + skip connection from encoder
                in_channels = decoder_widths[i-1] + decoder_widths[i]  # Skip connection doubles the channels
                
                # Add upsampler between decoder blocks
                self.upsamples.append(
                    UpSample(
                        in_channels=decoder_widths[i-1],
                        out_channels=decoder_widths[i-1]
                    )
                )
            
            # Add blocks for this decoder stage
            for j in range(block_depth):
                decoder_blocks.append(
                    ResidualBlock(
                        in_channels=in_channels if j == 0 else decoder_widths[i],
                        out_channels=decoder_widths[i],
                        time_emb_dim=time_embedding_dim,
                        feature_emb_dim=feature_embedding_dim
                    )
                )
            
            self.decoder_blocks.append(nn.ModuleList(decoder_blocks))
            
        # Final output convolution to map to a single channel
        self.final_conv = nn.Conv2d(widths[0], 1, kernel_size=3, padding=1)
        
        # Initialize the dataset statistics for normalization
        self.register_buffer("data_mean", torch.tensor(0.1307))
        self.register_buffer("data_std", torch.tensor(0.3081))
        
        # Move model to device
        self.to(self.device)
        self._print_model_summary()
        
    def _print_model_summary(self):
        """Print a summary of the model architecture."""
        print(f"DiffusionModel using device: {self.device}")
    
    def _diffusion_schedule(self, diffusion_times):
        """
        Generates alpha values for the diffusion process at the specified timesteps.
        
        Args:
            diffusion_times: Tensor of shape [batch_size, 1] or [batch_size] with values in [0, 1]
                             where 0 = no noise, 1 = pure noise
                             
        Returns:
            Tensor of alpha values at the diffusion times
        """
        # Ensure diffusion_times is properly shaped
        if diffusion_times.dim() == 1:
            diffusion_times = diffusion_times.unsqueeze(-1)
        
        # Use beta schedule from 1e-4 to 0.02 as per DDPM paper
        start_beta = 1e-4
        end_beta = 0.02
        
        # Linear beta schedule
        betas = start_beta + diffusion_times * (end_beta - start_beta)
        
        # Calculate alpha and alpha_cumprod as in DDPM
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=-1)
        
        return alpha_cumprod.squeeze(-1)
    
    def normalize(self, images):
        """Normalize images to mean 0 and std 1."""
        return (images - self.data_mean) / self.data_std
    
    def denormalize(self, images):
        """Convert normalized images back to pixel values."""
        images = images * self.data_std + self.data_mean
        return torch.clamp(images, 0.0, 1.0)
    
    def adapt_input_data(self, data_loader):
        """Compute mean and std from a data loader."""
        print("Computing dataset statistics...")
        
        # Get a batch of images
        images, _ = next(iter(data_loader))  # Assuming (images, features) tuple
        
        # Compute statistics
        self.data_mean = images.mean().to(self.device)
        self.data_std = images.std().to(self.device)
        
        print(f"Dataset statistics - Mean: {self.data_mean.item():.4f}, Std: {self.data_std.item():.4f}")
    
    def forward(self, noisy_images, diffusion_times, feature_vectors):
        """Forward pass through the U-Net denoising model."""
        # Embed diffusion timesteps
        t_emb = self.time_embed(diffusion_times)
        
        # Embed feature vectors for conditioning
        f_emb = self.feature_embed(feature_vectors)
        
        # Initial convolution
        x = self.initial_conv(noisy_images)
        
        # U-Net encoder (downsampling path)
        skip_connections = []
        
        for i, blocks in enumerate(self.encoder_blocks):
            # Process through residual blocks at this level
            for block in blocks:
                x = block(x, t_emb, f_emb)
            
            # Store skip connection
            skip_connections.append(x)
            
            # Downsample if not the last block
            if i < len(self.encoder_blocks) - 1:
                x = self.downsamples[i](x)
        
        # U-Net bottleneck
        for block in self.bottleneck_blocks:
            x = block(x, t_emb, f_emb)
        
        # U-Net decoder (upsampling path)
        for i, blocks in enumerate(self.decoder_blocks):
            # Skip the upsampling for the first block (bottleneck)
            if i > 0:
                # Upsample
                x = self.upsamples[i-1](x)
                
                # Add skip connection from the corresponding encoder level
                # The connections are in reverse order - we use negative indexing
                skip_idx = len(skip_connections) - i
                skip = skip_connections[skip_idx - 1]
                x = torch.cat([x, skip], dim=1)
            
            # Process through residual blocks at this level
            for block in blocks:
                x = block(x, t_emb, f_emb)
        
        # Final convolution to get output image
        x = self.final_conv(x)
        
        return x
    
    def training_step(self, batch, optimizer):
        """Execute a single training step with a batch of data."""
        # Unpack the batch into images and features
        images, feature_vectors = batch
        images = images.to(self.device)
        feature_vectors = feature_vectors.to(self.device)
        
        # Ensure images have the correct size
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            print(f"Warning: Resizing images from {images.shape} to expected size {self.image_size}x{self.image_size}")
            images = F.interpolate(images, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        # Normalize the images
        images = self.normalize(images)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Sample random timesteps
        batch_size = images.shape[0]
        diffusion_times = torch.rand(batch_size, 1, device=self.device)
        
        # Get noise schedule
        alphas = self._diffusion_schedule(diffusion_times)
        
        # Reshape alphas for proper broadcasting
        alphas_expanded = alphas.view(-1, 1, 1, 1)
        
        # Sample noise
        noise = torch.randn_like(images)

        # Create noisy images
        sqrt_alphas = torch.sqrt(alphas_expanded)
        
        term1 = sqrt_alphas * images
        
        sqrt_1_minus_alphas = torch.sqrt(1 - alphas_expanded)
        
        term2 = sqrt_1_minus_alphas * noise
        
        noisy_images = term1 + term2
        
        # Use the model to predict the noise
        pred_noise = self(noisy_images, diffusion_times, feature_vectors)
        
        # Calculate loss (MSE between predicted and true noise)
        loss = F.mse_loss(pred_noise, noise)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def add_noise(self, images, diffusion_time):
        """
        Add noise to images at a given diffusion timestep.
        
        Args:
            images: Tensor of input images [batch_size, 1, height, width]
            diffusion_time: Float between 0.0 and 1.0 indicating how much noise to add
            
        Returns:
            Tuple of (noisy_images, noise) tensors
        """
        # Ensure images are on the correct device and have the expected size
        images = images.to(self.device)
        
        # Check if the images need resizing
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            print(f"Warning: Resizing images from {images.shape} to expected size {self.image_size}x{self.image_size}")
            images = F.interpolate(images, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        # Normalize images
        images = self.normalize(images)
        
        # Convert diffusion_time to a tensor and reshape for broadcasting
        diffusion_times = torch.tensor([diffusion_time], device=self.device).repeat(images.shape[0])
        
        # Get alpha values from diffusion schedule
        alphas = self._diffusion_schedule(diffusion_times)
        
        # For broadcasting, reshape alphas to [batch_size, 1, 1, 1]
        alphas_reshaped = alphas.view(-1, 1, 1, 1)
        
        # Sample noise from normal distribution
        noise = torch.randn_like(images)
        
        # Apply diffusion at specified timestep: 
        # noisy_image = sqrt(alpha) * image + sqrt(1-alpha) * noise
        sqrt_alphas = torch.sqrt(alphas_reshaped)
        sqrt_one_minus_alphas = torch.sqrt(1.0 - alphas_reshaped)
        
        noisy_images = sqrt_alphas * images + sqrt_one_minus_alphas * noise
        
        return noisy_images, noise
    
    @torch.no_grad()
    def generate(
        self,
        feature_vectors,
        batch_size=16,
        num_steps=50,
        progress_callback=None,
    ):
        """
        Generate images conditioned on feature vectors using DDIM sampling.
        
        Args:
            feature_vectors: Tensor of feature vectors to condition on [batch_size, latent_dim]
            batch_size: Number of images to generate (if feature_vectors is None)
            num_steps: Number of denoising steps
            progress_callback: Optional callback function called after each step
            
        Returns:
            Tensor of generated images [batch_size, 1, height, width]
        """
        # Determine how many images to generate
        if feature_vectors is not None:
            batch_size = feature_vectors.shape[0]
        else:
            # If no feature vectors provided, generate random ones
            feature_vectors = torch.randn(batch_size, 256, device=self.device)
        
        # Ensure feature vectors are on the correct device
        feature_vectors = feature_vectors.to(self.device)
        
        # Start with random noise
        x = torch.randn(batch_size, 1, self.image_size, self.image_size, device=self.device)
        
        # Timesteps for denoising (from 1.0 to 0.0)
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)[:-1]
        
        # DDIM sampling loop
        for i, t in enumerate(timesteps):
            # Create a batch of timestep embeddings
            timestep_batch = torch.ones(batch_size, 1, device=self.device) * t
            
            # Predict noise using the model
            pred_noise = self(x, timestep_batch, feature_vectors)
            
            # Get alpha values for current timestep
            alpha = self._diffusion_schedule(torch.tensor([t.item()], device=self.device)).view(-1, 1, 1, 1)
            
            # Get alpha values for next timestep
            next_t = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(0.0, device=self.device)
            alpha_next = self._diffusion_schedule(torch.tensor([next_t.item()], device=self.device)).view(-1, 1, 1, 1)
            
            # DDIM update step
            x0_pred = (x - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
            x = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * pred_noise
            
            # Call progress callback if provided
            if progress_callback and callable(progress_callback):
                progress_callback(i, num_steps, self.denormalize(x))
        
        # Denormalize the final images
        generated_images = self.denormalize(x)
        
        return generated_images
    
    @torch.no_grad()
    def denoise_with_latent(self, noisy_images, latent_vectors, diffusion_time, num_steps=20):
        """
        Progressively denoise images conditioned on latent vectors.
        
        Args:
            noisy_images: Tensor of noisy images [batch_size, 1, height, width]
            latent_vectors: Tensor of feature vectors to condition on [batch_size, latent_dim]
            diffusion_time: Float between 0.0 and 1.0 indicating the starting noise level
            num_steps: Number of denoising steps
            
        Returns:
            Tensor of denoised images [batch_size, 1, height, width]
        """
        # Ensure inputs are on the correct device
        noisy_images = noisy_images.to(self.device)
        latent_vectors = latent_vectors.to(self.device)
        
        # Check if the images need resizing
        if noisy_images.shape[-1] != self.image_size or noisy_images.shape[-2] != self.image_size:
            print(f"Warning: Resizing images from {noisy_images.shape} to expected size {self.image_size}x{self.image_size}")
            noisy_images = F.interpolate(noisy_images, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
            # Normalize after resizing
            noisy_images = self.normalize(noisy_images)
        
        # Calculate denoising schedule
        start_step = int(diffusion_time * num_steps)
        timesteps = torch.linspace(start_step / num_steps, 0, num_steps - start_step + 1, device=self.device)
        
        # Start with the provided noisy images
        x = noisy_images
        
        # Progressively denoise the images
        for time in timesteps:
            # Create a batch of time tensors
            t_batch = torch.ones(noisy_images.shape[0], 1, device=self.device) * time
            
            # Predict noise using the model
            pred_noise = self(x, t_batch, latent_vectors)
            
            # If not the last step, add the predicted noise
            if time > 0:
                # Calculate the noise scaling factor for this timestep
                alpha = self._diffusion_schedule(torch.tensor([time.item()], device=self.device)).view(-1, 1, 1, 1)
                alpha_prev = self._diffusion_schedule(torch.tensor([max(0, time.item() - 1/num_steps)], device=self.device)).view(-1, 1, 1, 1)
                
                # DDIM update step
                x0_pred = (x - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
                x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * pred_noise
            else:
                # Last step - use the direct prediction
                x = pred_noise
        
        # Denormalize to get pixel values
        denoised_images = self.denormalize(x)
        
        return denoised_images
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'data_mean': self.data_mean,
            'data_std': self.data_std,
        }, path)
        print(f"Model saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.data_mean = checkpoint.get('data_mean', torch.tensor(0.1307, device=self.device))
        self.data_std = checkpoint.get('data_std', torch.tensor(0.3081, device=self.device))
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