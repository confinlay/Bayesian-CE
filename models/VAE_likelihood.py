import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import numpy as np
import os

class ConvVAELikelihoodEstimator(nn.Module):
    """
    A Convolutional Variational Autoencoder for MNIST that can compute
    importance-weighted estimates of the log-likelihood for assessing
    the realism of images.
    """
    
    def __init__(self, latent_dim=20, device='cpu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        
        # Encoder (q(z|x))
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Fully connected layers for mean and variance
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_var = nn.Linear(64 * 7 * 7, latent_dim)
        
        # Decoder first layer (from latent to initial volume)
        self.decoder_input = nn.Linear(latent_dim, 64 * 7 * 7)
        
        # Decoder (p(x|z))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # For MNIST (binary/grayscale pixels)
        )
        
        self.to(device)
    
    def encode(self, x):
        """Encode input into parameters of q(z|x)"""
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        log_var = self.fc_var(h_flat)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Sample z from q(z|x) using the reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode z to reconstruct input x"""
        h = self.decoder_input(z)
        h = h.view(-1, 64, 7, 7)  # Reshape to spatial volume
        x_recon = self.decoder(h)
        return x_recon
    
    def forward(self, x):
        """Full VAE forward pass"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def log_likelihood(self, x, k=100):
        """
        Compute a k-sample importance-weighted estimate of log p(x),
        which gives a tighter lower bound than ELBO.
        
        Args:
            x: Input image tensor [batch_size, 1, 28, 28]
            k: Number of importance samples
            
        Returns:
            log_likelihood: Estimated log-likelihood for each input
        """
        batch_size = x.size(0)
        
        # Create a repeat function for expanding tensors along batch dimension
        def expand_for_iw(tensor):
            return tensor.unsqueeze(1).expand(-1, k, *tensor.shape[1:]).reshape(batch_size * k, *tensor.shape[1:])
        
        # Expand input for importance weighting
        x_expanded = expand_for_iw(x)
        
        with torch.no_grad():
            # Encode and sample z
            mu, log_var = self.encode(x_expanded)
            z = self.reparameterize(mu, log_var)
            
            # Decode to get p(x|z)
            x_recon = self.decode(z)
            
            # Flatten images for easier computations
            x_expanded_flat = x_expanded.view(batch_size * k, -1)
            x_recon_flat = x_recon.view(batch_size * k, -1)
            
            # Compute log p(x|z) - Bernoulli likelihood for MNIST
            log_p_x_given_z = torch.sum(
                x_expanded_flat * torch.log(x_recon_flat + 1e-8) + 
                (1 - x_expanded_flat) * torch.log(1 - x_recon_flat + 1e-8),
                dim=1
            )
            
            # Compute log p(z) - prior
            log_p_z = torch.sum(
                -0.5 * torch.pow(z, 2) - 0.5 * np.log(2 * np.pi),
                dim=1
            )
            
            # Compute log q(z|x) - encoder distribution
            log_q_z_given_x = torch.sum(
                -0.5 * (log_var + torch.pow(z - mu, 2) / torch.exp(log_var)) - 
                0.5 * np.log(2 * np.pi),
                dim=1
            )
            
            # Compute importance weights: log w = log p(x|z) + log p(z) - log q(z|x)
            log_w = log_p_x_given_z + log_p_z - log_q_z_given_x
            
            # Reshape to [batch_size, k]
            log_w = log_w.view(batch_size, k)
            
            # Compute log-sum-exp trick for numerical stability
            log_likelihood = torch.logsumexp(log_w, dim=1) - np.log(k)
            
        return log_likelihood
    
    def compute_realism_score(self, x, normalized=True):
        """
        Compute a realism score based on the estimated log-likelihood.
        
        Args:
            x: Input image tensor [batch_size, 1, 28, 28]
            normalized: If True, return a score between 0 and 1
                        If False, return the raw log-likelihood
                        
        Returns:
            realism_score: Higher values indicate more realistic images
        """
        log_likelihood = self.log_likelihood(x)
        
        if normalized:
            # Normalize to [0,1] using a sigmoid transformation
            # The constant may need adjustment based on your trained model
            realism_score = torch.sigmoid((log_likelihood + 300) / 50)
            return realism_score
        
        return log_likelihood


def train_vae_for_likelihood(trainloader, val_loader=None, epochs=20, latent_dim=20, 
                            device='cuda', lr=1e-3, patience=5, model_saves_dir=None):
    """
    Train a convolutional VAE on MNIST for likelihood estimation with optimal training strategies
    
    Args:
        trainloader: DataLoader with MNIST training data
        val_loader: Optional validation DataLoader
        epochs: Number of training epochs
        latent_dim: Dimension of the latent space
        device: 'cpu' or 'cuda'
        lr: Learning rate
        patience: Number of epochs to wait before early stopping
        model_saves_dir: Directory to save the best model
        
    Returns:
        vae: Trained VAE model
    """
    vae = ConvVAELikelihoodEstimator(latent_dim=latent_dim, device=device).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                    patience=2, verbose=True)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        vae.train()
        train_loss = 0
        recon_loss = 0
        kl_loss = 0
        
        for batch_idx, (data, _) in enumerate(trainloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, log_var = vae(data)
            
            # Compute loss
            # Reconstruction loss (binary cross entropy for MNIST)
            BCE = F.binary_cross_entropy(
                recon_batch.view(-1, 784), 
                data.view(-1, 784), 
                reduction='sum'
            )
            
            # KL divergence
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Total loss
            loss = BCE + KLD
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss += BCE.item()
            kl_loss += KLD.item()
        
        # Calculate average training losses
        avg_train_loss = train_loss / len(trainloader.dataset)
        avg_recon = recon_loss / len(trainloader.dataset)
        avg_kl = kl_loss / len(trainloader.dataset)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {avg_train_loss:.4f} '
              f'(Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})')
        
        # Validation phase
        val_loss = avg_train_loss  # Default to train loss if no validation set
        if val_loader is not None:
            vae.eval()
            val_running_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(device)
                    recon_batch, mu, log_var = vae(data)
                    
                    # Compute validation loss
                    BCE = F.binary_cross_entropy(
                        recon_batch.view(-1, 784), 
                        data.view(-1, 784), 
                        reduction='sum'
                    )
                    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                    loss = BCE + KLD
                    
                    val_running_loss += loss.item()
                    val_recon_loss += BCE.item()
                    val_kl_loss += KLD.item()
            
            val_loss = val_running_loss / len(val_loader.dataset)
            val_recon = val_recon_loss / len(val_loader.dataset)
            val_kl = val_kl_loss / len(val_loader.dataset)
            
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Val Loss: {val_loss:.4f} '
                  f'(Recon: {val_recon:.4f}, KL: {val_kl:.4f})')
        
        # Learning rate scheduling based on validation loss
        scheduler.step(val_loss)
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            
            # Save the best model
            if model_saves_dir:
                os.makedirs(model_saves_dir, exist_ok=True)
                save_path = os.path.join(model_saves_dir, f"vae_likelihood_estimator_{latent_dim}.pt")
                torch.save(vae.state_dict(), save_path)
                print(f"Saved best model to: {save_path}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return vae