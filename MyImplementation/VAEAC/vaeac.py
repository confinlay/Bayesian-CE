import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from src.layers import ResBlock  # Assuming this exists

class VAEAC(nn.Module):
    def __init__(self, input_dim=784, num_classes=10, hidden_dim=400, latent_dim=20):
        """VAEAC with joint (x,y) modeling for counterfactual evaluation"""
        super(VAEAC, self).__init__()
        
        # Joint input dimensions (x + y)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.joint_dim = input_dim + num_classes
        
        # Recognition network (x,y) -> z
        self.recognition_net = nn.Sequential(
            nn.Linear(self.joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.recognition_mu = nn.Linear(hidden_dim, latent_dim)
        self.recognition_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Prior network (masked_x, mask) -> z
        self.prior_net = nn.Sequential(
            nn.Linear(self.joint_dim * 2, hidden_dim),  # x + y + mask
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.prior_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder network (z -> x, y)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.joint_dim),  # Reconstruct both x and y
            nn.Sigmoid()
        )
        
        # Classifier for p(y|x)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=1)
        )
        
        # Hyperparameters for prior regularization
        self.sigma_mu = 1e4
        self.sigma_sigma = 1e-4
        
    def apply_mask(self, x, mask):
        """Apply mask to input tensor. True values in mask are unobserved."""
        masked_x = x.clone()
        masked_x[mask] = 0
        return masked_x
        
    def recognition_encode(self, x):
        """Encode complete input to approximate posterior."""
        h = self.recognition_net(x)
        mu = self.recognition_mu(h)
        logvar = self.recognition_logvar(h)
        return mu, logvar
        
    def prior_encode(self, x, mask):
        """Encode masked input to prior distribution."""
        # Mask input and concatenate with mask
        masked_x = self.apply_mask(x, mask)
        h = torch.cat([masked_x, mask.float()], dim=1)
        h = self.prior_net(h)
        mu = self.prior_mu(h)
        logvar = self.prior_logvar(h)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        """Decode latent vector to reconstruction."""
        return self.decoder(z)
        
    def reg_cost(self, prior_mu, prior_logvar):
        """Matches MNISTconv_bern's implementation exactly"""
        mu_regularizer = -(prior_mu ** 2).sum(-1) / 2 / (self.sigma_mu ** 2)
        sigma_regularizer = (prior_logvar - torch.exp(prior_logvar)).sum(-1) * self.sigma_sigma
        return mu_regularizer + sigma_regularizer
        
    def forward(self, x, y, mask):
        """Forward pass with joint (x,y) handling"""
        # Convert y to one-hot and concatenate with x
        y_oh = F.one_hot(y, self.num_classes).float()
        xy = torch.cat([x.view(x.size(0), -1), y_oh], dim=1)
        
        # Get recognition parameters
        rec_mu, rec_logvar = self.recognition_encode(xy)
        
        # Get prior parameters (mask both x and y)
        prior_mu, prior_logvar = self.prior_encode(xy, mask)
        
        # Sample and decode
        z = self.reparameterize(rec_mu, rec_logvar)
        recon_xy = self.decode(z)
        
        return recon_xy, rec_mu, rec_logvar, prior_mu, prior_logvar
        
    def loss_function(self, recon_x, x, recognition_mu, recognition_logvar, prior_mu, prior_logvar):
        """Compute VAEAC loss function.
        
        Returns:
            total_loss: Combined loss
            reconstruction_loss: BCE reconstruction term
            kl_div: KL divergence term
            reg_loss: Prior parameter regularization
        """
        # Reconstruction loss (binary cross entropy)
        BCE = F.binary_cross_entropy(recon_x, x.view(x.size(0), -1), reduction='sum')
        
        # KL divergence between recognition and prior
        recognition = Normal(recognition_mu, torch.exp(0.5 * recognition_logvar))
        prior = Normal(prior_mu, torch.exp(0.5 * prior_logvar))
        KLD = kl_divergence(recognition, prior).sum()
        
        # Prior parameter regularization
        reg_loss = self.reg_cost(prior_mu, prior_logvar).sum()
        
        return BCE + KLD - reg_loss, BCE, KLD, reg_loss
        
    def sample(self, x, mask, num_samples=1):
        """Sample completions given masked input.
        
        Args:
            x: Input tensor
            mask: Boolean mask (True = unobserved)
            num_samples: Number of samples to generate
            
        Returns:
            samples: [num_samples, batch_size, input_dim] tensor of samples
        """
        self.eval()
        with torch.no_grad():
            x_flat = x.view(x.size(0), -1)
            mask_flat = mask.view(mask.size(0), -1)
            
            # Get prior parameters
            prior_mu, prior_logvar = self.prior_encode(x_flat, mask_flat)
            
            samples = []
            for _ in range(num_samples):
                # Sample from prior
                z = self.reparameterize(prior_mu, prior_logvar)
                # Decode
                sample = self.decode(z)
                samples.append(sample)
                
            return torch.stack(samples)

    def conditional_entropy(self, x, n_samples=100):
        """Compute H(y|x) using Monte Carlo samples"""
        with torch.no_grad():
            # Encode input
            x_flat = x.view(x.size(0), -1)
            z_mu, z_logvar = self.recognition_net(x_flat)
            
            # Sample multiple latent vectors
            probs = []
            for _ in range(n_samples):
                z = self.reparameterize(z_mu, z_logvar)
                class_probs = self.classifier(z)
                probs.append(class_probs)
            
            # Compute entropy
            prob_mean = torch.mean(torch.stack(probs), dim=0)
            entropy = -torch.sum(prob_mean * torch.log(prob_mean + 1e-10), dim=1)
            return entropy

    def evaluate_counterfactual(self, x_orig, y_orig, x_cf, y_cf, mask):
        """Compute paper's evaluation metrics"""
        # Compute log p(x_cf,y_cf)
        xy_cf = torch.cat([x_cf, F.one_hot(y_cf, self.num_classes).float()], 1)
        recon, _, _, _, _ = self(x_cf, y_cf, mask)
        log_p = -F.binary_cross_entropy(recon, xy_cf, reduction='sum')
        
        # Compute distance metrics
        l1_dist = torch.norm(x_orig - x_cf, p=1, dim=1)
        
        # Compute uncertainty reduction
        h_orig = self.conditional_entropy(x_orig)
        h_cf = self.conditional_entropy(x_cf)
        
        return {
            'log_p(xy_cf)': log_p,
            'l1_distance': l1_dist,
            'delta_entropy': h_orig - h_cf
        }

if __name__ == '__main__':
    # Test VAEAC on random MNIST-like data
    model = VAEAC()
    x = torch.randn(16, 1, 28, 28)  # batch of 16 MNIST images
    # Create random mask (True = unobserved)
    mask = torch.rand(16, 1, 28, 28) > 0.5
    
    # Forward pass
    recon_x, rec_mu, rec_logvar, prior_mu, prior_logvar = model(x, mask)
    loss, bce, kld, reg = model.loss_function(recon_x, x, rec_mu, rec_logvar, prior_mu, prior_logvar)
    print(f'Loss: {loss.item():.2f}, BCE: {bce.item():.2f}, KLD: {kld.item():.2f}, Reg: {reg.item():.2f}')
    
    # Test sampling
    samples = model.sample(x, mask, num_samples=5)
    print(f'Generated {samples.shape[0]} samples of shape {samples.shape[1:]}') 