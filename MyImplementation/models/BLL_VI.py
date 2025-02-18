import torch
import torch.nn as nn
import torch.nn.functional as F
import torchbnn as bnn
import copy
from .torchbnn_functional import bayesian_kl_loss


class BayesianLastLayerVI(nn.Module):
    """
    A Bayesian last-layer model using Variational Inference (VI) via torchbnn.
    The backbone is frozen during training of the last layer's variational posterior.
    """

    def __init__(self, backbone, input_dim, output_dim, prior_mu=0.0, prior_sigma=0.1,
                 kl_weight=0.1, device=None):
        """
        Args:
            backbone: Pretrained nn.Module up to penultimate layer
            input_dim: Dimension of backbone's output (penultimate features)
            output_dim: Number of classes
            prior_mu: Mean of Gaussian prior on weights
            prior_sigma: Standard deviation of Gaussian prior
            kl_weight: Weight of KL divergence term in loss
            device: 'cpu', 'cuda', or 'mps' (if available)
        """
        super().__init__()
        
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # Set up model components
        self.backbone = backbone.to(self.device)
        self.last_layer = bnn.BayesLinear(
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            in_features=input_dim,
            out_features=output_dim
        ).to(self.device)

        self.to(self.device)
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Store hyperparameters
        self.kl_weight = kl_weight
        
        # Set up losses - ensure they're on the correct device
        self.ce_loss = nn.CrossEntropyLoss().to(self.device)
        self.bayesian_kl_loss = bayesian_kl_loss  # Store the function reference
        
        # Register a buffer for tracking best validation performance
        self.register_buffer('best_val_loss', torch.tensor(float('inf')))

    def forward(self, x):
        """Full forward pass through backbone and last layer."""
        features = self.extract_features(x)
        return self.last_layer(features)

    @torch.no_grad()
    def extract_features(self, x):
        """Extract features from the frozen backbone."""
        self.backbone.eval()
        output = self.backbone(x.to(self.device))
        # Handle both single tensor and tuple outputs
        if isinstance(output, tuple):
            features, _ = output
        else:
            features = output
        return features

    def fit(self, x, y):
        """Single training step with VI objective."""
        self.train()  # Ensure we're in training mode for sampling
        y = y.long().to(self.device)
        
        # Forward pass
        features = self.extract_features(x)
        logits = self.last_layer(features)
        
        # Compute losses
        ce_loss = self.ce_loss(logits, y)
        kl_div = self.bayesian_kl_loss(self, reduction='mean')  # Call the function
        loss = ce_loss + self.kl_weight * kl_div
        
        # Compute metrics
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            err = preds.ne(y).sum().item()
            
        # Return loss tensor first for backprop, then metrics
        return loss, err, ce_loss.item(), kl_div.item()

    @torch.no_grad()
    def evaluate(self, x, y, n_samples=10):
        """Evaluate with multiple forward passes for uncertainty."""
        self.train()  
        x, y = x.to(self.device), y.long().to(self.device)
        
        outputs = []
        for _ in range(n_samples):
            features = self.extract_features(x)
            logits = self.last_layer(features)
            outputs.append(logits.unsqueeze(0))
            
        # Stack predictions
        outputs = torch.cat(outputs, dim=0)  # [n_samples, batch_size, n_classes]
        
        # Compute mean prediction and uncertainty
        mean_logits = outputs.mean(dim=0)
        uncertainty = outputs.std(dim=0)
        
        # Compute metrics
        cost = F.cross_entropy(mean_logits, y, reduction='sum')
        _, predicted = mean_logits.max(1)
        err = predicted.ne(y).sum().item()
        probs = F.softmax(mean_logits, dim=1)
        
        return cost.item(), err, probs, uncertainty

    def save_checkpoint(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'backbone_state': self.backbone.state_dict(),
            'last_layer_state': self.last_layer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, path)
        print(f" [save_checkpoint] Saved model state to {path}")

    def load_checkpoint(self, path):
        """Load a saved checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.backbone.load_state_dict(checkpoint['backbone_state'])
        self.last_layer.load_state_dict(checkpoint['last_layer_state'])
        self.best_val_loss.copy_(checkpoint['best_val_loss'])
        
        print(f" [load_checkpoint] Loaded checkpoint from {path}")

    @torch.no_grad()
    def sample_predict_z(self, z, Nsamples=None):
        """Match BLL's interface for CLUE compatibility."""
        self.train()  # Keep in train mode for sampling
        outputs = []
        n_samples = Nsamples if Nsamples is not None else 10
        
        for _ in range(n_samples):
            logits = self.last_layer(z)
            probs = F.softmax(logits, dim=1)
            outputs.append(probs)
        
        return torch.stack(outputs, dim=0)  # [n_samples, batch_size, n_classes] 