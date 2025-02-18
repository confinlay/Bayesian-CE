import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from sampler import H_SA_SGHMC

class BayesianLastLayerCat(nn.Module):
    """
    A Bayesian last-layer model that handles both the deterministic backbone
    and the Bayesian linear layer. The backbone is frozen during SGHMC sampling
    of the last layer's posterior.
    """

    def __init__(self, backbone, input_dim, output_dim, N_train,
                 lr=1e-2, base_C=0.05, gauss_sig=0.1,
                 device=None, seed=42, schedule=None):
        """
        Args:
            backbone     : Pretrained nn.Module up to penultimate layer.
            input_dim   : Dimension of backbone's output (penultimate features).
            output_dim  : Number of classes (categorical outputs).
            N_train    : Size of training set (for scaling the loss).
            lr, base_C, gauss_sig : Hyperparameters for SGHMC.
            device     : 'cpu', 'cuda', or 'mps' (if available).
            seed      : Random seed for reproducibility.
            schedule  : List of epochs at which to decay learning rate.
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

        # Set random seeds
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)

        # Store training parameters
        self.N_train = N_train
        self.lr = lr  # Store initial learning rate
        self.schedule = schedule  # Store learning rate schedule
        self.epoch = 0  # Initialize epoch counter

        # Set up model components
        self.backbone = backbone.to(self.device)
        self.last_layer = nn.Linear(input_dim, output_dim).to(self.device)
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # SGHMC optimizer (only for last layer)
        self.optimizer = H_SA_SGHMC(self.last_layer.parameters(),
                                  lr=lr, base_C=base_C, gauss_sig=gauss_sig)

        # Initialize ensemble storage
        self.ensemble_last_layers = []
        
        # Register a buffer for tracking best validation performance
        self.register_buffer('best_val_loss', torch.tensor(float('inf')))
        
        # Optional: gradient tracking for adaptive clipping
        self.grad_history = []
        self.max_grad = 1e20
        self.grad_std_mul = 30

    def forward(self, x):
        """Full forward pass through backbone and last layer."""
        features, _ = self.backbone(x.to(self.device))
        return self.last_layer(features)

    @torch.no_grad()
    def extract_features(self, x):
        """Extract features from the frozen backbone."""
        self.backbone.eval()
        with torch.no_grad():
            output = self.backbone(x.to(self.device))
            # Handle both single tensor and tuple outputs
            if isinstance(output, tuple):
                features, _ = output
            else:
                features = output
            return features

    def fit(self, x, y, burn_in=False, resample_momentum=False, resample_prior=False):
        """Single SGHMC update step."""
        self.train()
        # Ensure y is on correct device and type
        y = y.long().to(self.device)
        
        features = self.extract_features(x)
        
        # Forward pass through last layer using extracted features
        self.optimizer.zero_grad()
        logits = self.last_layer(features)
        
        # Compute scaled loss (for proper SGHMC gradient scaling)
        loss = F.cross_entropy(logits, y, reduction='mean') * self.N_train
        loss.backward()
        
        # Gradient clipping (optional)
        if self.grad_history is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.last_layer.parameters(),
                max_norm=self.max_grad
            )
            self.grad_history.append(float(grad_norm))
            
            # Update max_grad adaptively
            if len(self.grad_history) > 1000:
                grad_tensor = torch.tensor(self.grad_history)
                self.max_grad = float(
                    grad_tensor.mean() + self.grad_std_mul * grad_tensor.std()
                )
                self.grad_history.pop(0)
        
        # SGHMC step
        self.optimizer.step(
            burn_in=burn_in,
            resample_momentum=resample_momentum,
            resample_prior=resample_prior
        )
        
        # Compute metrics
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            err = preds.ne(y).sum().item()
            
        return loss.item() / self.N_train, err

    @torch.no_grad()
    def evaluate_full_dataset(self, dataloader):
        """Evaluate the current model on a full dataset."""
        self.eval()
        total_loss = 0
        total_err = 0
        n_samples = 0
        
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            features = self.extract_features(x)
            logits = self.last_layer(features)
            
            loss = F.cross_entropy(logits, y, reduction='sum')
            preds = logits.argmax(dim=1)
            err = preds.ne(y).sum()
            
            total_loss += loss.item()
            total_err += err.item()
            n_samples += x.size(0)
            
        return total_loss / n_samples, total_err / n_samples
    
    def evaluate(self, x, y):
        """Evaluate the model on a batch and return cost, error, and predictions."""
        self.eval()
        x, y = x.to(self.device), y.long().to(self.device)
        with torch.no_grad():
            features = self.extract_features(x)
            logits = self.last_layer(features)
            cost = F.cross_entropy(logits, y, reduction='sum')
            _, predicted = logits.max(1)
            err = (predicted != y).sum().item()
            probs = F.softmax(logits, dim=1)
        return cost.item(), err, probs

    def save_sampled_net(self, max_samples=None):
        """Store the current last layer as a posterior sample."""
        if max_samples and len(self.ensemble_last_layers) >= max_samples:
            self.ensemble_last_layers.pop(0)
            
        net_copy = copy.deepcopy(self.last_layer)
        net_copy.eval()
        self.ensemble_last_layers.append(net_copy)
        
        print(f" [save_sampled_net] Ensemble size = {len(self.ensemble_last_layers)}")
    
    @torch.no_grad()
    def sample_predict(self, x, Nsamples=None):
        """
        Make predictions using all ensemble members.
        
        Args:
            x: Input tensor
            Nsamples: Number of samples to use (if None, uses all samples)
        
        Returns:
            prob_stack: Tensor of shape [n_ensemble, batch_size, n_classes]
                       containing softmax probabilities from each ensemble member
        """
        self.eval()
        features = self.extract_features(x)
        
        # Determine how many samples to use
        if Nsamples is not None:
            ensemble = self.ensemble_last_layers[:Nsamples]
        else:
            ensemble = self.ensemble_last_layers
        
        # Collect predictions from all ensemble members
        all_probs = []
        for last_layer in ensemble:
            logits = last_layer(features)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)
        
        # Stack into [n_ensemble, batch_size, n_classes]
        prob_stack = torch.stack(all_probs, dim=0)
        
        return prob_stack

    @torch.no_grad()
    def predict_with_uncertainty(self, x):
        """
        Make predictions using the entire ensemble.
        Returns:
            mean_probs: Average probabilities across ensemble
            uncertainty: Dictionary containing total, aleatoric and epistemic uncertainty
        """
        self.eval()
        features = self.extract_features(x)
        
        # Collect predictions from all ensemble members
        all_probs = []
        for last_layer in self.ensemble_last_layers:
            logits = last_layer(features)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)
            
        # Stack into [n_ensemble, batch_size, n_classes]
        prob_stack = torch.stack(all_probs, dim=0)
        
        # Compute mean probabilities
        mean_probs = prob_stack.mean(dim=0)
        
        # Compute uncertainty decomposition (total, aleatoric, epistemic)
        eps = 1e-10
        total_entropy = -(mean_probs * torch.log(mean_probs + eps)).sum(dim=1)
        
        sample_entropy = -(prob_stack * torch.log(prob_stack + eps)).sum(dim=2)
        aleatoric_entropy = sample_entropy.mean(dim=0)
        
        epistemic_entropy = total_entropy - aleatoric_entropy
        
        return mean_probs, {
            'total_entropy': total_entropy,
            'aleatoric_entropy': aleatoric_entropy, 
            'epistemic_entropy': epistemic_entropy
        }

    def save_checkpoint(self, path):
        """Save model checkpoint including backbone, current last layer and ensemble."""
        checkpoint = {
            'backbone_state': self.backbone.state_dict(),
            'current_last_layer': self.last_layer.state_dict(),
            'ensemble_last_layers': [ll.state_dict() for ll in self.ensemble_last_layers],
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epoch': self.epoch,
            'lr': self.lr
        }
        torch.save(checkpoint, path)
        print(f" [save_checkpoint] Saved model state to {path}")

    def load_checkpoint(self, path):
        """Load a saved checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.backbone.load_state_dict(checkpoint['backbone_state'])
        self.last_layer.load_state_dict(checkpoint['current_last_layer'])
        
        # Rebuild ensemble
        self.ensemble_last_layers = []
        for state_dict in checkpoint['ensemble_last_layers']:
            layer = copy.deepcopy(self.last_layer)
            layer.load_state_dict(state_dict)
            layer.eval()
            self.ensemble_last_layers.append(layer)
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.best_val_loss.copy_(checkpoint['best_val_loss'])
        self.epoch = checkpoint['epoch']
        self.lr = checkpoint['lr']
        
        print(f" [load_checkpoint] Loaded checkpoint from {path}")

    def update_lr(self, epoch, gamma=0.99):
        """Update learning rate according to schedule or decay."""
        self.epoch = epoch  # Update current epoch
        
        # Check if we should decay the learning rate
        should_decay = (
            self.schedule is None or  # Always decay if no schedule
            len(self.schedule) == 0 or  # Always decay if empty schedule
            epoch in self.schedule  # Decay at scheduled epochs
        )
        
        if should_decay:
            self.lr *= gamma
            print(f" [update_lr] Learning rate: {self.lr:.6f}  (epoch {epoch})")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    def save_weights(self, path):
        """Save the current model state dict and ensemble samples."""
        save_dict = {
            'backbone_state': self.backbone.state_dict(),
            'current_last_layer': self.last_layer.state_dict(),
            'ensemble_last_layers': [ll.state_dict() for ll in self.ensemble_last_layers]
        }
        torch.save(save_dict, path)
        print(f" [save_weights] Saved model state and {len(self.ensemble_last_layers)} ensemble samples to {path}")

    
