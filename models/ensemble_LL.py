import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split

class EnsembleLastLayer(nn.Module):
    """
    An Ensemble of Last Layers with the same interface as BayesianLastLayerVI.
    The backbone is frozen during training of the ensemble members.
    """

    def __init__(self, backbone, input_dim, output_dim, n_members=10, device=None):
        """
        Args:
            backbone: Pretrained nn.Module up to penultimate layer
            input_dim: Dimension of backbone's output (penultimate features)
            output_dim: Number of classes
            n_members: Number of ensemble members
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
        
        # Create ensemble of last layers
        self.ensemble = nn.ModuleList([
            nn.Linear(input_dim, output_dim).to(self.device)
            for _ in range(n_members)
        ])
        
        self.n_members = n_members
        self.current_member = 0  # Track which member is being trained
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Set up loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        # Register a buffer for tracking best validation performance
        self.register_buffer('best_val_loss', torch.tensor(float('inf')))

    def forward(self, x):
        """Forward pass using current ensemble member during training, or average during inference."""
        features = self.extract_features(x)
        if self.training:
            # Use current member during training
            return self.ensemble[self.current_member](features)
        else:
            # Average predictions during inference
            logits = [layer(features) for layer in self.ensemble]
            return torch.stack(logits, dim=0).mean(dim=0)

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
        """Single training step for current ensemble member."""
        self.train()
        y = y.long().to(self.device)
        
        # Forward pass with current member
        features = self.extract_features(x)
        logits = self.ensemble[self.current_member](features)
        
        # Compute loss
        loss = self.criterion(logits, y)
        
        # Compute metrics
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            err = preds.ne(y).sum().item()
            
        # Return loss tensor first for backprop, then metrics
        return loss, err, loss.item(), 0.0  # No KL term for ensemble

    def train_ensemble(self, train_loader, val_loader=None, epochs=10, lr=0.001, 
                       bootstrap=True, optimizer_class=torch.optim.Adam):
        """
        Train all ensemble members.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            epochs: Number of epochs per member
            lr: Learning rate
            bootstrap: Whether to use bootstrapping (sampling with replacement)
            optimizer_class: PyTorch optimizer to use
        """
        optimizers = [
            optimizer_class(member.parameters(), lr=lr)
            for member in self.ensemble
        ]
        
        print(f"Training ensemble with {self.n_members} members")
        for member_idx in range(self.n_members):
            self.current_member = member_idx
            optimizer = optimizers[member_idx]
            
            print(f"\nTraining ensemble member {member_idx+1}/{self.n_members}")
            
            # Create bootstrap sample if requested
            if bootstrap:
                train_loader_member = self._create_bootstrap_loader(train_loader)
            else:
                train_loader_member = train_loader
            
            for epoch in range(epochs):
                # Training phase
                self.train()
                running_loss = 0.0
                running_err = 0
                total = 0
                
                for batch_x, batch_y in train_loader_member:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    loss, err, _, _ = self.fit(batch_x, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    batch_size = batch_y.size(0)
                    running_loss += loss.item() * batch_size
                    running_err += err
                    total += batch_size
                
                train_loss = running_loss / total
                train_err = running_err / total
                
                # Validation phase if provided
                if val_loader:
                    val_loss, val_err = self._evaluate_member(val_loader, member_idx)
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Err: {train_err:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Err: {val_err:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Err: {train_err:.4f}")

    def _create_bootstrap_loader(self, dataloader):
        """Create a bootstrap sample for diversity in ensemble."""
        # Extract dataset from dataloader
        dataset = dataloader.dataset
        data_size = len(dataset)
        
        # Generate bootstrap indices (sampling with replacement)
        indices = np.random.choice(np.arange(data_size), size=data_size, replace=True)
        
        # Create bootstrap dataset
        bootstrap_dataset = Subset(dataset, indices)
        
        # Create new dataloader with same batch size and workers
        bootstrap_loader = DataLoader(
            bootstrap_dataset, 
            batch_size=dataloader.batch_size,
            shuffle=True,
            num_workers=dataloader.num_workers,
            drop_last=dataloader.drop_last
        )
        
        return bootstrap_loader

    @torch.no_grad()
    def _evaluate_member(self, dataloader, member_idx):
        """Evaluate a single ensemble member."""
        self.eval()
        running_loss = 0.0
        running_err = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            features = self.extract_features(batch_x)
            logits = self.ensemble[member_idx](features)
            loss = self.criterion(logits, batch_y)
            
            _, predicted = logits.max(1)
            err = predicted.ne(batch_y).sum().item()
            
            batch_size = batch_y.size(0)
            running_loss += loss.item() * batch_size
            running_err += err
            total += batch_size
        
        return running_loss / total, running_err / total

    @torch.no_grad()
    def evaluate(self, x, y, n_samples=None):
        """Evaluate with all ensemble members."""
        self.eval()
        x, y = x.to(self.device), y.long().to(self.device)
        
        features = self.extract_features(x)
        
        # Get predictions from all ensemble members
        outputs = []
        for member in self.ensemble:
            logits = member(features)
            outputs.append(logits.unsqueeze(0))
            
        # Stack predictions
        outputs = torch.cat(outputs, dim=0)  # [n_members, batch_size, n_classes]
        
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
            'ensemble_state': [member.state_dict() for member in self.ensemble],
            'n_members': self.n_members,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, path)
        print(f" [save_checkpoint] Saved ensemble model state to {path}")

    def load_checkpoint(self, path):
        """Load a saved checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.backbone.load_state_dict(checkpoint['backbone_state'])
        
        # Handle mismatch in ensemble size
        n_members = checkpoint['n_members']
        if n_members != self.n_members:
            print(f"Warning: loaded checkpoint has {n_members} members, but model has {self.n_members}")
            # Resize the ensemble if needed
            if n_members > self.n_members:
                # Only load the first self.n_members
                for i in range(self.n_members):
                    self.ensemble[i].load_state_dict(checkpoint['ensemble_state'][i])
            else:
                # Load what we can
                for i in range(n_members):
                    self.ensemble[i].load_state_dict(checkpoint['ensemble_state'][i])
        else:
            # Sizes match, load all members
            for i, member_state in enumerate(checkpoint['ensemble_state']):
                self.ensemble[i].load_state_dict(member_state)
        
        self.best_val_loss.copy_(checkpoint['best_val_loss'])
        print(f" [load_checkpoint] Loaded checkpoint from {path}")

    def sample_predict(self, x, Nsamples=None):
        """
        Make predictions using all ensemble members.
        
        Args:
            x: Input tensor
            Nsamples: If not None, use only this many members (for fair comparison with BLL)
        
        Returns:
            prob_stack: Tensor of shape [n_members, batch_size, n_classes]
                      containing softmax probabilities from each member
        """
        self.eval()
        features = self.extract_features(x)
        
        # Determine how many members to use
        n_use = self.n_members if Nsamples is None else min(Nsamples, self.n_members)
        
        # Collect predictions from ensemble members
        all_probs = []
        for i in range(n_use):
            logits = self.ensemble[i](features)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)
        
        # If we need more samples than members, resample with replacement
        if Nsamples is not None and Nsamples > self.n_members:
            # Randomly sample with replacement to reach Nsamples
            additional_samples = Nsamples - self.n_members
            indices = torch.randint(0, self.n_members, (additional_samples,))
            for idx in indices:
                all_probs.append(all_probs[idx])
        
        # Stack into [n_samples, batch_size, n_classes]
        prob_stack = torch.stack(all_probs, dim=0)
        
        return prob_stack

    def sample_predict_z(self, z, Nsamples=None):
        """
        Make predictions using all ensemble members for a latent representation.
        
        Args:
            z: Input latent tensor 
            Nsamples: If not None, use only this many members
        
        Returns:
            prob_stack: Tensor of shape [n_members, batch_size, n_classes]
                      containing softmax probabilities from each member
        """
        self.eval()
        features = z
        
        # Determine how many members to use
        n_use = self.n_members if Nsamples is None else min(Nsamples, self.n_members)
        
        # Collect predictions from ensemble members
        all_probs = []
        for i in range(n_use):
            logits = self.ensemble[i](features)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)
        
        # If we need more samples than members, resample with replacement
        if Nsamples is not None and Nsamples > self.n_members:
            # Randomly sample with replacement to reach Nsamples
            additional_samples = Nsamples - self.n_members
            indices = torch.randint(0, self.n_members, (additional_samples,))
            for idx in indices:
                all_probs.append(all_probs[idx])
        
        # Stack into [n_samples, batch_size, n_classes]
        prob_stack = torch.stack(all_probs, dim=0)
        
        return prob_stack
    
    def predict_with_uncertainty(self, x):
        """
        Make predictions using the entire ensemble.
        Returns:
            mean_probs: Average probabilities across ensemble
            uncertainty: Dictionary containing total, aleatoric and epistemic uncertainty
        """
        self.eval()
            
        # Stack into [n_ensemble, batch_size, n_classes]
        prob_stack = self.sample_predict(x)
        
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