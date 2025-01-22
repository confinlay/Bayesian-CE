import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from laplace import Laplace

class MLP(nn.Module):
    """Simple MLP for MNIST classification."""
    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10, device=None):
        super().__init__()

        if device is None:
            device = get_device()
        
        self.device = device
        self.to(self.device)
        
        # Build layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        # Add final layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.to(self.device)
        self.to(self.device)

        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        return self.layers(x)
        
    def fit(self, train_loader, epochs=10):
        """Train the MLP model."""
        
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters())
        self.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for x, y in train_loader:
                # Ensure tensors are on the correct device
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                out = self(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
                
                # Move tensors to CPU for numerical operations
                total_loss += loss.cpu().item()
                pred = out.argmax(dim=1)
                correct += (pred == y).cpu().sum().item()
                total += y.size(0)
                
            acc = correct / total
            print(f'Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, Acc = {acc:.4f}')
        
        return self

class BayesianMLP:
    """Wrapper for MLP with last-layer Laplace approximation."""
    def __init__(self, base_model):
        self.base_model = base_model
        self.la = None
        
    def fit(self, train_loader):
        """Fit the Laplace approximation."""
        # Check if the current device is MPS and switch to CPU
        if torch.device('mps') == self.base_model.device:
            self.base_model.device = torch.device('cpu')

        
        # Initialize Laplace with last-layer setting
        self.la = Laplace(
            self.base_model,
            'classification',
            subset_of_weights='last_layer',
            hessian_structure='kron'
        )
        
        # Move data to the correct device
        for x, y in train_loader:
            x, y = x.to(self.base_model.device), y.to(self.base_model.device)
        
        # Fit the Laplace approximation
        self.la.fit(train_loader)
        
        # Optimize the prior precision
        self.la.optimize_prior_precision(method='marglik')
        
    def predict(self, x, link='softmax'):
        """Get predictions with uncertainty."""
        if self.la is None:
            raise RuntimeError("Model needs to be fit first!")
        
        x = x.to(self.base_model.device)
        # Get predictions with uncertainty
        pred = self.la(x)
        # if link == 'softmax':
        #     return F.softmax(pred, dim=-1)
        return pred

def get_device():
    """Get the appropriate device (MPS for Mac, CUDA for NVIDIA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
