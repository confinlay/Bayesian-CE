import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from laplace import Laplace

class MLP(nn.Module):
    """Simple MLP for MNIST classification."""
    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10):
        super().__init__()
        
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
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        return self.layers(x)

class BayesianMLP:
    """Wrapper for MLP with last-layer Laplace approximation."""
    def __init__(self, base_model):
        self.base_model = base_model
        self.la = None
        
    def fit(self, train_loader):
        """Fit the Laplace approximation."""
        # Initialize Laplace with last-layer setting
        self.la = Laplace(
            self.base_model,
            'classification',
            subset_of_weights='last_layer',
            hessian_structure='kron'
        )
        
        # Fit the Laplace approximation
        self.la.fit(train_loader)
        
        # Optimize the prior precision
        self.la.optimize_prior_precision()
        
    def predict(self, x, link='softmax'):
        """Get predictions with uncertainty."""
        if self.la is None:
            raise RuntimeError("Model needs to be fit first!")
            
        # Get predictions with uncertainty
        pred = self.la(x)
        if link == 'softmax':
            return F.softmax(pred, dim=-1)
        return pred

def get_device():
    """Get the appropriate device (MPS for Mac, CUDA for NVIDIA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_mlp(model, train_loader, test_loader, epochs=10, device=None):
    """Train the base MLP model."""
    if device is None:
        device = get_device()
    
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            # Ensure tensors are on the correct device
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
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
    
    return model

def main():
    # Get appropriate device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    # Create and train base model
    mlp = MLP().to(device)
    mlp = train_mlp(mlp, train_loader, test_loader, device=device)
    
    # Create Bayesian version and fit LA
    bayes_mlp = BayesianMLP(mlp)
    bayes_mlp.fit(train_loader)
    
    # Test predictions with uncertainty
    x_test, y_test = next(iter(test_loader))
    x_test = x_test.to(device)
    
    # Get predictions with uncertainty
    pred_probs = bayes_mlp.predict(x_test)
    print("\nPredictive distribution shape:", pred_probs.shape)
    
    # Move to CPU for printing
    max_probs = pred_probs.max(dim=1)[0][:5].cpu()
    print("Max probability:", max_probs)  # Show first 5 confidence scores

if __name__ == '__main__':
    main()
