import torch
import torch.nn as nn

class MNISTClassifierBLL(nn.Module):
    """MNIST classifier backbone for BLL."""
    
    def __init__(self, device='cpu'):
        nn.Module.__init__(self)
        
        # Store device
        self.device = torch.device(device)
        
        # Define encoder layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 3 * 3, 256)
        
        # Output layer
        self.classifier = nn.Linear(256, 10)
        
        # Move model to device
        self.to(self.device)
        
    def forward(self, x):
        # Ensure input is on the correct device and type
        x = x.to(self.device, dtype=torch.float32)
        
        # Encoder path
        x = self.pool(self.relu(self.conv1(x)))  # 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 7x7
        x = self.pool(self.relu(self.conv3(x)))  # 3x3
        x = self.flatten(x)
        z = self.fc(x)  # Features
        
        # Classification head
        y = self.classifier(z)
        
        return z, y
        
    def load_weights(self, path):
        """Load model weights from a saved state dict."""
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.eval()
        print(f" [load_weights] Loaded backbone weights from {path}")