import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self, device='cpu'):
        super(MNISTClassifier, self).__init__()
        self.device = device
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7x7
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),  # 3x3
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.Linear(256, 10)  # Output layer for 10 MNIST classes
        )
        self.to(device)
        
    def forward(self, x):
        return self.classifier(x)