import torch
import torch.nn as nn
import torch.optim as optim

# This file contains the implementations of the models for ReGene, from the paper "Classify and Generate". 
# This allows us to generate images from the latent space of a classifier.


class Classifier(nn.Module):
    """
    Classifier model which saves the latent representation of the input image for generation later.
    """
    def __init__(self, latent_dim=32, num_classes=10, device='cpu'):
        super(Classifier, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        y = self.classifier(z)
        return z, y  # Return latent representation and class prediction
    
    def train_classifier(self, train_loader, num_epochs=5, lr=0.001):
        """ Train the classifier """
        self.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                z, y_pred = self(images)
                loss = criterion(y_pred, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
class Decoder(nn.Module):
    """
    Decoder model which generates an image from the latent representation of the classifier.
    """
    def __init__(self, latent_dim=32, device='cpu'):
        super(Decoder, self).__init__()
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * 7 * 7),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Normalize output to [0,1]
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 32, 7, 7)
        x = self.deconv(x)
        return x
    
    def train_decoder(self, train_loader, classifier, num_epochs=5, lr=0.001):
        """ Train the decoder using the classifier's latent space """
        self.to(self.device)
        classifier.to(self.device)
        decoder_optimizer = optim.Adam(self.parameters(), lr=lr)
        reconstruction_loss = nn.MSELoss()

        for epoch in range(num_epochs):
            for images, _ in train_loader:
                images = images.to(self.device)
                z, _ = classifier(images)
                x_reconstructed = self(z)
                loss = reconstruction_loss(x_reconstructed, images)
                decoder_optimizer.zero_grad()
                loss.backward()
                decoder_optimizer.step()
            print(f"Decoder Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    def convex_combine(self, z1, z2, alphas = [0.5]):
        """ Convex combinations of two latent representations """
        x_combined = []

        for alpha in alphas:
            z_combined = alpha * z1 + (1 - alpha) * z2
            x_combined.append(self(z_combined))

        return x_combined