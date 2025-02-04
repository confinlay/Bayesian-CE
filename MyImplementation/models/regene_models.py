import torch
import torch.nn as nn
import torch.optim as optim

# This file contains the implementations of the models for ReGene, from the paper "Classify and Generate". 
# This allows us to generate images from the latent space of a classifier.


class Classifier(nn.Module):
    """
    Classifier model which saves the latent representation of the input image for generation later.
    """
    def __init__(self, latent_dim=128, num_classes=10, device='cpu'):
        super(Classifier, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
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
            nn.Linear(128 * 3 * 3, latent_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.to(device)
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
        self.to(device)

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

def train_joint(classifier, decoder, train_loader, num_epochs=5, lr=0.001, lambda_recon=0.5):
    """Train both classifier and decoder jointly with combined loss"""
    classifier.to(classifier.device)
    decoder.to(decoder.device)
    
    # Setup optimizers
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    
    # Loss functions
    classification_criterion = nn.CrossEntropyLoss()
    reconstruction_criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(classifier.device)
            labels = labels.to(classifier.device)
            
            # Zero gradients
            classifier_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            # Forward pass through classifier
            z, y_pred = classifier(images)
            
            # Forward pass through decoder
            x_reconstructed = decoder(z)
            
            # Calculate losses
            classification_loss = classification_criterion(y_pred, labels)
            reconstruction_loss = reconstruction_criterion(x_reconstructed, images)
            
            # Combined loss
            total_loss = lambda_recon * reconstruction_loss + (1 - lambda_recon) * classification_loss
            
            # Backward pass and optimization
            total_loss.backward()
            classifier_optimizer.step()
            decoder_optimizer.step()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Total Loss: {total_loss.item():.4f}, "
              f"Classification Loss: {classification_loss.item():.4f}, "
              f"Reconstruction Loss: {reconstruction_loss.item():.4f}")
        
def train_autoencoder(classifier, decoder, train_loader, num_epochs=5, lr=0.001):
    """Train classifier encoder and decoder to minimize reconstruction loss"""
    classifier.to(classifier.device)
    decoder.to(decoder.device)
    
    # Setup optimizer
    params = list(classifier.parameters()) + list(decoder.parameters()) 
    optimizer = optim.Adam(params, lr=lr)
    
    # Loss function
    reconstruction_criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, _ in train_loader:
            images = images.to(classifier.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            z, _ = classifier(images)  # Get latent representation from classifier
            reconstructed = decoder(z)
            
            # Calculate reconstruction loss
            loss = reconstruction_criterion(reconstructed, images)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Reconstruction Loss: {avg_loss:.4f}")

        # Print progress similar to train_classifier
        print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss:.4f}")

def train_classifier_only(classifier, train_loader, num_epochs=5, lr=0.001):
    """Train only the classification layer, keeping encoder weights frozen"""
    classifier.to(classifier.device)
    # Freeze encoder weights
    for param in classifier.encoder.parameters():
        param.requires_grad = False
    # Only optimize classifier layer
    optimizer = optim.Adam(classifier.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(classifier.device), labels.to(classifier.device)
            optimizer.zero_grad()
            _, y_pred = classifier(images)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            
    # Unfreeze encoder for future use
    for param in classifier.encoder.parameters():
        param.requires_grad = True
