import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# This file contains the implementations of the models for ReGene, from the paper "Classify and Generate". 
# This allows us to generate images from the latent space of a classifier.


class SVAE(nn.Module):
    """
    Supervised Variational Autoencoder that combines classification with reconstruction.
    """
    def __init__(self, latent_dim=128, num_classes=10, device='cpu'):
        super(SVAE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder network (recognition model)
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
            nn.Linear(128 * 3 * 3, 256)
        )
        
        # Mean and variance for latent space
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        
        # Classifier head
        self.classifier = nn.Linear(latent_dim, num_classes)
        
        # Decoder network (generation model)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        self.to(device)

    def encode(self, x):
        """Encode input to get mean and log variance of latent distribution"""
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick to sample from latent distribution"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """Forward pass through the entire model"""
        # Encode
        mu, log_var = self.encode(x)
        
        # Sample from latent space
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_recon = self.decoder(z)
        
        # Classify
        y_pred = self.classifier(z)
        
        return x_recon, y_pred, mu, log_var
    
    def loss_function(self, x_recon, x, y_pred, y, mu, log_var, beta=1.0, alpha=1.0):
        """
        Compute the total loss:
        - Reconstruction loss (BCE)
        - KL divergence loss
        - Classification loss (CE)
        
        Args:
            beta: Weight for KL divergence term
            alpha: Weight for classification loss
        """
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Classification loss
        class_loss = F.cross_entropy(y_pred, y, reduction='sum')
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss + alpha * class_loss
        
        return total_loss, recon_loss, kl_loss, class_loss
    
    def train_model(self, train_loader, num_epochs=5, lr=0.001, beta=1.0, alpha=1.0):
        """Train the SVAE"""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            total_loss = 0
            recon_total = 0
            kl_total = 0
            class_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                x_recon, y_pred, mu, log_var = self(images)
                
                # Calculate loss
                loss, recon_loss, kl_loss, class_loss = self.loss_function(
                    x_recon, images, y_pred, labels, mu, log_var, beta, alpha
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Accumulate losses
                total_loss += loss.item()
                recon_total += recon_loss.item()
                kl_total += kl_loss.item()
                class_total += class_loss.item()
            
            # Print epoch statistics
            avg_loss = total_loss / len(train_loader.dataset)
            avg_recon = recon_total / len(train_loader.dataset)
            avg_kl = kl_total / len(train_loader.dataset)
            avg_class = class_total / len(train_loader.dataset)
            
            print(f"Epoch [{epoch+1}/{num_epochs}]:")
            print(f"Total Loss: {avg_loss:.4f}")
            print(f"Reconstruction Loss: {avg_recon:.4f}")
            print(f"KL Loss: {avg_kl:.4f}")
            print(f"Classification Loss: {avg_class:.4f}")

def train_classifier(classifier, train_loader, num_epochs=5, lr=0.001):
    """Train the classifier"""
    classifier.to(classifier.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(classifier.device), labels.to(classifier.device)
            optimizer.zero_grad()
            _, y_pred = classifier(images)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def train_decoder(decoder, train_loader, classifier, num_epochs=5, lr=0.001):
    """Train the decoder using the classifier's latent space"""
    decoder.to(decoder.device)
    classifier.to(classifier.device)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    reconstruction_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        for images, _ in train_loader:
            images = images.to(decoder.device)
            z, _ = classifier(images)
            x_reconstructed = decoder(z)
            loss = reconstruction_loss(x_reconstructed, images)
            decoder_optimizer.zero_grad()
            loss.backward()
            decoder_optimizer.step()
        print(f"Decoder Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def convex_combine(decoder, z1, z2, alphas = [0.5]):
    """Convex combinations of two latent representations"""
    x_combined = []

    for alpha in alphas:
        z_combined = alpha * z1 + (1 - alpha) * z2
        x_combined.append(decoder(z_combined))

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
