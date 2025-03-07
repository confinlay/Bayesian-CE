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
            # First block: 28×28×1 -> 14×14×32
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second block: 14×14×32 -> 7×7×64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third block: 7×7×64 -> 4×4×128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=1),  # Padding to get 4×4 output
            
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.to(device)
    def forward(self, x):
        z = self.encoder(x)
        y = self.classifier(z)
        return z, y  # Return latent representation and class prediction
    
    def train_classifier(self, train_loader, num_epochs=5, lr=0.001, patience=5, model_saves_dir=None):
        """ Train the classifier """
        self.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        
        best_loss = float('inf')
        early_stopping_counter = 0
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                z, y_pred = self(images)
                loss = criterion(y_pred, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            
            # Learning rate scheduling
            scheduler.step(epoch_loss)
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                early_stopping_counter = 0
                
                # Save the best model
                if model_saves_dir:
                    import os
                    os.makedirs(model_saves_dir, exist_ok=True)
                    save_path = os.path.join(model_saves_dir, f"classifier_dominated_classifier_{self.encoder[-2].out_features}.pt")
                    torch.save(self.state_dict(), save_path)
                    print(f"Saved best model to: {save_path}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
    
class Decoder(nn.Module):
    """
    Decoder model which generates an image from the latent representation of the classifier.
    """
    def __init__(self, latent_dim=128, device='cpu'):
        super(Decoder, self).__init__()
        self.device = device
        
        # Project from latent space to match flattened dimensions (128*4*4)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU()
        )
        
        # Mirrored convolutional structure to match the encoder
        self.deconv = nn.Sequential(
            # First upsampling block: 4×4×128 -> 7×7×64
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Second upsampling block: 7×7×64 -> 14×14×32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Third upsampling block: 14×14×32 -> 28×28×1
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Final activation for pixel values in [0,1]
        )
        
        self.to(device)

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 4, 4)
        x = self.deconv(x)
        return x
    
    def train_decoder(self, train_loader, classifier, num_epochs=5, lr=0.001, patience=5, model_saves_dir=None):
        """ Train the decoder using the classifier's latent space """
        self.to(self.device)
        classifier.to(self.device)
        decoder_optimizer = optim.Adam(self.parameters(), lr=lr)
        reconstruction_loss = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        
        best_loss = float('inf')
        early_stopping_counter = 0

        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, _ in train_loader:
                images = images.to(self.device)
                z, _ = classifier(images)
                x_reconstructed = self(z)
                loss = reconstruction_loss(x_reconstructed, images)
                decoder_optimizer.zero_grad()
                loss.backward()
                decoder_optimizer.step()
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            print(f"Decoder Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            
            # Learning rate scheduling
            scheduler.step(epoch_loss)
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                early_stopping_counter = 0
                
                # Save the best model
                if model_saves_dir:
                    import os
                    os.makedirs(model_saves_dir, exist_ok=True)
                    save_path = os.path.join(model_saves_dir, f"classifier_dominated_decoder_{self.fc[0].in_features}.pt")
                    torch.save(self.state_dict(), save_path)
                    print(f"Saved best model to: {save_path}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

def train_joint(classifier, decoder, train_loader, num_epochs=5, lr=0.001, lambda_recon=0.5, patience=5, model_saves_dir=None):
    """Train both classifier and decoder jointly with combined loss"""
    classifier.to(classifier.device)
    decoder.to(decoder.device)
    
    # Setup optimizers
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(classifier_optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Loss functions
    classification_criterion = nn.CrossEntropyLoss()
    reconstruction_criterion = nn.MSELoss()
    
    best_loss = float('inf')
    early_stopping_counter = 0
    
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
            
        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Total Loss: {total_loss.item():.4f}, "
              f"Classification Loss: {classification_loss.item():.4f}, "
              f"Reconstruction Loss: {reconstruction_loss.item():.4f}")
        
        # Learning rate scheduling
        scheduler.step(epoch_loss)
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            early_stopping_counter = 0
            
            # Save the best model
            if model_saves_dir:
                import os
                os.makedirs(model_saves_dir, exist_ok=True)
                save_path = os.path.join(model_saves_dir, f"joint_model_{classifier.encoder[-2].out_features}.pt")
                torch.save(classifier.state_dict(), save_path)
                torch.save(decoder.state_dict(), save_path)
                print(f"Saved best model to: {save_path}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

def train_autoencoder(classifier, decoder, train_loader, num_epochs=5, lr=0.001, patience=5, model_saves_dir=None):
    """Train classifier encoder and decoder to minimize reconstruction loss"""
    classifier.to(classifier.device)
    decoder.to(decoder.device)
    
    # Setup optimizer
    params = list(classifier.parameters()) + list(decoder.parameters()) 
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Loss function
    reconstruction_criterion = nn.MSELoss()
    
    best_loss = float('inf')
    early_stopping_counter = 0
    
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

        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stopping_counter = 0
            
            # Save the best model
            if model_saves_dir:
                import os
                os.makedirs(model_saves_dir, exist_ok=True)
                save_path = os.path.join(model_saves_dir, f"autoencoder_{classifier.encoder[-2].out_features}.pt")
                torch.save(classifier.state_dict(), save_path)
                torch.save(decoder.state_dict(), save_path)
                print(f"Saved best model to: {save_path}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

def train_classifier_only(classifier, train_loader, num_epochs=5, lr=0.001, patience=5, model_saves_dir=None):
    """Train only the classification layer, keeping encoder weights frozen"""
    classifier.to(classifier.device)
    # Freeze encoder weights
    for param in classifier.encoder.parameters():
        param.requires_grad = False
    # Only optimize classifier layer
    optimizer = optim.Adam(classifier.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    best_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(classifier.device), labels.to(classifier.device)
            optimizer.zero_grad()
            _, y_pred = classifier(images)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(epoch_loss)
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            early_stopping_counter = 0
            
            # Save the best model
            if model_saves_dir:
                import os
                os.makedirs(model_saves_dir, exist_ok=True)
                save_path = os.path.join(model_saves_dir, f"classifier_only_{classifier.encoder[-2].out_features}.pt")
                torch.save(classifier.state_dict(), save_path)
                print(f"Saved best model to: {save_path}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Unfreeze encoder for future use
    for param in classifier.encoder.parameters():
        param.requires_grad = True
