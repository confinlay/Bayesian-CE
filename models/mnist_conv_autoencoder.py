import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=256, device='mps', model_save_path='models/mnist_conv_autoencoder.pth'):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.model_save_path = model_save_path
        
        # Encoder - uses convolutional layers
        self.encoder = nn.Sequential(
            # Input: 1x28x28
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            # Size: 128 * 7 * 7 = 6272
            nn.Linear(128 * 7 * 7, latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder - uses transposed convolutions for upsampling
        self.decoder_dense = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.decoder_conv = nn.Sequential(
            # Size: 128 * 7 * 7 -> 128x7x7
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 28x28
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Final activation to get pixel values in [0,1]
        )
        self.to(device)
    
    def encode(self, x):
        """Encode input to latent representation"""
        x = x.to(self.device)
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to reconstruction"""
        z = z.to(self.device)
        z_decoded = self.decoder_dense(z)
        z_reshaped = z_decoded.view(-1, 128, 7, 7)
        return self.decoder_conv(z_reshaped)
    
    def forward(self, x):
        """Full forward pass through the autoencoder"""
        x = x.to(self.device)
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
    
    def train_model(self, train_loader, val_loader, num_epochs=20, lr=0.001, patience=5, factor=0.5, min_lr=1e-6, loss_type='mse'):
        """Train the autoencoder with early stopping and learning rate reduction"""
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=5, 
            verbose=True, min_lr=min_lr
        )
        
        # Select loss function based on loss_type parameter
        if loss_type.lower() == 'bce':
            criterion = nn.BCELoss()
        else:  # default to MSE
            criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            running_loss = 0.0
            for data in train_loader:
                img, _ = data
                img = img.to(self.device)
                
                # Forward pass 
                recon, _ = self(img)
                loss = criterion(recon, img)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            train_loss = running_loss / len(train_loader)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in val_loader:
                    img, _ = data
                    img = img.to(self.device)
                    recon, _ = self(img)
                    loss = criterion(recon, img)
                    val_loss += loss.item()
            
            val_loss = val_loss / len(val_loader)
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Print statistics
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                # Save best model
                self.save_model()
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        print('Finished Training')
        return self
    
    def save_model(self):
        torch.save(self.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")
    
    def load_model(self):
        self.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        print(f"Model loaded from {self.model_save_path}")
        return self