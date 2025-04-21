import torch
import torch.nn as nn

class MNISTIM1Autoencoder(nn.Module):
    def __init__(self, latent_dim=32, device='cuda', model_save_path='models/mnist_im1_autoencoder.pth'):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.model_save_path = model_save_path
        
        # Encoder - uses a simpler architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder_dense = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.Sigmoid()  # Keep sigmoid for output between 0 and 1
        )
        
        self.to(device)
    
    def encode(self, x):
        x = x.to(self.device)
        return self.encoder(x)
    
    def decode(self, z):
        z = z.to(self.device)
        z_decoded = self.decoder_dense(z)
        z_reshaped = z_decoded.view(-1, 32, 7, 7)
        return self.decoder_conv(z_reshaped)
    
    def forward(self, x):
        x = x.to(self.device)
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
    
    def train_model(self, train_loader, val_loader, num_epochs=50, lr=0.001, patience=5, factor=0.5, min_lr=1e-6):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=5, 
            verbose=True, min_lr=min_lr
        )
        
        # Switch to MSE Loss which works better for grayscale MNIST
        criterion = nn.MSELoss()
        print("Using Mean Squared Error Loss")
        
        # Add noise during training (denoising autoencoder approach)
        noise_level = 0.2  # Add 20% noise
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            running_loss = 0.0
            for data in train_loader:
                img, _ = data
                img = img.to(self.device)
                
                # Add random noise to create a denoising task
                noisy_img = img + noise_level * torch.randn_like(img)
                noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
                
                # Forward pass
                recon, _ = self(noisy_img)
                
                # Calculate MSE loss
                loss = criterion(recon, img)  # Reconstruct the clean image
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Add gradient clipping to improve stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
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
        self.load_state_dict(torch.load(self.model_save_path))
        print(f"Model loaded from {self.model_save_path}")
        return self