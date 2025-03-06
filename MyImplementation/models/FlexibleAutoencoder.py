import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FlexibleAutoencoder(nn.Module):
    """
    A flexible autoencoder for MNIST that supports variable hidden dimensions.
    Automatically constructs appropriate intermediate layer sizes based on the 
    specified hidden dimension.
    """
    
    def __init__(self, hidden_dim=32, input_shape=(1, 28, 28), device=None):
        """
        Args:
            hidden_dim: Dimension of the latent space (32-784)
            input_shape: Shape of input images (channels, height, width)
            device: Device to run the model on ('cpu', 'cuda', or 'mps')
        """
        super().__init__()
        
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        # Store parameters
        self.hidden_dim = hidden_dim
        self.input_shape = input_shape
        self.input_dim = np.prod(input_shape)  # Flattened input dimension
        
        # Validate hidden dimension
        if hidden_dim > self.input_dim:
            raise ValueError(f"Hidden dimension ({hidden_dim}) cannot exceed input dimension ({self.input_dim})")
        if hidden_dim < 2:
            raise ValueError(f"Hidden dimension must be at least 2, got {hidden_dim}")
            
        # Design network architecture
        self._build_architecture()
        self.to(self.device)
        
    def _build_architecture(self):
        """
        Constructs the encoder and decoder architecture based on hidden_dim.
        Creates a smooth transition from input dimension to hidden dimension.
        """
        # Calculate intermediate layer sizes
        # We'll use 3 intermediate layers for encoder (and decoder)
        layer_sizes = self._calculate_layer_sizes(self.input_dim, self.hidden_dim, 3)
        
        # Build encoder
        encoder_layers = []
        prev_size = self.input_dim
        
        for size in layer_sizes:
            encoder_layers.extend([
                nn.Linear(prev_size, size),
                nn.BatchNorm1d(size),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            prev_size = size
            
        # Final encoder layer to hidden dimension
        encoder_layers.extend([
            nn.Linear(prev_size, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (reverse of encoder)
        decoder_layers = []
        prev_size = self.hidden_dim
        
        # First decoder layer from hidden dimension
        decoder_layers.extend([
            nn.Linear(self.hidden_dim, layer_sizes[-1]),
            nn.BatchNorm1d(layer_sizes[-1]),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        prev_size = layer_sizes[-1]
        
        # Intermediate decoder layers
        for size in reversed(layer_sizes[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_size, size),
                nn.BatchNorm1d(size),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            prev_size = size
            
        # Output layer
        decoder_layers.append(nn.Linear(prev_size, self.input_dim))
        # Sigmoid for pixel values in [0,1]
        decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def _calculate_layer_sizes(self, input_dim, hidden_dim, num_layers):
        """
        Calculates intermediate layer sizes using a logarithmic scale.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of intermediate layers
            
        Returns:
            List of intermediate layer sizes
        """
        # Use logarithmic scaling for smoother transition
        log_input = np.log(input_dim)
        log_hidden = np.log(hidden_dim)
        log_step = (log_input - log_hidden) / (num_layers + 1)
        
        sizes = []
        for i in range(1, num_layers + 1):
            size = int(np.exp(log_input - i * log_step))
            # Ensure size is at least hidden_dim
            size = max(size, hidden_dim)
            sizes.append(size)
        
        return sizes
    
    def encode(self, x):
        """Encode input to latent representation"""
        # Reshape input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x.to(self.device))
    
    def decode(self, z):
        """Decode latent representation to reconstruction"""
        recon = self.decoder(z)
        # Reshape to original dimensions
        return recon.view(-1, *self.input_shape)
    
    def forward(self, x):
        """Full forward pass through the autoencoder"""
        # Reshape input if needed
        original_shape = x.shape
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Pass through encoder and decoder
        z = self.encoder(x.to(self.device))
        recon = self.decoder(z)
        
        # Reshape output to match input shape
        if len(original_shape) > 2:
            recon = recon.view(original_shape)
            
        return recon, z
    
    def fit(self, x, optimizer=None, criterion=None):
        """
        Perform a single training step
        
        Args:
            x: Input data
            optimizer: Optional optimizer, creates one if None
            criterion: Optional loss function, defaults to MSE
            
        Returns:
            loss: Reconstruction loss
        """
        self.train()
        
        # Use default optimizer if none provided
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        # Use default criterion if none provided
        if criterion is None:
            criterion = nn.MSELoss()
            
        # Forward pass
        x = x.to(self.device)
        recon, _ = self(x)
        
        # If input is not flattened, flatten for loss calculation
        if len(x.shape) > 2:
            x_flat = x.view(x.size(0), -1)
            recon_flat = recon.view(recon.size(0), -1)
            loss = criterion(recon_flat, x_flat)
        else:
            loss = criterion(recon, x)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def train_model(self, train_loader, num_epochs=10, lr=1e-3, weight_decay=1e-5, 
                   val_loader=None, early_stopping_patience=5, verbose=True, save_path=None, model_name=None):
        """
        Train the autoencoder on a dataset
        
        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of epochs to train for
            lr: Learning rate
            weight_decay: L2 regularization strength
            val_loader: Optional DataLoader for validation data
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            verbose: Whether to print progress
            save_path: Path to save the best model to (if None, model won't be saved to disk)
            model_name: Name of the model (if None, model won't be saved to disk)   
        Returns:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch (if val_loader provided)
        """
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopped = False
        if model_name is None:
            model_name = "autoencoder_" + str(self.hidden_dim)
        
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                # Handle different data formats (tuple vs tensor)
                if isinstance(batch, (list, tuple)):
                    x = batch[0]  # Assume first element is the data
                else:
                    x = batch
                
                x = x.to(self.device)
                
                # Forward pass
                recon, _ = self(x)
                
                # Calculate loss
                x_flat = x.view(x.size(0), -1)
                recon_flat = recon.view(recon.size(0), -1)
                loss = criterion(recon_flat, x_flat)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Calculate average epoch loss
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase if validation data is provided
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                num_val_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        # Handle different data formats
                        if isinstance(batch, (list, tuple)):
                            x = batch[0]
                        else:
                            x = batch
                        
                        x = x.to(self.device)
                        
                        # Forward pass
                        recon, _ = self(x)
                        
                        # Calculate loss
                        x_flat = x.view(x.size(0), -1)
                        recon_flat = recon.view(recon.size(0), -1)
                        loss = criterion(recon_flat, x_flat)
                        
                        val_loss += loss.item()
                        num_val_batches += 1
                
                avg_val_loss = val_loss / num_val_batches
                val_losses.append(avg_val_loss)
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model in memory
                    best_model_state = self.state_dict()
                    
                    # Save to disk if path provided
                    if save_path:
                        self.save(save_path + "/" + model_name + "_best_model.pth")
                        if verbose:
                            print(f"Saved best model to {save_path + "/" + model_name + "_best_model.pth"}")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        # Restore best model
                        self.load_state_dict(best_model_state)
                        early_stopped = True
                        break
                
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}")
                # Save the latest model if no validation data is provided
                if save_path:
                    self.save(save_path + "/" + model_name + "_latest_model.pth")
                    if verbose:
                        print(f"Saved model to {save_path + "/" + model_name + "_latest_model.pth"}")
        
        # End of training - make sure we have the best model loaded
        if val_loader is not None and not early_stopped:
            # Load the best model if we didn't early stop
            self.load_state_dict(best_model_state)
        
        # Final save if requested and we didn't already save the best model
        if save_path and (val_loader is None or not early_stopped):
            self.save(save_path + "/" + model_name + "_final_model.pth")
            if verbose:
                print(f"Saved final model to {save_path + "/" + model_name + "_final_model.pth"}")
        
        return train_losses, val_losses if val_loader is not None else None
    
    @torch.no_grad()
    def evaluate(self, x, criterion=None):
        """
        Evaluate the model on a batch of data
        
        Args:
            x: Input data
            criterion: Optional loss function, defaults to MSE
            
        Returns:
            loss: Reconstruction loss
            reconstructions: Reconstructed data
        """
        self.eval()
        
        # Use default criterion if none provided
        if criterion is None:
            criterion = nn.MSELoss()
            
        # Forward pass
        x = x.to(self.device)
        recon, _ = self(x)
        
        # If input is not flattened, flatten for loss calculation
        if len(x.shape) > 2:
            x_flat = x.view(x.size(0), -1)
            recon_flat = recon.view(recon.size(0), -1)
            loss = criterion(recon_flat, x_flat)
        else:
            loss = criterion(recon, x)
        
        return loss.item(), recon
    
    def save(self, path):
        """Save model state"""
        torch.save({
            'hidden_dim': self.hidden_dim,
            'input_shape': self.input_shape,
            'state_dict': self.state_dict()
        }, path)
        
    @classmethod
    def load(cls, path, device=None):
        """Load model from saved state"""
        if device is None:
            device = torch.device('cpu')
            
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            hidden_dim=checkpoint['hidden_dim'],
            input_shape=checkpoint['input_shape'],
            device=device
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model