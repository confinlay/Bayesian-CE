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
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(
            hidden_dim=checkpoint['hidden_dim'],
            input_shape=checkpoint['input_shape'],
            device=device
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model