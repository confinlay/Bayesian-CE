import torch
import torch.nn as nn
from pythae.models import VAE, VAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.benchmarks.mnist import Encoder_Conv_VAE_MNIST, Decoder_Conv_AE_MNIST
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines import TrainingPipeline

class MNISTVAE:
    def __init__(self, input_dim=(1, 28, 28), latent_dim=10, device=None):
        # Determine the device
        if device is None:
            self.device = (
                "mps" if torch.backends.mps.is_available() 
                else "cuda" if torch.cuda.is_available() 
                else "cpu"
            )
        else:
            self.device = device
        print(f"Using device: {self.device}")

        # Set up the model configuration
        self.model_config = VAEConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            reconstruction_loss="bce"  # Binary cross entropy for MNIST
        )

        # Initialize encoder and decoder
        self.encoder = Encoder_Conv_VAE_MNIST(self.model_config)
        self.decoder = Decoder_Conv_AE_MNIST(self.model_config)

        # Build the VAE model
        self.model = VAE(
            model_config=self.model_config,
            encoder=self.encoder,
            decoder=self.decoder
        )
        
        # Move model to appropriate device
        self.model.to(self.device)
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def setup_training(self, 
                      num_epochs=50,
                      learning_rate=1e-3,
                      batch_size=64,
                      output_dir="mnist_vae_training"):
        """Set up the training configuration"""
        
        training_config = BaseTrainerConfig(
            output_dir=output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            optimizer_cls="Adam",
            optimizer_params={"weight_decay": 0.0},
            scheduler_cls="ReduceLROnPlateau",
            scheduler_params={
                "patience": 5,
                "factor": 0.5
            }
        )

        self.pipeline = TrainingPipeline(
            training_config=training_config,
            model=self.model
        )

    def train(self, train_data, eval_data=None):
        """Train the VAE model"""
        if not hasattr(self, 'pipeline'):
            raise RuntimeError("Call setup_training before training")
            
        # Ensure model is on correct device before training
        self.model.to(self.device)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
            
        self.pipeline(
            train_data=train_data,
            eval_data=eval_data
        )

    def encode(self, x):
        """Encode input data to latent representation"""
        x = self._ensure_tensor(x)
        return self.model.encode(x)

    def decode(self, z):
        """Decode latent representation to reconstructed input"""
        z = self._ensure_tensor(z)
        return self.model.decode(z)

    def reconstruct(self, x):
        """Reconstruct input through the VAE"""
        x = self._ensure_tensor(x)
        return self.model.reconstruct(x)

    def _ensure_tensor(self, x):
        """Ensure input is a tensor on the correct device"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x.to(self.device)

    def generate(self, num_samples=1):
        """Generate new samples from the learned distribution"""
        samples = self.model.sample(num_samples=num_samples)
        return samples.to(self.device)

    def save(self, path):
        """Save the model"""
        self.model.save(path)

    @classmethod
    def load(cls, path):
        """Load a saved model"""
        loaded_model = VAE.load_from_folder(path)
        vae = cls()
        vae.model = loaded_model
        vae.model.to(vae.device)  # Ensure loaded model is on correct device
        return vae

# Example usage
if __name__ == "__main__":
    # Import MPS-related imports
    import torch.backends.mps

    # Create VAE instance
    vae = MNISTVAE(input_dim=(1, 28, 28), latent_dim=10)
    
    # Setup training configuration
    vae.setup_training(
        num_epochs=50,
        learning_rate=1e-3,
        batch_size=64,
        output_dir="mnist_vae_results"
    )
    
    # Training would look like:
    # vae.train(train_data, eval_data)
