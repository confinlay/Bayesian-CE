# Minimal DDIM for MNIST with Feature Conditioning

This is a minimal implementation of a Diffusion Autoencoder using DDIM (Denoising Diffusion Implicit Model) for the MNIST dataset using PyTorch. The model is conditioned on feature vectors from a separate autoencoder, allowing it to generate MNIST digits guided by latent representations.

## Overview

The implementation consists of two main components:

1. **SimpleAutoencoder**: A convolutional autoencoder that generates feature vectors from MNIST images.
2. **DiffusionModel**: A DDIM implementation that uses a U-Net architecture to denoise images, conditioned on both diffusion timesteps and feature vectors from the autoencoder.

## Features

- **Latent Space Conditioning**: Denoise images conditioned on feature vectors
- **Diffusion Autoencoder Approach**: Encode images, add noise, then denoise with conditioning
- **Latent Space Manipulation**: Perturb latent vectors to see how it affects generation
- **Cross-Platform Support**: Works on CUDA GPUs, Apple Silicon (MPS), and CPU

## Requirements

- PyTorch (>=1.12.0 for MPS support on macOS)
- torchvision
- NumPy
- Matplotlib

## Integration with Existing PyTorch Codebase

This implementation is designed to integrate seamlessly with the existing PyTorch codebase, including the BLL_VI (Bayesian Last Layer Variational Inference) and CLUE implementations. There's no need for cross-framework conversions since everything is implemented in PyTorch.

## Usage

### Training

To train both the autoencoder and the diffusion model:

```bash
cd DDIM
python train_ddim_mnist.py
```

This will:
1. Train a simple autoencoder on MNIST for 5 epochs
2. Generate feature vectors for all MNIST images
3. Train the DDIM model for 30 epochs
4. Save generated samples in the `samples` directory every 5 epochs
5. Generate a comparison of original and generated images in `conditioned_samples.png`
6. Create a demonstration of the diffusion autoencoder with perturbed latent vectors in `diffusion_autoencoder_demo.png`

### Using Pre-trained Models

If you have pre-trained models (`autoencoder_weights.pt` and `diffusion_model.pt`), the script will load them instead of training from scratch.

## Diffusion Autoencoder Workflow

This implementation follows the diffusion autoencoder approach:

1. **Encoding**: An image is encoded by the autoencoder to obtain a latent vector
2. **Noising**: The original image has noise added at various timesteps
3. **Denoising with Conditioning**: The model denoises the image conditioned on the latent vector

This workflow enables:
- Exploring the latent space by perturbing latent vectors
- Controlling the denoising process through latent manipulation
- Observing how changes to the latent representation affect the generated output

## Platform Support

The implementation automatically detects and uses the best available hardware:

- **CUDA**: For NVIDIA GPUs
- **MPS**: For Apple Silicon (M1/M2/M3) Macs
- **CPU**: As fallback for any platform

## Customization

You can customize the models by modifying parameters in `train_ddim_mnist.py`:

### Autoencoder

- `latent_dim`: Dimensionality of the feature vectors (default: 256)
- `device`: Device to use for training/inference (automatically selected)

### Diffusion Model

- `time_embedding_dim`: Dimensionality of the timestep embeddings (default: 32)
- `feature_embedding_dim`: Dimensionality of the feature embeddings (default: 64)
- `widths`: Channel widths for the U-Net layers (default: [32, 64, 128])
- `block_depth`: Number of residual blocks per U-Net stage (default: 2)
- `device`: Device to use for training/inference (automatically selected)

## Example: Manipulating Latent Vectors

```python
import torch
from DDIM.ddim_mnist import DiffusionModel, SimpleAutoencoder

# Load models
autoencoder = SimpleAutoencoder()
autoencoder.load_checkpoint("DDIM/autoencoder_weights.pt")

diffusion_model = DiffusionModel()
diffusion_model.load_checkpoint("DDIM/diffusion_model.pt")

# Load an image and get its latent representation
image = torch.randn(1, 1, 28, 28)  # Replace with your image
latent = autoencoder.encode(image)

# Add noise to the image
noisy_image, _ = diffusion_model.add_noise(image, diffusion_time=0.7)

# Create perturbed versions of the latent vector
perturbed_latent = latent + torch.randn_like(latent) * 0.5

# Denoise with original and perturbed latent vectors
original_denoised = diffusion_model.denoise_with_latent(noisy_image, latent, diffusion_time=0.7)
perturbed_denoised = diffusion_model.denoise_with_latent(noisy_image, perturbed_latent, diffusion_time=0.7)
```

## Using with BLL_VI and CLUE

To use the DDIM model's feature vectors with your existing BLL_VI and CLUE implementations:

```python
# Load the trained autoencoder
from DDIM.ddim_mnist import SimpleAutoencoder

autoencoder = SimpleAutoencoder()
autoencoder.load_checkpoint("DDIM/autoencoder_weights.pt")

# Extract features for use with BLL_VI
with torch.no_grad():
    features = autoencoder.encode(images)  # images is your MNIST tensor

# Now you can use these features with your BLL_VI model
```

## References

- [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502)
- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [Diffusion Autoencoders](https://arxiv.org/abs/2205.14772) 