import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCLUE:
    """
    A simplified version of the CLUE algorithm that optimizes a single latent point
    directly in the latent space of a VAE. At each optimization step, the VAE
    generates an image from the current latent code which is then fed into a classifier.
    The objective is to increase the classifier's predictive uncertainty (via softmax entropy)
    while keeping the latent code close to its original value.

    Thie version requires two models. The VAE decodes the latent point at each step so that 
    it can be fed into the classifier.
    
    The loss being minimized is:
        L(z) = uncertainty_weight * H(y|x_generated) + distance_weight * ||z - z0||_2
    where x_generated = VAE.decoder(z)
    """
    
    def __init__(self, vae, classifier, z0, uncertainty_weight=1.0, distance_weight=1.0, lr=0.1, device='cpu'):
        """
        Args:
            vae: A VAE model with a decoder. It should implement vae.decoder(z).
            classifier: A classifier model that accepts an image and outputs logits.
            z0: The initial latent code (array-like, torch.Tensor, or tuple) for the data point to explain.
                Typically obtained from the VAE's encode() method. If a tuple (mu, log_var) is passed,
                the first element (mu) is used.
            uncertainty_weight: Weight for the uncertainty (entropy) term.
            distance_weight: Weight for the distance penalty term.
            lr: Learning rate for the Adam optimizer.
            device: 'cpu' or 'cuda'.
        """
        self.device = device
        
        # If z0 is a tuple (e.g. (mu, log_var)), take the first element (mu)
        if isinstance(z0, tuple):
            z0 = z0[0]
        # Ensure z0 is a tensor
        if not torch.is_tensor(z0):
            z0 = torch.tensor(z0, dtype=torch.float)
        self.z0 = z0.to(self.device)
        
        # Ensure z0 has a batch dimension [1, latent_dim]
        if len(self.z0.shape) == 1:
            self.z0 = self.z0.unsqueeze(0)
        # The latent variable to be optimized is initialized to z0.
        self.z = torch.nn.Parameter(self.z0.clone())
        
        self.vae = vae.to(self.device)
        self.classifier = classifier.to(self.device)
        self.uncertainty_weight = uncertainty_weight
        self.distance_weight = distance_weight
        self.lr = lr
        self.optimizer = torch.optim.Adam([self.z], lr=lr)

    def predict_uncertainty(self):
        """
        Generates an image using the VAE's decoder from the current latent code,
        obtains the classifier logits for that image, and computes the softmax entropy.
        
        Returns:
            A scalar tensor representing the uncertainty.
        """
        # Generate an image from the current latent code.
        x_generated = self.vae.decoder(self.z)
        # Get classifier predictions on the generated image.
        logits = self.classifier(x_generated)
        probs = torch.nn.functional.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        return entropy

    def optimize(self, steps=25):
        """
        Optimizes the latent code by minimizing:
            loss = uncertainty_weight * H(y|x_generated) + distance_weight * ||z - z0||_2
        
        Args:
            steps: Number of gradient steps to perform.
        
        Returns:
            The optimized latent code (as a torch.Tensor with no gradient).
        """
        for step in range(steps):
            self.optimizer.zero_grad()
            entropy = self.predict_uncertainty()
            distance = torch.norm(self.z - self.z0, p=2)
            loss = self.uncertainty_weight * entropy + self.distance_weight * distance
            loss.backward()
            self.optimizer.step()
            print(f"Step {step:02d}: Loss: {loss.item():.4f}, Entropy: {entropy.item():.4f}, Distance: {distance.item():.4f}")
        return self.z.detach()