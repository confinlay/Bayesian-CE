import torch
import torch.nn as nn
import torch.optim as optim

class NewCLUE:
    """
    A simplified version of the CLUE algorithm that optimizes a single latent point
    directly in the intermediate feature space (latent space) produced, for example, by the
    encoder of your regene_models.Classifier. The objective is a combination of reducing the
    classifier's predictive uncertainty (via softmax entropy) while keeping the change
    from the original latent code minimal.

    This version accepts a classification layer, and directly optimises the latent code based on
    the uncertainty of the classifier's predictions on the latent code. No need for a decoder.

    The loss being minimized is:
        L(z) = uncertainty_weight * H(y|z) + distance_weight * || z - z0 ||_2
    """

    def __init__(self, classifier, z0, uncertainty_weight=1.0, distance_weight=1.0, lr=0.1, device='cpu'):
        """
        Args:
            classifier: A classification layer which transforms the latent code into a logit.
            z0: The initial latent code (array-like or torch.Tensor) for the data point to explain.
                Typically obtained from the encoder (shape: [1, latent_dim]).
            uncertainty_weight: Weight for the uncertainty (entropy) term.
            distance_weight: Weight for the distance penalty term, which ensures the explanation remains
                             close to the original latent code.
            lr: Learning rate for the Adam optimizer.
            device: 'cpu' or 'cuda'.
        """
        self.device = device
        if not torch.is_tensor(z0):
            z0 = torch.tensor(z0, dtype=torch.float)
        self.z0 = z0.to(self.device)
        # Ensure z0 is of shape [1, latent_dim] (batch dimension)
        if len(self.z0.shape) == 1:
            self.z0 = self.z0.unsqueeze(0)
        # The latent variable to be optimized is initialized to z0
        self.z = torch.nn.Parameter(self.z0.clone())
        self.classifier = classifier.to(self.device)
        self.uncertainty_weight = uncertainty_weight
        self.distance_weight = distance_weight
        self.lr = lr
        self.optimizer = torch.optim.Adam([self.z], lr=lr)

    def predict_uncertainty(self):
        """
        Computes the uncertainty from the classifier's predictions on the latent code.
        Here we use the entropy of the softmax output of the classifier head.
        
        Returns:
            A scalar tensor representing the uncertainty.
        """
        # Directly obtain prediction logits from the classifier head.
        logits = self.classifier(self.z)      
        probs = torch.nn.functional.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        return entropy

    def optimize(self, steps=25):
        """
        Optimizes the latent code by minimizing the objective:
            loss = uncertainty_weight * H(y|z) + distance_weight * ||z - z0||_2

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