import torch
import torch.nn as nn
import torch.optim as optim

class AE_CLUE:
    """
    A simplified version of the CLUE algorithm that optimizes a single input point
    directly to reduce the classifier's predictive uncertainty (via softmax entropy) 
    while keeping the change from the original input minimal.

    This version accepts a classifier, and directly optimises the input based on
    the uncertainty of the classifier's predictions on the input. Works with any input dimension.

    The loss being minimized is:
        L(x) = uncertainty_weight * H(y|x) + distance_weight * || x - x0 ||_2
    """

    def __init__(self, classifier, x0, uncertainty_weight=1.0, distance_weight=1.0, lr=0.1, device='cpu', bayesian=False, verbose=True):
        """
        Args:
            classifier: A classifier which transforms the input into predictions.
            x0: The initial input (array-like or torch.Tensor) for the data point to explain.
            uncertainty_weight: Weight for the uncertainty (entropy) term.
            distance_weight: Weight for the distance penalty term, which ensures the explanation remains
                             close to the original input.
            lr: Learning rate for the Adam optimizer.
            device: 'cpu' or 'cuda'.
        """
        self.device = device
        if not torch.is_tensor(x0):
            x0 = torch.tensor(x0, dtype=torch.float)
        self.x0 = x0.to(self.device)
        # Ensure x0 has a batch dimension
        if len(self.x0.shape) == 1:
            self.x0 = self.x0.unsqueeze(0)
        # The input variable to be optimized is initialized to x0
        self.x = torch.nn.Parameter(self.x0.clone())
        self.classifier = classifier.to(self.device)
        self.uncertainty_weight = uncertainty_weight
        self.distance_weight = distance_weight
        self.lr = lr
        self.optimizer = torch.optim.Adam([self.x], lr=lr)
        self.bayesian = bayesian
        self.verbose = verbose

    def predict_uncertainty(self):
        """
        Computes the uncertainty from the classifier's predictions on the input.
        Here we use the entropy of the softmax output of the classifier.
        
        Returns:
            A scalar tensor representing the uncertainty.
        """
        # Directly obtain prediction logits from the classifier
        logits = self.classifier(self.x)      
        probs = torch.nn.functional.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        return entropy

    def predict_uncertainty_bayesian(self, num_samples=None):
        """Compute total, aleatoric and epistemic uncertainty from Bayesian samples."""
        # Get samples - these are already probabilities from classifier's sample_predict
        probs = self.classifier.sample_predict(self.x, num_samples)  # [num_samples, batch_size, num_classes]
        
        # Compute decomposed uncertainties
        posterior_preds = probs.mean(dim=0, keepdim=False)  # [batch_size, num_classes]
        total_entropy = -(posterior_preds * torch.log(posterior_preds + 1e-10)).sum(dim=1)
        
        sample_preds_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=2)
        aleatoric_entropy = sample_preds_entropy.mean(dim=0)
        
        epistemic_entropy = total_entropy - aleatoric_entropy
        
        return total_entropy, aleatoric_entropy, epistemic_entropy


    def optimize(self, steps=25):
        """
        Optimizes the input by minimizing the objective:
            loss = uncertainty_weight * H(y|x) + distance_weight * ||x - x0||_2

        Args:
            steps: Number of gradient steps to perform.

        Returns:
            The optimized input (as a torch.Tensor with no gradient).
        """
        
        for step in range(steps):
            self.optimizer.zero_grad()

            if self.bayesian:
                total_entropy, aleatoric_entropy, epistemic_entropy = self.predict_uncertainty_bayesian()
            else:
                total_entropy = self.predict_uncertainty()
                aleatoric_entropy = total_entropy  # For non-Bayesian case, no uncertainty decomposition
                epistemic_entropy = torch.tensor(0.0).to(self.device)
                
            distance = torch.norm(self.x - self.x0, p=2)
            loss = self.uncertainty_weight * total_entropy + self.distance_weight * distance
            loss.backward()
            self.optimizer.step()
            if self.verbose:
                print(f"Step {step:02d}: Loss: {loss.item():.4f}, Total Entropy: {total_entropy.item():.4f}, Epistemic Entropy: {epistemic_entropy.item():.4f}, Aleatoric Entropy: {aleatoric_entropy.item():.4f}, Distance: {distance.item():.4f}")
        return self.x.detach()