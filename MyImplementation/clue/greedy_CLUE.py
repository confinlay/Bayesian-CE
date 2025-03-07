import torch
import torch.nn as nn
import torch.optim as optim

class GreedyCLUE:
    """
    A simplified version of the CLUE algorithm that optimizes a single latent point
    by greedily updating the most salient dimension by a fixed delta.
    """

    def __init__(self, classifier, z0, uncertainty_weight=1.0, distance_weight=1.0, delta=0.1, device='cpu', bayesian=False, verbose=True):
        """
        Args:
            classifier: A classification layer which transforms the latent code into a logit.
            z0: The initial latent code (array-like or torch.Tensor) for the data point to explain.
                Typically obtained from the encoder (shape: [1, latent_dim]).
            uncertainty_weight: Weight for the uncertainty (entropy) term.
            distance_weight: Weight for the distance penalty term.
            delta: Size of the step to take in each dimension.
            device: 'cpu' or 'cuda'.
            bayesian: Whether to use Bayesian uncertainty estimation.
            verbose: Whether to print progress information.
        """
        self.device = device
        if not torch.is_tensor(z0):
            z0 = torch.tensor(z0, dtype=torch.float)
        self.z0 = z0.to(self.device)
        if len(self.z0.shape) == 1:
            self.z0 = self.z0.unsqueeze(0)
        self.z = torch.nn.Parameter(self.z0.clone())
        self.classifier = classifier.to(self.device)
        self.uncertainty_weight = uncertainty_weight
        self.distance_weight = distance_weight
        self.delta = delta
        self.bayesian = bayesian
        self.verbose = verbose

    def predict_uncertainty(self):
        """Computes the uncertainty from the classifier's predictions."""
        logits = self.classifier(self.z)      
        probs = torch.nn.functional.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        return entropy

    def predict_uncertainty_bayesian(self):
        """Compute total, aleatoric and epistemic uncertainty from Bayesian samples."""
        probs = self.classifier.sample_predict_z(self.z)
        
        posterior_preds = probs.mean(dim=0, keepdim=False)
        total_entropy = -(posterior_preds * torch.log(posterior_preds + 1e-10)).sum(dim=1)
        
        sample_preds_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=2)
        aleatoric_entropy = sample_preds_entropy.mean(dim=0)
        
        epistemic_entropy = total_entropy - aleatoric_entropy
        
        return total_entropy, aleatoric_entropy, epistemic_entropy

    def compute_loss(self, z):
        """Compute the loss for a given latent code z."""
        if self.bayesian:
            total_entropy, _, _ = self.predict_uncertainty_bayesian()
        else:
            total_entropy = self.predict_uncertainty()
        distance = torch.norm(z - self.z0, p=2)
        loss = self.uncertainty_weight * total_entropy + self.distance_weight * distance
        return loss

    def get_confidence(self):
        """Get the confidence score for the current latent code."""
        with torch.no_grad():
            if self.bayesian:
                probs = self.classifier.sample_predict_z(self.z)
                mean_probs = probs.mean(dim=0)
                confidence = mean_probs.max(dim=1)[0]
            else:
                logits = self.classifier(self.z)
                probs = torch.nn.functional.softmax(logits, dim=1)
                confidence = probs.max(dim=1)[0]
        return confidence

    def optimize(self, steps=25, max_changes_per_dim=3, confidence_threshold=0.99):
        """
        Optimizes the latent code by iteratively updating the most salient dimension.
        
        Args:
            steps: Maximum number of gradient steps to perform.
            max_changes_per_dim: Maximum number of times each dimension can be modified.
            confidence_threshold: Stop when confidence exceeds this threshold.
            
        Returns:
            The optimized latent code (as a torch.Tensor with no gradient).
        """
        best_z = self.z.clone()
        best_loss = self.compute_loss(self.z)
        
        # Track number of changes per dimension (P in Algorithm 1)
        changes_per_dim = torch.zeros(self.z.shape[1], dtype=torch.int, device=self.device)
        
        for step in range(steps):
            # Check if we've reached the confidence threshold
            confidence = self.get_confidence()
            if confidence >= confidence_threshold:
                if self.verbose:
                    print(f"Reached confidence threshold {confidence_threshold:.4f} after {step} steps. Current confidence: {confidence.item():.4f}")
                break
                
            # Compute loss and gradients
            if self.bayesian:
                total_entropy, aleatoric_entropy, epistemic_entropy = self.predict_uncertainty_bayesian()
            else:
                total_entropy = self.predict_uncertainty()
                aleatoric_entropy = total_entropy
                epistemic_entropy = torch.tensor(0.0).to(self.device)
                
            distance = torch.norm(self.z - self.z0, p=2)
            loss = self.uncertainty_weight * total_entropy + self.distance_weight * distance
            
            # Compute gradients
            loss.backward()
            
            # Find dimension with largest gradient magnitude that hasn't reached max changes
            with torch.no_grad():
                grad_magnitudes = torch.abs(self.z.grad[0])
                
                # Create a mask for dimensions that can still be changed
                valid_dims = torch.ones_like(grad_magnitudes, dtype=torch.bool)
                for dim in range(valid_dims.shape[0]):
                    if changes_per_dim[dim] >= max_changes_per_dim:
                        valid_dims[dim] = False
                
                # Apply mask to gradients
                masked_grads = grad_magnitudes.clone()
                masked_grads[~valid_dims] = 0.0
                
                # If all dimensions are at max changes, break or reset counters
                if masked_grads.max() == 0:
                    if self.verbose:
                        print(f"All dimensions reached maximum changes limit ({max_changes_per_dim}). Stopping.")
                    break
                
                # Select best dimension and update
                best_dim = torch.argmax(masked_grads)
                update = -self.delta * torch.sign(self.z.grad[0, best_dim])
                self.z.data[0, best_dim] += update
                
                # Update change counter for this dimension
                changes_per_dim[best_dim] += 1
                
                if self.verbose:
                    print(f"Step {step:02d}: Loss: {loss.item():.4f}, Total Entropy: {total_entropy.item():.4f}, "
                          f"Updated dim: {best_dim.item()}, Delta: {update.item():.4f}, Changes: {changes_per_dim[best_dim].item()}/{max_changes_per_dim}")
                
                # Zero gradients for next iteration
                self.z.grad.zero_()
                
                # Keep track of best result
                current_loss = self.compute_loss(self.z)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_z = self.z.clone()
        
        return best_z.detach()