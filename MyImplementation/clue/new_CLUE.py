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
        If target_class is None:
            L(z) = uncertainty_weight * H(y|z) + distance_weight * || z - z0 ||_2
        If target_class is specified:
            L(z) = uncertainty_weight * (-log(p[target_class])) + distance_weight * || z - z0 ||_2
    """

    def __init__(self, classifier, z0, uncertainty_weight=1.0, distance_weight=1.0, lr=0.1, device='cpu', 
                 bayesian=False, verbose=True, target_class=None):
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
            bayesian: Whether to use Bayesian uncertainty measures.
            verbose: Whether to print progress during optimization.
            target_class: Optional target class index. If provided, optimization will aim to maximize
                         the probability of this class instead of minimizing entropy.
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
        self.bayesian = bayesian
        self.verbose = verbose
        self.target_class = target_class

    def get_class_logit(self):
        """
        Computes the logit value for the target class.
        
        Returns:
            A scalar tensor representing the logit for the target class.
        """
        logits = self.classifier.classifier(self.z)
        return logits[0, self.target_class]

    def get_class_probability(self):
        """
        Computes the post-softmax probability for the target class.
        
        Returns:
            A scalar tensor representing the probability for the target class.
        """
        logits = self.classifier.classifier(self.z)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs[0, self.target_class]

    def predict_uncertainty(self):
        """
        Computes the uncertainty from the classifier's predictions on the latent code.
        Here we use the entropy of the softmax output of the classifier head.
        
        Returns:
            A scalar tensor representing the uncertainty.
        """
        # Directly obtain prediction logits from the classifier head.
        logits = self.classifier.classifier(self.z)      
        probs = torch.nn.functional.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        return entropy

    def predict_uncertainty_bayesian(self, num_samples=None):
        """Compute total, aleatoric and epistemic uncertainty from Bayesian samples."""
        # Get samples - these are already probabilities from BLL's sample_predict_z
        probs = self.classifier.sample_predict_z(self.z, num_samples)  # [num_samples, 1, num_classes]
        
        # No need to apply softmax since BLL already returns probabilities
        # probs = torch.nn.functional.softmax(samples, dim=2)  # Remove this line
        
        # Compute decomposed uncertainties
        posterior_preds = probs.mean(dim=0, keepdim=False)  # [1, num_classes]
        total_entropy = -(posterior_preds * torch.log(posterior_preds + 1e-10)).sum(dim=1)
        
        sample_preds_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=2)
        aleatoric_entropy = sample_preds_entropy.mean(dim=0)
        
        epistemic_entropy = total_entropy - aleatoric_entropy
        
        return total_entropy, aleatoric_entropy, epistemic_entropy

    def get_target_class_probability_bayesian(self, num_samples=None):
        """Get the probability of the target class using Bayesian sampling."""
        probs = self.classifier.sample_predict_z(self.z, num_samples)  # [num_samples, 1, num_classes]
        posterior_preds = probs.mean(dim=0, keepdim=False)  # [1, num_classes]
        return posterior_preds[0, self.target_class]

    def optimize(self, steps=25):
        """
        Optimizes the latent code by minimizing the objective:
            If target_class is None:
                loss = uncertainty_weight * H(y|z) + distance_weight * ||z - z0||_2
            If target_class is specified:
                loss = uncertainty_weight * (-log(p[target_class])) + distance_weight * ||z - z0||_2

        Args:
            steps: Number of gradient steps to perform.

        Returns:
            The optimized latent code (as a torch.Tensor with no gradient).
        """
        
        for step in range(steps):
            self.optimizer.zero_grad()
            
            distance = torch.norm(self.z - self.z0, p=2)
            
            if self.target_class is not None:
                # Optimize for target class
                if self.bayesian:
                    # For Bayesian case, maximize probability of target class
                    target_prob = self.get_target_class_probability_bayesian()
                    # Negative log probability (to be minimized)
                    target_term = -torch.log(target_prob + 1e-10)
                    loss = self.uncertainty_weight * target_term + self.distance_weight * distance
                    
                    if self.verbose:
                        print(f"Step {step:02d}: Loss: {loss.item():.4f}, Target Class Prob: {target_prob.item():.4f}, Distance: {distance.item():.4f}")
                else:
                    # For non-Bayesian case, maximize probability
                    target_prob = self.get_class_probability()
                    # Negative log probability (to be minimized)
                    target_term = -torch.log(target_prob + 1e-10)
                    loss = self.uncertainty_weight * target_term + self.distance_weight * distance
                    
                    if self.verbose:
                        print(f"Step {step:02d}: Loss: {loss.item():.4f}, Target Class Prob: {target_prob.item():.4f}, Distance: {distance.item():.4f}")
            else:
                # Original entropy-based optimization
                if self.bayesian:
                    total_entropy, aleatoric_entropy, epistemic_entropy = self.predict_uncertainty_bayesian()
                else:
                    total_entropy = self.predict_uncertainty()
                    aleatoric_entropy = total_entropy  # For non-Bayesian case, no uncertainty decomposition
                    epistemic_entropy = torch.tensor(0.0).to(self.device)
                    
                loss = self.uncertainty_weight * total_entropy + self.distance_weight * distance
                
                if self.verbose:
                    print(f"Step {step:02d}: Loss: {loss.item():.4f}, Total Entropy: {total_entropy.item():.4f}, Epistemic Entropy: {epistemic_entropy.item():.4f}, Aleatoric Entropy: {aleatoric_entropy.item():.4f}, Distance: {distance.item():.4f}")
            
            loss.backward()
            self.optimizer.step()

        return self.z.detach()