from __future__ import division
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import BCEWithLogitsLoss
from torch.distributions import Normal, kl_divergence
from src.layers import SkipConnection
from src.utils import BaseNet, to_variable, cprint
from src.probability import normal_parse_params
from src.radam import RAdam

class ConvMNISTVAE(nn.Module):
    """Convolutional Variational Autoencoder for MNIST with Bernoulli likelihood."""
    def __init__(self, latent_dim, encoder, decoder):
        super(ConvMNISTVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder  # Recognition network
        self.decoder = decoder  # Generation network
        self.reconstruction_loss = BCEWithLogitsLoss(reduction='none')

    def encode(self, x):
        """Encode input into approximate posterior distribution."""
        params = self.encoder(x)
        return normal_parse_params(params, eps=1e-3)

    def decode(self, z):
        """Decode latent variable to reconstruction parameters."""
        return self.decoder(z)

    def compute_vlb(self, prior, posterior, x, rec_params):
        """Compute the Variational Lower Bound (VLB)."""
        reconstruction = -self.reconstruction_loss(rec_params, x).view(x.size(0), -1).sum(dim=1)
        kl = kl_divergence(posterior, prior).view(x.size(0), -1).sum(dim=1)
        return reconstruction - kl

    def compute_iwlb(self, prior, posterior, x, num_samples=50):
        """Compute the Importance Weighted Lower Bound (IWLB)."""
        estimates = []
        for _ in range(num_samples):
            z = posterior.rsample()
            rec_params = self.decode(z)
            rec_loglik = -self.reconstruction_loss(rec_params, x).view(x.size(0), -1).sum(dim=1)

            prior_logprob = prior.log_prob(z).view(x.size(0), -1).sum(dim=1)
            posterior_logprob = posterior.log_prob(z).view(x.size(0), -1).sum(dim=1)

            estimate = rec_loglik + prior_logprob - posterior_logprob
            estimates.append(estimate.unsqueeze(1))

        stacked_estimates = torch.cat(estimates, dim=1)
        return torch.logsumexp(stacked_estimates, dim=1) - np.log(num_samples)

class MNISTVAEModel(BaseNet):
    """VAE model for MNIST using ConvMNISTVAE with Laplace approximation."""
    def __init__(self, latent_dim, encoder, decoder, learning_rate=1e-3, device=None):
        super(MNISTVAEModel, self).__init__()
        cprint('y', 'Initializing VAE Model')

        self.device = device or self.get_default_device()
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        self._initialize_model(encoder, decoder)
        self._initialize_optimizer()
        self.epoch = 0
        self.scheduler = None

        self.prior = Normal(
            loc=torch.zeros(latent_dim, device=self.device),
            scale=torch.ones(latent_dim, device=self.device)
        )
        self.vlb_scale = 1 / 784  # Normalize VLB

    def get_default_device(self):
        """Select the best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def _initialize_model(self, encoder, decoder):
        """Set up the VAE model."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        self.model = ConvMNISTVAE(self.latent_dim, encoder, decoder).to(self.device)
        if self.device.type == 'cuda':
            cudnn.benchmark = True
        print(f'    Total parameters: {self.get_num_parameters() / 1e6:.2f}M')

    def get_num_parameters(self):
        """Calculate total number of parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def _initialize_optimizer(self):
        """Set up the optimizer."""
        self.optimizer = RAdam(self.model.parameters(), lr=self.learning_rate)

    def fit(self, x):
        """Train the VAE on a batch of data."""
        self.set_train_mode(True)
        x = x.to(self.device)

        self.optimizer.zero_grad()
        posterior = self.model.encode(x)
        z = posterior.rsample()
        rec_params = self.model.decode(z)

        vlb = self.model.compute_vlb(self.prior, posterior, x, rec_params)
        loss = (-vlb * self.vlb_scale).mean()

        loss.backward()
        self.optimizer.step()

        return vlb.mean().item(), torch.sigmoid(rec_params)

    def evaluate(self, x, sample=False):
        """Evaluate the VAE on a batch of data."""
        self.set_train_mode(False)
        x = x.to(self.device)

        posterior = self.model.encode(x)
        z = posterior.sample() if sample else posterior.loc
        rec_params = self.model.decode(z)

        vlb = self.model.compute_vlb(self.prior, posterior, x, rec_params)
        return vlb.mean().item(), torch.sigmoid(rec_params)

    def evaluate_iw(self, x, num_samples=50):
        """Evaluate the VAE using Importance Weighted estimation."""
        self.set_train_mode(False)
        x = x.to(self.device)

        posterior = self.model.encode(x)
        iw_lb = self.model.compute_iwlb(self.prior, posterior, x, num_samples)
        return iw_lb.mean().item()

    def recognition(self, x, requires_grad=False):
        """Get the approximate posterior for input x."""
        self.set_train_mode(False)
        if requires_grad:
            x.requires_grad_(True)
        else:
            x = x.to(self.device)
        return self.model.encode(x)

    def regenerate(self, z, requires_grad=False):
        """Regenerate data from latent variable z."""
        self.set_train_mode(False)
        if requires_grad:
            z.requires_grad_(True)
        else:
            z = z.to(self.device)
        out = self.model.decode(z)
        return torch.sigmoid(out) if requires_grad else torch.sigmoid(out.detach())

# Aliases for easier access
conv_VAE_bern = ConvMNISTVAE
conv_VAE_bern_net = MNISTVAEModel