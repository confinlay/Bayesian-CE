from __future__ import division

import sys
import os
from pathlib import Path

# 1. Navigate up from current directory to project root
current_dir = Path(__file__).parent.absolute()  # /MyImplementation/VAEAC
project_root = current_dir.parent.parent  # Assuming structure: /root/MyImplementation/VAEAC

# 2. Add old src directory to Python path
old_src_path = project_root / "OldStuff"
sys.path.insert(0, str(old_src_path))

# 3. Verify imports work
try:
    from src.utils import BaseNet, to_variable  # Should now work
    print("Successfully imported from src!")
except ImportError:
    print(f"Path configuration failed. Current sys.path: {sys.path}")


import torch
import numpy as np
from src.utils import BaseNet, to_variable, cprint
from src.probability import normal_parse_params
from src.radam import RAdam
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.distributions import kl_divergence
import torch.backends.cudnn as cudnn
from .models import MNIST_recognition_resnet, MNIST_generator_resnet, MNIST_prior_resnet


class MNIST_VAEAC_bern(nn.Module):
    def __init__(self, latent_dim, targets=False, recognition_net=None, generator_net=None, prior_net=None):
        super(MNIST_VAEAC_bern, self).__init__()
        self.latent_dim = latent_dim
        self.pred_sig = False

        if recognition_net is None:
            self.recognition_net = MNIST_recognition_resnet(latent_dim, targets)
        else:
            self.recognition_net = recognition_net

        if generator_net is None:
            self.generator_net = MNIST_generator_resnet(latent_dim, targets)
        else:
            self.generator_net = generator_net

        if prior_net is None:
            self.prior_net = MNIST_prior_resnet(latent_dim, targets)
        else:
            self.prior_net = prior_net


        self.m_rec_loglike = BCEWithLogitsLoss(reduction='none')  # GaussianLoglike(min_sigma=1e-2)
        self.sigma_mu = 1e4
        self.sigma_sigma = 1e-4

    @staticmethod
    def apply_mask(x, mask):
        observed = x.clone()  # torch.tensor(x)
        observed[mask.bool()] = 0
        return observed

    def recognition_encode(self, x):
        approx_post_params = self.recognition_net(x)
        approx_post = normal_parse_params(approx_post_params, 1e-3)
        return approx_post

    def prior_encode(self, x, mask):
        x = self.apply_mask(x, mask)
        x = torch.cat([x, mask], dim=1)
        prior_params = self.prior_net(x)
        prior = normal_parse_params(prior_params, 1e-3)
        return prior

    def decode(self, z_sample):
        rec_params = self.generator_net(z_sample)
        return rec_params

    def reg_cost(self, prior):
        num_objects = prior.mean.shape[0]
        mu = prior.mean.view(num_objects, -1)
        sigma = prior.scale.view(num_objects, -1)
        mu_regularizer = -(mu ** 2).sum(-1) / 2 / (self.sigma_mu ** 2)
        sigma_regularizer = (sigma.log() - sigma).sum(-1) * self.sigma_sigma
        return mu_regularizer + sigma_regularizer

    def vlb(self, prior, approx_post, x, rec_params):
        rec = -self.m_rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)
        kl = kl_divergence(approx_post, prior).view(x.shape[0], -1).sum(-1)
        prior_regularization = self.reg_cost(prior).view(x.shape[0], -1).sum(-1)
        return rec - kl + prior_regularization  # signs for prior regulariser are already taken into account

    def iwlb(self, prior, approx_post, x, K=50):
        estimates = []
        for i in range(K):
            latent = approx_post.rsample()
            rec_params = self.decode(latent)
            rec_loglike = -self.m_rec_loglike(rec_params, x).view(x.shape[0], -1).sum(-1)

            prior_log_prob = prior.log_prob(latent)
            prior_log_prob = prior_log_prob.view(x.shape[0], -1)
            prior_log_prob = prior_log_prob.sum(-1)

            proposal_log_prob = approx_post.log_prob(latent)
            proposal_log_prob = proposal_log_prob.view(x.shape[0], -1)
            proposal_log_prob = proposal_log_prob.sum(-1)

            estimate = rec_loglike + prior_log_prob - proposal_log_prob
            estimates.append(estimate[:, None])

        return torch.logsumexp(torch.cat(estimates, 1), 1) - np.log(K)


def get_best_device(use_gpu=True):
    """
    Determines best available device with priority:
    1. CUDA (if available and on Linux/Windows)
    2. MPS (if available and on macOS)
    3. CPU (fallback)
    """
    if not use_gpu:
        return torch.device('cpu')
        
    # Check CUDA first (preferred for non-macOS)
    if torch.cuda.is_available():
        return torch.device('cuda')
        
    # Then check MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        return torch.device('mps')
        
    # Fallback to CPU
    return torch.device('cpu')

class VAEAC_bern_net(BaseNet):
    def __init__(self, input_dim, latent_dim, lr=1e-4, targets=False,
                 recognition_net=None, generator_net=None, prior_net=None, use_gpu=True):
        super(VAEAC_bern_net, self).__init__()
        cprint('y', 'VAE_bern_net')
        
        # Get the best available device
        self.device = get_best_device(use_gpu)
        self.use_gpu = self.device.type in ['cuda', 'mps']
        
        # Print device info
        print(f"Using device: {self.device}")
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.targets = targets
        self.lr = lr

        self.create_net(recognition_net, generator_net, prior_net)
        self.create_opt()
        self.epoch = 0
        self.schedule = None

        self.vlb_scale = 1 / input_dim

    def create_net(self, recognition_net, generator_net, prior_net):
        torch.manual_seed(42)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(42)
        
        self.model = MNIST_VAEAC_bern(self.latent_dim, targets=self.targets,
                                     recognition_net=recognition_net,
                                     generator_net=generator_net,
                                     prior_net=prior_net)
        
        if self.use_gpu:
            self.model = self.model.to(self.device)
            if self.device.type == 'cuda':
                cudnn.benchmark = True
            
        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        self.optimizer = RAdam(self.model.parameters(), lr=self.lr) # torch.optim.Adam

    def fit(self, x, mask):
        self.set_mode_train(train=True)
        
        try:
            # Move data to MPS if available
            x = x.to(self.device)
            mask = mask.to(self.device)
            
            self.optimizer.zero_grad()

            prior = self.model.prior_encode(x, mask)
            approx_post = self.model.recognition_encode(x)
            z_sample = approx_post.rsample()
            rec_params = self.model.decode(z_sample)

            vlb = self.model.vlb(prior, approx_post, x, rec_params)
            loss = (- vlb * self.vlb_scale).mean()

            loss.backward()
            self.optimizer.step()

            return vlb.mean().item(), torch.sigmoid(rec_params.data)
            
        except RuntimeError as e:
            print(f"Runtime error during training: {e}")
            raise

    def evaluate(self, x, mask, sample=False):
        """Evaluate the model (formerly eval())"""
        self.set_mode_train(train=False)
        
        x = x.to(self.device)
        mask = mask.to(self.device)

        prior = self.model.prior_encode(x, mask)
        approx_post = self.model.recognition_encode(x)
        
        if sample:
            z_sample = approx_post.sample()
        else:
            z_sample = approx_post.loc
            
        rec_params = self.model.decode(z_sample)
        vlb = self.model.vlb(prior, approx_post, x, rec_params)

        return vlb.mean().item(), torch.sigmoid(rec_params.data)

    def eval_iw(self, x, mask, k=50):
        self.set_mode_train(train=False)
        x, mask = to_variable(var=(x, mask), device=self.device)
        
        prior = self.model.prior_encode(x, mask)
        approx_post = self.model.recognition_encode(x)
        
        iw_lb = self.model.iwlb(prior, approx_post, x, k)
        return iw_lb.mean().item()

    def get_prior(self, x, mask):
        self.set_mode_train(train=False)
        x, mask = to_variable(var=(x, mask), device=self.device)
        prior = self.model.prior_encode(x, mask)
        return prior

    def get_post(self, x):
        self.set_mode_train(train=False)
        x, = to_variable(var=(x,), device=self.device)
        approx_post = self.model.recognition_encode(x)
        return approx_post

    def inpaint(self, x, mask, Nsample=1, z_mean=False, logits=False):
        self.set_mode_train(train=False)
        
        x = x.to(self.device)
        mask = mask.to(self.device)
        
        prior = self.model.prior_encode(x, mask)
        out = []
        
        for i in range(Nsample):
            if z_mean:
                z_sample = prior.loc.data
            else:
                z_sample = prior.sample()
            rec_params = self.model.decode(z_sample)
            out.append(rec_params.data)
            
        out = torch.stack(out, dim=0)
        
        if logits:
            return out
        else:
            return torch.sigmoid(out)

    def regenerate(self, z, grad=False):
        self.set_mode_train(train=False)
        if grad:
            if not z.requires_grad:
                z.requires_grad = True
            z = z.to(self.device)
        else:
            z, = to_variable(var=(z,), device=self.device)
        out = self.model.decode(z)
        return torch.sigmoid(out.data)

    def save(self, filename):
        """Save method with proper device handling and error checking"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Move model to CPU for saving
            model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            optimizer_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                              for k, v in self.optimizer.state_dict().items()}
            
            state_dict = {
                'model_state': model_state,
                'optimizer_state': optimizer_state,
                'epoch': self.epoch,
                'device_type': self.device.type,
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'targets': self.targets,
                'lr': self.lr
            }
            
            torch.save(state_dict, filename)
            print(f"Model saved successfully to {filename}")
            
        except Exception as e:
            print(f"Error saving model to {filename}: {str(e)}")
            raise

    def load(self, filename):
        """Load method with proper device handling and error checking"""
        try:
            # Load state dict to CPU first
            state_dict = torch.load(filename, map_location='cpu')
            
            # Load model parameters
            self.model.load_state_dict(state_dict['model_state'])
            self.optimizer.load_state_dict(state_dict['optimizer_state'])
            
            # Load other attributes
            self.epoch = state_dict['epoch']
            self.input_dim = state_dict.get('input_dim', self.input_dim)
            self.latent_dim = state_dict.get('latent_dim', self.latent_dim)
            self.targets = state_dict.get('targets', self.targets)
            self.lr = state_dict.get('lr', self.lr)
            
            # Move model to correct device
            self.model = self.model.to(self.device)
            
            # Move optimizer states to correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
                        
            print(f"Model loaded successfully from {filename}")
            
        except Exception as e:
            print(f"Error loading model from {filename}: {str(e)}")
            raise

