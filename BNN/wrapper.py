from __future__ import division
import copy
import torch
import torch.nn as nn
from .models import MLP, MNIST_small_cnn
from src.probability import diagonal_gauss_loglike, get_rms, get_loglike
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from .sampler import H_SA_SGHMC
from src.utils import BaseNet, to_variable, cprint, save_object, load_object
import os
from src.probability import decompose_entropy_cat

class BNN_cat(BaseNet):  # for categorical distributions
    def __init__(self, model, N_train, lr=1e-2, grad_std_mul=30, seed=None, device=None):
        super(BNN_cat, self).__init__()

        cprint('y', 'BNN categorical output')
        # Determine the device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.lr = lr
        self.model = model
        self.seed = seed

        self.model.to(self.device)

        self.N_train = N_train
        self.create_net()
        self.create_opt()
        self.schedule = None  # [] #[50,200,400,600]
        self.epoch = 0

        self.grad_buff = []
        self.max_grad = 1e20
        self.grad_std_mul = grad_std_mul

        self.weight_set_samples = []

    def create_net(self):
        # Set the seed for CPU
        if self.seed is None:
            torch.manual_seed(42)
        else:
            torch.manual_seed(self.seed)

        # Set the seed for CUDA if using CUDA
        if self.device.type == 'cuda':
            if self.seed is None:
                torch.cuda.manual_seed(42)
            else:
                torch.cuda.manual_seed(self.seed)

            # Set cudnn.benchmark for performance optimization
            cudnn.benchmark = True

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        """This optimiser incorporates the gaussian prior term automatically. The prior variance is gibbs sampled from
        its posterior using a gamma hyper-prior."""
        self.optimizer = H_SA_SGHMC(params=self.model.parameters(), lr=self.lr, base_C=0.05, gauss_sig=0.1)  # this last parameter does nothing

    def fit(self, x, y, burn_in=False, resample_momentum=False, resample_prior=False):
        self.set_mode_train(train=True)
        x, y = to_variable(var=(x, y.long()), cuda=self.device.type=='cuda')
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = F.cross_entropy(out, y, reduction='mean')
        loss = loss * self.N_train  # We use mean because we treat as an estimation of whole dataset
        loss.backward()

        if len(self.grad_buff) > 1000:
            grad_array = torch.tensor(self.grad_buff).cpu()
            self.max_grad = float(grad_array.mean() + self.grad_std_mul * grad_array.std())
            self.grad_buff.pop(0)

        grad_norm = nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                     max_norm=self.max_grad, norm_type=2)
        
        self.grad_buff.append(float(grad_norm.cpu()))
        
        if self.grad_buff[-1] >= self.max_grad:
            print(self.max_grad, self.grad_buff[-1])
            self.grad_buff.pop()
        
        self.optimizer.step(burn_in=burn_in, resample_momentum=resample_momentum, resample_prior=resample_prior)

        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data * x.shape[0] / self.N_train, err

    def eval(self, x, y, train=False):
        self.set_mode_train(train=False)
        x, y = to_variable(var=(x, y.long()), cuda=self.device.type=='cuda')

        out = self.model(x)
        loss = F.cross_entropy(out, y, reduction='sum')
        probs = F.softmax(out, dim=1).data.cpu()

        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def save_sampled_net(self, max_samples):

        if len(self.weight_set_samples) >= max_samples:
            self.weight_set_samples.pop(0)

        self.weight_set_samples.append(copy.deepcopy(self.model.state_dict()))

        cprint('c', ' saving weight samples %d/%d' % (len(self.weight_set_samples), max_samples))
        return None

    def predict(self, x):
        self.set_mode_train(train=False)
        x, = to_variable(var=(x, ), cuda=self.device.type=='cuda')
        out = self.model(x)
        probs = F.softmax(out, dim=1).data.cpu()
        return probs.data

    def sample_predict(self, x, Nsamples, grad=False):
        """return predictions using multiple samples from posterior"""
        self.set_mode_train(train=False)
        if Nsamples == 0:
            Nsamples = len(self.weight_set_samples)
        x, = to_variable(var=(x, ), cuda=self.device.type=='cuda')

        if grad:
            self.optimizer.zero_grad()
            if not x.requires_grad:
                x.requires_grad = True

        outputs = []
        with torch.no_grad():
            original_state = copy.deepcopy(self.model.state_dict())
            for idx, weight_dict in enumerate(self.weight_set_samples[:Nsamples]):
                self.model.load_state_dict(weight_dict)
                output = self.model(x)
                # Maintain the original logic for handling outputs
                if grad:
                    outputs.append(output.clone())
                else:
                    outputs.append(output.detach())

            # Restore original weights
            self.model.load_state_dict(original_state)

        out = torch.stack(outputs, dim=0)
        if grad:
            out.requires_grad_(True)
        
        # Softmax is applied over the classes here as it's not done in the model structur
        # (see MLP class in models.py)
        prob_out = F.softmax(out, dim=2)

        return prob_out
        # out = x.data.new(Nsamples, x.shape[0], self.model.output_dim)



        # # iterate over all saved weight configuration samples

        # for idx, weight_dict in enumerate(self.weight_set_samples):

        #     if idx == Nsamples:

        #         break

        #     self.model.load_state_dict(weight_dict)

        #     out[idx] = self.model(x)



        # out = out[:idx]

        # prob_out = F.softmax(out, dim=2)



        # if grad:

        #     return prob_out

        # else:

        #     return prob_out.data

    def get_weight_samples(self, Nsamples=0):
        """return weight samples from posterior in a single-column array"""
        weight_vec = []

        if Nsamples == 0 or Nsamples > len(self.weight_set_samples):
            Nsamples = len(self.weight_set_samples)

        for idx, state_dict in enumerate(self.weight_set_samples):
            if idx == Nsamples:
                break

            for key in state_dict.keys():
                if 'weight' in key:
                    weight_mtx = state_dict[key].cpu().data
                    for weight in weight_mtx.view(-1):
                        weight_vec.append(weight)

        return np.array(weight_vec)

    def evaluate_uncertainty(self, probs, true_labels):
        """
        Evaluate correlations between different types of entropy and prediction errors.
        
        Args:
            probs (torch.Tensor): Prediction probabilities (Nsamples, batch_size, classes).
            true_labels (torch.Tensor): True labels (batch_size).
        
        Returns:
            dict: A dictionary containing correlations between prediction errors and
                 total, aleatoric, epistemic entropies, and sample disagreement.
        """
        # Ensure all tensors are on the same device
        true_labels = true_labels.to(self.device)
        
        total_entropy, aleatoric_entropy, epistemic_entropy = decompose_entropy_cat(probs)
        
        # Calculate prediction errors
        mean_probs = probs.mean(dim=0)
        pred_labels = mean_probs.argmax(dim=1)
        prediction_errors = (pred_labels != true_labels).float()
        
        # Get sample disagreement uncertainty
        sample_uncertainty = self.estimate_uncertainty_from_predictions(probs)
        
        # Calculate correlations with prediction errors
        total_correlation = torch.corrcoef(torch.stack((total_entropy, prediction_errors)))[0, 1].item()
        aleatoric_correlation = torch.corrcoef(torch.stack((aleatoric_entropy, prediction_errors)))[0, 1].item()
        epistemic_correlation = torch.corrcoef(torch.stack((epistemic_entropy, prediction_errors)))[0, 1].item()
        sample_correlation = torch.corrcoef(torch.stack((sample_uncertainty, prediction_errors)))[0, 1].item()
        # Calculate correlation for sum of total and sample uncertainties
        combined_uncertainty = total_entropy + sample_uncertainty
        combined_correlation = torch.corrcoef(torch.stack((combined_uncertainty, prediction_errors)))[0, 1].item()
        
        return (total_correlation, aleatoric_correlation, epistemic_correlation, sample_correlation, combined_correlation)
    
    
        
    def save_weights(self, filename):
        """Save the list of weight samples to a .pt file."""
        cprint('c', f'Saving weight samples to {filename}')
        torch.save(self.weight_set_samples, filename)

    def load_weights(self, filename, subsample=1):
        """Load the list of weight samples from a .pt file."""
        cprint('c', f'Loading weight samples from {filename}')
        
        # Check if the file exists and print its size
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file {filename} does not exist.")
        file_size = os.path.getsize(filename)
        print(f"File size: {file_size / (1024 * 1024):.2f} MB")

        # Load the weight samples
        self.weight_set_samples = torch.load(filename, map_location=self.device, weights_only=True)
        
        # Check if the loaded object is None
        if self.weight_set_samples is None:
            raise ValueError(f"Loaded object is None. Please check the file content of {filename}.")
        
        # Print the type and length of the loaded object
        print(f"Loaded object type: {type(self.weight_set_samples)}")
        if isinstance(self.weight_set_samples, list):
            print(f"Number of weight samples: {len(self.weight_set_samples)}")
        
        # Subsample if necessary
        self.weight_set_samples = self.weight_set_samples[::subsample]
    
    def estimate_uncertainty_from_predictions(self, probs):
        """
        Estimates uncertainty based on the fraction of samples predicting the most common class.
        
        Args:
            probs (torch.Tensor): Prediction probabilities (Nsamples, batch_size, classes)
            
        Returns:
            torch.Tensor: Uncertainty scores for each sample in the batch (batch_size)
                         where 0 means all samples agree (certain) and values closer to 1 
                         mean high disagreement (uncertain)
        """
        # Get predicted class for each sample
        pred_classes = torch.argmax(probs, dim=2)  # (Nsamples, batch_size)
        
        # Calculate the most common predicted class manually
        mode_classes = []
        for i in range(pred_classes.size(1)):  # Iterate over batch size
            unique, counts = torch.unique(pred_classes[:, i], return_counts=True)
            mode_class = unique[counts.argmax()]  # Get the class with the highest count
            mode_classes.append(mode_class)
        
        mode_classes = torch.stack(mode_classes)  # (batch_size)
        
        # Calculate fraction of predictions that disagree with mode
        disagreement = (pred_classes != mode_classes).float().mean(dim=0)  # (batch_size)
        
        return disagreement
    
