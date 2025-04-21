import math
import torch

from torchbnn.modules import *

def _kl_loss(mu_0, log_sigma_0, mu_1, log_sigma_1) :
    """
    An method for calculating KL divergence between two Normal distribtuion.
    """
    kl = log_sigma_1 - log_sigma_0 + \
    (torch.exp(log_sigma_0)**2 + (mu_0-mu_1)**2)/(2*math.exp(log_sigma_1)**2) - 0.5
    return kl.sum()

def bayesian_kl_loss(model, reduction='mean', last_layer_only=False) :
    """
    An method for calculating KL divergence of whole layers in the model.
    """
    # Get the device of the first parameter
    device = next(model.parameters()).device
    
    kl = torch.tensor([0.], device=device)
    kl_sum = torch.tensor([0.], device=device)
    n = torch.tensor([0.], device=device)

    for m in model.modules() :
        if isinstance(m, (BayesLinear, BayesConv2d)):
            # Ensure all tensors are on the same device
            weight_mu = m.weight_mu.to(device)
            weight_log_sigma = m.weight_log_sigma.to(device)
            prior_mu = torch.tensor(m.prior_mu, device=device)
            prior_log_sigma = torch.tensor(m.prior_log_sigma, device=device)
            
            kl = _kl_loss(weight_mu, weight_log_sigma, prior_mu, prior_log_sigma)
            kl_sum += kl
            n += len(m.weight_mu.view(-1))

            if m.bias:
                bias_mu = m.bias_mu.to(device)
                bias_log_sigma = m.bias_log_sigma.to(device)
                kl = _kl_loss(bias_mu, bias_log_sigma, prior_mu, prior_log_sigma)
                kl_sum += kl
                n += len(m.bias_mu.view(-1))

        if isinstance(m, BayesBatchNorm2d):
            if m.affine:
                weight_mu = m.weight_mu.to(device)
                weight_log_sigma = m.weight_log_sigma.to(device)
                prior_mu = torch.tensor(m.prior_mu, device=device)
                prior_log_sigma = torch.tensor(m.prior_log_sigma, device=device)
                
                kl = _kl_loss(weight_mu, weight_log_sigma, prior_mu, prior_log_sigma)
                kl_sum += kl
                n += len(m.weight_mu.view(-1))

                bias_mu = m.bias_mu.to(device)
                bias_log_sigma = m.bias_log_sigma.to(device)
                kl = _kl_loss(bias_mu, bias_log_sigma, prior_mu, prior_log_sigma)
                kl_sum += kl                
                n += len(m.bias_mu.view(-1))
            
    if last_layer_only or n == 0:
        return kl
    
    if reduction == 'mean':
        return kl_sum/n
    elif reduction == 'sum':
        return kl_sum
    else:
        raise ValueError(reduction + " is not valid") 