from __future__ import division, print_function
import numpy as np
import torch
import torch.nn.functional as F
from src.utils import to_variable
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def evaluate_BNN_net_cat(net, valloader, samples=0, flat_ims=False):

    test_cost = 0  # Note that these are per sample
    test_err = 0
    nb_samples = 0

    for j, (x, y) in enumerate(valloader):
        y, = to_variable(var=(y.long(),), cuda=net.device.type=='cuda')
        if flat_ims:
            x = x.view(x.shape[0], -1)
        probs_samples = net.sample_predict(x, Nsamples=samples, grad=False).data
        probs = probs_samples.mean(dim=0)

        log_probs = torch.log(probs)
        loss = F.nll_loss(log_probs, y, reduction='sum')
        pred = probs.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        test_cost += loss.item()
        test_err += err.cpu().numpy()
        nb_samples += len(x)

    test_cost /= nb_samples
    test_err /= nb_samples
    print('Loglike = %6.6f, err = %1.6f\n' % (-test_cost, test_err))
    return -test_cost, test_err


def evaluate_BNN_net_gauss(net, valloader, y_means, y_stds, samples=0, gmm_sig=False, flat_ims=False):
    mu_vec = []
    sigma_vec = []
    y_vec = []

    for x,y in valloader:
        y, = to_variable(var=(y,), cuda=net.device.type=='cuda')
        if flat_ims:
            x = x.view(x.shape[0], -1)
        mu, sig = net.sample_predict(x, Nsamples=samples, grad=False)
        mu_vec.append(mu.data.cpu())
        sigma_vec.append(sig.data.cpu())
        y_vec.append(y.data.cpu())

    mu_vec = torch.cat(mu_vec, dim=1)
    sigma_vec = torch.cat(sigma_vec, dim=1)
    y_vec = torch.cat(y_vec, dim=0)

    from src.probability import marginal_std
    mu_mean = mu_vec.mean(dim=0)
    sigma_mean = sigma_vec.mean(dim=0)
    marg_sigma = marginal_std(mu_vec, sigma_vec)

    if gmm_sig:
        rms, ll = net.unnormalised_eval(mu_mean, marg_sigma, y_vec, y_mu=y_means, y_std=y_stds)
    else:
        rms, ll = net.unnormalised_eval(mu_mean, sigma_mean, y_vec, y_mu=y_means, y_std=y_stds)

    print('rms', rms, 'll', ll)

    return ll, rms


def evaluate_uncertainty_accuracy(net, valloader, samples=10, flat_ims=False):
    all_probs = []
    all_targets = []

    for x, y in valloader:
        y, = to_variable(var=(y.long(),), cuda=net.device.type=='cuda')
        if flat_ims:
            x = x.view(x.shape[0], -1)
        
        # Sample predictions
        probs_samples = net.sample_predict(x, Nsamples=samples, grad=False).data
        probs = probs_samples.mean(dim=0)
        
        all_probs.append(probs.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Brier score
    brier_score = np.mean(np.sum((all_probs - np.eye(all_probs.shape[1])[all_targets])**2, axis=1))
    print(f'Brier Score: {brier_score}')

    # Calibration plot
    prob_true, prob_pred = calibration_curve(all_targets, all_probs[:, 1], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration plot')
    plt.legend()
    plt.show()

    return brier_score





# def evaluate_BNN_net_logit_gauss(net, valloader, y_means, y_stds, samples=0, gmm_sig=False):
#     mu_vec = []
#     sigma_vec = []
#     y_vec = []
#
#     for x,y in valloader:
#         mu, sig = net.sample_predict(x, Nsamples=samples, grad=False)
#         mu_vec.append(mu.data.cpu())
#         sigma_vec.append(sig.data.cpu())
#         y_vec.append(y.data.cpu())
#
#     mu_vec = torch.cat(mu_vec, dim=1)
#     sigma_vec = torch.cat(sigma_vec, dim=1)
#     y_vec = torch.cat(y_vec, dim=0)
#
#     from src.probability import marginal_std
#     mu_mean = mu_vec.mean(dim=0)
#     sigma_mean = sigma_vec.mean(dim=0)
#     marg_sigma = marginal_std(mu_vec, sigma_vec)
#
#     if gmm_sig:
#         rms, ll = net.unnormalised_eval(mu_mean, marg_sigma, y_vec, y_mu=y_means, y_std=y_stds)
#     else:
#         rms, ll = net.unnormalised_eval(mu_mean, sigma_mean, y_vec, y_mu=y_means, y_std=y_stds)
#     print('rms', rms, 'll', ll)
#
#     return ll, rms