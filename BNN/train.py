from __future__ import print_function
from __future__ import division
import torch, time
import torch.utils.data
import matplotlib
import matplotlib.pyplot as plt

from torchvision.utils import save_image, make_grid
from src.utils import *
from numpy.random import normal


def train_BNN_classification(net, name, batch_size, nb_epochs, trainset, valset, cuda,
                         burn_in, sim_steps, N_saves, resample_its, resample_prior_its,
                         re_burn, flat_ims=False, nb_its_dev=1):
    """
    Train a Bayesian Neural Network for classification tasks
    
    Args:
        net: The BNN model to train
        name: Name prefix for saving models and results
        batch_size: Mini-batch size for training
        nb_epochs: Number of training epochs
        trainset: Training dataset
        valset: Validation dataset
        cuda: Whether to use GPU acceleration
        burn_in: Number of burn-in iterations for MCMC
        sim_steps: How often to save model samples
        N_saves: Maximum number of model samples to save
        resample_its: How often to resample momentum
        resample_prior_its: How often to resample prior
        re_burn: Re-burn-in period
        flat_ims: Whether to flatten input images
        nb_its_dev: How often to evaluate on validation set
    """
    # Create directories for saving models and results
    models_dir = name + '_models'
    results_dir = name + '_results'
    mkdir(models_dir)
    mkdir(results_dir)

    # Set up data loaders with appropriate settings for CPU/GPU
    if cuda:
        trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True,
            num_workers=3,
            persistent_workers=True,
            multiprocessing_context='fork'
        )
        valloader = torch.utils.data.DataLoader(
            valset, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=True,
            num_workers=3,
            persistent_workers=True,
            multiprocessing_context='fork'
        )

    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Initialize training variables
    cprint('c', '\nNetwork:')
    epoch = 0
    it_count = 0

    # Initialize arrays to track metrics
    cprint('c', '\nTrain:')
    print('  init cost variables:')
    cost_train = np.zeros(nb_epochs)  # Training cost per epoch
    err_train = np.zeros(nb_epochs)   # Training error per epoch
    cost_dev = np.zeros(nb_epochs)    # Validation cost per epoch
    err_dev = np.zeros(nb_epochs)     # Validation error per epoch
    best_cost = np.inf
    best_err = np.inf

    # Main training loop
    tic0 = time.time()
    for i in range(epoch, nb_epochs):
        net.set_mode_train(True)
        tic = time.time()
        nb_samples = 0

        # Training pass
        for x, y in trainloader:
            # Flatten images if requested
            if flat_ims:
                x = x.view(x.shape[0], -1)

            # Train on batch and get metrics
            cost_pred, err = net.fit(x, y, burn_in=(i % re_burn < burn_in),
                                     resample_momentum=(it_count % resample_its == 0),
                                     resample_prior=(it_count % resample_prior_its == 0))
            it_count += 1
            err_train[i] += err
            cost_train[i] += cost_pred
            nb_samples += len(x)

        # Calculate epoch averages
        cost_train[i] /= nb_samples
        err_train[i] /= nb_samples
        toc = time.time()

        # Print training progress
        print("it %d/%d, Jtr_pred = %f, err = %f, " % (i, nb_epochs, cost_train[i], err_train[i]), end="")
        cprint('r', '   time: %f seconds\n' % (toc - tic))
        net.update_lr(i)

        # Save model samples after burn-in period
        if i % re_burn >= burn_in and i % sim_steps == 0:
            net.save_sampled_net(max_samples=N_saves)

        # Validation pass
        if i % nb_its_dev == 0:
            nb_samples = 0
            for j, (x, y) in enumerate(valloader):
                if flat_ims:
                    x = x.view(x.shape[0], -1)

                # Get validation metrics
                cost, err, probs = net.eval(x, y)

                cost_dev[i] += cost
                err_dev[i] += err
                nb_samples += len(x)

            # Calculate validation averages
            cost_dev[i] /= nb_samples
            err_dev[i] /= nb_samples

            # Print validation metrics
            cprint('g', '    Jdev = %f, err = %f\n' % (cost_dev[i], err_dev[i]))
            if err_dev[i] < best_err:
                best_err = err_dev[i]
                cprint('b', 'best test error')

    # Calculate and print total runtime
    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(nb_epochs)
    cprint('r', '   average time: %f seconds\n' % runtime_per_it)

    # Save final model weights
    net.save_weights(models_dir + '/state_dicts.pkl')

    # Plot training curves
    textsize = 15
    marker = 5

    # Plot cross entropy loss
    plt.figure(dpi=100)
    fig, ax1 = plt.subplots()
    ax1.plot(np.clip(cost_train, a_min=-5, a_max=5), 'r--')
    ax1.plot(range(0, nb_epochs, nb_its_dev), np.clip(cost_dev[::nb_its_dev], a_min=-5, a_max=5), 'b-')
    ax1.set_ylabel('Cross Entropy')
    plt.xlabel('epoch')
    plt.grid(True, which='major')
    plt.grid(True, which='minor')
    lgd = plt.legend(['train error', 'test error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('classification costs')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir + '/cost.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot classification error rate
    plt.figure(dpi=100)
    fig2, ax2 = plt.subplots()
    ax2.set_ylabel('% error')
    ax2.semilogy(range(0, nb_epochs, nb_its_dev), err_dev[::nb_its_dev], 'b-')
    ax2.semilogy(err_train, 'r--')
    ax2.set_ylim(bottom=0.01, top=1.0)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=10))
    ax2.grid(True, which='both', linestyle='-', alpha=0.2)
    plt.xlabel('epoch')
    lgd = plt.legend(['test error', 'train error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir + '/err.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    plt.close()

    return cost_train, cost_dev, err_train, err_dev


##########

def train_BNN_regression(net, name, batch_size, nb_epochs, trainset, valset, cuda,
                                 burn_in, sim_steps, N_saves, resample_its, resample_prior_its,
                                 re_burn, flat_ims=False, nb_its_dev=1, y_mu=0, y_std=1):
    """
    Train a Bayesian Neural Network for regression tasks
    
    Similar to train_BNN_classification but adapted for regression with additional
    metrics like RMS error and log likelihood
    """
    # Create directories for saving models and results
    models_dir = name + '_models'
    results_dir = name + '_results'
    mkdir(models_dir)
    mkdir(results_dir)

    # Set up data loaders with appropriate settings for CPU/GPU
    if cuda:
        trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True,
            num_workers=3,
            persistent_workers=True,
            multiprocessing_context='fork'
        )
        valloader = torch.utils.data.DataLoader(
            valset, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=True,
            num_workers=3,
            persistent_workers=True,
            multiprocessing_context='fork'
        )

    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Initialize training variables
    cprint('c', '\nNetwork:')
    epoch = 0
    it_count = 0

    # Initialize arrays to track metrics
    cprint('c', '\nTrain:')
    print('  init cost variables:')
    cost_train = np.zeros(nb_epochs)  # Training cost per epoch
    cost_dev = np.zeros(nb_epochs)    # Validation cost per epoch
    ll_dev = np.zeros(nb_epochs)      # Log likelihood on validation set
    rms_dev = np.zeros(nb_epochs)     # RMS error on validation set
    best_cost = np.inf

    # Main training loop
    tic0 = time.time()
    for i in range(epoch, nb_epochs):
        net.set_mode_train(True)
        tic = time.time()
        nb_samples = 0

        # Training pass
        for x, y in trainloader:
            if flat_ims:
                x = x.view(x.shape[0], -1)

            # Train on batch and get cost
            cost_pred, _, _ = net.fit(x, y, burn_in=(i % re_burn < burn_in),
                                      resample_momentum=(it_count % resample_its == 0),
                                      resample_prior=(it_count % resample_prior_its == 0))
            it_count += 1
            cost_train[i] += cost_pred
            nb_samples += len(x)

        # Calculate epoch average
        cost_train[i] /= nb_samples
        toc = time.time()

        # Print training progress
        print("it %d/%d, Jtr_pred = %f, " % (i, nb_epochs, cost_train[i]), end="")
        cprint('r', '   time: %f seconds\n' % (toc - tic))
        net.update_lr(i)

        # Save model samples after burn-in period
        if i % re_burn >= burn_in and i % sim_steps == 0:
            net.save_sampled_net(max_samples=N_saves)

        # Validation pass
        if i % nb_its_dev == 0:
            nb_samples = 0
            mu_vec = []      # Predicted means
            sigma_vec = []   # Predicted standard deviations
            y_vec = []       # True values
            
            # Collect predictions and targets
            for j, (x, y) in enumerate(valloader):
                if flat_ims:
                    x = x.view(x.shape[0], -1)

                cost, mu, sigma = net.eval(x, y)
                mu_vec.append(mu.data.cpu())
                sigma_vec.append(sigma.data.cpu())
                y_vec.append(y.data.cpu())

                cost_dev[i] += cost
                nb_samples += len(x)

            # Concatenate predictions and calculate metrics
            mu_vec = torch.cat(mu_vec)
            sigma_vec = torch.cat(sigma_vec)
            y_vec = torch.cat(y_vec)
            rms, ll = net.unnormalised_eval(mu_vec, sigma_vec, y_vec, y_mu, y_std)
            ll_dev[i] = ll
            rms_dev[i] = rms
            cost_dev[i] /= nb_samples

            # Print validation metrics
            cprint('g', '    Jdev = %f \n' % cost_dev[i])
            cprint('g', ' Loglike = %.4f, rms = %.4f\n' % (ll, rms))
            if cost_dev[i] < best_cost:
                best_cost = cost_dev[i]
                cprint('b', 'best test error')

    # Calculate and print total runtime
    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(nb_epochs)
    cprint('r', '   average time: %f seconds\n' % runtime_per_it)

    # Save final model weights
    net.save_weights(models_dir + '/state_dicts.pkl')

    # Plot training curves
    textsize = 15
    marker = 5

    # Plot normalized negative log likelihood
    plt.figure(dpi=100)
    fig, ax1 = plt.subplots()
    ax1.plot(np.clip(cost_train, a_min=-5, a_max=5), 'r--')
    ax1.plot(range(0, nb_epochs, nb_its_dev), np.clip(cost_dev[::nb_its_dev], a_min=-5, a_max=5), 'b-')
    ax1.set_ylabel('Cross Entropy')
    plt.xlabel('epoch')
    plt.grid(True, which='major')
    plt.grid(True, which='minor')
    lgd = plt.legend(['train error', 'test error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('regression costs')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir + '/cost.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot unnormalized log likelihood
    plt.figure(dpi=100)
    fig, ax1 = plt.subplots()
    ax1.plot(range(0, nb_epochs, nb_its_dev), ll_dev[::nb_its_dev], 'b-')
    ax1.set_ylabel('-loglike')
    plt.xlabel('epoch')
    plt.grid(True, which='major')
    plt.grid(True, which='minor')
    ax = plt.gca()
    plt.title('un-normalised')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir + '/ll.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot RMS error
    plt.figure(dpi=100)
    fig, ax1 = plt.subplots()
    ax1.plot(range(0, nb_epochs, nb_its_dev), rms_dev[::nb_its_dev], 'b-')
    ax1.set_ylabel('rms')
    plt.xlabel('epoch')
    plt.grid(True, which='major')
    plt.grid(True, which='minor')
    ax = plt.gca()
    plt.title('un-normalised')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir + '/rms.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    plt.close()
    plt.show()
    return cost_train, cost_dev, rms_dev, ll_dev

