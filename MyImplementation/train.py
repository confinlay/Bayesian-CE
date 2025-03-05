from __future__ import print_function
from __future__ import division
import torch, time
import torch.utils.data
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
from torchvision.utils import save_image, make_grid
from numpy.random import normal
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy


def train_BLL_classification(net, name, batch_size, nb_epochs, trainset, valset, device,
                         burn_in, sim_steps, N_saves, resample_its, resample_prior_its,
                         re_burn, flat_ims=False, nb_its_dev=1, model_saves_dir=None):
    """
    Train a Bayesian Neural Network for classification tasks
    
    Args:
        net: The BLL model to train
        name: Name prefix for saving models and results
        batch_size: Mini-batch size for training
        nb_epochs: Number of training epochs
        trainset: Training dataset
        valset: Validation dataset
        device: Device to run the training on
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
    models_dir = os.path.join(model_saves_dir, name + '_models')
    results_dir = os.path.join(model_saves_dir, name + '_results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Set up data loaders with appropriate settings for CPU/GPU
    if device == torch.device('cuda'):
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
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

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
        net.train()
        tic = time.time()
        nb_samples = 0

        # Training pass
        for x, y in trainloader:
            # Move data to device
            x, y = x.to(device), y.to(device)
            
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
                # Move data to device
                x, y = x.to(device), y.to(device)
                
                if flat_ims:
                    x = x.view(x.shape[0], -1)

                # Get validation metrics
                cost, err, probs = net.evaluate(x, y)

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
                cprint('b', 'best validation error')

    # Calculate and print total runtime
    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(nb_epochs)
    cprint('r', '   average time: %f seconds\n' % runtime_per_it)

    # Save final model weights
    rand_id = np.random.randint(0, 10000)
    save_path = f'{models_dir}/BLL_checkpoint_{rand_id}.pth'
    net.save_checkpoint(save_path)
    print(f'Saved final model to: {save_path}')

    # Plot training curves
    textsize = 15
    marker = 5

    # Plot cross entropy loss
    plt.figure(dpi=100)
    fig, ax1 = plt.subplots()
    ax1.plot(cost_train, 'r--')
    ax1.plot(cost_dev, 'b-')
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
    ax2.plot(err_train, 'r--')        # Plot training error first
    ax2.plot(err_dev, 'b-')           # Plot validation error
    ax2.set_ylabel('Error Rate')
    plt.xlabel('epoch')
    plt.grid(True, which='major')
    plt.grid(True, which='minor')
    # Set y-axis limits based on data range with some padding
    ymin, ymax = min(min(err_train), min(err_dev)), max(max(err_train), max(err_dev))
    padding = (ymax - ymin) * 0.1  # 10% padding
    ax2.set_ylim(ymin - padding, ymax + padding)
    lgd = plt.legend(['train error', 'test error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('classification errors')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir + '/err.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    plt.close()

    return cost_train, cost_dev, err_train, err_dev


def train_backbone(net, name, batch_size, nb_epochs, trainset, valset, device,
                  lr=0.001, patience=5, nb_its_dev=1, model_saves_dir=None):
    """
    Train a deterministic backbone network before BNN last layer training.
    
    Args:
        net: The backbone model to train
        name: Name prefix for saving models and results
        batch_size: Mini-batch size for training
        nb_epochs: Number of training epochs
        trainset: Training dataset
        valset: Validation dataset
        device: Device to run the training on
        lr: Initial learning rate
        patience: Early stopping patience
        nb_its_dev: How often to evaluate on validation set
    """
    # Create directories for saving models and results
    models_dir = os.path.join(model_saves_dir, name + '_models')
    results_dir = os.path.join(model_saves_dir, name + '_results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Set up data loaders
    if device == torch.device('cuda'):
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
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Initialize tracking variables
    cprint('c', '\nBackbone Network:')
    cost_train = np.zeros(nb_epochs)
    err_train = np.zeros(nb_epochs)
    cost_dev = np.zeros(nb_epochs)
    err_dev = np.zeros(nb_epochs)
    best_err = float('inf')
    best_state = None
    patience_counter = 0

    # Main training loop
    tic0 = time.time()
    for i in range(nb_epochs):
        net.train()
        tic = time.time()
        nb_samples = 0

        # Training pass
        for x, y in trainloader:
            # Move data to device
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            # Handle tuple output from backbone
            _, outputs = net(x)  # Unpack features and logits

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # Track metrics
            _, predicted = outputs.max(1)
            err = (predicted != y).float().mean().item()
            
            cost_train[i] += loss.item()
            err_train[i] += err
            nb_samples += len(x)

        # Calculate epoch averages
        cost_train[i] /= nb_samples
        err_train[i] /= nb_samples
        toc = time.time()

        # Print training progress
        print("it %d/%d, Jtr = %f, err = %f, " % (i, nb_epochs, cost_train[i], err_train[i]), end="")
        cprint('r', '   time: %f seconds\n' % (toc - tic))

        # Validation pass
        if i % nb_its_dev == 0:
            net.eval()
            nb_samples = 0
            with torch.no_grad():
                for x, y in valloader:
                    # Move data to device
                    x, y = x.to(device), y.to(device)
                    
                    # Handle tuple output from backbone
                    _, outputs = net(x)  # Unpack features and logits
                    loss = criterion(outputs, y)
                    _, predicted = outputs.max(1)
                    err = (predicted != y).float().mean().item()

                    cost_dev[i] += loss.item()
                    err_dev[i] += err
                    nb_samples += len(x)

            # Calculate validation averages
            cost_dev[i] /= nb_samples
            err_dev[i] /= nb_samples

            # Learning rate scheduling
            scheduler.step(cost_dev[i])

            # Print validation metrics
            cprint('g', '    Jdev = %f, err = %f\n' % (cost_dev[i], err_dev[i]))
            
            # Track best model and early stopping
            if err_dev[i] < best_err:
                best_err = err_dev[i]
                best_state = copy.deepcopy(net.state_dict())
                patience_counter = 0
                cprint('b', 'best validation error')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    cprint('y', f'\nEarly stopping triggered after {i+1} epochs')
                    break

    # Calculate and print total runtime
    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(i + 1)
    cprint('r', '   average time: %f seconds\n' % runtime_per_it)

    # Restore best model
    net.load_state_dict(best_state)
    rand_id = np.random.randint(0, 10000)
    save_path = f'{models_dir}/backbone_best_{rand_id}.pt'
    torch.save(best_state, save_path)
    print(f'Saved best model to: {save_path}')

    # Plot training curves (similar to BNN plotting code)
    textsize = 15
    marker = 5

    # Plot cross entropy loss
    plt.figure(dpi=100)
    fig, ax1 = plt.subplots()
    ax1.plot(cost_train, 'r--')  # Remove clip to see actual values
    ax1.plot(cost_dev, 'b-')     # Plot full array, not just every nb_its_dev
    ax1.set_ylabel('Cross Entropy')
    plt.xlabel('epoch')
    plt.grid(True, which='major')
    plt.grid(True, which='minor')
    lgd = plt.legend(['train error', 'test error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title('backbone training costs')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(results_dir + '/backbone_cost.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    plt.close()

    return cost_train[:i+1], cost_dev[:i+1], err_train[:i+1], err_dev[:i+1], best_err


def train_BLL_VI_classification(net, name, batch_size, nb_epochs, trainset, valset, device,
                              lr=1e-3, patience=5, nb_its_dev=1, model_saves_dir=None):
    """
    Train a Bayesian Last Layer using Variational Inference for classification tasks.
    
    Args:
        net: The BLL_VI model to train
        name: Name prefix for saving models and results
        batch_size: Mini-batch size for training
        nb_epochs: Number of training epochs
        trainset: Training dataset
        valset: Validation dataset
        device: Device to run the training on
        lr: Initial learning rate
        patience: Early stopping patience
        nb_its_dev: How often to evaluate on validation set
        model_saves_dir: Directory to save models and results
    """
    # Create directories for saving models and results
    models_dir = os.path.join(model_saves_dir, name + '_models')
    results_dir = os.path.join(model_saves_dir, name + '_results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Set up data loaders
    if device == torch.device('cuda'):
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
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize optimizer
    optimizer = torch.optim.Adam(net.last_layer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Initialize tracking variables
    cprint('c', '\nBayesian Last Layer (VI):')
    cost_train = np.zeros(nb_epochs)
    err_train = np.zeros(nb_epochs)
    kl_train = np.zeros(nb_epochs)
    cost_dev = np.zeros(nb_epochs)
    err_dev = np.zeros(nb_epochs)
    best_err = float('inf')
    best_state = None
    patience_counter = 0

    # Main training loop
    tic0 = time.time()
    for i in range(nb_epochs):
        net.train()
        tic = time.time()
        nb_samples = 0

        # Training pass
        for x, y in trainloader:
            # Move data to device
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            # Get loss tensor and metrics
            loss, err, ce_loss, kl_div = net.fit(x, y)
            # Backward pass on the loss tensor
            loss.backward()
            optimizer.step()

            # Track metrics
            cost_train[i] += ce_loss  # Already detached in fit()
            err_train[i] += err
            kl_train[i] += kl_div
            nb_samples += len(x)

        # Calculate epoch averages
        cost_train[i] /= nb_samples
        err_train[i] /= nb_samples
        kl_train[i] /= len(trainloader)
        toc = time.time()

        # Print training progress
        print("it %d/%d, Jtr = %.3f, err = %.3f, KL = %.3f, " % 
              (i, nb_epochs, cost_train[i], err_train[i], kl_train[i]), end="")
        cprint('r', '   time: %f seconds\n' % (toc - tic))

        # Validation pass
        if i % nb_its_dev == 0:
            net.eval()
            nb_samples = 0
            with torch.no_grad():
                for x, y in valloader:
                    # Move data to device
                    x, y = x.to(device), y.to(device)
                    
                    # Get validation metrics with uncertainty
                    cost, err, probs, uncertainty = net.evaluate(x, y)
                    
                    cost_dev[i] += cost
                    err_dev[i] += err
                    nb_samples += len(x)

            # Calculate validation averages
            cost_dev[i] /= nb_samples
            err_dev[i] /= nb_samples

            # Learning rate scheduling
            scheduler.step(cost_dev[i])

            # Print validation metrics
            cprint('g', '    Jdev = %.3f, err = %.3f\n' % (cost_dev[i], err_dev[i]))
            
            # Track best model and early stopping
            if err_dev[i] < best_err:
                best_err = err_dev[i]
                best_state = copy.deepcopy(net.state_dict())
                patience_counter = 0
                cprint('b', 'best validation error')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    cprint('y', f'\nEarly stopping triggered after {i+1} epochs')
                    break

    # Calculate and print total runtime
    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(i + 1)
    cprint('r', '   average time: %f seconds\n' % runtime_per_it)

    # Restore best model
    net.load_state_dict(best_state)
    rand_id = np.random.randint(0, 10000)
    save_path = f'{models_dir}/BLL_VI_best_{rand_id}.pt'
    net.save_checkpoint(save_path)
    print(f'Saved best model to: {save_path}')

    # Plot training curves
    textsize = 15
    marker = 5

    # Plot cross entropy loss
    plt.figure(dpi=100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(cost_train, 'r--')
    ax1.plot(cost_dev, 'b-')
    ax1.set_ylabel('Cross Entropy')
    ax1.set_xlabel('epoch')
    ax1.grid(True)
    ax1.legend(['train', 'validation'])
    ax1.set_title('Classification Loss')
    
    # KL divergence plot
    ax2.plot(kl_train, 'g-')
    ax2.set_ylabel('KL Divergence')
    ax2.set_xlabel('epoch')
    ax2.grid(True)
    ax2.set_title('KL Divergence')
    
    plt.tight_layout()
    plt.savefig(results_dir + '/vi_training.png')
    plt.close()

    # Plot classification error rate
    plt.figure(dpi=100)
    plt.plot(err_train, 'r--')
    plt.plot(err_dev, 'b-')
    plt.ylabel('Error Rate')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(['train error', 'validation error'])
    plt.title('Classification Errors')
    plt.savefig(results_dir + '/vi_error.png')
    plt.close()

    return cost_train[:i+1], cost_dev[:i+1], err_train[:i+1], err_dev[:i+1], kl_train[:i+1]

def train_BNN_VI_classification(net, name, batch_size, nb_epochs, trainset, valset, device,
                              lr=1e-3, patience=5, nb_its_dev=1, model_saves_dir=None):
    """
    Customized training function for BayesianNeuralNetworkVI models.
    """
    # Create directories for saving models and results
    models_dir = os.path.join(model_saves_dir, name + '_models')
    results_dir = os.path.join(model_saves_dir, name + '_results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Set up data loaders - same as original function
    if device == torch.device('cuda'):
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, 
            pin_memory=True, num_workers=3, persistent_workers=True,
            multiprocessing_context='fork'
        )
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=False, 
            pin_memory=True, num_workers=3, persistent_workers=True,
            multiprocessing_context='fork'
        )
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize optimizer - Modified for BNN_VI
    # Get only the parameters we want to train (Bayesian layers)
    bayesian_params = list(net.bayes_layer1.parameters()) + list(net.bayes_layer2.parameters())
    optimizer = torch.optim.Adam(bayesian_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Initialize tracking variables
    cprint('c', '\nBayesian Neural Network (VI):')
    cost_train = np.zeros(nb_epochs)
    err_train = np.zeros(nb_epochs)
    kl_train = np.zeros(nb_epochs)
    cost_dev = np.zeros(nb_epochs)
    err_dev = np.zeros(nb_epochs)
    best_err = float('inf')
    best_state = None
    patience_counter = 0

    # Main training loop
    tic0 = time.time()
    for i in range(nb_epochs):
        net.train()
        tic = time.time()
        nb_samples = 0

        # Training pass
        for x, y in trainloader:
            # Move data to device - no need to explicitly move as BNN_VI handles this
            
            optimizer.zero_grad()
            # Get loss tensor and metrics from BNN_VI.fit
            loss, err, ce_loss, kl_div = net.fit(x, y)
            # Backward pass on the loss tensor
            loss.backward()
            optimizer.step()

            # Track metrics
            cost_train[i] += ce_loss
            err_train[i] += err
            kl_train[i] += kl_div
            nb_samples += len(x)

        # Calculate epoch averages
        cost_train[i] /= nb_samples
        err_train[i] /= nb_samples
        kl_train[i] /= len(trainloader)
        toc = time.time()

        # Print training progress
        print("it %d/%d, Jtr = %.3f, err = %.3f, KL = %.3f, " % 
              (i, nb_epochs, cost_train[i], err_train[i], kl_train[i]), end="")
        cprint('r', '   time: %f seconds\n' % (toc - tic))

        # Validation pass
        if i % nb_its_dev == 0:
            net.eval()
            nb_samples = 0
            with torch.no_grad():
                for x, y in valloader:
                    # BNN_VI.evaluate function matches our needs
                    cost, err, probs, uncertainty = net.evaluate(x, y)
                    
                    cost_dev[i] += cost
                    err_dev[i] += err
                    nb_samples += len(x)

            # Calculate validation averages
            cost_dev[i] /= nb_samples
            err_dev[i] /= nb_samples

            # Learning rate scheduling
            scheduler.step(cost_dev[i])

            # Print validation metrics
            cprint('g', '    Jdev = %.3f, err = %.3f\n' % (cost_dev[i], err_dev[i]))
            
            # Track best model and early stopping
            if err_dev[i] < best_err:
                best_err = err_dev[i]
                best_state = copy.deepcopy(net.state_dict())
                patience_counter = 0
                cprint('b', 'best validation error')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    cprint('y', f'\nEarly stopping triggered after {i+1} epochs')
                    break

    # Calculate and print total runtime
    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(i + 1)
    cprint('r', '   average time: %f seconds\n' % runtime_per_it)

    # Restore best model
    net.load_state_dict(best_state)
    rand_id = np.random.randint(0, 10000)
    save_path = f'{models_dir}/BNN_VI_best_{rand_id}.pt'
    # Use net.save_checkpoint method
    net.save_checkpoint(save_path)
    print(f'Saved best model to: {save_path}')

    # Plot training curves
    textsize = 15
    marker = 5

    # Plot cross entropy loss
    plt.figure(dpi=100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(cost_train, 'r--')
    ax1.plot(cost_dev, 'b-')
    ax1.set_ylabel('Cross Entropy')
    ax1.set_xlabel('epoch')
    ax1.grid(True)
    ax1.legend(['train', 'validation'])
    ax1.set_title('Classification Loss')
    
    # KL divergence plot
    ax2.plot(kl_train, 'g-')
    ax2.set_ylabel('KL Divergence')
    ax2.set_xlabel('epoch')
    ax2.grid(True)
    ax2.set_title('KL Divergence')
    
    plt.tight_layout()
    plt.savefig(results_dir + '/vi_training.png')
    plt.close()

    # Plot classification error rate
    plt.figure(dpi=100)
    plt.plot(err_train, 'r--')
    plt.plot(err_dev, 'b-')
    plt.ylabel('Error Rate')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(['train error', 'validation error'])
    plt.title('Classification Errors')
    plt.savefig(results_dir + '/vi_error.png')
    plt.close()

    return cost_train[:i+1], cost_dev[:i+1], err_train[:i+1], err_dev[:i+1], kl_train[:i+1]


def cprint(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()

def fine_tune_backbone(bayesian_classifier, decoder, name, train_loader, val_loader, num_epochs=5, 
                      lr=0.001, lambda_recon=0.5, model_saves_dir=None, patience=5):
    """
    Train both classifier and decoder jointly with combined loss
    
    Args:
        bayesian_classifier: The BLL model to fine-tune
        decoder: Decoder network
        name: Name prefix for saving models and results
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        lambda_recon: Weight for reconstruction loss
        model_saves_dir: Directory to save models and results
        patience: Early stopping patience
    """
    # Create directories for saving models and results
    models_dir = os.path.join(model_saves_dir, name + '_models')
    results_dir = os.path.join(model_saves_dir, name + '_results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    bayesian_classifier.to(bayesian_classifier.device)
    decoder.to(decoder.device)

    # Freeze the weights of the last layer
    for param in bayesian_classifier.last_layer.parameters():
        param.requires_grad = False
    
    # Setup optimizers and schedulers
    classifier_optimizer = optim.Adam(bayesian_classifier.backbone.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    
    # Add learning rate schedulers
    classifier_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        classifier_optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        decoder_optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Loss functions
    classification_criterion = nn.CrossEntropyLoss()
    reconstruction_criterion = nn.MSELoss()
    
    # Initialize tracking variables
    train_losses = {'total': [], 'classification': [], 'reconstruction': []}
    val_losses = {'total': [], 'classification': [], 'reconstruction': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = {
        'backbone': None,
        'decoder': None
    }

    tic0 = time.time()
    for epoch in range(num_epochs):
        # Training phase
        bayesian_classifier.train()
        decoder.train()
        tic = time.time()
        epoch_losses = {'total': 0.0, 'classification': 0.0, 'reconstruction': 0.0}
        nb_samples = 0
        
        for images, labels in train_loader:
            images = images.to(bayesian_classifier.device)
            labels = labels.to(bayesian_classifier.device)
            
            # Zero gradients
            classifier_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            # Forward passes
            z = bayesian_classifier.extract_features(images)
            y_pred = bayesian_classifier.last_layer(z)
            x_reconstructed = decoder(z)
            
            # Calculate losses
            classification_loss = classification_criterion(y_pred, labels)
            reconstruction_loss = reconstruction_criterion(x_reconstructed, images)
            total_loss = lambda_recon * reconstruction_loss + (1 - lambda_recon) * classification_loss
            
            # Backward pass and optimization
            total_loss.backward()
            classifier_optimizer.step()
            decoder_optimizer.step()
            
            # Accumulate batch losses
            epoch_losses['total'] += total_loss.item() * len(images)
            epoch_losses['classification'] += classification_loss.item() * len(images)
            epoch_losses['reconstruction'] += reconstruction_loss.item() * len(images)
            nb_samples += len(images)
        
        # Calculate epoch averages
        for key in epoch_losses:
            epoch_losses[key] /= nb_samples
            train_losses[key].append(epoch_losses[key])
        
        toc = time.time()
        
        # Print training progress
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Total Loss: {epoch_losses['total']:.4f}, "
              f"Class Loss: {epoch_losses['classification']:.4f}, "
              f"Recon Loss: {epoch_losses['reconstruction']:.4f}, ", end="")
        cprint('r', f'time: {toc - tic:.2f} seconds\n')
        
        # Validation phase
        bayesian_classifier.eval()
        decoder.eval()
        val_epoch_losses = {'total': 0.0, 'classification': 0.0, 'reconstruction': 0.0}
        nb_val_samples = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(bayesian_classifier.device)
                labels = labels.to(bayesian_classifier.device)
                
                z = bayesian_classifier.extract_features(images)
                y_pred = bayesian_classifier.last_layer(z)
                x_reconstructed = decoder(z)
                
                classification_loss = classification_criterion(y_pred, labels)
                reconstruction_loss = reconstruction_criterion(x_reconstructed, images)
                total_loss = lambda_recon * reconstruction_loss + (1 - lambda_recon) * classification_loss
                
                val_epoch_losses['total'] += total_loss.item() * len(images)
                val_epoch_losses['classification'] += classification_loss.item() * len(images)
                val_epoch_losses['reconstruction'] += reconstruction_loss.item() * len(images)
                nb_val_samples += len(images)
        
        # Calculate validation averages
        for key in val_epoch_losses:
            val_epoch_losses[key] /= nb_val_samples
            val_losses[key].append(val_epoch_losses[key])
        
        # Print validation metrics
        cprint('g', f'    Val Total Loss: {val_epoch_losses["total"]:.4f}, '
               f'Class Loss: {val_epoch_losses["classification"]:.4f}, '
               f'Recon Loss: {val_epoch_losses["reconstruction"]:.4f}\n')
        
        # Save best model
        if val_epoch_losses['total'] < best_val_loss:
            best_val_loss = val_epoch_losses['total']
            best_state['backbone'] = copy.deepcopy(bayesian_classifier.backbone.state_dict())
            best_state['decoder'] = copy.deepcopy(decoder.state_dict())
            patience_counter = 0
            cprint('b', 'best validation loss')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                cprint('y', f'\nEarly stopping triggered after {epoch+1} epochs')
                break
        
        # After validation phase, update schedulers
        classifier_scheduler.step(val_epoch_losses['classification'])
        decoder_scheduler.step(val_epoch_losses['reconstruction'])
    
    # Calculate and print total runtime
    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(epoch + 1)
    cprint('r', f'   average time: {runtime_per_it:.2f} seconds\n')
    
    # Restore best model
    bayesian_classifier.backbone.load_state_dict(best_state['backbone'])
    decoder.load_state_dict(best_state['decoder'])
    
    # Save best models
    rand_id = np.random.randint(0, 10000)
    # Save full BLL model checkpoint
    bll_path = f'{models_dir}/BLL_finetuned_{rand_id}.pth'
    bayesian_classifier.save_checkpoint(bll_path)
    # Save decoder separately
    decoder_path = f'{models_dir}/decoder_finetuned_{rand_id}.pt'
    torch.save(decoder.state_dict(), decoder_path)
    print(f'Saved best models to:\n{bll_path}\n{decoder_path}')
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    # Total Loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses['total'], 'r--')
    plt.plot(val_losses['total'], 'b-')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.grid(True)
    
    # Classification Loss
    plt.subplot(1, 3, 2)
    plt.plot(train_losses['classification'], 'r--')
    plt.plot(val_losses['classification'], 'b-')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.grid(True)
    
    # Reconstruction Loss
    plt.subplot(1, 3, 3)
    plt.plot(train_losses['reconstruction'], 'r--')
    plt.plot(val_losses['reconstruction'], 'b-')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/fine_tuning_losses.png')
    
    plt.close()
    
    return train_losses, val_losses