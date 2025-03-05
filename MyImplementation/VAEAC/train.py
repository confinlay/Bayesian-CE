from __future__ import print_function
from __future__ import division
import torch, time
import torch.utils.data
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision.utils import save_image, make_grid
from src.utils import *
from numpy.random import normal
import os

# TODO: implement image masks

def create_platform_safe_loader(dataset, batch_size, shuffle, device_type, **kwargs):
    """
    Creates a DataLoader that's appropriate for the platform and device
    """
    # # Determine number of workers based on platform and device
    # if device_type == 'mps':  # macOS with Metal
    #     num_workers = 0  # No multiprocessing on MPS
    # elif device_type == 'cuda':  # CUDA device
    #     num_workers = 4  # Or another appropriate number
    # else:  # CPU
    #     num_workers = 0  # Safe default
    num_workers = 0
        
    # Configure DataLoader
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device_type != 'cpu'),
        **kwargs
    )

def safe_save_model(net, filepath):
    """Safely save model with proper directory creation and error handling"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        net.save(filepath)
        print(f"Successfully saved model to {filepath}")
        
    except Exception as e:
        print(f"Warning: Failed to save model to {filepath}")
        print(f"Error: {str(e)}")

def train_VAEAC(net, masker, name, batch_size, nb_epochs, trainset, valset, cuda,
                flat_ims=False, train_plot=False, Nclass=None, early_stop=None, script_mode=False):

    models_dir = name + '_models'
    results_dir = name + '_results'
    mkdir(models_dir)
    mkdir(results_dir)

    # Use the actual device type from the model
    device_type = net.device.type
    
    # Create appropriate DataLoader
    trainloader = create_platform_safe_loader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True,
        device_type=device_type
    )
    
    valloader = create_platform_safe_loader(
        valset, 
        batch_size=batch_size, 
        shuffle=False,
        device_type=device_type
    )

## ---------------------------------------------------------------------------------------------------------------------
# net dims
    cprint('c', '\nNetwork:')

    epoch = 0

    ## ---------------------------------------------------------------------------------------------------------------------
    # train
    cprint('c', '\nTrain:')

    print('  init cost variables:')
    vlb_train = np.zeros(nb_epochs)
    vlb_dev = np.zeros(nb_epochs)
    iwlb_dev = np.zeros(nb_epochs)
    best_vlb = -np.inf
    best_epoch = 0

    nb_its_dev = 1

    tic0 = time.time()

    # Detect if we're in a notebook environment
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from tqdm.notebook import tqdm as notebook_tqdm
        else:
            from tqdm import tqdm as notebook_tqdm
    except (ImportError, AttributeError):
        from tqdm import tqdm as notebook_tqdm

    for i in range(epoch, nb_epochs):
        net.set_mode_train(True)
        tic = time.time()
        nb_samples = 0
        
        # Add progress bar - use notebook version for better display
        n_batches = len(trainset) // batch_size
        pbar = notebook_tqdm(total=n_batches, desc=f'Epoch {i+1}/{nb_epochs}')
        
        for x, y in trainloader:
            if flat_ims:
                x = x.view(x.shape[0], -1)
            if Nclass is not None:
                y_oh = torch_onehot(y, Nclass).type(x.type())
                x = torch.cat([x, y_oh], 1)

            mask = masker(x)
            cost, _ = net.fit(x, mask)
            vlb_train[i] += cost * len(x)
            nb_samples += len(x)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'vlb': f'{vlb_train[i]/nb_samples:.4f}'})

        pbar.close()
        vlb_train[i] /= nb_samples
        toc = time.time()
        
        print(f"Epoch {i+1}/{nb_epochs}, vlb {vlb_train[i]:.4f}, time: {toc-tic:.2f}s")
        net.update_lr(i)

        # ---- dev
        if i % nb_its_dev == 0:
            nb_samples = 0
            for j, (x, y) in enumerate(valloader):

                if flat_ims:
                    x = x.view(x.shape[0], -1)
                if Nclass is not None:
                    y_oh = torch_onehot(y, Nclass).type(x.type())
                    x = torch.cat([x, y_oh], 1)

                mask = masker(x)
                cost, rec_mean = net.evaluate(x, mask)
                # iwlb = net.eval_iw(x, mask, 25)

                vlb_dev[i] += cost * len(x)
                # iwlb_dev[i] += iwlb
                nb_samples += len(x)

            vlb_dev[i] /= nb_samples
            # iwlb_dev[i] /= nb_samples

            cprint('g', '    vlb %f (%f)\n' % (vlb_dev[i], best_vlb))

            if train_plot:
                xm = net.model.apply_mask(x, mask)
                
                xr = x.cpu()
                rec_inpaint = net.inpaint(xm, mask)
                try:
                    o = rec_mean.cpu()
                    rec_inpaint = rec_inpaint[0].cpu()
                except:
                    o = rec_mean.loc.cpu()
                    rec_inpaint = rec_inpaint[0].loc.cpu()

                if Nclass is not None:
                    xm = xm[:, :-Nclass]
                    rec_inpaint = rec_inpaint[:, :-Nclass]
                    xr = xr[:, :-Nclass]
                    o = o[:, :-Nclass]

                if len(x.shape) == 2:
                    side = int(np.sqrt(xm.shape[1]))
                    xm = xm.view(-1, 1, side, side).data
                    rec_inpaint = rec_inpaint.view(-1, 1, side, side).data
                    xr = xr.view(-1, 1, side, side).data
                    o = o.view(-1, 1, side, side).data

                import matplotlib.pyplot as plt
                plt.figure()
                dd = make_grid(torch.cat([xr[:10], o[:10]]), nrow=10).numpy()
                plt.imshow(np.transpose(dd, (1, 2, 0)), interpolation='nearest')
                plt.title('reconstruct')
                if script_mode:
                    plt.savefig(results_dir + '/rec%d.png' % i)
                else:
                    plt.show()

                import matplotlib.pyplot as plt
                plt.figure()
                dd = make_grid(torch.cat([xm[:10], rec_inpaint[:10]]), nrow=10).numpy()
                plt.imshow(np.transpose(dd, (1, 2, 0)), interpolation='nearest')
                plt.title('inpaint')
                if script_mode:
                    plt.savefig(results_dir + '/inp%d.png' % i)
                else:
                    plt.show()

        if vlb_dev[i] > best_vlb:
            best_vlb = vlb_dev[i]
            best_epoch = i
            safe_save_model(net, os.path.join(models_dir, 'theta_best.dat'))

        if early_stop is not None and (i - best_epoch) > early_stop:
            break


    safe_save_model(net, os.path.join(models_dir, 'theta_last.dat'))
    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(nb_epochs)
    cprint('r', '   average time: %f seconds\n' % runtime_per_it)

    ## ---------------------------------------------------------------------------------------------------------------------
    # results
    cprint('c', '\nRESULTS:')
    nb_parameters = net.get_nb_parameters()
    best_cost_dev = np.max(vlb_dev)
    # best_iw_dev = np.max(iwlb_dev)
    best_cost_train = np.max(vlb_train)

    print('  best_vlb_dev: %f' % best_cost_dev)
    # print('  best_iwlb_dev: %f' % best_iw_dev)
    print('  best_vlb_train: %f' % best_cost_train)
    print('  nb_parameters: %d (%s)\n' % (nb_parameters, humansize(nb_parameters)))

    ## ---------------------------------------------------------------------------------------------------------------------
    # fig cost vs its
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(np.clip(vlb_train, -1000, 1000), 'r')
    plt.plot(np.clip(vlb_dev[::nb_its_dev], -1000, 1000), 'b')
    plt.legend(['cost_train', 'cost_dev'])
    plt.ylabel('vlb')
    plt.xlabel('it')
    plt.grid(True)
    plt.savefig(results_dir+'/train_cost.png')

    # plt.figure()
    # plt.plot(np.clip(iwlb_dev[::nb_its_dev], -1000, 1000), 'b')
    # plt.ylabel('dev iwlb')
    # plt.xlabel('it')
    # plt.grid(True)
    # plt.savefig(results_dir + '/train_iwlb.png')
    if train_plot:
        plt.show()
    return vlb_train, vlb_dev

def train_under_VAEAC(under_net, base_vaeac, name, batch_size, nb_epochs, trainset, valset,
                       flat_ims=False, train_plot=False, Nclass=None, early_stop=None, script_mode=False):
    """
    Train the under_VAEAC network using latent vectors from base_vaeac.
    
    Parameters:
    -----------
    under_net : under_VAEAC
        The under_VAEAC network to be trained
    base_vaeac : VAEAC_bern_net
        The pretrained base VAEAC network
    name : str
        Name prefix for saved models and results
    batch_size : int
        Batch size for training
    nb_epochs : int
        Number of epochs to train for
    trainset : torch.utils.data.Dataset
        Dataset for training
    valset : torch.utils.data.Dataset
        Dataset for validation
    flat_ims : bool
        Whether to flatten images
    train_plot : bool
        Whether to plot training progress
    Nclass : int
        Number of classes if using class labels
    early_stop : int
        Number of epochs to wait before early stopping
    script_mode : bool
        Whether running in script mode (vs interactive)
    """
    models_dir = name + '_under_models'
    results_dir = name + '_under_results'
    mkdir(models_dir)
    mkdir(results_dir)

    # Use the device from the undernet
    device_type = under_net.device.type
    
    # Create appropriate DataLoader
    trainloader = create_platform_safe_loader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True,
        device_type=device_type
    )
    
    valloader = create_platform_safe_loader(
        valset, 
        batch_size=batch_size, 
        shuffle=False,
        device_type=device_type
    )

    # net dims
    cprint('c', '\nTraining under_VAEAC:')

    epoch = 0

    # train
    cprint('c', '\nTrain:')

    print('  init cost variables:')
    vlb_train = np.zeros(nb_epochs)
    vlb_dev = np.zeros(nb_epochs)
    best_vlb = -np.inf
    best_epoch = 0

    nb_its_dev = 1

    tic0 = time.time()
    
    # Set base VAEAC to evaluation mode
    base_vaeac.set_mode_train(False)
    
    # Detect if we're in a notebook environment
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from tqdm.notebook import tqdm as notebook_tqdm
        else:
            from tqdm import tqdm as notebook_tqdm
    except (ImportError, AttributeError):
        from tqdm import tqdm as notebook_tqdm

    for i in range(epoch, nb_epochs):
        under_net.set_mode_train(True)
        tic = time.time()
        nb_samples = 0
        
        # Add progress bar - use notebook version for better display
        n_batches = len(trainset) // batch_size
        pbar = notebook_tqdm(total=n_batches, desc=f'Epoch {i+1}/{nb_epochs}')
        
        for x, y in trainloader:
            if flat_ims:
                x = x.view(x.shape[0], -1)
            if Nclass is not None:
                y_oh = torch_onehot(y, Nclass).type(x.type())
                x = torch.cat([x, y_oh], 1)
                
            # MODIFIED: Don't pass z_sample directly to fit
            # Instead, store it in under_net.z_cache and call fit() with the original x
            # Get latent vectors from base VAEAC
            z_sample = base_vaeac.get_post(x).sample()
            
            # Store z_sample in the under_net for use in fit
            under_net.z_cache = z_sample
            
            # Call fit with original data - the under_net will use z_cache internally
            # instead of re-computing latent vectors
            cost, _ = under_net.fit(x)
            
            vlb_train[i] += cost * len(x)
            nb_samples += len(x)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'vlb': f'{vlb_train[i]/nb_samples:.4f}'})

        pbar.close()
        vlb_train[i] /= nb_samples
        toc = time.time()
        
        print(f"Epoch {i+1}/{nb_epochs}, vlb {vlb_train[i]:.4f}, time: {toc-tic:.2f}s")
        under_net.update_lr(i)  # Now using the new update_lr method

        # Validation
        if i % nb_its_dev == 0:
            nb_samples = 0
            for j, (x, y) in enumerate(valloader):
                if flat_ims:
                    x = x.view(x.shape[0], -1)
                if Nclass is not None:
                    y_oh = torch_onehot(y, Nclass).type(x.type())
                    x = torch.cat([x, y_oh], 1)
                
                # MODIFIED: Same approach for validation
                z_sample = base_vaeac.get_post(x).sample()
                under_net.z_cache = z_sample
                
                # Evaluate under_VAEAC
                cost, rec_mean = under_net.eval(x)
                vlb_dev[i] += cost * len(x)
                nb_samples += len(x)

            vlb_dev[i] /= nb_samples
            cprint('g', f'    vlb {vlb_dev[i]:.4f} (best: {best_vlb:.4f})\n')
            
            # Visualization (optional)
            if train_plot and j == 0:  # Only use the last batch for visualization
                # Select a few examples for visualization
                x_subset = x[:10]  # Device handling done by methods
                
                # Get original reconstructions from base VAEAC
                z_original = base_vaeac.get_post(x_subset).sample()
                x_recon_base = base_vaeac.regenerate(z_original)
                
                # Get reconstructions through under_VAEAC
                u_approx = under_net.recongnition(z_original).sample()
                z_recon = under_net.regenerate(u_approx)
                x_recon_under = base_vaeac.regenerate(z_recon)
                
                # Display original and reconstructions
                if Nclass is not None:
                    x_subset = x_subset[:, :-Nclass]
                    x_recon_base = x_recon_base[:, :-Nclass]
                    x_recon_under = x_recon_under[:, :-Nclass]
                
                # Reshape for visualization if needed
                if len(x_subset.shape) == 2:
                    side = int(np.sqrt(x_subset.shape[1]))
                    x_subset = x_subset.view(-1, 1, side, side).data
                    x_recon_base = x_recon_base.view(-1, 1, side, side).data
                    x_recon_under = x_recon_under.view(-1, 1, side, side).data
                
                # Move tensors to CPU for plotting
                x_subset = x_subset.cpu()
                x_recon_base = x_recon_base.cpu()
                x_recon_under = x_recon_under.cpu()
                
                plt.figure()
                dd = make_grid(torch.cat([x_subset, x_recon_base, x_recon_under]), nrow=len(x_subset)).numpy()
                plt.imshow(np.transpose(dd, (1, 2, 0)), interpolation='nearest')
                plt.title('Original / Base Recon / Under Recon')
                if script_mode:
                    plt.savefig(results_dir + f'/under_recon_{i}.png')
                else:
                    plt.show()
                
                # Plot latent space if 2D
                if under_net.latent_dim == 2:
                    plt.figure(figsize=(10,8))
                    # Get a larger batch of latent vectors
                    x_large = x[:200]  # Device handling done by methods
                    z_large = base_vaeac.get_post(x_large).sample()
                    u_posterior = under_net.recongnition(z_large)
                    u_samples = u_posterior.sample().detach().cpu().numpy()
                    
                    plt.scatter(u_samples[:,0], u_samples[:,1], alpha=0.5)
                    plt.title(f'Under_VAEAC latent space (epoch {i+1})')
                    if script_mode:
                        plt.savefig(f'{results_dir}/latent_epoch{i+1}.png')
                    else:
                        plt.show()

        # Save best model
        if vlb_dev[i] > best_vlb:
            best_vlb = vlb_dev[i]
            best_epoch = i
            safe_save_model(under_net, os.path.join(models_dir, 'under_theta_best.dat'))

        # Early stopping
        if early_stop is not None and (i - best_epoch) > early_stop:
            break

    # Save final model
    safe_save_model(under_net, os.path.join(models_dir, 'under_theta_last.dat'))
    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(nb_epochs)
    cprint('r', f'   average time: {runtime_per_it:.4f} seconds\n')

    # Results
    cprint('c', '\nRESULTS:')
    nb_parameters = under_net.get_nb_parameters()
    best_cost_dev = np.max(vlb_dev)
    best_cost_train = np.max(vlb_train)

    print(f'  best_vlb_dev: {best_cost_dev:.4f}')
    print(f'  best_vlb_train: {best_cost_train:.4f}')
    print(f'  nb_parameters: {nb_parameters} ({humansize(nb_parameters)})\n')

    # Plot training curves
    plt.figure()
    plt.plot(np.clip(vlb_train, -1000, 1000), 'r')
    plt.plot(np.clip(vlb_dev, -1000, 1000), 'b')
    plt.legend(['cost_train', 'cost_dev'])
    plt.ylabel('vlb')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.savefig(results_dir+'/under_train_cost.png')
    
    if train_plot:
        plt.show()
        
    return vlb_train, vlb_dev




