

def evaluate_clue_counterfactuals(
    images, 
    bayesian_model, 
    decoder, 
    vae, 
    uncertainty_weight=1.0,
    distance_weight=0.005,
    lr=0.01,
    steps=200,
    device='cuda',
    bayesian=True,
    verbose=False,
    k_samples=100  # Number of importance samples for VAE likelihood estimation
):
    """
    Evaluates CLUE counterfactuals on a set of images and calculates comprehensive metrics
    including entropy reduction and VAE likelihood scores.
    
    Args:
        images: Tensor of images to analyze [N, 1, 28, 28]
        bayesian_model: The Bayesian model used to extract features and make predictions
        decoder: Decoder model to visualize latent representations
        vae: Variational Autoencoder for likelihood estimation
        uncertainty_weight: Weight for uncertainty term in CLUE optimization
        distance_weight: Weight for distance term in CLUE optimization
        lr: Learning rate for CLUE optimization
        steps: Number of optimization steps
        device: Device to run computation on ('cuda' or 'cpu')
        bayesian: Whether to use Bayesian uncertainty in CLUE
        verbose: Print detailed progress
        k_samples: Number of importance samples for VAE likelihood estimation
        
    Returns:
        results: Dictionary containing comprehensive metrics and individual image results
    """
    from clue import new_CLUE
    import torch
    import numpy as np
    
    # Move models to the specified device
    bayesian_model.eval()
    decoder.eval()
    vae.eval()
    
    # Lists to store metrics
    latent_entropy_reductions = []
    recon_entropy_reductions = []
    latent_distances = []
    likelihood_original = []
    likelihood_counterfactual = []
    likelihood_differences = []
    likelihood_ratios = []
    realism_original = []
    realism_counterfactual = []
    realism_differences = []
    
    # Lists to store individual results
    individual_results = []
    
    # Process each image
    with torch.no_grad():
        # Move images to device if not already
        if not isinstance(images, torch.Tensor):
            images = torch.stack([img for img in images]).to(device)
        elif images.device != device:
            images = images.to(device)
    
    for i in range(len(images)):
        image = images[i:i+1]  # Keep batch dimension
        
        if verbose:
            print(f"Processing image {i+1}/{len(images)}")
        
        # Get latent representation
        with torch.no_grad():
            z0 = bayesian_model.extract_features(image)
        
        # Initialize CLUE
        clue = new_CLUE.NewCLUE(
            classifier=bayesian_model,
            z0=z0,
            uncertainty_weight=uncertainty_weight,
            distance_weight=distance_weight,
            lr=lr,
            device=device,
            bayesian=bayesian,
            verbose=verbose
        )
        
        # Optimize to find explanation
        z_explained = clue.optimize(steps=steps)
        
        # Calculate distance between original and explained latent codes
        distance = torch.norm(z0 - z_explained).item()
        latent_distances.append(distance)
        
        # Generate reconstructions 
        with torch.no_grad():
            # Original reconstruction
            original_recon = decoder(z0)
            # CLUE reconstruction  
            clue_recon = decoder(z_explained)
            
            # Get predictions and uncertainties
            if bayesian:
                # Bayesian predictions from latent codes
                original_probs_latent = bayesian_model.sample_predict_z(z0)
                explained_probs_latent = bayesian_model.sample_predict_z(z_explained)
                
                # Bayesian predictions from reconstructions
                original_probs_recon = bayesian_model.sample_predict(original_recon)
                explained_probs_recon = bayesian_model.sample_predict(clue_recon)
                
                # Calculate mean probabilities and entropies for latent predictions
                original_mean_probs_latent = original_probs_latent.mean(dim=0)
                explained_mean_probs_latent = explained_probs_latent.mean(dim=0)
                
                # Calculate mean probabilities and entropies for reconstructions
                original_mean_probs_recon = original_probs_recon.mean(dim=0)
                explained_mean_probs_recon = explained_probs_recon.mean(dim=0)
            else:
                # Non-Bayesian predictions from latent codes
                original_logits_latent = bayesian_model.classifier(z0)
                explained_logits_latent = bayesian_model.classifier(z_explained)
                
                original_mean_probs_latent = torch.nn.functional.softmax(original_logits_latent, dim=1)
                explained_mean_probs_latent = torch.nn.functional.softmax(explained_logits_latent, dim=1)
                
                # Non-Bayesian predictions from reconstructions
                _, original_logits_recon = bayesian_model(original_recon)
                _, explained_logits_recon = bayesian_model(clue_recon)
                
                original_mean_probs_recon = torch.nn.functional.softmax(original_logits_recon, dim=1)
                explained_mean_probs_recon = torch.nn.functional.softmax(explained_logits_recon, dim=1)
            
            # Calculate entropies for latent predictions
            original_entropy_latent = -(original_mean_probs_latent * torch.log(original_mean_probs_latent + 1e-10)).sum(dim=1)
            explained_entropy_latent = -(explained_mean_probs_latent * torch.log(explained_mean_probs_latent + 1e-10)).sum(dim=1)
            
            # Calculate entropies for reconstruction predictions
            original_entropy_recon = -(original_mean_probs_recon * torch.log(original_mean_probs_recon + 1e-10)).sum(dim=1)
            explained_entropy_recon = -(explained_mean_probs_recon * torch.log(explained_mean_probs_recon + 1e-10)).sum(dim=1)
            
            # Calculate entropy reductions
            latent_entropy_reduction = (original_entropy_latent - explained_entropy_latent).item()
            recon_entropy_reduction = (original_entropy_recon - explained_entropy_recon).item()
            
            latent_entropy_reductions.append(latent_entropy_reduction)
            recon_entropy_reductions.append(recon_entropy_reduction)
            
            # Calculate VAE likelihood estimates
            original_ll = vae.log_likelihood(image, k=k_samples).item()
            counterfactual_ll = vae.log_likelihood(clue_recon, k=k_samples).item()
            
            likelihood_original.append(original_ll)
            likelihood_counterfactual.append(counterfactual_ll)
            likelihood_diff = original_ll - counterfactual_ll
            likelihood_differences.append(likelihood_diff)
            
            # Calculate likelihood ratio (how many times less likely is the counterfactual)
            # Convert from log space to normal space for ratio
            ratio = np.exp(original_ll) / np.exp(counterfactual_ll)
            likelihood_ratios.append(ratio)
            
            # Calculate realism scores
            original_realism = vae.compute_realism_score(image).item()
            counterfactual_realism = vae.compute_realism_score(clue_recon).item()
            realism_diff = original_realism - counterfactual_realism
            
            realism_original.append(original_realism)
            realism_counterfactual.append(counterfactual_realism)
            realism_differences.append(realism_diff)
            
            # Store individual results
            individual_results.append({
                'image_index': i,
                'original_image': image.cpu(),
                'counterfactual_image': clue_recon.cpu(),
                'original_latent': z0.cpu(),
                'counterfactual_latent': z_explained.cpu(),
                'latent_distance': distance,
                'original_entropy_latent': original_entropy_latent.item(),
                'counterfactual_entropy_latent': explained_entropy_latent.item(),
                'latent_entropy_reduction': latent_entropy_reduction,
                'original_entropy_recon': original_entropy_recon.item(),
                'counterfactual_entropy_recon': explained_entropy_recon.item(),
                'recon_entropy_reduction': recon_entropy_reduction,
                'original_log_likelihood': original_ll,
                'counterfactual_log_likelihood': counterfactual_ll,
                'log_likelihood_difference': likelihood_diff,
                'likelihood_ratio': ratio,
                'original_realism_score': original_realism,
                'counterfactual_realism_score': counterfactual_realism,
                'realism_score_difference': realism_diff,
                'original_class_probs': original_mean_probs_latent.cpu().numpy(),
                'counterfactual_class_probs': explained_mean_probs_latent.cpu().numpy()
            })
    
    # Calculate aggregate metrics
    results = {
        'avg_latent_entropy_reduction': np.mean(latent_entropy_reductions),
        'avg_recon_entropy_reduction': np.mean(recon_entropy_reductions),
        'avg_latent_distance': np.mean(latent_distances),
        'avg_original_log_likelihood': np.mean(likelihood_original),
        'avg_counterfactual_log_likelihood': np.mean(likelihood_counterfactual),
        'avg_log_likelihood_difference': np.mean(likelihood_differences),
        'median_log_likelihood_difference': np.median(likelihood_differences),
        'avg_likelihood_ratio': np.mean(likelihood_ratios),
        'median_likelihood_ratio': np.median(likelihood_ratios),
        'avg_original_realism_score': np.mean(realism_original),
        'avg_counterfactual_realism_score': np.mean(realism_counterfactual),
        'avg_realism_score_difference': np.mean(realism_differences),
        'median_realism_score_difference': np.median(realism_differences),
        'individual_results': individual_results
    }
    
    # Print results if verbose
    if verbose:
        print(f"\nResults over {len(images)} images:")
        print(f"Average latent entropy reduction: {results['avg_latent_entropy_reduction']:.3f}")
        print(f"Average reconstruction entropy reduction: {results['avg_recon_entropy_reduction']:.3f}")
        print(f"Average latent distance: {results['avg_latent_distance']:.3f}")
        print(f"Average original log likelihood: {results['avg_original_log_likelihood']:.3f}")
        print(f"Average counterfactual log likelihood: {results['avg_counterfactual_log_likelihood']:.3f}")
        print(f"Average log likelihood difference: {results['avg_log_likelihood_difference']:.3f}")
        print(f"Median log likelihood difference: {results['median_log_likelihood_difference']:.3f}")
        print(f"Average likelihood ratio: {results['avg_likelihood_ratio']:.3f}")
        print(f"Median likelihood ratio: {results['median_likelihood_ratio']:.3f}")
        print(f"Average original realism score: {results['avg_original_realism_score']:.3f}")
        print(f"Average counterfactual realism score: {results['avg_counterfactual_realism_score']:.3f}")
        print(f"Average realism score difference: {results['avg_realism_score_difference']:.3f}")
        print(f"Median realism score difference: {results['median_realism_score_difference']:.3f}")
    
    return results


def find_uncertain_images(model, dataloader, n=50, device='cuda'):
    """
    Find the n most uncertain images in a dataset according to a Bayesian model.
    
    Args:
        model: Bayesian model with predict_with_uncertainty method
        dataloader: DataLoader for the dataset to evaluate
        n: Number of uncertain images to return
        device: Device to run computation on
        
    Returns:
        uncertain_images: Tensor of uncertain images [n, 1, 28, 28]
        uncertain_indices: Indices of uncertain images in the dataset
    """
    import torch
    import numpy as np
    
    # Get uncertainty scores for all data points
    uncertainties = []
    indices = []
    
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            # Get predictions and uncertainties
            _, uncertainty_dict = model.predict_with_uncertainty(images)
            
            # Store total uncertainties and indices
            uncertainties.extend(uncertainty_dict['total_entropy'].cpu().numpy())
            indices.extend(range(i * len(images), i * len(images) + len(images)))
    
    # Convert to numpy arrays
    uncertainties = np.array(uncertainties)
    indices = np.array(indices)
    
    # Sort by uncertainty (descending order)
    sorted_idx = np.argsort(-uncertainties)
    sorted_indices = indices[sorted_idx]
    
    # Get the n most uncertain indices
    uncertain_indices = sorted_indices[:n]
    
    # Get the corresponding images
    uncertain_images = torch.stack([dataloader.dataset[idx][0] for idx in uncertain_indices])
    
    return uncertain_images, uncertain_indices

def visualize_counterfactual_results(results, n=5, figsize=(18, 12)):
    """
    Visualize counterfactual results with original and counterfactual images,
    along with metrics for each.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Only visualize n examples
    n = min(n, len(results['individual_results']))
    
    # Select examples with the largest entropy reduction
    sorted_indices = np.argsort([-r['latent_entropy_reduction'] for r in results['individual_results']])
    selected_indices = sorted_indices[:n]
    
    fig, axs = plt.subplots(n, 5, figsize=figsize)
    
    # Print aggregate results first
    print(f"\nAggregate Results over {len(results['individual_results'])} images:")
    print(f"Average latent entropy reduction: {results['avg_latent_entropy_reduction']:.3f}")
    print(f"Average reconstruction entropy reduction: {results['avg_recon_entropy_reduction']:.3f}")
    print(f"Average latent distance: {results['avg_latent_distance']:.3f}")
    print(f"Average log likelihood difference: {results['avg_log_likelihood_difference']:.3f}")
    print(f"Average realism score difference: {results['avg_realism_score_difference']:.3f}\n")
    
    for i, idx in enumerate(selected_indices):
        result = results['individual_results'][idx]
        
        # Print individual result details
        print(f"\nExample {i+1} (Image Index: {result['image_index']}):")
        print(f"Latent entropy reduction: {result['latent_entropy_reduction']:.3f}")
        print(f"Latent distance: {result['latent_distance']:.3f}")
        print(f"Log likelihood difference: {result['log_likelihood_difference']:.3f}")
        
        # Original image
        axs[i, 0].imshow(result['original_image'][0, 0].numpy(), cmap='gray')
        axs[i, 0].set_title(f"Original\nEntropy: {result['original_entropy_latent']:.3f}\nLL: {result['original_log_likelihood']:.1f}")
        
        # Counterfactual image
        axs[i, 1].imshow(result['counterfactual_image'][0, 0].numpy(), cmap='gray')
        axs[i, 1].set_title(f"Counterfactual\nEntropy: {result['counterfactual_entropy_latent']:.3f}\nLL: {result['counterfactual_log_likelihood']:.1f}")
        
        # Difference map
        diff = result['counterfactual_image'][0, 0].numpy() - result['original_image'][0, 0].numpy()
        axs[i, 2].imshow(diff, cmap='coolwarm', vmin=-1, vmax=1)
        axs[i, 2].set_title(f"Difference\nDistance: {result['latent_distance']:.3f}\nLL-diff: {result['log_likelihood_difference']:.1f}")
        
        # Original reconstruction (using counterfactual_image as it's already the reconstruction)
        axs[i, 3].imshow(result['counterfactual_image'][0, 0].numpy(), cmap='gray')
        axs[i, 3].set_title("Original Reconstruction")
        
        # Class probability changes
        top_indices = np.argsort(-result['counterfactual_class_probs'][0])[:5]
        orig_probs = result['original_class_probs'][0][top_indices]
        new_probs = result['counterfactual_class_probs'][0][top_indices]
        
        x = np.arange(len(top_indices))
        width = 0.35
        axs[i, 4].bar(x - width/2, orig_probs, width, label='Original')
        axs[i, 4].bar(x + width/2, new_probs, width, label='Counterfactual')
        axs[i, 4].set_xticks(x)
        axs[i, 4].set_xticklabels(top_indices)
        axs[i, 4].set_title("Top class probabilities")
        
        # Only add legend and labels to the first row
        if i == 0:
            axs[i, 4].legend()
            axs[i, 4].set_xlabel("Digit Class")
            axs[i, 4].set_ylabel("Probability")
    
    # Turn off axes for all subplots except probability plots
    for ax in axs.flat:
        ax.set_axis_off()
    
    # Turn on axes and labels for the first row
    for j in range(5):
        axs[0, j].set_axis_on()
        axs[0, j].set_xticks([])
        axs[0, j].set_yticks([])
    
    # Set column labels for the first row
    axs[0, 0].set_xlabel("Original")
    axs[0, 1].set_xlabel("Counterfactual")
    axs[0, 2].set_xlabel("Difference")
    axs[0, 3].set_xlabel("Original Reconstruction")
    
    # Set axes for probability plot in first row
    axs[0, 4].set_axis_on()
    axs[0, 4].set_xlabel("Digit Class")
    axs[0, 4].set_ylabel("Probability")
    
    plt.tight_layout()
    return fig

def evaluate_single_clue_counterfactual(
    image, 
    bayesian_model, 
    decoder, 
    vae=None,
    true_label=None,
    uncertainty_weight=1.0,
    distance_weight=0.005,
    lr=0.1,
    steps=200,
    device='cuda',
    bayesian=True,
    k_samples=100,
    figsize=(15, 10),
    show_plot=True,
    verbose=False
):
    """
    Evaluates CLUE counterfactual on a single image, calculates metrics and visualizes the results.
    
    Args:
        image: Single image tensor [1, 1, 28, 28] or [1, 28, 28]
        bayesian_model: The Bayesian model used to extract features and make predictions
        decoder: Decoder model to visualize latent representations
        vae: Optional VAE for likelihood estimation
        true_label: Optional ground truth label for the image
        uncertainty_weight: Weight for uncertainty term in CLUE optimization
        distance_weight: Weight for distance term in CLUE optimization
        lr: Learning rate for CLUE optimization
        steps: Number of optimization steps
        device: Device to run computation on ('cuda', 'mps', or 'cpu')
        bayesian: Whether to use Bayesian uncertainty in CLUE
        k_samples: Number of importance samples for VAE likelihood estimation
        figsize: Size of the figure
        show_plot: Whether to display the plot immediately
        verbose: Print detailed progress
        
    Returns:
        results: Dictionary containing metrics
        fig: Matplotlib figure object
    """
    from clue import new_CLUE
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Move models to the specified device
    bayesian_model.eval()
    decoder.eval()
    if vae is not None:
        vae.eval()
    
    # Ensure image is a proper tensor with batch dimension
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)
    
    # Add channel dimension if needed
    if image.dim() == 2:  # [H, W]
        image = image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif image.dim() == 3 and (image.shape[0] == 1 or image.shape[0] == 28):  # [1, H, W] or [H, W, 1]
        if image.shape[0] == 28:  # Likely [H, W, C]
            image = image.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        else:  # [C, H, W]
            image = image.unsqueeze(0)  # [1, C, H, W]
    
    # Move to device
    image = image.to(device)
    
    # Get latent representation
    with torch.no_grad():
        z0 = bayesian_model.extract_features(image)
    
    # Initialize CLUE
    clue = new_CLUE.NewCLUE(
        classifier=bayesian_model,
        z0=z0,
        uncertainty_weight=uncertainty_weight,
        distance_weight=distance_weight,
        lr=lr,
        device=device,
        bayesian=bayesian,
        verbose=verbose
    )
    
    # Optimize to find explanation
    z_explained = clue.optimize(steps=steps)
    
    # Calculate distance between original and explained latent codes
    distance = torch.norm(z0 - z_explained).item()
    
    # Generate reconstructions 
    with torch.no_grad():
        # Original reconstruction
        original_recon = decoder(z0)
        # CLUE reconstruction  
        clue_recon = decoder(z_explained)
        
        # Get predictions and uncertainties
        if bayesian:
            # Bayesian predictions from latent codes
            original_probs_latent = bayesian_model.sample_predict_z(z0)
            explained_probs_latent = bayesian_model.sample_predict_z(z_explained)
            
            # Bayesian predictions from reconstructions
            original_probs_recon = bayesian_model.sample_predict(original_recon)
            explained_probs_recon = bayesian_model.sample_predict(clue_recon)
            
            # Calculate mean probabilities for latent predictions
            original_mean_probs_latent = original_probs_latent.mean(dim=0)
            explained_mean_probs_latent = explained_probs_latent.mean(dim=0)
            
            # Calculate mean probabilities for reconstructions
            original_mean_probs_recon = original_probs_recon.mean(dim=0)
            explained_mean_probs_recon = explained_probs_recon.mean(dim=0)
        else:
            # Non-Bayesian predictions from latent codes
            original_logits_latent = bayesian_model.classifier(z0)
            explained_logits_latent = bayesian_model.classifier(z_explained)
            
            original_mean_probs_latent = torch.nn.functional.softmax(original_logits_latent, dim=1)
            explained_mean_probs_latent = torch.nn.functional.softmax(explained_logits_latent, dim=1)
            
            # Non-Bayesian predictions from reconstructions
            _, original_logits_recon = bayesian_model(original_recon)
            _, explained_logits_recon = bayesian_model(clue_recon)
            
            original_mean_probs_recon = torch.nn.functional.softmax(original_logits_recon, dim=1)
            explained_mean_probs_recon = torch.nn.functional.softmax(explained_logits_recon, dim=1)
        
        # Calculate entropies for latent predictions
        original_entropy_latent = -(original_mean_probs_latent * torch.log(original_mean_probs_latent + 1e-10)).sum(dim=1)
        explained_entropy_latent = -(explained_mean_probs_latent * torch.log(explained_mean_probs_latent + 1e-10)).sum(dim=1)
        
        # Calculate entropies for reconstruction predictions
        original_entropy_recon = -(original_mean_probs_recon * torch.log(original_mean_probs_recon + 1e-10)).sum(dim=1)
        explained_entropy_recon = -(explained_mean_probs_recon * torch.log(explained_mean_probs_recon + 1e-10)).sum(dim=1)
        
        # Calculate entropy reductions
        latent_entropy_reduction = (original_entropy_latent - explained_entropy_latent).item()
        recon_entropy_reduction = (original_entropy_recon - explained_entropy_recon).item()
        
        # Get predicted classes
        original_pred = original_mean_probs_recon.argmax(dim=1).item()
        explained_pred = explained_mean_probs_recon.argmax(dim=1).item()
        
        # Calculate VAE likelihood estimates and realism scores if VAE is provided
        likelihood_metrics = {}
        if vae is not None:
            original_ll = vae.log_likelihood(image, k=k_samples).item()
            counterfactual_ll = vae.log_likelihood(clue_recon, k=k_samples).item()
            likelihood_diff = original_ll - counterfactual_ll
            likelihood_ratio = np.exp(counterfactual_ll) / np.exp(original_ll)
            
            # Calculate realism scores
            original_realism = vae.compute_realism_score(image).item()
            counterfactual_realism = vae.compute_realism_score(clue_recon).item()
            realism_diff = original_realism - counterfactual_realism
            
            likelihood_metrics = {
                'original_log_likelihood': original_ll,
                'counterfactual_log_likelihood': counterfactual_ll,
                'log_likelihood_difference': likelihood_diff,
                'likelihood_ratio': likelihood_ratio,
                'original_realism_score': original_realism,
                'counterfactual_realism_score': counterfactual_realism,
                'realism_score_difference': realism_diff
            }
    
    # Create visualization
    fig = plt.figure(figsize=figsize)
    
    plt.subplot(231)
    plt.imshow(image[0, 0].cpu(), cmap='gray')
    plt.title(f'Original Image\nPredicted: {original_pred}' + 
              (f' (True: {true_label})' if true_label is not None else '') + 
              f'\nEntropy: {original_entropy_recon[0]:.3f}')
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(clue_recon[0, 0].cpu().detach(), cmap='gray')
    plt.title(f'Counterfactual\nPredicted: {explained_pred}\nEntropy: {explained_entropy_recon[0]:.3f}')
    plt.axis('off')
    
    plt.subplot(233)
    diff = clue_recon[0, 0].cpu().detach() - image[0, 0].cpu()
    plt.imshow(diff, cmap='RdBu', vmin=-1, vmax=1)  # Fixed scale -1 to 1
    plt.title('Counterfactual vs Original\nDifference\n(Red: Removed, Blue: Added)')
    plt.axis('off')
    
    plt.subplot(234)
    plt.imshow(original_recon[0, 0].cpu().detach(), cmap='gray')
    plt.title('Original Reconstruction')
    plt.axis('off')
    
    plt.subplot(235)
    recon_diff = clue_recon[0, 0].cpu().detach() - original_recon[0, 0].cpu().detach()
    plt.imshow(recon_diff, cmap='RdBu', vmin=-1, vmax=1)  # Fixed scale -1 to 1
    plt.title('Counterfactual vs Original\nReconstruction Difference')
    plt.axis('off')
    
    # Plot top class probabilities
    plt.subplot(236)
    top_indices = np.argsort(-original_mean_probs_recon.cpu().numpy()[0])[:5]
    x = np.arange(len(top_indices))
    width = 0.35
    
    orig_probs = original_mean_probs_recon.cpu().numpy()[0][top_indices]
    new_probs = explained_mean_probs_recon.cpu().numpy()[0][top_indices]
    
    plt.bar(x - width/2, orig_probs, width, label='Original')
    plt.bar(x + width/2, new_probs, width, label='Counterfactual')
    plt.xticks(x, [str(i) for i in top_indices])
    plt.xlabel('Digit Class')
    plt.ylabel('Probability')
    plt.title('Top Class Probabilities')
    plt.legend()
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        
    # Compile results
    results = {
        'latent_distance': distance,
        'original_entropy_latent': original_entropy_latent.item(),
        'counterfactual_entropy_latent': explained_entropy_latent.item(),
        'latent_entropy_reduction': latent_entropy_reduction,
        'original_entropy_recon': original_entropy_recon.item(),
        'counterfactual_entropy_recon': explained_entropy_recon.item(),
        'recon_entropy_reduction': recon_entropy_reduction,
        'original_pred': original_pred,
        'counterfactual_pred': explained_pred,
        'original_probs': original_mean_probs_recon.cpu().numpy()[0],
        'counterfactual_probs': explained_mean_probs_recon.cpu().numpy()[0],
        'original_latent': z0.cpu(),
        'counterfactual_latent': z_explained.cpu(),
        'original_image': image.cpu(),
        'counterfactual_image': clue_recon.cpu(),
        **likelihood_metrics  # Add likelihood metrics if available
    }
    
    # Print results if verbose
    if verbose:
        print("\nCLUE Counterfactual Results:")
        print(f"Latent Distance: {distance:.3f}")
        print(f"Latent Entropy Reduction: {latent_entropy_reduction:.3f}")
        print(f"Reconstruction Entropy Reduction: {recon_entropy_reduction:.3f}")
        print(f"\nClass probabilities:")
        print(f"Original (Predicted: {original_pred}" + 
              (f", True: {true_label}" if true_label is not None else "") + 
              f"): {original_mean_probs_recon.cpu().numpy()[0].round(3)}")
        print(f"Counterfactual (Predicted: {explained_pred}): {explained_mean_probs_recon.cpu().numpy()[0].round(3)}")
        
        if vae is not None:
            print(f"\nLikelihood metrics:")
            print(f"Original log-likelihood: {results['original_log_likelihood']:.2f}")
            print(f"Counterfactual log-likelihood: {results['counterfactual_log_likelihood']:.2f}")
            print(f"Log-likelihood difference: {results['log_likelihood_difference']:.2f}")
            print(f"Likelihood ratio: {results['likelihood_ratio']:.2f}x more likely")
            print(f"Original realism score: {results['original_realism_score']:.3f}")
            print(f"Counterfactual realism score: {results['counterfactual_realism_score']:.3f}")
            print(f"Realism score difference: {results['realism_score_difference']:.3f}")
    
    return results, fig