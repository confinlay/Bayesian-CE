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
    likelihood_reconstruction = []  # Added for original reconstruction
    likelihood_counterfactual = []
    likelihood_differences = []
    likelihood_ratios = []
    recon_likelihood_differences = []  # Added for reconstruction vs counterfactual
    recon_likelihood_ratios = []  # Added for reconstruction vs counterfactual
    realism_original = []
    realism_reconstruction = []  # Added for original reconstruction
    realism_counterfactual = []
    realism_differences = []
    recon_realism_differences = []  # Added for reconstruction vs counterfactual
    
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
                
                # Get class predictions
                original_class_latent = torch.argmax(original_mean_probs_latent, dim=1).item()
                original_class_recon = torch.argmax(original_mean_probs_recon, dim=1).item()
                explained_class_latent = torch.argmax(explained_mean_probs_latent, dim=1).item()
                explained_class_recon = torch.argmax(explained_mean_probs_recon, dim=1).item()
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
                
                # Get class predictions
                original_class_latent = torch.argmax(original_mean_probs_latent, dim=1).item()
                original_class_recon = torch.argmax(original_mean_probs_recon, dim=1).item()
                explained_class_latent = torch.argmax(explained_mean_probs_latent, dim=1).item()
                explained_class_recon = torch.argmax(explained_mean_probs_recon, dim=1).item()
            
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
            reconstruction_ll = vae.log_likelihood(original_recon, k=k_samples).item()  # Added for original reconstruction
            counterfactual_ll = vae.log_likelihood(clue_recon, k=k_samples).item()
            
            likelihood_original.append(original_ll)
            likelihood_reconstruction.append(reconstruction_ll)  # Added for original reconstruction
            likelihood_counterfactual.append(counterfactual_ll)
            likelihood_diff = original_ll - counterfactual_ll
            likelihood_differences.append(likelihood_diff)
            
            # Calculate reconstruction vs counterfactual likelihood difference
            recon_likelihood_diff = reconstruction_ll - counterfactual_ll
            recon_likelihood_differences.append(recon_likelihood_diff)
            
            # Calculate likelihood ratio (how many times less likely is the counterfactual)
            # Convert from log space to normal space for ratio
            ratio = np.exp(original_ll) / np.exp(counterfactual_ll)
            likelihood_ratios.append(ratio)
            
            # Calculate reconstruction vs counterfactual likelihood ratio
            recon_ratio = np.exp(reconstruction_ll) / np.exp(counterfactual_ll)
            recon_likelihood_ratios.append(recon_ratio)
            
            # Calculate realism scores
            original_realism = vae.compute_realism_score(image).item()
            reconstruction_realism = vae.compute_realism_score(original_recon).item()  # Added for original reconstruction
            counterfactual_realism = vae.compute_realism_score(clue_recon).item()
            realism_diff = original_realism - counterfactual_realism
            recon_realism_diff = reconstruction_realism - counterfactual_realism  # Added for reconstruction vs counterfactual
            
            realism_original.append(original_realism)
            realism_reconstruction.append(reconstruction_realism)  # Added for original reconstruction
            realism_counterfactual.append(counterfactual_realism)
            realism_differences.append(realism_diff)
            recon_realism_differences.append(recon_realism_diff)  # Added for reconstruction vs counterfactual
            
            # Store individual results
            individual_results.append({
                'image_index': i,
                'original_image': image.cpu(),
                'original_reconstruction': original_recon.cpu(),  # Store original reconstruction
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
                'reconstruction_log_likelihood': reconstruction_ll,  # Added for original reconstruction
                'counterfactual_log_likelihood': counterfactual_ll,
                'log_likelihood_difference': likelihood_diff,
                'recon_counterfactual_log_likelihood_difference': recon_likelihood_diff,  # Added for reconstruction vs counterfactual
                'likelihood_ratio': ratio,
                'recon_counterfactual_likelihood_ratio': recon_ratio,  # Added for reconstruction vs counterfactual
                'original_realism_score': original_realism,
                'reconstruction_realism_score': reconstruction_realism,  # Added for original reconstruction
                'counterfactual_realism_score': counterfactual_realism,
                'realism_score_difference': realism_diff,
                'recon_counterfactual_realism_difference': recon_realism_diff,  # Added for reconstruction vs counterfactual
                'original_class_probs': original_mean_probs_latent.cpu().numpy(),
                'original_recon_class_probs': original_mean_probs_recon.cpu().numpy(),  # Added class probs for original reconstruction
                'counterfactual_class_probs': explained_mean_probs_latent.cpu().numpy(),
                'counterfactual_recon_class_probs': explained_mean_probs_recon.cpu().numpy(),  # Added class probs for counterfactual reconstruction
                'original_class_latent': original_class_latent,  # Added class prediction for original latent
                'original_class_recon': original_class_recon,  # Added class prediction for original reconstruction
                'counterfactual_class_latent': explained_class_latent,  # Added class prediction for counterfactual latent
                'counterfactual_class_recon': explained_class_recon  # Added class prediction for counterfactual reconstruction
            })
    
    # Calculate aggregate metrics
    results = {
        'avg_latent_entropy_reduction': np.mean(latent_entropy_reductions),
        'avg_recon_entropy_reduction': np.mean(recon_entropy_reductions),
        'avg_latent_distance': np.mean(latent_distances),
        'avg_original_log_likelihood': np.mean(likelihood_original),
        'avg_reconstruction_log_likelihood': np.mean(likelihood_reconstruction),  # Added for original reconstruction
        'avg_counterfactual_log_likelihood': np.mean(likelihood_counterfactual),
        'avg_log_likelihood_difference': np.mean(likelihood_differences),
        'avg_recon_counterfactual_log_likelihood_difference': np.mean(recon_likelihood_differences),  # Added for reconstruction vs counterfactual
        'median_log_likelihood_difference': np.median(likelihood_differences),
        'median_recon_counterfactual_log_likelihood_difference': np.median(recon_likelihood_differences),  # Added for reconstruction vs counterfactual
        'avg_likelihood_ratio': np.mean(likelihood_ratios),
        'avg_recon_counterfactual_likelihood_ratio': np.mean(recon_likelihood_ratios),  # Added for reconstruction vs counterfactual
        'median_likelihood_ratio': np.median(likelihood_ratios),
        'median_recon_counterfactual_likelihood_ratio': np.median(recon_likelihood_ratios),  # Added for reconstruction vs counterfactual
        'avg_original_realism_score': np.mean(realism_original),
        'avg_reconstruction_realism_score': np.mean(realism_reconstruction),  # Added for original reconstruction
        'avg_counterfactual_realism_score': np.mean(realism_counterfactual),
        'avg_realism_score_difference': np.mean(realism_differences),
        'avg_recon_counterfactual_realism_difference': np.mean(recon_realism_differences),  # Added for reconstruction vs counterfactual
        'median_realism_score_difference': np.median(realism_differences),
        'median_recon_counterfactual_realism_difference': np.median(recon_realism_differences),  # Added for reconstruction vs counterfactual
        'individual_results': individual_results
    }
    
    # Print results if verbose
    if verbose:
        print(f"\nResults over {len(images)} images:")
        print(f"Average latent entropy reduction: {results['avg_latent_entropy_reduction']:.3f}")
        print(f"Average reconstruction entropy reduction: {results['avg_recon_entropy_reduction']:.3f}")
        print(f"Average latent distance: {results['avg_latent_distance']:.3f}")
        print(f"Average original log likelihood: {results['avg_original_log_likelihood']:.3f}")
        print(f"Average reconstruction log likelihood: {results['avg_reconstruction_log_likelihood']:.3f}")  # Added for original reconstruction
        print(f"Average counterfactual log likelihood: {results['avg_counterfactual_log_likelihood']:.3f}")
        print(f"Average log likelihood difference: {results['avg_log_likelihood_difference']:.3f}")
        print(f"Average recon-counterfactual log likelihood difference: {results['avg_recon_counterfactual_log_likelihood_difference']:.3f}")  # Added for reconstruction vs counterfactual
        print(f"Median log likelihood difference: {results['median_log_likelihood_difference']:.3f}")
        print(f"Median recon-counterfactual log likelihood difference: {results['median_recon_counterfactual_log_likelihood_difference']:.3f}")  # Added for reconstruction vs counterfactual
        print(f"Average likelihood ratio: {results['avg_likelihood_ratio']:.3f}")
        print(f"Average recon-counterfactual likelihood ratio: {results['avg_recon_counterfactual_likelihood_ratio']:.3f}")  # Added for reconstruction vs counterfactual
        print(f"Median likelihood ratio: {results['median_likelihood_ratio']:.3f}")
        print(f"Median recon-counterfactual likelihood ratio: {results['median_recon_counterfactual_likelihood_ratio']:.3f}")  # Added for reconstruction vs counterfactual
        print(f"Average original realism score: {results['avg_original_realism_score']:.3f}")
        print(f"Average reconstruction realism score: {results['avg_reconstruction_realism_score']:.3f}")  # Added for original reconstruction
        print(f"Average counterfactual realism score: {results['avg_counterfactual_realism_score']:.3f}")
        print(f"Average realism score difference: {results['avg_realism_score_difference']:.3f}")
        print(f"Average recon-counterfactual realism difference: {results['avg_recon_counterfactual_realism_difference']:.3f}")  # Added for reconstruction vs counterfactual
        print(f"Median realism score difference: {results['median_realism_score_difference']:.3f}")
        print(f"Median recon-counterfactual realism difference: {results['median_recon_counterfactual_realism_difference']:.3f}")  # Added for reconstruction vs counterfactual
    
    return results


def find_uncertain_images(model, dataloader, n=50, device='cuda', bayesian=True):
    """
    Find the n most uncertain images in a dataset according to a model.
    
    Args:
        model: Model with predict_with_uncertainty method
        dataloader: DataLoader for the dataset to evaluate
        n: Number of uncertain images to return
        device: Device to run computation on
        bayesian: If True, use Bayesian uncertainty. If False, use entropy of a single prediction.
        
    Returns:
        uncertain_images: Tensor of uncertain images [n, 1, 28, 28]
        uncertain_indices: Indices of uncertain images in the dataset
    """
    import torch
    import numpy as np
    import torch.nn.functional as F
    
    # Get uncertainty scores for all data points
    uncertainties = []
    indices = []
    
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            if bayesian:
                # Get predictions and uncertainties using Bayesian approach
                _, uncertainty_dict = model.predict_with_uncertainty(images)
                batch_uncertainties = uncertainty_dict['total_entropy'].cpu().numpy()
            else:
                # Regular forward pass for non-Bayesian uncertainty
                z, logits = model(images)
                probs = F.softmax(logits, dim=1)
                # Calculate entropy of the prediction
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                batch_uncertainties = entropy.cpu().numpy()
            
            # Store uncertainties and indices
            uncertainties.extend(batch_uncertainties)
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
    
    Args:
        results: Results dictionary from evaluate_clue_counterfactuals
        n: Number of examples to show (default: 5)
        figsize: Size of the figure
        
    Returns:
        None - figures are displayed directly in the notebook
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    
    # Close any existing figures to prevent duplicate displays
    plt.close('all')
    
    # Only visualize n examples
    n = min(n, len(results['individual_results']))
    
    # Select examples with the largest entropy reduction
    sorted_indices = np.argsort([-r['latent_entropy_reduction'] for r in results['individual_results']])
    selected_indices = sorted_indices[:n]
    
    # Maximum number of images per figure
    max_images_per_fig = 5
    
    # Calculate number of figures needed
    num_figures = math.ceil(n / max_images_per_fig)
    
    # Print aggregate results first
    print(f"\nAggregate Results over {len(results['individual_results'])} images:")
    print(f"Average latent entropy reduction: {results['avg_latent_entropy_reduction']:.3f}")
    print(f"Average reconstruction entropy reduction: {results['avg_recon_entropy_reduction']:.3f}")
    print(f"Average latent distance: {results['avg_latent_distance']:.3f}")
    print(f"Average log likelihood difference: {results['avg_log_likelihood_difference']:.3f}")
    print(f"Average realism score difference: {results['avg_realism_score_difference']:.3f}\n")
    
    # Create list to store figures (just for reference)
    figs = []
    
    # Process each figure
    for fig_idx in range(num_figures):
        # Get indices for this figure
        start_idx = fig_idx * max_images_per_fig
        end_idx = min((fig_idx + 1) * max_images_per_fig, n)
        current_indices = selected_indices[start_idx:end_idx]
        num_images_in_fig = len(current_indices)
        
        # Create a new figure
        fig = plt.figure(figsize=figsize)
        
        # Add figure title
        fig.suptitle(f"Counterfactual Results (Figure {fig_idx+1}/{num_figures})", fontsize=16)
        
        # Process each image for this figure
        for i, idx in enumerate(current_indices):
            result = results['individual_results'][idx]
            
            # Print individual result details
            print(f"\nExample {start_idx + i + 1} (Image Index: {result['image_index']}):")
            print(f"Latent entropy reduction: {result['latent_entropy_reduction']:.3f}")
            print(f"Latent distance: {result['latent_distance']:.3f}")
            print(f"Log likelihood difference: {result['log_likelihood_difference']:.3f}")
            print(f"Original class prediction: {result['original_class_latent']}")
            print(f"Counterfactual class prediction: {result['counterfactual_class_latent']}")
            
            # Original image
            ax1 = plt.subplot(num_images_in_fig, 5, i*5 + 1)
            ax1.imshow(result['original_image'][0, 0].numpy(), cmap='gray')
            ax1.set_title(f"Original\nClass: {result['original_class_latent']}\nEntropy: {result['original_entropy_latent']:.3f}\nLL: {result['original_log_likelihood']:.1f}")
            ax1.set_axis_off()
            
            # Original reconstruction
            ax2 = plt.subplot(num_images_in_fig, 5, i*5 + 2)
            ax2.imshow(result['original_reconstruction'][0, 0].numpy(), cmap='gray')
            ax2.set_title(f"Original Reconstruction\nClass: {result['original_class_recon']}\nEntropy: {result['original_entropy_recon']:.3f}\nLL: {result['reconstruction_log_likelihood']:.1f}")
            ax2.set_axis_off()
            
            # Counterfactual image
            ax3 = plt.subplot(num_images_in_fig, 5, i*5 + 3)
            ax3.imshow(result['counterfactual_image'][0, 0].numpy(), cmap='gray')
            ax3.set_title(f"Counterfactual\nClass: {result['counterfactual_class_latent']}\nLatent entropy: {result['counterfactual_entropy_latent']:.3f}\nReconstruction entropy: {result['counterfactual_entropy_recon']:.3f}\nLL: {result['counterfactual_log_likelihood']:.1f}")
            ax3.set_axis_off()
            
            # Difference map
            ax4 = plt.subplot(num_images_in_fig, 5, i*5 + 4)
            diff = result['original_reconstruction'][0, 0].numpy() - result['counterfactual_image'][0, 0].numpy()
            ax4.imshow(diff, cmap='coolwarm', vmin=-1, vmax=1)
            ax4.set_title(f"Difference\nDistance: {result['latent_distance']:.3f}\nLL-diff: {result['log_likelihood_difference']:.1f}")
            ax4.set_axis_off()
            
            # Class probability changes
            ax5 = plt.subplot(num_images_in_fig, 5, i*5 + 5)
            top_indices = np.argsort(-result['counterfactual_recon_class_probs'][0])[:5]
            orig_probs = result['original_recon_class_probs'][0][top_indices]
            new_probs = result['counterfactual_recon_class_probs'][0][top_indices]
            
            x = np.arange(len(top_indices))
            width = 0.35
            ax5.bar(x - width/2, orig_probs, width, label='Original')
            ax5.bar(x + width/2, new_probs, width, label='Counterfactual')
            ax5.set_xticks(x)
            ax5.set_xticklabels(top_indices)
            ax5.set_title("Top class probabilities")
            
            # Only add legend and labels to the first row of each figure
            if i == 0:
                ax5.legend()
                ax5.set_xlabel("Digit Class")
                ax5.set_ylabel("Probability")
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to account for suptitle
        
        # Add to list but don't return them
        figs.append(fig)
    
    # Return None so no additional output is displayed
    return None

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

def save_counterfactual_results_to_csv(results, output_dir="./results", suffix='clue_results'):
    """
    Saves counterfactual evaluation results to CSV files.
    
    Args:
        results: Results dictionary from evaluate_clue_counterfactuals
        output_dir: Directory to save the CSV files to
        prefix: Prefix for the CSV filenames
        
    Returns:
        tuple: Paths to the individual results CSV and aggregate results CSV
    """
    import pandas as pd
    import os
    import numpy as np
    from datetime import datetime

    suffix = suffix + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, suffix), exist_ok=True)
    results_directory = os.path.join(output_dir, suffix)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare individual results for CSV
    individual_data = []
    for result in results['individual_results']:
        # Convert tensors to numpy and extract scalar values
        result_dict = {}
        
        # Extract scalar metrics
        scalar_metrics = [
            'image_index', 'latent_distance', 'original_entropy_latent', 'counterfactual_entropy_latent',
            'latent_entropy_reduction', 'original_entropy_recon', 'counterfactual_entropy_recon',
            'recon_entropy_reduction', 'original_log_likelihood', 'reconstruction_log_likelihood',
            'counterfactual_log_likelihood', 'log_likelihood_difference', 
            'recon_counterfactual_log_likelihood_difference', 'likelihood_ratio',
            'recon_counterfactual_likelihood_ratio', 'original_realism_score',
            'reconstruction_realism_score', 'counterfactual_realism_score',
            'realism_score_difference', 'recon_counterfactual_realism_difference',
            'original_class_latent', 'original_class_recon',
            'counterfactual_class_latent', 'counterfactual_class_recon'
        ]
        
        # Add scalar metrics to result dictionary
        for metric in scalar_metrics:
            if metric in result:
                result_dict[metric] = result[metric]
        
        # Add class probabilities
        if 'original_class_probs' in result:
            for i, prob in enumerate(result['original_class_probs'][0]):
                result_dict[f'original_prob_class_{i}'] = prob
                
        if 'counterfactual_class_probs' in result:
            for i, prob in enumerate(result['counterfactual_class_probs'][0]):
                result_dict[f'counterfactual_prob_class_{i}'] = prob
        
        individual_data.append(result_dict)
    
    # Create DataFrame for individual results
    individual_df = pd.DataFrame(individual_data)
    
    # Extract aggregate metrics
    aggregate_metrics = {}
    for key, value in results.items():
        if key != 'individual_results' and not isinstance(value, list):
            aggregate_metrics[key] = [value]  # Wrap in list to create DataFrame row
    
    # Create DataFrame for aggregate results
    aggregate_df = pd.DataFrame(aggregate_metrics)
    
    # Save to CSV
    individual_csv_path = os.path.join(results_directory, f"individual_results.csv")
    aggregate_csv_path = os.path.join(results_directory, f"aggregate_results.csv")
    
    individual_df.to_csv(individual_csv_path, index=False)
    aggregate_df.to_csv(aggregate_csv_path, index=False)
    
    print(f"Individual results saved to: {individual_csv_path}")
    print(f"Aggregate results saved to: {aggregate_csv_path}")
    
    return individual_csv_path, aggregate_csv_path