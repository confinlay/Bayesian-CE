import torch
import torch.utils.data as data

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
    verbose=False,
    target_class=None,
    ReconstructionOnly=False
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
        target_class: Optional target class for the counterfactual
        ReconstructionOnly: If True, only shows reconstructions and not original images
        
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
        verbose=verbose,
        target_class=target_class
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
            reconstruction_ll = vae.log_likelihood(original_recon, k=k_samples).item()
            counterfactual_ll = vae.log_likelihood(clue_recon, k=k_samples).item()
            
            # Calculate differences and ratios
            likelihood_diff = original_ll - counterfactual_ll
            likelihood_ratio = np.exp(counterfactual_ll) / np.exp(original_ll)
            
            # Reconstruction vs counterfactual
            recon_cf_ll_diff = reconstruction_ll - counterfactual_ll
            recon_cf_ll_ratio = np.exp(counterfactual_ll) / np.exp(reconstruction_ll)
            
            # Calculate realism scores
            original_realism = vae.compute_realism_score(image).item()
            reconstruction_realism = vae.compute_realism_score(original_recon).item()
            counterfactual_realism = vae.compute_realism_score(clue_recon).item()
            
            # Calculate differences
            realism_diff = original_realism - counterfactual_realism
            recon_cf_realism_diff = reconstruction_realism - counterfactual_realism
            
            likelihood_metrics = {
                'original_log_likelihood': original_ll,
                'reconstruction_log_likelihood': reconstruction_ll,
                'counterfactual_log_likelihood': counterfactual_ll,
                'log_likelihood_difference': likelihood_diff,
                'recon_counterfactual_log_likelihood_difference': recon_cf_ll_diff,
                'likelihood_ratio': likelihood_ratio,
                'recon_counterfactual_likelihood_ratio': recon_cf_ll_ratio,
                'original_realism_score': original_realism,
                'reconstruction_realism_score': reconstruction_realism,
                'counterfactual_realism_score': counterfactual_realism,
                'realism_score_difference': realism_diff,
                'recon_counterfactual_realism_difference': recon_cf_realism_diff
            }
    
    # Create visualization
    fig = plt.figure(figsize=figsize)
    
    if ReconstructionOnly:
        # Only show reconstructions, not original images
        ax1 = plt.subplot(221)
        ax1.imshow(original_recon[0, 0].cpu().detach(), cmap='gray')
        ax1.set_title(f'Original Reconstruction\nPredicted: {original_pred}' + 
                  (f' (True: {true_label})' if true_label is not None else '') + 
                  f'\nEntropy: {original_entropy_recon[0]:.3f}')
        ax1.axis('off')
        
        ax2 = plt.subplot(222)
        ax2.imshow(clue_recon[0, 0].cpu().detach(), cmap='gray')
        ax2.set_title(f'Counterfactual (Target: Class {target_class})\nPredicted: {explained_pred}\nEntropy: {explained_entropy_recon[0]:.3f}')
        ax2.axis('off')
        
        ax3 = plt.subplot(223)
        recon_diff = clue_recon[0, 0].cpu().detach() - original_recon[0, 0].cpu().detach()
        ax3.imshow(recon_diff, cmap='RdBu', vmin=-1, vmax=1)  # Fixed scale -1 to 1
        ax3.set_title('Counterfactual vs Original\nReconstruction Difference')
        ax3.axis('off')
        
        # Plot top class probabilities
        ax4 = plt.subplot(224)
    else:
        ax1 = plt.subplot(231)
        ax1.imshow(image[0, 0].cpu(), cmap='gray')
        ax1.set_title(f'Original Image\nPredicted: {original_pred}' + 
                  (f' (True: {true_label})' if true_label is not None else '') + 
                  f'\nEntropy: {original_entropy_recon[0]:.3f}')
        ax1.axis('off')
        
        ax2 = plt.subplot(232)
        ax2.imshow(clue_recon[0, 0].cpu().detach(), cmap='gray')
        ax2.set_title(f'Counterfactual (Target: Class {target_class})\nPredicted: {explained_pred}\nEntropy: {explained_entropy_recon[0]:.3f}')
        ax2.axis('off')
        
        ax3 = plt.subplot(233)
        diff = clue_recon[0, 0].cpu().detach() - image[0, 0].cpu()
        ax3.imshow(diff, cmap='RdBu', vmin=-1, vmax=1)  # Fixed scale -1 to 1
        ax3.set_title(f'Counterfactual vs Original\nDifference\n(Red: Removed, Blue: Added)\nClass Change: {original_pred} → {explained_pred}')
        ax3.axis('off')
        
        ax4 = plt.subplot(234)
        ax4.imshow(original_recon[0, 0].cpu().detach(), cmap='gray')
        ax4.set_title(f'Original Reconstruction\nPredicted: {original_pred}\nEntropy: {original_entropy_recon[0]:.3f}')
        ax4.axis('off')
        
        ax5 = plt.subplot(235)
        recon_diff = clue_recon[0, 0].cpu().detach() - original_recon[0, 0].cpu().detach()
        ax5.imshow(recon_diff, cmap='RdBu', vmin=-1, vmax=1)  # Fixed scale -1 to 1
        ax5.set_title('Counterfactual vs Original\nReconstruction Difference')
        ax5.axis('off')
        
        # Plot top class probabilities
        ax6 = plt.subplot(236)
        ax6 = plt.subplot(236) if not ReconstructionOnly else ax4
    
    # Get indices of max probabilities for original and counterfactual
    orig_max_idx = np.argmax(original_mean_probs_recon.cpu().numpy()[0])
    cf_max_idx = np.argmax(explained_mean_probs_recon.cpu().numpy()[0])
    
    # Get top indices from original prediction, ensuring max indices are included
    top_indices = np.argsort(-original_mean_probs_recon.cpu().numpy()[0])[:5].tolist()
    
    # Make sure both max indices are included
    if orig_max_idx not in top_indices:
        top_indices = top_indices[:-1] + [orig_max_idx]
    if cf_max_idx not in top_indices and cf_max_idx != orig_max_idx:
        top_indices = top_indices[:-1] + [cf_max_idx]
    # Make sure target class is included if it exists
    if target_class is not None and target_class not in top_indices and target_class != orig_max_idx and target_class != cf_max_idx:
        top_indices = top_indices[:-1] + [target_class]
    
    # Convert to numpy array for indexing
    top_indices = np.array(top_indices)
    
    x = np.arange(len(top_indices))
    width = 0.35
    
    orig_probs = original_mean_probs_recon.cpu().numpy()[0][top_indices]
    new_probs = explained_mean_probs_recon.cpu().numpy()[0][top_indices]
    
    # Use the appropriate axis based on ReconstructionOnly
    ax_prob = ax4 if ReconstructionOnly else ax6
    
    ax_prob.bar(x - width/2, orig_probs, width, label='Original')
    ax_prob.bar(x + width/2, new_probs, width, label='Counterfactual')
    ax_prob.set_xticks(x)
    ax_prob.set_xticklabels(top_indices)
    ax_prob.set_title(f'CF Pred: {explained_pred} (Target: {target_class}, Entropy: {explained_entropy_recon[0]:.4f})')
    ax_prob.set_xlabel('Digit Class')
    ax_prob.set_ylabel('Probability')
    ax_prob.legend()
    
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
        print(f"Counterfactual (Predicted: {explained_pred}, Target: {target_class}): {explained_mean_probs_recon.cpu().numpy()[0].round(3)}")
        
        if vae is not None:
            print(f"\nLikelihood metrics:")
            print(f"Original log-likelihood: {results['original_log_likelihood']:.2f}")
            print(f"Reconstruction log-likelihood: {results['reconstruction_log_likelihood']:.2f}")
            print(f"Counterfactual log-likelihood: {results['counterfactual_log_likelihood']:.2f}")
            print(f"Log-likelihood difference (original vs CF): {results['log_likelihood_difference']:.2f}")
            print(f"Log-likelihood difference (recon vs CF): {results['recon_counterfactual_log_likelihood_difference']:.2f}")
            print(f"Likelihood ratio (CF/original): {results['likelihood_ratio']:.2f}x")
            print(f"Likelihood ratio (CF/recon): {results['recon_counterfactual_likelihood_ratio']:.2f}x")
            print(f"Original realism score: {results['original_realism_score']:.3f}")
            print(f"Reconstruction realism score: {results['reconstruction_realism_score']:.3f}")
            print(f"Counterfactual realism score: {results['counterfactual_realism_score']:.3f}")
            print(f"Realism score difference (original vs CF): {results['realism_score_difference']:.3f}")
            print(f"Realism score difference (recon vs CF): {results['recon_counterfactual_realism_difference']:.3f}")
    
    # Add target_class to result_dict
    results['target_class'] = target_class
    
    return results, fig

def evaluate_class_counterfactuals(
    images, 
    bayesian_model, 
    decoder, 
    vae, 
    target_classes,
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
    Evaluates targeted class counterfactuals on a set of images and calculates comprehensive metrics
    including target class confidence and VAE likelihood scores.
    
    Args:
        images: Tensor of images to analyze [N, 1, 28, 28]
        bayesian_model: The Bayesian model used to extract features and make predictions
        decoder: Decoder model to visualize latent representations
        vae: Variational Autoencoder for likelihood estimation
        target_classes: List of target classes to optimize towards, one per image
        uncertainty_weight: Weight for uncertainty term in optimization
        distance_weight: Weight for distance term in optimization
        lr: Learning rate for optimization
        steps: Number of optimization steps
        device: Device to run computation on ('cuda' or 'cpu')
        bayesian: Whether to use Bayesian uncertainty
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
    
    # Ensure target_classes is the same length as images
    assert len(target_classes) == len(images), "Number of target classes must match number of images"
    
    # Lists to store metrics
    target_confidences = []
    latent_target_confidences = []  # NEW: Store confidence for latent representation
    latent_distances = []
    likelihood_original = []
    likelihood_reconstruction = []
    likelihood_counterfactual = []
    likelihood_differences = []
    likelihood_ratios = []
    recon_likelihood_differences = []
    recon_likelihood_ratios = []
    realism_original = []
    realism_reconstruction = []
    realism_counterfactual = []
    realism_differences = []
    recon_realism_differences = []
    
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
        target_class = target_classes[i]
        
        if verbose:
            print(f"Processing image {i+1}/{len(images)}, target class: {target_class}")
        
        # Get latent representation
        with torch.no_grad():
            z0 = bayesian_model.extract_features(image)
        
        # Initialize CLUE for target class optimization
        clue = new_CLUE.NewCLUE(
            classifier=bayesian_model,
            z0=z0,
            uncertainty_weight=uncertainty_weight,
            distance_weight=distance_weight,
            lr=lr,
            device=device,
            bayesian=bayesian,
            verbose=verbose,
            target_class=target_class  # Set the target class
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
            
            # Get predictions for latent representations and reconstructions
            if bayesian:
                # Bayesian predictions from latent codes
                original_probs_latent = bayesian_model.sample_predict_z(z0)
                explained_probs_latent = bayesian_model.sample_predict_z(z_explained)
                
                # Calculate mean probabilities for latent predictions
                original_mean_probs_latent = original_probs_latent.mean(dim=0)
                explained_mean_probs_latent = explained_probs_latent.mean(dim=0)
                
                # Bayesian predictions from reconstructions
                original_probs_recon = bayesian_model.sample_predict(original_recon)
                explained_probs_recon = bayesian_model.sample_predict(clue_recon)
                
                # Calculate mean probabilities
                original_mean_probs_recon = original_probs_recon.mean(dim=0)
                explained_mean_probs_recon = explained_probs_recon.mean(dim=0)
                
                # Get class predictions
                original_class_latent = torch.argmax(original_mean_probs_latent, dim=1).item()
                explained_class_latent = torch.argmax(explained_mean_probs_latent, dim=1).item()
                original_class_recon = torch.argmax(original_mean_probs_recon, dim=1).item()
                explained_class_recon = torch.argmax(explained_mean_probs_recon, dim=1).item()
                
                # Get target class confidence for reconstruction
                target_confidence = explained_mean_probs_recon[0, target_class].item()
                
                # NEW: Get target class confidence for latent representation
                latent_target_confidence = explained_mean_probs_latent[0, target_class].item()
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
                explained_class_latent = torch.argmax(explained_mean_probs_latent, dim=1).item()
                original_class_recon = torch.argmax(original_mean_probs_recon, dim=1).item()
                explained_class_recon = torch.argmax(explained_mean_probs_recon, dim=1).item()
                
                # Get target class confidence for reconstruction
                target_confidence = explained_mean_probs_recon[0, target_class].item()
                
                # NEW: Get target class confidence for latent representation
                latent_target_confidence = explained_mean_probs_latent[0, target_class].item()
            
            target_confidences.append(target_confidence)
            latent_target_confidences.append(latent_target_confidence)  # NEW
            
            # Calculate VAE likelihood estimates
            original_ll = vae.log_likelihood(image, k=k_samples).item()
            reconstruction_ll = vae.log_likelihood(original_recon, k=k_samples).item()
            counterfactual_ll = vae.log_likelihood(clue_recon, k=k_samples).item()
            
            likelihood_original.append(original_ll)
            likelihood_reconstruction.append(reconstruction_ll)
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
            reconstruction_realism = vae.compute_realism_score(original_recon).item()
            counterfactual_realism = vae.compute_realism_score(clue_recon).item()
            realism_diff = original_realism - counterfactual_realism
            recon_realism_diff = reconstruction_realism - counterfactual_realism
            
            realism_original.append(original_realism)
            realism_reconstruction.append(reconstruction_realism)
            realism_counterfactual.append(counterfactual_realism)
            realism_differences.append(realism_diff)
            recon_realism_differences.append(recon_realism_diff)
            
            # Store individual results
            individual_results.append({
                'image_index': i,
                'target_class': target_class,
                'target_class_confidence': target_confidence,
                'latent_target_class_confidence': latent_target_confidence,  # NEW
                'original_image': image.cpu(),
                'original_reconstruction': original_recon.cpu(),
                'counterfactual_image': clue_recon.cpu(),
                'original_latent': z0.cpu(),
                'counterfactual_latent': z_explained.cpu(),
                'latent_distance': distance,
                'original_class_latent': original_class_latent,  # NEW
                'counterfactual_class_latent': explained_class_latent,  # NEW
                'original_latent_class_probs': original_mean_probs_latent.cpu().numpy(),  # NEW
                'counterfactual_latent_class_probs': explained_mean_probs_latent.cpu().numpy(),  # NEW
                'original_log_likelihood': original_ll,
                'reconstruction_log_likelihood': reconstruction_ll,
                'counterfactual_log_likelihood': counterfactual_ll,
                'log_likelihood_difference': likelihood_diff,
                'recon_counterfactual_log_likelihood_difference': recon_likelihood_diff,
                'likelihood_ratio': ratio,
                'recon_counterfactual_likelihood_ratio': recon_ratio,
                'original_realism_score': original_realism,
                'reconstruction_realism_score': reconstruction_realism,
                'counterfactual_realism_score': counterfactual_realism,
                'realism_score_difference': realism_diff,
                'recon_counterfactual_realism_difference': recon_realism_diff,
                'original_class_probs': original_mean_probs_recon.cpu().numpy(),
                'counterfactual_class_probs': explained_mean_probs_recon.cpu().numpy(),
                'original_class_recon': original_class_recon,
                'counterfactual_class_recon': explained_class_recon
            })
    
    # Calculate aggregate metrics
    results = {
        'avg_target_class_confidence': np.mean(target_confidences),
        'avg_latent_target_class_confidence': np.mean(latent_target_confidences),  # NEW
        'avg_latent_distance': np.mean(latent_distances),
        'avg_original_log_likelihood': np.mean(likelihood_original),
        'avg_reconstruction_log_likelihood': np.mean(likelihood_reconstruction),
        'avg_counterfactual_log_likelihood': np.mean(likelihood_counterfactual),
        'avg_log_likelihood_difference': np.mean(likelihood_differences),
        'avg_recon_counterfactual_log_likelihood_difference': np.mean(recon_likelihood_differences),
        'median_log_likelihood_difference': np.median(likelihood_differences),
        'median_recon_counterfactual_log_likelihood_difference': np.median(recon_likelihood_differences),
        'avg_likelihood_ratio': np.mean(likelihood_ratios),
        'avg_recon_counterfactual_likelihood_ratio': np.mean(recon_likelihood_ratios),
        'median_likelihood_ratio': np.median(likelihood_ratios),
        'median_recon_counterfactual_likelihood_ratio': np.median(recon_likelihood_ratios),
        'avg_original_realism_score': np.mean(realism_original),
        'avg_reconstruction_realism_score': np.mean(realism_reconstruction),
        'avg_counterfactual_realism_score': np.mean(realism_counterfactual),
        'avg_realism_score_difference': np.mean(realism_differences),
        'avg_recon_counterfactual_realism_difference': np.mean(recon_realism_differences),
        'median_realism_score_difference': np.median(realism_differences),
        'median_recon_counterfactual_realism_difference': np.median(recon_realism_differences),
        'individual_results': individual_results
    }
    
    # Print results if verbose
    if verbose:
        print(f"\nResults over {len(images)} images:")
        print(f"Average target class confidence: {results['avg_target_class_confidence']:.3f}")
        print(f"Average latent target class confidence: {results['avg_latent_target_class_confidence']:.3f}")  # NEW
        print(f"Average latent distance: {results['avg_latent_distance']:.3f}")
        print(f"Average original log likelihood: {results['avg_original_log_likelihood']:.3f}")
        print(f"Average reconstruction log likelihood: {results['avg_reconstruction_log_likelihood']:.3f}")
        print(f"Average counterfactual log likelihood: {results['avg_counterfactual_log_likelihood']:.3f}")
        print(f"Average log likelihood difference: {results['avg_log_likelihood_difference']:.3f}")
        print(f"Average recon-counterfactual log likelihood difference: {results['avg_recon_counterfactual_log_likelihood_difference']:.3f}")
        print(f"Median log likelihood difference: {results['median_log_likelihood_difference']:.3f}")
        print(f"Median recon-counterfactual log likelihood difference: {results['median_recon_counterfactual_log_likelihood_difference']:.3f}")
        print(f"Average likelihood ratio: {results['avg_likelihood_ratio']:.3f}")
        print(f"Average recon-counterfactual likelihood ratio: {results['avg_recon_counterfactual_likelihood_ratio']:.3f}")
        print(f"Median likelihood ratio: {results['median_likelihood_ratio']:.3f}")
        print(f"Median recon-counterfactual likelihood ratio: {results['median_recon_counterfactual_likelihood_ratio']:.3f}")
        print(f"Average original realism score: {results['avg_original_realism_score']:.3f}")
        print(f"Average reconstruction realism score: {results['avg_reconstruction_realism_score']:.3f}")
        print(f"Average counterfactual realism score: {results['avg_counterfactual_realism_score']:.3f}")
        print(f"Average realism score difference: {results['avg_realism_score_difference']:.3f}")
        print(f"Average recon-counterfactual realism difference: {results['avg_recon_counterfactual_realism_difference']:.3f}")
        print(f"Median realism score difference: {results['median_realism_score_difference']:.3f}")
        print(f"Median recon-counterfactual realism difference: {results['median_recon_counterfactual_realism_difference']:.3f}")
    
    return results


def visualize_class_counterfactual_results(results, n=5, figsize=(18, 12)):
    """
    Visualize class counterfactual results with original and counterfactual images,
    along with metrics for each.
    
    Args:
        results: Results dictionary from evaluate_class_counterfactuals
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
    
    # Select examples with the largest target class confidence
    sorted_indices = np.argsort([-r['target_class_confidence'] for r in results['individual_results']])
    selected_indices = sorted_indices[:n]
    
    # Maximum number of images per figure
    max_images_per_fig = 5
    
    # Calculate number of figures needed
    num_figures = math.ceil(n / max_images_per_fig)
    
    # Print aggregate results first
    print(f"\nAggregate Results over {len(results['individual_results'])} images:")
    print(f"Average target class confidence: {results['avg_target_class_confidence']:.3f}")
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
        fig.suptitle(f"Class Counterfactual Results (Figure {fig_idx+1}/{num_figures})", fontsize=16)
        
        # Process each image for this figure
        for i, idx in enumerate(current_indices):
            result = results['individual_results'][idx]
            
            # Print individual result details
            print(f"\nExample {start_idx + i + 1} (Image Index: {result['image_index']}):")
            print(f"Target class confidence: {result['target_class_confidence']:.3f}")
            print(f"Latent distance: {result['latent_distance']:.3f}")
            print(f"Log likelihood difference: {result['log_likelihood_difference']:.3f}")
            print(f"Original class prediction: {result['original_class_recon']}")
            print(f"Target class: {result['target_class']}")
            
            # Original image
            ax1 = plt.subplot(num_images_in_fig, 5, i*5 + 1)
            ax1.imshow(result['original_image'][0, 0].numpy(), cmap='gray')
            ax1.set_title(f"Original\nClass: {result['original_class_recon']}\nLL: {result['original_log_likelihood']:.1f}")
            ax1.axis('off')
            
            # Original reconstruction
            ax2 = plt.subplot(num_images_in_fig, 5, i*5 + 2)
            ax2.imshow(result['original_reconstruction'][0, 0].numpy(), cmap='gray')
            ax2.set_title(f"Original Reconstruction\nLL: {result['reconstruction_log_likelihood']:.1f}")
            ax2.axis('off')
            
            # Counterfactual image
            ax3 = plt.subplot(num_images_in_fig, 5, i*5 + 3)
            ax3.imshow(result['counterfactual_image'][0, 0].numpy(), cmap='gray')
            ax3.set_title(f"Counterfactual\nTarget: {result['target_class']}\nConfidence: {result['target_class_confidence']:.3f}\nLL: {result['counterfactual_log_likelihood']:.1f}")
            ax3.axis('off')
            
            # Difference map
            ax4 = plt.subplot(num_images_in_fig, 5, i*5 + 4)
            diff = result['original_reconstruction'][0, 0].numpy() - result['counterfactual_image'][0, 0].numpy()
            ax4.imshow(diff, cmap='coolwarm', vmin=-1, vmax=1)
            ax4.set_title(f"Difference\nDistance: {result['latent_distance']:.3f}\nLL-diff: {result['log_likelihood_difference']:.1f}")
            ax4.set_axis_off()
            
            # Class probability changes
            ax5 = plt.subplot(num_images_in_fig, 5, i*5 + 5)
            
            # Get top classes by probability
            top_indices = np.argsort(-result['counterfactual_class_probs'][0])[:5]
            orig_probs = result['original_class_probs'][0][top_indices]
            new_probs = result['counterfactual_class_probs'][0][top_indices]
            
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
                ax5.set_xlabel("Class")
                ax5.set_ylabel("Probability")
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to account for suptitle
        
        # Add to list but don't return them
        figs.append(fig)
    
    # Return None so no additional output is displayed
    return None

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

def save_class_counterfactual_results_to_csv(results, output_dir="./results", suffix='class_counterfactual_results'):
    """
    Saves class counterfactual evaluation results to CSV files.
    
    Args:
        results: Results dictionary from evaluate_class_counterfactuals
        output_dir: Directory to save the CSV files to
        suffix: Suffix for the results directory
        
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
            'original_class_recon', 'target_class', 'counterfactual_class_recon',
            'original_class_latent', 'counterfactual_class_latent',  # NEW
            'target_class_confidence', 'latent_target_class_confidence'  # NEW
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
        
        # NEW: Add latent class probabilities
        if 'original_latent_class_probs' in result:
            for i, prob in enumerate(result['original_latent_class_probs'][0]):
                result_dict[f'original_latent_prob_class_{i}'] = prob
                
        if 'counterfactual_latent_class_probs' in result:
            for i, prob in enumerate(result['counterfactual_latent_class_probs'][0]):
                result_dict[f'counterfactual_latent_prob_class_{i}'] = prob
        
        individual_data.append(result_dict)
    
    # Create DataFrame for individual results
    individual_df = pd.DataFrame(individual_data)
    
    # Extract aggregate metrics for the simplified aggregate results
    aggregate_data = []
    for result in results['individual_results']:
        agg_result = {}
        
        # Original class prediction
        agg_result['original_class_recon'] = result.get('original_class_recon')
        agg_result['original_class_latent'] = result.get('original_class_latent')  # NEW
        
        # Target class
        agg_result['target_class'] = result.get('target_class')
        
        # Target class probability in original prediction
        if 'original_class_probs' in result and 'target_class' in result:
            target_class = result['target_class']
            agg_result['original_target_class_prob'] = result['original_class_probs'][0][target_class]
        
        # NEW: Target class probability in original latent space
        if 'original_latent_class_probs' in result and 'target_class' in result:
            target_class = result['target_class']
            agg_result['original_latent_target_class_prob'] = result['original_latent_class_probs'][0][target_class]
        
        # Counterfactual class prediction
        agg_result['counterfactual_class_recon'] = result.get('counterfactual_class_recon')
        agg_result['counterfactual_class_latent'] = result.get('counterfactual_class_latent')  # NEW
        
        # Target class probability in counterfactual prediction
        if 'counterfactual_class_probs' in result and 'target_class' in result:
            target_class = result['target_class']
            agg_result['counterfactual_target_class_prob'] = result['counterfactual_class_probs'][0][target_class]
            
            # Difference in target class probability
            if 'original_class_probs' in result:
                orig_prob = result['original_class_probs'][0][target_class]
                cf_prob = result['counterfactual_class_probs'][0][target_class]
                agg_result['target_class_prob_difference'] = cf_prob - orig_prob
        
        # NEW: Target class probability in counterfactual latent space
        if 'counterfactual_latent_class_probs' in result and 'target_class' in result:
            target_class = result['target_class']
            agg_result['counterfactual_latent_target_class_prob'] = result['counterfactual_latent_class_probs'][0][target_class]
            
            # Difference in latent target class probability
            if 'original_latent_class_probs' in result:
                orig_prob = result['original_latent_class_probs'][0][target_class]
                cf_prob = result['counterfactual_latent_class_probs'][0][target_class]
                agg_result['latent_target_class_prob_difference'] = cf_prob - orig_prob
        
        # Save the direct confidence values
        agg_result['target_class_confidence'] = result.get('target_class_confidence')
        agg_result['latent_target_class_confidence'] = result.get('latent_target_class_confidence')  # NEW
        
        # Likelihood metrics
        agg_result['original_log_likelihood'] = result.get('original_log_likelihood')
        agg_result['counterfactual_log_likelihood'] = result.get('counterfactual_log_likelihood')
        agg_result['log_likelihood_difference'] = result.get('log_likelihood_difference')
        
        aggregate_data.append(agg_result)
    
    # Create DataFrame for aggregate results
    aggregate_df = pd.DataFrame(aggregate_data)
    
    # Save to CSV
    individual_csv_path = os.path.join(results_directory, f"individual_results.csv")
    aggregate_csv_path = os.path.join(results_directory, f"aggregate_results.csv")
    
    individual_df.to_csv(individual_csv_path, index=False)
    aggregate_df.to_csv(aggregate_csv_path, index=False)
    
    print(f"Individual results saved to: {individual_csv_path}")
    print(f"Aggregate results saved to: {aggregate_csv_path}")
    
    return individual_csv_path, aggregate_csv_path

def save_uncertain_dataset(uncertain_images, uncertain_indices, save_path="uncertain_dataset"):
    """
    Save uncertain images and their indices for later use.
    
    Args:
        uncertain_images: Tensor of uncertain images [n, channels, height, width]
        uncertain_indices: Indices of uncertain images in the original dataset
        save_path: Directory to save the dataset
    """
    import os
    import torch
    import json
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save the images as a tensor file
    torch.save(uncertain_images, os.path.join(save_path, "uncertain_images.pt"))
    
    # Save the indices as JSON
    with open(os.path.join(save_path, "uncertain_indices.json"), "w") as f:
        json.dump(uncertain_indices.tolist(), f)
    
    print(f"Dataset saved to {save_path}")
    print(f"Images shape: {uncertain_images.shape}")
    print(f"Number of indices: {len(uncertain_indices)}")


def load_uncertain_dataset(load_path="uncertain_dataset"):
    """
    Load previously saved uncertain images and their indices.
    
    Args:
        load_path: Directory containing the saved dataset
    
    Returns:
        uncertain_images: Tensor of uncertain images
        uncertain_indices: Indices of uncertain images in the original dataset
    """
    import os
    import torch
    import json
    import numpy as np
    
    # Load the images tensor
    images_path = os.path.join(load_path, "uncertain_images.pt")
    uncertain_images = torch.load(images_path)
    
    # Load the indices
    indices_path = os.path.join(load_path, "uncertain_indices.json")
    with open(indices_path, "r") as f:
        uncertain_indices = np.array(json.load(f))
    
    print(f"Dataset loaded from {load_path}")
    print(f"Images shape: {uncertain_images.shape}")
    print(f"Number of indices: {len(uncertain_indices)}")
    
    return uncertain_images, uncertain_indices


# Create a PyTorch Dataset for the uncertain images
class UncertainImagesDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for uncertain images.
    Can use original labels from a parent dataset if provided.
    """
    def __init__(self, images, indices, parent_dataset=None, transform=None):
        self.images = images
        self.indices = indices
        self.parent_dataset = parent_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # If parent dataset provided, get original label
        if self.parent_dataset is not None:
            _, label = self.parent_dataset[self.indices[idx]]
            return image, label
        
        # Otherwise just return the image with index as placeholder
        return image, self.indices[idx]

def save_class_counterfactual_images(
    results, 
    output_dir="./counterfactual_images",
    n=None,  # Number of examples to save (all by default)
    figsize=(18, 12),
    sort_by="confidence",  # Can be "confidence", "latent_confidence", "distance", "likelihood"
    include_metadata=True,
    dpi=150
):
    """
    Save visualizations of class counterfactual results to disk.
    
    Args:
        results: Results dictionary from evaluate_class_counterfactuals
        output_dir: Directory to save the images to
        n: Number of examples to save (all by default)
        figsize: Size of the figure
        sort_by: How to sort the examples ("confidence", "latent_confidence", "distance", "likelihood")
        include_metadata: Whether to include metadata in a text file
        dpi: Resolution of saved images
        
    Returns:
        list: Paths to the saved image files
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import datetime
    
    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of examples to save
    if n is None:
        n = len(results['individual_results'])
    else:
        n = min(n, len(results['individual_results']))
    
    # Use original order of examples (no sorting)
    selected_indices = list(range(n))
    
    # Prepare to collect paths of saved images
    saved_paths = []
    
    # Save aggregate results summary
    if include_metadata:
        summary_path = os.path.join(output_dir, "aggregate_results.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Aggregate Results over {len(results['individual_results'])} images:\n")
            f.write(f"Average target class confidence (reconstruction): {results['avg_target_class_confidence']:.4f}\n")
            if 'avg_latent_target_class_confidence' in results:
                f.write(f"Average target class confidence (latent): {results['avg_latent_target_class_confidence']:.4f}\n")
            f.write(f"Average latent distance: {results['avg_latent_distance']:.4f}\n")
            f.write(f"Average log likelihood difference: {results['avg_log_likelihood_difference']:.4f}\n")
            f.write(f"Average realism score difference: {results['avg_realism_score_difference']:.4f}\n")
    
    # Create and save a figure for each example
    for i, idx in enumerate(selected_indices):
        result = results['individual_results'][idx]
        
        # Create visualization
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"Class Counterfactual Result (Image {result['image_index']})", fontsize=16)
        
        # Original image
        ax1 = plt.subplot(151)
        ax1.imshow(result['original_image'][0, 0].cpu().numpy(), cmap='gray')
        ax1.set_title(f"Original\nClass: {result['original_class_recon']}\nLL: {result['original_log_likelihood']:.1f}")
        ax1.axis('off')
        
        # Original reconstruction
        ax2 = plt.subplot(152)
        ax2.imshow(result['original_reconstruction'][0, 0].cpu().numpy(), cmap='gray')
        ax2.set_title(f"Original Reconstruction\nLatent class: {result.get('original_class_latent', 'N/A')}\nLL: {result['reconstruction_log_likelihood']:.1f}")
        ax2.axis('off')
        
        # Counterfactual image
        ax3 = plt.subplot(153)
        ax3.imshow(result['counterfactual_image'][0, 0].cpu().numpy(), cmap='gray')
        title_text = f"Counterfactual\nTarget: {result['target_class']}\n" + \
                     f"Confidence: {result['target_class_confidence']:.3f}"
        
        if 'latent_target_class_confidence' in result:
            title_text += f"\nLatent confidence: {result['latent_target_class_confidence']:.3f}"
            
        title_text += f"\nLL: {result['counterfactual_log_likelihood']:.1f}"
        ax3.set_title(title_text)
        ax3.axis('off')
        
        # Difference map
        ax4 = plt.subplot(154)
        diff = result['original_reconstruction'][0, 0].cpu().numpy() - result['counterfactual_image'][0, 0].cpu().numpy()
        ax4.imshow(diff, cmap='coolwarm', vmin=-1, vmax=1)
        ax4.set_title(f"Difference\nDistance: {result['latent_distance']:.3f}\nLL-diff: {result['log_likelihood_difference']:.1f}")
        ax4.set_axis_off()
        
        # Class probability changes
        ax5 = plt.subplot(155)
        
        # Get top classes by probability, always including the target class
        top_indices = np.argsort(-result['counterfactual_class_probs'][0])[:4].tolist()
        if result['target_class'] not in top_indices:
            top_indices.append(result['target_class'])
        
        # Sort indices to ensure consistent ordering
        top_indices = sorted(top_indices)
        
        x = np.arange(len(top_indices))
        width = 0.25
        
        # Plot original reconstruction probabilities
        orig_recon_probs = result['original_class_probs'][0][top_indices]
        ax5.bar(x - width, orig_recon_probs, width, label='Original (Recon)')
        
        # Plot latent probabilities if available
        if 'counterfactual_latent_class_probs' in result:
            orig_latent_probs = result['original_latent_class_probs'][0][top_indices]
            cf_latent_probs = result['counterfactual_latent_class_probs'][0][top_indices]
            ax5.bar(x, cf_latent_probs, width, label='Counterfactual (Latent)')
        
        # Plot counterfactual reconstruction probabilities
        cf_recon_probs = result['counterfactual_class_probs'][0][top_indices]
        ax5.bar(x + width, cf_recon_probs, width, label='Counterfactual (Recon)')
        
        ax5.set_xticks(x)
        ax5.set_xticklabels(top_indices)
        ax5.set_title("Class probabilities")
        ax5.legend(loc='upper left', fontsize='small')
        ax5.set_xlabel("Class")
        ax5.set_ylabel("Probability")
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to account for suptitle
        
        # Save figure
        img_path = os.path.join(output_dir, f"counterfactual_{i+1:03d}_img{result['image_index']:04d}_target{result['target_class']}.png")
        plt.savefig(img_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        saved_paths.append(img_path)
        
        # Save metadata if requested
        if include_metadata:
            metadata_path = os.path.join(output_dir, f"counterfactual_{i+1:03d}_img{result['image_index']:04d}_target{result['target_class']}_metadata.txt")
            with open(metadata_path, 'w') as f:
                f.write(f"Image Index: {result['image_index']}\n")
                f.write(f"Target Class: {result['target_class']}\n\n")
                
                f.write("Classification:\n")
                f.write(f"Original Class (reconstruction): {result['original_class_recon']}\n")
                if 'original_class_latent' in result:
                    f.write(f"Original Class (latent): {result['original_class_latent']}\n")
                f.write(f"Counterfactual Class (reconstruction): {result['counterfactual_class_recon']}\n")
                if 'counterfactual_class_latent' in result:
                    f.write(f"Counterfactual Class (latent): {result['counterfactual_class_latent']}\n\n")
                
                f.write("Confidence:\n")
                f.write(f"Target Class Confidence (reconstruction): {result['target_class_confidence']:.6f}\n")
                if 'latent_target_class_confidence' in result:
                    f.write(f"Target Class Confidence (latent): {result['latent_target_class_confidence']:.6f}\n\n")
                
                f.write("Metrics:\n")
                f.write(f"Latent Distance: {result['latent_distance']:.6f}\n")
                f.write(f"Original Log Likelihood: {result['original_log_likelihood']:.6f}\n")
                f.write(f"Reconstruction Log Likelihood: {result['reconstruction_log_likelihood']:.6f}\n")
                f.write(f"Counterfactual Log Likelihood: {result['counterfactual_log_likelihood']:.6f}\n")
                f.write(f"Log Likelihood Difference (original vs counterfactual): {result['log_likelihood_difference']:.6f}\n")
                if 'recon_counterfactual_log_likelihood_difference' in result:
                    f.write(f"Log Likelihood Difference (reconstruction vs counterfactual): {result['recon_counterfactual_log_likelihood_difference']:.6f}\n")
                
                # Add probability distributions
                f.write("\nClass Probability Distributions:\n")
                f.write("Original (reconstruction):\n")
                for class_idx, prob in enumerate(result['original_class_probs'][0]):
                    if prob > 0.01:  # Only show significant probabilities
                        f.write(f"  Class {class_idx}: {prob:.6f}\n")
                
                if 'original_latent_class_probs' in result:
                    f.write("\nOriginal (latent):\n")
                    for class_idx, prob in enumerate(result['original_latent_class_probs'][0]):
                        if prob > 0.01:  # Only show significant probabilities
                            f.write(f"  Class {class_idx}: {prob:.6f}\n")
                
                f.write("\nCounterfactual (reconstruction):\n")
                for class_idx, prob in enumerate(result['counterfactual_class_probs'][0]):
                    if prob > 0.01:  # Only show significant probabilities
                        f.write(f"  Class {class_idx}: {prob:.6f}\n")
                
                if 'counterfactual_latent_class_probs' in result:
                    f.write("\nCounterfactual (latent):\n")
                    for class_idx, prob in enumerate(result['counterfactual_latent_class_probs'][0]):
                        if prob > 0.01:  # Only show significant probabilities
                            f.write(f"  Class {class_idx}: {prob:.6f}\n")
    
    print(f"Saved {len(saved_paths)} counterfactual visualizations to {output_dir}")
    return saved_paths