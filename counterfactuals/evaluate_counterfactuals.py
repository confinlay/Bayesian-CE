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
    from clue import counterfactual_optimizer
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
        clue = counterfactual_optimizer.CounterfactualOptimizer(
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

def evaluate_single_counterfactual(
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
    ReconstructionOnly=False,
    counterfactual_scorer=None
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
        counterfactual_scorer: Optional instance of counterfactual_scorer.py for IM1 score calculation
        
    Returns:
        results: Dictionary containing metrics
        fig: Matplotlib figure object
    """
    from clue import counterfactual_optimizer
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
    clue = counterfactual_optimizer.CounterfactualOptimizer(
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
            
            # Get predicted classes from latent
            original_pred_latent = original_mean_probs_latent.argmax(dim=1).item()
            explained_pred_latent = explained_mean_probs_latent.argmax(dim=1).item()
            # Get predicted classes from reconstruction
            original_pred_recon = original_mean_probs_recon.argmax(dim=1).item()
            explained_pred_recon = explained_mean_probs_recon.argmax(dim=1).item()
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
            
            # Get predicted classes from latent
            original_pred_latent = original_mean_probs_latent.argmax(dim=1).item()
            explained_pred_latent = explained_mean_probs_latent.argmax(dim=1).item()
            # Get predicted classes from reconstruction
            original_pred_recon = original_mean_probs_recon.argmax(dim=1).item()
            explained_pred_recon = explained_mean_probs_recon.argmax(dim=1).item()
        
        # Calculate entropies for latent predictions
        original_entropy_latent = -(original_mean_probs_latent * torch.log(original_mean_probs_latent + 1e-10)).sum(dim=1)
        explained_entropy_latent = -(explained_mean_probs_latent * torch.log(explained_mean_probs_latent + 1e-10)).sum(dim=1)
        
        # Calculate entropies for reconstruction predictions
        original_entropy_recon = -(original_mean_probs_recon * torch.log(original_mean_probs_recon + 1e-10)).sum(dim=1)
        explained_entropy_recon = -(explained_mean_probs_recon * torch.log(explained_mean_probs_recon + 1e-10)).sum(dim=1)
        
        # Calculate entropy reductions
        latent_entropy_reduction = (original_entropy_latent - explained_entropy_latent).item()
        recon_entropy_reduction = (original_entropy_recon - explained_entropy_recon).item()
        
        # Get predicted classes (using reconstruction-based predictions as primary)
        original_pred = original_pred_recon 
        explained_pred = explained_pred_recon
        
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
        
        # Calculate IM1 score if counterfactual_scorer is provided
        im1_metrics = {}
        if counterfactual_scorer is not None and target_class is not None:
            # Calculate IM1 score using the scorer's calculate_im1 function
            im1_score = counterfactual_scorer.calculate_im1(
                x_prime=clue_recon,
                original_class=original_pred,
                counterfactual_class=target_class
            )
            
            im1_metrics = {
                'im1_score': im1_score
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
        'original_class_latent': original_pred_latent,
        'counterfactual_class_latent': explained_pred_latent,
        'original_latent_probs': original_mean_probs_latent.cpu().numpy()[0],
        'counterfactual_latent_probs': explained_mean_probs_latent.cpu().numpy()[0],
        'original_latent': z0.cpu(),
        'counterfactual_latent': z_explained.cpu(),
        'original_image': image.cpu(),
        'original_reconstruction': original_recon.cpu(),
        'counterfactual_image': clue_recon.cpu(),
        **likelihood_metrics,  # Add likelihood metrics if available
        **im1_metrics  # Add IM1 metrics if available
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
        
        if counterfactual_scorer is not None and target_class is not None:
            print(f"\nIM1 score: {results['im1_score']:.5f}")
    
    # Add target_class to result_dict
    results['target_class'] = target_class
    
    return results, fig

def evaluate_single_auxiliary_counterfactual(
    image, 
    bayesian_model, 
    autoencoder,  # Changed from decoder to full autoencoder
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
    ReconstructionOnly=False,
    counterfactual_scorer=None
):
    """
    Evaluates CLUE counterfactual on a single image using an auxiliary autoencoder, calculates metrics and visualizes the results.
    
    Args:
        image: Single image tensor [1, 1, 28, 28] or [1, 28, 28]
        bayesian_model: The Bayesian model used to make predictions
        autoencoder: Full autoencoder model with encoder and decoder components
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
        counterfactual_scorer: Optional scorer for IM1 metric calculation
        
    Returns:
        results: Dictionary containing metrics
        fig: Matplotlib figure object
    """
    from clue import counterfactual_optimizer
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Move models to the specified device
    bayesian_model.eval()
    autoencoder.eval()
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
    
    # Get latent representation using the autoencoder's encoder
    with torch.no_grad():
        z0 = autoencoder.encode(image)  # Using autoencoder to encode instead of bayesian_model
    
    # Initialize CLUE
    clue = counterfactual_optimizer.CounterfactualOptimizer(
        classifier=bayesian_model,
        z0=z0,
        uncertainty_weight=uncertainty_weight,
        distance_weight=distance_weight,
        lr=lr,
        device=device,
        bayesian=bayesian,
        verbose=verbose,
        target_class=target_class,
        autoencoder=autoencoder
    )
    
    # Optimize to find explanation
    z_explained = clue.auxiliary_optimize(steps=steps)
    
    # Calculate distance between original and explained latent codes
    distance = torch.norm(z0 - z_explained).item()
    
    # Generate reconstructions 
    with torch.no_grad():
        # Original reconstruction
        original_recon = autoencoder.decode(z0)  # Using autoencoder's decoder
        # CLUE reconstruction  
        clue_recon = autoencoder.decode(z_explained)  # Using autoencoder's decoder
        
        # Get predictions and uncertainties
        if bayesian:
            # Bayesian predictions from reconstructions
            original_probs_recon = bayesian_model.sample_predict(original_recon)
            explained_probs_recon = bayesian_model.sample_predict(clue_recon)
            
            # Calculate mean probabilities for reconstructions
            original_mean_probs_recon = original_probs_recon.mean(dim=0)
            explained_mean_probs_recon = explained_probs_recon.mean(dim=0)
        else:
            # Non-Bayesian predictions from reconstructions
            _, original_logits_recon = bayesian_model(original_recon)
            _, explained_logits_recon = bayesian_model(clue_recon)
            
            original_mean_probs_recon = torch.nn.functional.softmax(original_logits_recon, dim=1)
            explained_mean_probs_recon = torch.nn.functional.softmax(explained_logits_recon, dim=1)
        
        # Calculate entropies for latent predictions (using reconstructions)
        original_entropy_latent = -(original_mean_probs_recon * torch.log(original_mean_probs_recon + 1e-10)).sum(dim=1)
        explained_entropy_latent = -(explained_mean_probs_recon * torch.log(explained_mean_probs_recon + 1e-10)).sum(dim=1)
        
        # Calculate entropies for reconstruction predictions (same as latent in this case)
        original_entropy_recon = original_entropy_latent.clone()
        explained_entropy_recon = explained_entropy_latent.clone()
        
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
            
        # Calculate IM1 score if scorer is provided and target class is specified
        im1_metrics = {}
        if counterfactual_scorer is not None and target_class is not None:
            im1_score = counterfactual_scorer.calculate_im1(
                x_prime=clue_recon,
                original_class=original_pred,
                counterfactual_class=target_class
            )
            im1_metrics = {'im1_score': im1_score}
    
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
        'original_reconstruction': original_recon.cpu(),
        'counterfactual_image': clue_recon.cpu(),
        **likelihood_metrics,  # Add likelihood metrics if available
        **im1_metrics  # Add IM1 metrics if available
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
        
        if counterfactual_scorer is not None and target_class is not None:
            print(f"\nIM1 score: {results['im1_score']:.5f}")
    
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
    k_samples=100,  # Number of importance samples for VAE likelihood estimation
    autoencoder=None,  # Optional autoencoder for auxiliary approach
    counterfactual_scorer=None  # Optional counterfactual scorer for IM1 metric calculation
):
    """
    Evaluates targeted class counterfactuals on a set of images and calculates comprehensive metrics
    including target class confidence and VAE likelihood scores.
    
    Args:
        images: Tensor of images to analyze [N, 1, 28, 28]
        bayesian_model: The Bayesian model used to extract features and make predictions
        decoder: Decoder model to visualize latent representations (used if autoencoder is None)
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
        autoencoder: Optional full autoencoder model (if provided, uses auxiliary approach)
        
    Returns:
        results: Dictionary containing comprehensive metrics and individual image results
    """
    from clue import counterfactual_optimizer
    import torch
    import numpy as np
    
    # Move models to the specified device
    bayesian_model.eval()
    if autoencoder is not None:
        autoencoder.eval()
    else:
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
    im1_scores = []
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
        
        # If autoencoder is provided, use the auxiliary approach
        if autoencoder is not None:
            # Use the auxiliary function for a single image
            single_result, _ = evaluate_single_auxiliary_counterfactual(
                image=image,
                bayesian_model=bayesian_model,
                autoencoder=autoencoder,
                vae=vae,
                target_class=target_class,
                uncertainty_weight=uncertainty_weight,
                distance_weight=distance_weight,
                lr=lr,
                steps=steps,
                device=device,
                bayesian=bayesian,
                k_samples=k_samples,
                show_plot=False,
                verbose=verbose,
                counterfactual_scorer=counterfactual_scorer
            )
            
            # Extract metrics from the result
            target_confidences.append(single_result['counterfactual_probs'][target_class])
            latent_distances.append(single_result['latent_distance'])
            # In auxiliary case, 'latent' probs are the same as recon probs
            latent_target_confidences.append(single_result['counterfactual_probs'][target_class]) 
            
            # If available, add more metrics
            if 'original_log_likelihood' in single_result:
                likelihood_original.append(single_result['original_log_likelihood'])
                likelihood_reconstruction.append(single_result['reconstruction_log_likelihood'])
                likelihood_counterfactual.append(single_result['counterfactual_log_likelihood'])
                likelihood_differences.append(single_result['log_likelihood_difference'])
                recon_likelihood_differences.append(single_result['recon_counterfactual_log_likelihood_difference'])
                likelihood_ratios.append(single_result['likelihood_ratio'])
                recon_likelihood_ratios.append(single_result['recon_counterfactual_likelihood_ratio'])
                realism_original.append(single_result['original_realism_score'])
                realism_reconstruction.append(single_result['reconstruction_realism_score'])
                realism_counterfactual.append(single_result['counterfactual_realism_score'])
                realism_differences.append(single_result['realism_score_difference'])
                recon_realism_differences.append(single_result['recon_counterfactual_realism_difference'])
            
            # Prepare the individual result dictionary
            result_dict = {
                'image_index': i,
                'target_class': target_class,
                'target_class_confidence': single_result['counterfactual_probs'][target_class], # Confidence from recon
                'latent_target_class_confidence': single_result['counterfactual_probs'][target_class], # Confidence from 'latent' (same as recon here)
                'original_image': single_result['original_image'],
                'counterfactual_image': single_result['counterfactual_image'],
                'original_latent': single_result['original_latent'],
                'counterfactual_latent': single_result['counterfactual_latent'],
                'latent_distance': single_result['latent_distance'],
                'original_class_recon': single_result['original_pred'], # Use 'original_pred' key
                'counterfactual_class_recon': single_result['counterfactual_pred'], # Use 'counterfactual_pred' key
                 # Use recon probs for these in auxiliary case
                'original_recon_class_probs': single_result['original_probs'],
                'counterfactual_recon_class_probs': single_result['counterfactual_probs'],
                # Placeholder/Copy for consistency - derived from recon in aux case
                'original_class_latent': single_result['original_pred'], 
                'counterfactual_class_latent': single_result['counterfactual_pred'],
                'original_latent_class_probs': single_result['original_probs'], 
                'counterfactual_latent_class_probs': single_result['counterfactual_probs'],
                'original_reconstruction': single_result.get('original_reconstruction', single_result['original_image'].clone()) # Use get with default
            }
            
            # Add original reconstruction if available (for consistent format)
            if 'reconstruction_image' in single_result:
                result_dict['original_reconstruction'] = single_result['reconstruction_image']
            else:
                # Create a placeholder reconstruction to maintain consistent format
                result_dict['original_reconstruction'] = single_result['original_image'].clone()
            
            # Add likelihood metrics if available
            if 'original_log_likelihood' in single_result:
                likelihood_metrics = {
                    'original_log_likelihood': single_result['original_log_likelihood'],
                    'reconstruction_log_likelihood': single_result['reconstruction_log_likelihood'],
                    'counterfactual_log_likelihood': single_result['counterfactual_log_likelihood'],
                    'log_likelihood_difference': single_result['log_likelihood_difference'],
                    'recon_counterfactual_log_likelihood_difference': single_result['recon_counterfactual_log_likelihood_difference'],
                    'likelihood_ratio': single_result['likelihood_ratio'],
                    'recon_counterfactual_likelihood_ratio': single_result['recon_counterfactual_likelihood_ratio'],
                    'original_realism_score': single_result['original_realism_score'],
                    'reconstruction_realism_score': single_result['reconstruction_realism_score'],
                    'counterfactual_realism_score': single_result['counterfactual_realism_score'],
                    'realism_score_difference': single_result['realism_score_difference'],
                    'recon_counterfactual_realism_difference': single_result['recon_counterfactual_realism_difference']
                }
                result_dict.update(likelihood_metrics)
            
            # Add IM1 metrics if available
            if 'im1_score' in single_result:
                im1_metrics = {
                    'im1_score': single_result['im1_score']
                }
                result_dict.update(im1_metrics)
                im1_scores.append(single_result['im1_score'])
            individual_results.append(result_dict)
            
        else:
            # Use original approach with decoder
            # Call evaluate_single_clue_counterfactual for each image
            single_result, _ = evaluate_single_counterfactual(
                image=image,
                bayesian_model=bayesian_model,
                decoder=decoder,
                vae=vae,
                true_label=None,  # We don't have true labels in this context
                uncertainty_weight=uncertainty_weight,
                distance_weight=distance_weight,
                lr=lr,
                steps=steps,
                device=device,
                bayesian=bayesian,
                k_samples=k_samples,
                figsize=(15, 10),
                show_plot=False,  # Don't show individual plots
                verbose=verbose,
                target_class=target_class,
                ReconstructionOnly=False,
                counterfactual_scorer=counterfactual_scorer
            )
            
            # Extract metrics from the result
            distance = single_result['latent_distance']
            latent_distances.append(distance)
            
            # --- CORRECTED ACCESS ---
            target_confidence = single_result['counterfactual_probs'][target_class] 
            target_confidences.append(target_confidence)
            
            # --- CORRECTED ACCESS (using newly added latent probs key) ---
            latent_target_confidence = single_result['counterfactual_latent_probs'][target_class]
            latent_target_confidences.append(latent_target_confidence)
            # --- END CORRECTIONS ---
            
            # Extract likelihood metrics
            if 'original_log_likelihood' in single_result:
                original_ll = single_result['original_log_likelihood']
                reconstruction_ll = single_result['reconstruction_log_likelihood']
                counterfactual_ll = single_result['counterfactual_log_likelihood']
                likelihood_diff = single_result['log_likelihood_difference']
                recon_likelihood_diff = single_result['recon_counterfactual_log_likelihood_difference']
                ratio = single_result['likelihood_ratio']
                recon_ratio = single_result['recon_counterfactual_likelihood_ratio']
                original_realism = single_result['original_realism_score']
                reconstruction_realism = single_result['reconstruction_realism_score']
                counterfactual_realism = single_result['counterfactual_realism_score']
                realism_diff = single_result['realism_score_difference']
                recon_realism_diff = single_result['recon_counterfactual_realism_difference']
                
                likelihood_original.append(original_ll)
                likelihood_reconstruction.append(reconstruction_ll)
                likelihood_counterfactual.append(counterfactual_ll)
                likelihood_differences.append(likelihood_diff)
                recon_likelihood_differences.append(recon_likelihood_diff)
                likelihood_ratios.append(ratio)
                recon_likelihood_ratios.append(recon_ratio)
                realism_original.append(original_realism)
                realism_reconstruction.append(reconstruction_realism)
                realism_counterfactual.append(counterfactual_realism)
                realism_differences.append(realism_diff)
                recon_realism_differences.append(recon_realism_diff)
            
            # Add IM1 score if available
            if 'im1_score' in single_result:
                im1_scores.append(single_result['im1_score'])
            
            # Store individual results
            # --- CORRECTED Population ---
            individual_results.append({
                'image_index': i,
                'target_class': target_class,
                'target_class_confidence': target_confidence, # Confidence based on reconstruction
                'latent_target_class_confidence': latent_target_confidence, # Confidence based on latent
                'original_image': single_result['original_image'].cpu(),
                'original_reconstruction': single_result['original_reconstruction'].cpu(), # Use the key added in step 1
                'counterfactual_image': single_result['counterfactual_image'].cpu(),
                'original_latent': single_result['original_latent'].cpu(),
                'counterfactual_latent': single_result['counterfactual_latent'].cpu(),
                'latent_distance': distance,
                'original_class_latent': single_result['original_class_latent'], # Use the key added in step 1
                'counterfactual_class_latent': single_result['counterfactual_class_latent'], # Use the key added in step 1
                'original_latent_class_probs': single_result['original_latent_probs'], # Use the key added in step 1
                'counterfactual_latent_class_probs': single_result['counterfactual_latent_probs'], # Use the key added in step 1
                'original_class_recon': single_result['original_pred'], # Use 'original_pred' key (recon based)
                'counterfactual_class_recon': single_result['counterfactual_pred'], # Use 'counterfactual_pred' key (recon based)
                'original_entropy_latent': single_result['original_entropy_latent'],
                'counterfactual_entropy_latent': single_result['counterfactual_entropy_latent'],
                'original_entropy_recon': single_result['original_entropy_recon'],
                'counterfactual_entropy_recon': single_result['counterfactual_entropy_recon'],
                'original_recon_class_probs': single_result['original_probs'], # Use 'original_probs' key (recon based)
                'counterfactual_recon_class_probs': single_result['counterfactual_probs'], # Use 'counterfactual_probs' key (recon based)
                'original_log_likelihood': single_result['original_log_likelihood'],
                'reconstruction_log_likelihood': single_result['reconstruction_log_likelihood'],
                'counterfactual_log_likelihood': single_result['counterfactual_log_likelihood'],
                'log_likelihood_difference': single_result['log_likelihood_difference'],
                'recon_counterfactual_log_likelihood_difference': single_result['recon_counterfactual_log_likelihood_difference'],
                'likelihood_ratio': single_result['likelihood_ratio'],
                'recon_counterfactual_likelihood_ratio': single_result['recon_counterfactual_likelihood_ratio'],
                'original_realism_score': single_result['original_realism_score'],
                'reconstruction_realism_score': single_result['reconstruction_realism_score'],
                'counterfactual_realism_score': single_result['counterfactual_realism_score'],
                'realism_score_difference': single_result['realism_score_difference'],
                'recon_counterfactual_realism_difference': single_result['recon_counterfactual_realism_difference'],
                'im1_score': single_result['im1_score']
            })
            # --- END CORRECTIONS ---
            
            # ... (Add likelihood/realism/IM1 metrics to individual_results[-1] as before) ...

    # Calculate aggregate metrics
    results = {
        'avg_target_class_confidence': np.mean(target_confidences),
        'avg_latent_target_class_confidence': np.mean(latent_target_confidences),
        'avg_latent_distance': np.mean(latent_distances),
        'individual_results': individual_results
    }
    
    # Add likelihood metrics if they were calculated
    if likelihood_original: 
        likelihood_metrics = {
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
            'median_recon_counterfactual_realism_difference': np.median(recon_realism_differences)
        }
        results.update(likelihood_metrics)

    if counterfactual_scorer is not None:
        results['avg_im1_score'] = np.mean(im1_scores)
    
    # Print results if verbose
    if verbose:
        print(f"\nResults over {len(images)} images:")
        print(f"Average target class confidence: {results['avg_target_class_confidence']:.3f}")
        print(f"Average latent target class confidence: {results['avg_latent_target_class_confidence']:.3f}")
        print(f"Average latent distance: {results['avg_latent_distance']:.3f}")
        
        if 'avg_original_log_likelihood' in results:
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
            top_indices = np.argsort(-result['counterfactual_recon_class_probs'])[:5]
            orig_probs = result['original_recon_class_probs'][top_indices]
            new_probs = result['counterfactual_recon_class_probs'][top_indices]
            
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
            'image_index', 'target_class', 'latent_distance', 
            'original_entropy_latent', 'counterfactual_entropy_latent',
            'latent_entropy_reduction', 'original_entropy_recon', 'counterfactual_entropy_recon',
            'recon_entropy_reduction', 'original_log_likelihood', 'reconstruction_log_likelihood',
            'counterfactual_log_likelihood', 'log_likelihood_difference', 
            'recon_counterfactual_log_likelihood_difference', 'likelihood_ratio',
            'recon_counterfactual_likelihood_ratio', 'original_realism_score',
            'reconstruction_realism_score', 'counterfactual_realism_score',
            'realism_score_difference', 'recon_counterfactual_realism_difference',
            'original_class_latent', 'original_class_recon',
            'counterfactual_class_latent', 'counterfactual_class_recon',
            'target_class_confidence', 'latent_target_class_confidence'
        ]
        
        # Add scalar metrics to result dictionary
        for metric in scalar_metrics:
            # Check if the metric exists and is not None before adding
            if metric in result and result[metric] is not None:
                result_dict[metric] = result[metric]
            # Optionally, add a placeholder if it's missing or None, e.g., result_dict[metric] = np.nan
        
        # Add class probabilities
        if 'original_class_probs' in result and result['original_class_probs'] is not None:
             # Iterate directly over the probability array/list
            for i, prob in enumerate(result['original_class_probs']):
                result_dict[f'original_prob_class_{i}'] = prob
                
        if 'counterfactual_class_probs' in result and result['counterfactual_class_probs'] is not None:
             # Iterate directly over the probability array/list
            for i, prob in enumerate(result['counterfactual_class_probs']):
                result_dict[f'counterfactual_prob_class_{i}'] = prob
        
        # Add latent class probabilities
        if 'original_latent_class_probs' in result and result['original_latent_class_probs'] is not None:
             # Iterate directly over the probability array/list
            for i, prob in enumerate(result['original_latent_class_probs']):
                result_dict[f'original_latent_prob_class_{i}'] = prob
                
        if 'counterfactual_latent_class_probs' in result and result['counterfactual_latent_class_probs'] is not None:
             # Iterate directly over the probability array/list
            for i, prob in enumerate(result['counterfactual_latent_class_probs']):
                result_dict[f'counterfactual_latent_prob_class_{i}'] = prob
        
        individual_data.append(result_dict)
    
    # Create DataFrame for individual results
    individual_df = pd.DataFrame(individual_data)
    
    # Extract aggregate metrics
    aggregate_metrics = {}
    for key, value in results.items():
        if key != 'individual_results' and not isinstance(value, list):
            # Ensure value is calculable (not None or NaN) before adding
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                 aggregate_metrics[key] = [value] # Wrap in list to create DataFrame row
    
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
    
    # Sort examples if requested
    # Use .get with default values for sorting keys that might be missing
    if sort_by == "confidence":
        sorted_indices = sorted(range(len(results['individual_results'])), 
                               key=lambda i: results['individual_results'][i].get('target_class_confidence', 0),
                               reverse=True)
    elif sort_by == "latent_confidence":
        sorted_indices = sorted(range(len(results['individual_results'])), 
                               key=lambda i: results['individual_results'][i].get('latent_target_class_confidence', 0),
                               reverse=True)
    elif sort_by == "distance":
        sorted_indices = sorted(range(len(results['individual_results'])), 
                               key=lambda i: results['individual_results'][i].get('latent_distance', float('inf'))) # Default large distance
    elif sort_by == "likelihood":
        sorted_indices = sorted(range(len(results['individual_results'])), 
                               key=lambda i: abs(results['individual_results'][i].get('log_likelihood_difference', float('inf')))) # Default large difference
    else:
        sorted_indices = list(range(len(results['individual_results'])))
    
    selected_indices = sorted_indices[:n]
    
    # Prepare to collect paths of saved images
    saved_paths = []
    
    # Save aggregate results summary using .get() for potentially missing keys
    if include_metadata:
        summary_path = os.path.join(output_dir, "aggregate_results.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Aggregate Results over {len(results['individual_results'])} images:\n")
            f.write(f"Average target class confidence (reconstruction): {results.get('avg_target_class_confidence', 'N/A'):.4f}\n")
            f.write(f"Average target class confidence (latent): {results.get('avg_latent_target_class_confidence', 'N/A'):.4f}\n")
            f.write(f"Average latent distance: {results.get('avg_latent_distance', 'N/A'):.4f}\n")
            f.write(f"Average log likelihood difference: {results.get('avg_log_likelihood_difference', 'N/A'):.4f}\n")
            f.write(f"Average realism score difference: {results.get('avg_realism_score_difference', 'N/A'):.4f}\n") # Assuming realism might also be missing
    
    # Create and save a figure for each example
    for i, idx in enumerate(selected_indices):
        result = results['individual_results'][idx]
        
        # Create visualization
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"Class Counterfactual Result (Image {result.get('image_index', 'N/A')})", fontsize=16)
        
        # Original image - Use .get() for keys that might be missing
        ax1 = plt.subplot(151)
        ax1.imshow(result['original_image'][0, 0].cpu().numpy(), cmap='gray')
        original_ll_str = f"{result.get('original_log_likelihood', float('nan')):.1f}" # Format NaN safely
        ax1.set_title(f"Original\nClass: {result.get('original_class_recon', 'N/A')}\nLL: {original_ll_str}")
        ax1.axis('off')
        
        # Original reconstruction - Use .get()
        ax2 = plt.subplot(152)
        ax2.imshow(result['original_reconstruction'][0, 0].cpu().numpy(), cmap='gray')
        recon_ll_str = f"{result.get('reconstruction_log_likelihood', float('nan')):.1f}"
        ax2.set_title(f"Original Reconstruction\nLatent class: {result.get('original_class_latent', 'N/A')}\nLL: {recon_ll_str}")
        ax2.axis('off')
        
        # Counterfactual image - Use .get()
        ax3 = plt.subplot(153)
        ax3.imshow(result['counterfactual_image'][0, 0].cpu().numpy(), cmap='gray')
        cf_ll_str = f"{result.get('counterfactual_log_likelihood', float('nan')):.1f}"
        title_text = f"Counterfactual\nTarget: {result.get('target_class', 'N/A')}\n" + \
                     f"Confidence: {result.get('target_class_confidence', float('nan')):.3f}"
        
        # Check and add latent confidence if available
        latent_conf = result.get('latent_target_class_confidence')
        if latent_conf is not None:
             title_text += f"\nLatent confidence: {latent_conf:.3f}"
            
        title_text += f"\nLL: {cf_ll_str}"
        ax3.set_title(title_text)
        ax3.axis('off')
        
        # Difference map - Use .get()
        ax4 = plt.subplot(154)
        diff = result['original_reconstruction'][0, 0].cpu().numpy() - result['counterfactual_image'][0, 0].cpu().numpy()
        ax4.imshow(diff, cmap='coolwarm', vmin=-1, vmax=1)
        ll_diff_str = f"{result.get('log_likelihood_difference', float('nan')):.1f}"
        ax4.set_title(f"Difference\nDistance: {result.get('latent_distance', float('nan')):.3f}\nLL-diff: {ll_diff_str}")
        ax4.axis('off') # CORRECTED METHOD
        
        # Class probability changes - Use .get()
        ax5 = plt.subplot(155)
        
        counterfactual_probs = result.get('counterfactual_recon_class_probs')
        target_class = result.get('target_class')

        if counterfactual_probs is not None and target_class is not None:
            top_indices = np.argsort(-counterfactual_probs)[:4].tolist()
            if target_class not in top_indices:
                 top_indices.append(target_class)
            top_indices = sorted(list(set(top_indices))) # Ensure unique and sorted
            
            x = np.arange(len(top_indices))
            width = 0.35
            
            # Plot original reconstruction probabilities safely
            orig_recon_probs_all = result.get('original_recon_class_probs')
            if orig_recon_probs_all is not None:
                orig_recon_probs = orig_recon_probs_all[top_indices]
                ax5.bar(x - width/2, orig_recon_probs, width, label='Original')
            else:
                ax5.bar(x - width/2, [0]*len(top_indices), width, label='Original (N/A)') # Placeholder
            
            # Plot counterfactual reconstruction probabilities
            cf_recon_probs = counterfactual_probs[top_indices]
            ax5.bar(x + width/2, cf_recon_probs, width, label='Counterfactual')
            
            ax5.set_xticks(x)
            ax5.set_xticklabels(top_indices)
            ax5.set_title("Class probabilities")
            ax5.legend(loc='upper left', fontsize='small')
            ax5.set_xlabel("Class")
            ax5.set_ylabel("Probability")
        else:
            ax5.set_title("Probabilities N/A")
            ax5.axis('off')

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to account for suptitle
        
        # Save figure
        img_path = os.path.join(output_dir, f"counterfactual_{i+1:03d}_img{result.get('image_index', 'NA'):04d}_target{result.get('target_class', 'NA')}.png")
        plt.savefig(img_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        saved_paths.append(img_path)
        
        # Save metadata if requested - using .get() extensively
        if include_metadata:
            metadata_path = os.path.join(output_dir, f"counterfactual_{i+1:03d}_img{result.get('image_index', 'NA'):04d}_target{result.get('target_class', 'NA')}_metadata.txt")
            with open(metadata_path, 'w') as f:
                f.write(f"Image Index: {result.get('image_index', 'N/A')}\n")
                f.write(f"Target Class: {result.get('target_class', 'N/A')}\n\n")
                
                f.write("Classification:\n")
                f.write(f"Original Class (reconstruction): {result.get('original_class_recon', 'N/A')}\n")
                f.write(f"Original Class (latent): {result.get('original_class_latent', 'N/A')}\n")
                f.write(f"Counterfactual Class (reconstruction): {result.get('counterfactual_class_recon', 'N/A')}\n")
                f.write(f"Counterfactual Class (latent): {result.get('counterfactual_class_latent', 'N/A')}\n\n")
                
                f.write("Confidence:\n")
                f.write(f"Target Class Confidence (reconstruction): {result.get('target_class_confidence', float('nan')):.6f}\n")
                f.write(f"Target Class Confidence (latent): {result.get('latent_target_class_confidence', float('nan')):.6f}\n\n")
                
                f.write("Metrics:\n")
                f.write(f"Latent Distance: {result.get('latent_distance', float('nan')):.6f}\n")
                f.write(f"Original Log Likelihood: {result.get('original_log_likelihood', float('nan')):.6f}\n")
                f.write(f"Reconstruction Log Likelihood: {result.get('reconstruction_log_likelihood', float('nan')):.6f}\n")
                f.write(f"Counterfactual Log Likelihood: {result.get('counterfactual_log_likelihood', float('nan')):.6f}\n")
                f.write(f"Log Likelihood Difference (original vs counterfactual): {result.get('log_likelihood_difference', float('nan')):.6f}\n")
                f.write(f"Log Likelihood Difference (reconstruction vs counterfactual): {result.get('recon_counterfactual_log_likelihood_difference', float('nan')):.6f}\n")
                f.write(f"IM1 Score: {result.get('im1_score', 'N/A')}\n\n") # Also check for IM1
                
                # Add probability distributions safely
                f.write("Class Probability Distributions:\n")
                orig_probs = result.get('original_class_probs')
                if orig_probs is not None:
                    f.write("Original (reconstruction):\n")
                    for class_idx, prob in enumerate(orig_probs):
                        if prob > 0.01:
                             f.write(f"  Class {class_idx}: {prob:.6f}\n")
                else:
                     f.write("Original (reconstruction): N/A\n")

                cf_probs = result.get('counterfactual_class_probs')
                if cf_probs is not None:
                    f.write("\nCounterfactual (reconstruction):\n")
                    for class_idx, prob in enumerate(cf_probs):
                        if prob > 0.01:
                            f.write(f"  Class {class_idx}: {prob:.6f}\n")
                else:
                    f.write("\nCounterfactual (reconstruction): N/A\n")
                    
                orig_latent_probs = result.get('original_latent_class_probs')
                if orig_latent_probs is not None:
                    f.write("\nOriginal (latent):\n")
                    for class_idx, prob in enumerate(orig_latent_probs):
                        if prob > 0.01:
                             f.write(f"  Class {class_idx}: {prob:.6f}\n")
                else:
                     f.write("\nOriginal (latent): N/A\n")

                cf_latent_probs = result.get('counterfactual_latent_class_probs')
                if cf_latent_probs is not None:
                    f.write("\nCounterfactual (latent):\n")
                    for class_idx, prob in enumerate(cf_latent_probs):
                        if prob > 0.01:
                            f.write(f"  Class {class_idx}: {prob:.6f}\n")
                else:
                     f.write("\nCounterfactual (latent): N/A\n")

    print(f"Saved {len(saved_paths)} counterfactual visualizations to {output_dir}")
    return saved_paths

def save_aggregate_comparison_to_csv(results_list, output_dir="./results", filename="aggregate_comparison.csv"):
    """
    Saves a comparison of aggregate metrics across different dimensions and model types to a CSV file.
    
    Args:
        results_list: Dictionary with structure results_list[dimension][model_type] = results
        output_dir: Directory to save the CSV file to
        filename: Name of the CSV file
        
    Returns:
        str: Path to the saved CSV file
    """
    import pandas as pd
    import os
    import numpy as np
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename_with_timestamp = f"{os.path.splitext(filename)[0]}_{timestamp}{os.path.splitext(filename)[1]}"
    
    # Prepare data for DataFrame
    comparison_data = []
    
    # Iterate through dimensions and model types
    for dim in sorted(results_list.keys()):
        for model_type, results in results_list[dim].items():
            # Extract the model configuration from the name
            is_bayesian = "Bayesian" in model_type
            is_supervised = "Supervised" in model_type
            
            # Create row dictionary
            row = {
                'Dimension': dim,
                'Model Type': model_type,
                'Bayesian': is_bayesian,
                'Supervised': is_supervised
            }
            
            # Extract key metrics, using get() to handle missing keys safely
            
            # 1. Class confidence metrics
            row['Target Class Confidence (After)'] = results.get('avg_target_class_confidence', np.nan)
            row['Latent Target Class Confidence (After)'] = results.get('avg_latent_target_class_confidence', np.nan)
            
            # 2. Confidence before optimization requires calculating from individual results
            before_confidences = []
            if 'individual_results' in results:
                for indiv_result in results['individual_results']:
                    target_class = indiv_result.get('target_class')
                    if target_class is not None:
                        # Try to get original confidence for the target class
                        orig_probs = indiv_result.get('original_class_probs')
                        if orig_probs is not None and len(orig_probs) > target_class:
                            before_confidences.append(orig_probs[target_class])
            
            if before_confidences:
                row['Target Class Confidence (Before)'] = np.mean(before_confidences)
                if 'Target Class Confidence (After)' in row and not np.isnan(row['Target Class Confidence (After)']):
                    row['Confidence Increase'] = row['Target Class Confidence (After)'] - row['Target Class Confidence (Before)']
            else:
                row['Target Class Confidence (Before)'] = np.nan
                row['Confidence Increase'] = np.nan
            
            # 3. Distance and realism metrics
            row['Avg Latent Distance'] = results.get('avg_latent_distance', np.nan)
            row['Avg Log Likelihood Difference'] = results.get('avg_log_likelihood_difference', np.nan)
            row['Avg Realism Score Difference'] = results.get('avg_realism_score_difference', np.nan)
            
            # 4. IM1 score (counterfactual plausibility)
            row['Avg IM1 Score'] = results.get('avg_im1_score', np.nan)
            
            # 5. Success rate - how often the counterfactual changed the prediction to the target class
            success_count = 0
            total_count = 0
            if 'individual_results' in results:
                for indiv_result in results['individual_results']:
                    if ('target_class' in indiv_result and 
                        'counterfactual_class_recon' in indiv_result and
                        indiv_result['counterfactual_class_recon'] == indiv_result['target_class']):
                        success_count += 1
                    total_count += 1
            
            if total_count > 0:
                row['Success Rate'] = success_count / total_count
            else:
                row['Success Rate'] = np.nan

            # Soft success rate - how often the target class probability was raised above 0.25
            soft_success_count = 0
            if 'individual_results' in results:
                for indiv_result in results['individual_results']:
                    if 'target_class' in indiv_result and 'counterfactual_recon_class_probs' in indiv_result:
                        target_class = indiv_result['target_class']
                        cf_probs = indiv_result['counterfactual_recon_class_probs']
                        if cf_probs is not None and cf_probs[target_class] > 0.25:
                            soft_success_count += 1
                    total_count += 1
            
            if total_count > 0:
                row['Soft Success Rate'] = soft_success_count / total_count
            else:
                row['Soft Success Rate'] = np.nan
                
            comparison_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Rearrange columns in a logical order
    column_order = [
        'Dimension', 'Model Type', 'Bayesian', 'Supervised',
        'Target Class Confidence (Before)', 'Target Class Confidence (After)', 
        'Confidence Increase', 'Latent Target Class Confidence (After)',
        'Success Rate', 'Soft Success Rate', 'Avg Latent Distance', 'Avg Log Likelihood Difference',
        'Avg Realism Score Difference', 'Avg IM1 Score'
    ]
    
    # Make sure we only include columns that exist
    final_columns = [col for col in column_order if col in df.columns]
    df = df[final_columns]
    
    # Save to CSV
    output_path = os.path.join(output_dir, filename_with_timestamp)
    df.to_csv(output_path, index=False)
    
    print(f"Comparison results saved to: {output_path}")
    return output_path

def visualize_latent_space_with_pca(
    dataloader, 
    model, 
    original_latent=None,
    counterfactual_latent=None,
    reconstruction_latent=None,
    n_components=2, 
    random_state=42, 
    batch_size=None, 
    device='cuda',
    figsize=(12, 10),
    point_size=30,
    alpha=0.5,
    highlight_size=150,
    original_color='blue',
    counterfactual_color='red',
    reconstruction_color='green',
    title="PCA with Original, Counterfactual and Reconstruction",
    cmap='tab10',
    show_legend=True,
    show_plot=True,
    show_arrow=True,
    variance_explained=True
):
    """
    Create a PCA visualization of latent space embeddings with highlighted special points.
    Unlike t-SNE, PCA preserves global distances better.
    
    Args:
        dataloader: DataLoader containing the dataset
        model: A model from regene_models.py with encoder functionality
        original_latent: Original point latent representation
        counterfactual_latent: Counterfactual point latent representation
        reconstruction_latent: Reconstruction point latent representation
        n_components: Number of components for PCA (usually 2)
        random_state: Random seed for reproducibility
        batch_size: If not None, limit to this many samples for faster computation
        device: Device to run computations on
        figsize: Figure size for the plot
        point_size: Size of regular data points
        alpha: Alpha transparency for regular points
        highlight_size: Size of highlighted special points
        original_color: Color for the original point
        counterfactual_color: Color for the counterfactual point
        reconstruction_color: Color for the reconstruction point
        title: Title for the plot
        cmap: Colormap for the dataset points
        show_legend: Whether to show the legend
        show_plot: Whether to display the plot
        show_arrow: Whether to show arrow from original to counterfactual
        variance_explained: Whether to show variance explained in axis labels
    
    Returns:
        - pca: Fitted PCA model
        - fig: Matplotlib figure (if show_plot=True)
        - ax: Matplotlib axis (if show_plot=True)
        - special_points_coords: Dictionary with coordinates of special points in PCA space
    """
    import torch
    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    # Set model to evaluation mode
    model.eval()
    
    # Collect embeddings and labels from dataloader
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
            # Limit batch processing if batch_size is specified
            if batch_size is not None and i * dataloader.batch_size >= batch_size:
                break
                
            images = images.to(device)
            
            # Extract embeddings based on model type
            if hasattr(model, 'extract_features'):
                embeddings = model.extract_features(images)
            elif hasattr(model, 'encoder'):
                # For models with encoder directly accessible
                embeddings, _ = model(images)
            else:
                raise ValueError("Model must have either extract_features method or encoder attribute")
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate embeddings and labels
    dataset_embeddings = np.vstack(all_embeddings)
    dataset_labels = np.concatenate(all_labels)
    
    # Prepare special points
    special_points = []
    special_point_labels = []
    
    # Add original latent if provided
    if original_latent is not None:
        if torch.is_tensor(original_latent):
            original_latent = original_latent.cpu().numpy()
        if len(original_latent.shape) == 1:
            original_latent = original_latent.reshape(1, -1)
        special_points.append(original_latent)
        special_point_labels.append("Original")
    
    # Add counterfactual latent if provided
    if counterfactual_latent is not None:
        if torch.is_tensor(counterfactual_latent):
            counterfactual_latent = counterfactual_latent.cpu().numpy()
        if len(counterfactual_latent.shape) == 1:
            counterfactual_latent = counterfactual_latent.reshape(1, -1)
        special_points.append(counterfactual_latent)
        special_point_labels.append("Counterfactual")
    
    # Add reconstruction latent if provided
    if reconstruction_latent is not None:
        if torch.is_tensor(reconstruction_latent):
            reconstruction_latent = reconstruction_latent.cpu().numpy()
        if len(reconstruction_latent.shape) == 1:
            reconstruction_latent = reconstruction_latent.reshape(1, -1)
        special_points.append(reconstruction_latent)
        special_point_labels.append("Reconstruction")
    
    # Combine all embeddings for PCA
    if special_points:
        special_points = np.vstack(special_points)
        all_points = np.vstack([dataset_embeddings, special_points])
    else:
        all_points = dataset_embeddings
    
    # Create and fit PCA model
    pca = PCA(n_components=n_components, random_state=random_state)
    print(f"Running PCA on {all_points.shape[0]} samples with {all_points.shape[1]} dimensions...")
    all_points_2d = pca.fit_transform(all_points)
    
    # Separate dataset points and special points in PCA space
    dataset_points_2d = all_points_2d[:len(dataset_embeddings)]
    special_points_2d = all_points_2d[len(dataset_embeddings):]
    
    # Create plot if requested
    fig = None
    ax = None
    special_points_coords = {}
    
    if show_plot:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot dataset points
        scatter = ax.scatter(
            dataset_points_2d[:, 0], dataset_points_2d[:, 1], 
            c=dataset_labels, cmap=cmap, alpha=alpha, 
            s=point_size, edgecolors='w', linewidths=0.5
        )
        
        # Plot special points
        special_markers = ['o', 'D', 's']  # Different markers for each special point
        special_colors = [original_color, counterfactual_color, reconstruction_color]
        
        for i, (label, coords) in enumerate(zip(special_point_labels, special_points_2d)):
            # Rename "Reconstruction" to "Counterfactual Reconstruction"
            display_label = "Counterfactual Reconstruction" if label == "Reconstruction" else label
            
            ax.scatter(
                coords[0], coords[1],
                color=special_colors[i], 
                s=point_size * 1.5, 
                marker=special_markers[i],
                edgecolors='black', 
                linewidths=1.5,
                label=display_label,
                zorder=5  # Make sure special points are on top
            )
            special_points_coords[label] = coords
            
            # Draw an arrow if both original and counterfactual exist and arrow is requested
            if show_arrow and label == "Counterfactual" and "Original" in special_points_coords:
                orig = special_points_coords["Original"]
                cf = coords
                
                # Calculate the Euclidean distance in PCA space
                pca_distance = np.sqrt(np.sum((orig - cf)**2))
                
                # Calculate the Euclidean distance in original latent space
                latent_distance = np.sqrt(np.sum(
                    (original_latent - counterfactual_latent)**2
                ))
                
                # Show the arrow with annotation
                ax.arrow(
                    orig[0], orig[1], cf[0]-orig[0], cf[1]-orig[1],
                    color='black', width=0.01, head_width=0.2, head_length=0.2,
                    length_includes_head=True, alpha=0.7, zorder=4
                )
                
                # Annotate with distances above the original point with a gap
                ax.annotate(
                    f"Latent dist: {latent_distance:.2f}\nPCA dist: {pca_distance:.2f}",
                    xy=orig, xytext=(orig[0], orig[1] + 1.0),  # Position above original point with gap
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="gray"),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    fontsize=9
                )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Class')
        
        # Add legend for special points
        if show_legend:
            ax.legend(loc='upper right')
        
        # Add labels with variance explained
        if variance_explained:
            explained_var_ratio = pca.explained_variance_ratio_
            ax.set_xlabel(f'PC1 ({explained_var_ratio[0]:.1%} variance explained)')
            ax.set_ylabel(f'PC2 ({explained_var_ratio[1]:.1%} variance explained)')
        else:
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
        
        ax.set_title(title)
        ax.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    if show_plot:
        return pca, fig, ax, special_points_coords
    else:
        return pca, special_points_coords

def visualize_latent_space_with_special_points(
    dataloader, 
    model, 
    original_latent=None,
    counterfactual_latent=None,
    reconstruction_latent=None,
    n_components=2, 
    perplexity=30, 
    n_iter=1000, 
    random_state=42, 
    batch_size=None, 
    device='cuda',
    figsize=(12, 10),
    point_size=30,
    alpha=0.5,
    highlight_size=150,
    original_color='blue',
    counterfactual_color='red',
    reconstruction_color='green',
    title="t-SNE with Original, Counterfactual and Reconstruction",
    cmap='tab10',
    show_legend=True,
    show_plot=True
):
    """
    Create a t-SNE visualization of latent space embeddings with highlighted special points.
    
    Args:
        dataloader: DataLoader containing the dataset
        model: A model from regene_models.py with encoder functionality
        original_latent: Original point latent representation
        counterfactual_latent: Counterfactual point latent representation
        reconstruction_latent: Reconstruction point latent representation
        n_components: Number of components for t-SNE (usually 2)
        perplexity: Perplexity parameter for t-SNE
        n_iter: Number of iterations for t-SNE
        random_state: Random seed for reproducibility
        batch_size: If not None, limit to this many samples for faster computation
        device: Device to run computations on
        figsize: Figure size for the plot
        point_size: Size of regular data points
        alpha: Alpha transparency for regular points
        highlight_size: Size of highlighted special points
        original_color: Color for the original point
        counterfactual_color: Color for the counterfactual point
        reconstruction_color: Color for the reconstruction point
        title: Title for the plot
        cmap: Colormap for the dataset points
        show_legend: Whether to show the legend
        show_plot: Whether to display the plot
    
    Returns:
        - tsne: Fitted t-SNE model
        - fig: Matplotlib figure (if show_plot=True)
        - ax: Matplotlib axis (if show_plot=True)
        - special_points_coords: Dictionary with coordinates of special points in t-SNE space
    """
    import torch
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    # Set model to evaluation mode
    model.eval()
    
    # Collect embeddings and labels from dataloader
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
            # Limit batch processing if batch_size is specified
            if batch_size is not None and i * dataloader.batch_size >= batch_size:
                break
                
            images = images.to(device)
            
            # Extract embeddings based on model type
            if hasattr(model, 'extract_features'):
                embeddings = model.extract_features(images)
            elif hasattr(model, 'encoder'):
                # For models with encoder directly accessible
                embeddings, _ = model(images)
            else:
                raise ValueError("Model must have either extract_features method or encoder attribute")
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate embeddings and labels
    dataset_embeddings = np.vstack(all_embeddings)
    dataset_labels = np.concatenate(all_labels)
    
    # Prepare special points
    special_points = []
    special_point_labels = []
    
    # Add original latent if provided
    if original_latent is not None:
        if torch.is_tensor(original_latent):
            original_latent = original_latent.cpu().numpy()
        if len(original_latent.shape) == 1:
            original_latent = original_latent.reshape(1, -1)
        special_points.append(original_latent)
        special_point_labels.append("Original")
    
    # Add counterfactual latent if provided
    if counterfactual_latent is not None:
        if torch.is_tensor(counterfactual_latent):
            counterfactual_latent = counterfactual_latent.cpu().numpy()
        if len(counterfactual_latent.shape) == 1:
            counterfactual_latent = counterfactual_latent.reshape(1, -1)
        special_points.append(counterfactual_latent)
        special_point_labels.append("Counterfactual")
    
    # Add reconstruction latent if provided
    if reconstruction_latent is not None:
        if torch.is_tensor(reconstruction_latent):
            reconstruction_latent = reconstruction_latent.cpu().numpy()
        if len(reconstruction_latent.shape) == 1:
            reconstruction_latent = reconstruction_latent.reshape(1, -1)
        special_points.append(reconstruction_latent)
        special_point_labels.append("Reconstruction")
    
    # Combine all embeddings for t-SNE
    if special_points:
        special_points = np.vstack(special_points)
        all_points = np.vstack([dataset_embeddings, special_points])
    else:
        all_points = dataset_embeddings
    
    # Create and fit t-SNE model
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    print(f"Running t-SNE on {all_points.shape[0]} samples with {all_points.shape[1]} dimensions...")
    all_points_2d = tsne.fit_transform(all_points)
    
    # Separate dataset points and special points in t-SNE space
    dataset_points_2d = all_points_2d[:len(dataset_embeddings)]
    special_points_2d = all_points_2d[len(dataset_embeddings):]
    
    # Create plot if requested
    fig = None
    ax = None
    special_points_coords = {}
    
    if show_plot:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot dataset points
        scatter = ax.scatter(
            dataset_points_2d[:, 0], dataset_points_2d[:, 1], 
            c=dataset_labels, cmap=cmap, alpha=alpha, 
            s=point_size, edgecolors='w', linewidths=0.5
        )
        
        # Plot special points
        special_markers = ['o', 'D', 's']  # Different markers for each special point
        special_colors = [original_color, counterfactual_color, reconstruction_color]
        
        for i, (label, coords) in enumerate(zip(special_point_labels, special_points_2d)):
            ax.scatter(
                coords[0], coords[1],
                color=special_colors[i], 
                s=point_size * 0.5,  # Much smaller highlight size, just 1.5x the regular point size
                marker=special_markers[i],
                edgecolors='black', 
                linewidths=1.0,
                label=label,
                zorder=5  # Make sure special points are on top
            )
            special_points_coords[label] = coords
            
            # Draw an arrow if both original and counterfactual exist
            if label == "Counterfactual" and "Original" in special_points_coords:
                orig = special_points_coords["Original"]
                cf = coords
                ax.arrow(
                    orig[0], orig[1], cf[0]-orig[0], cf[1]-orig[1],
                    color='black', width=0.01, head_width=0.1, head_length=0.1,
                    length_includes_head=True, alpha=0.7, zorder=4
                )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Class')
        
        # Add legend for special points
        if show_legend:
            ax.legend(loc='upper right')
        
        # Add labels and title
        ax.set_title(title)
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    if show_plot:
        return tsne, fig, ax, special_points_coords
    else:
        return tsne, special_points_coords
    
def evaluate_multiple_counterfactual_methods(
    image,
    bayesian_model,
    deterministic_model,
    decoder,
    autoencoder=None,
    vae=None,
    true_label=None,
    uncertainty_weight=1.0,
    distance_weight=0.005,
    lr=0.1,
    steps=200,
    device='cuda',
    k_samples=100,
    figsize=(20, 15),
    show_plot=True,
    verbose=False,
    target_class=None,
    counterfactual_scorer=None
):
    """
    Evaluates and compares four different counterfactual methods on a single image:
    1. Supervised Deterministic (standard CLUE with deterministic model)
    2. Unsupervised Deterministic (auxiliary CLUE with deterministic model)
    3. Supervised Bayesian (standard CLUE with Bayesian model)
    4. Unsupervised Bayesian (auxiliary CLUE with Bayesian model)
    
    Args:
        image: Single image tensor [1, 1, 28, 28] or [1, 28, 28]
        bayesian_model: Bayesian model (BLL)
        deterministic_model: Deterministic model (backbone)
        decoder: Decoder model for standard CLUE
        autoencoder: Full autoencoder model for auxiliary CLUE (optional)
        vae: Optional VAE for likelihood estimation
        true_label: Optional ground truth label for the image
        uncertainty_weight: Weight for uncertainty term in CLUE optimization
        distance_weight: Weight for distance term in CLUE optimization
        lr: Learning rate for CLUE optimization
        steps: Number of optimization steps
        device: Device to run computation on ('cuda', 'mps', or 'cpu')
        k_samples: Number of importance samples for VAE likelihood estimation
        figsize: Size of the figure
        show_plot: Whether to display the plot immediately
        verbose: Print detailed progress
        target_class: Target class for the counterfactual
        counterfactual_scorer: Optional counterfactual scorer for IM1 metric
        
    Returns:
        tuple: (results_dict, figure) where results_dict contains results from all methods
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Dictionary to store all results
    all_results = {}
    
    # Ensure image is proper shape and on device
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    # Common arguments for all evaluations
    common_args = {
        'image': image,
        'device': device,
        'verbose': verbose,
        'steps': steps,
        'distance_weight': distance_weight,
        'lr': lr,
        'target_class': target_class,
        'vae': vae,
        'counterfactual_scorer': counterfactual_scorer,
        'k_samples': k_samples,
        'show_plot': False,  # Don't show individual plots
        'true_label': true_label
    }
    
    # List of configurations to test
    configurations = [
        {
            'name': 'Supervised Deterministic',
            'model': deterministic_model,
            'bayesian': False,
            'auxiliary': False
        },
        {
            'name': 'Unsupervised Deterministic',
            'model': deterministic_model,
            'bayesian': False,
            'auxiliary': True
        },
        {
            'name': 'Supervised Bayesian',
            'model': bayesian_model,
            'bayesian': True,
            'auxiliary': False
        },
        {
            'name': 'Unsupervised Bayesian',
            'model': bayesian_model,
            'bayesian': True,
            'auxiliary': True
        }
    ]
    
    # Only run configurations that are possible with provided models
    valid_configurations = []
    for config in configurations:
        if config['auxiliary'] and autoencoder is None:
            print(f"Skipping {config['name']} because autoencoder is not provided")
            continue
        valid_configurations.append(config)
    
    if not valid_configurations:
        raise ValueError("No valid configurations to run. Please provide at least one of: decoder, autoencoder")
    
    # Run each configuration
    for config in valid_configurations:
        print(f"Evaluating {config['name']}...")
        
        if config['auxiliary']:
            # Use auxiliary CLUE
            results, _ = evaluate_single_auxiliary_counterfactual(
                **common_args,
                bayesian_model=config['model'],
                autoencoder=autoencoder,
                bayesian=config['bayesian'],
                uncertainty_weight=uncertainty_weight
            )
        else:
            # Use standard CLUE
            results, _ = evaluate_single_counterfactual(
                **common_args,
                bayesian_model=config['model'],
                decoder=decoder,
                bayesian=config['bayesian'],
                uncertainty_weight=uncertainty_weight
            )
        
        # Store results
        all_results[config['name']] = results
    
    # Create comparison figure
    if show_plot:
        # Create a figure with:
        # Row 1: Original image, reconstruction for each method
        # Row 2: Counterfactual for each method
        # Row 3: Difference maps
        # Row 4: Class probability bar charts
        
        num_methods = len(valid_configurations)
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"Counterfactual Comparison (Target Class: {target_class})", fontsize=16)
        
        # Row 1: Original image (col 1) and original reconstructions (cols 2+)
        ax_orig = plt.subplot(4, num_methods+1, 1)
        ax_orig.imshow(image[0, 0].cpu().numpy(), cmap='gray')
        ax_orig.set_title(f"Original Image\nTrue Label: {true_label if true_label is not None else 'N/A'}")
        ax_orig.axis('off')
        
        # Calculate metrics in advance
        method_metrics = {}
        for i, config in enumerate(valid_configurations):
            method_name = config['name']
            results = all_results[method_name]
            
            # Calculate target class confidence before
            target_class_before = 0
            if 'original_class_probs' in results and results['original_class_probs'] is not None:
                if target_class < len(results['original_class_probs']):
                    target_class_before = results['original_class_probs'][target_class]
            
            method_metrics[method_name] = {
                'orig_class': results.get('original_class_recon', 'N/A'),
                'cf_class': results.get('counterfactual_class_recon', 'N/A'),
                'target_confidence': results.get('target_class_confidence', 0),
                'target_before': target_class_before,
                'confidence_increase': results.get('target_class_confidence', 0) - target_class_before,
                'latent_distance': results.get('latent_distance', 0),
                'll_diff': results.get('log_likelihood_difference', 'N/A'),
                'im1': results.get('im1_score', 'N/A')
            }
        
        # Fill in the grid with comparison visualizations
        for i, config in enumerate(valid_configurations):
            method_name = config['name']
            results = all_results[method_name]
            metrics = method_metrics[method_name]
            col = i + 1  # +1 because first column is for original image
            
            # Original reconstruction
            ax_recon = plt.subplot(4, num_methods+1, col+1)
            ax_recon.imshow(results['original_reconstruction'][0, 0].cpu().numpy(), cmap='gray')
            if 'original_log_likelihood' in results:
                ax_recon.set_title(f"{method_name}\nReconstruction\nPred: {metrics['orig_class']}\nLL: {results.get('original_log_likelihood', 'N/A'):.1f}")
            else:
                ax_recon.set_title(f"{method_name}\nReconstruction\nPred: {metrics['orig_class']}")
            ax_recon.axis('off')
            
            # Counterfactual
            ax_cf = plt.subplot(4, num_methods+1, col+num_methods+2)
            ax_cf.imshow(results['counterfactual_image'][0, 0].cpu().numpy(), cmap='gray')
            if 'counterfactual_log_likelihood' in results:
                ax_cf.set_title(f"Counterfactual\nPred: {metrics['cf_class']}\nConf: {metrics['target_confidence']:.3f}\nLL: {results.get('counterfactual_log_likelihood', 'N/A'):.1f}")
            else:
                ax_cf.set_title(f"Counterfactual\nPred: {metrics['cf_class']}\nConf: {metrics['target_confidence']:.3f}")
            ax_cf.axis('off')
            
            # Difference map
            ax_diff = plt.subplot(4, num_methods+1, col+2*num_methods+3)
            diff = results['original_reconstruction'][0, 0].cpu().numpy() - results['counterfactual_image'][0, 0].cpu().numpy()
            ax_diff.imshow(diff, cmap='coolwarm', vmin=-1, vmax=1)
            ax_diff.set_title(f"Difference\nDistance: {metrics['latent_distance']:.3f}")
            ax_diff.axis('off')
            
            # Probability bars
            ax_probs = plt.subplot(4, num_methods+1, col+3*num_methods+4)
            
            orig_probs = results.get('original_class_probs')
            cf_probs = results.get('counterfactual_class_probs')
            
            if orig_probs is not None and cf_probs is not None:
                # Find classes with highest counterfactual probability + target class
                top_indices = list(np.argsort(-cf_probs)[:4])
                if target_class not in top_indices:
                    top_indices.append(target_class)
                top_indices = sorted(top_indices)
                
                # Extract probabilities for these classes
                orig_top_probs = orig_probs[top_indices]
                cf_top_probs = cf_probs[top_indices]
                
                # Plot bars
                x = np.arange(len(top_indices))
                width = 0.35
                ax_probs.bar(x - width/2, orig_top_probs, width, label='Original')
                ax_probs.bar(x + width/2, cf_top_probs, width, label='Counterfactual')
                ax_probs.set_xticks(x)
                ax_probs.set_xticklabels(top_indices)
                ax_probs.set_title("Class probabilities")
                
                # Only add legend to first plot
                if i == 0:
                    ax_probs.legend(loc='upper right', fontsize='small')
            else:
                ax_probs.set_title("Probabilities N/A")
                ax_probs.axis('off')
        
        # Add a table at the bottom comparing metrics
        metrics_table = []
        for config in valid_configurations:
            method_name = config['name']
            metrics = method_metrics[method_name]
            metrics_table.append([
                method_name, 
                f"{metrics['target_before']:.3f}",
                f"{metrics['target_confidence']:.3f}",
                f"{metrics['confidence_increase']:.3f}",
                f"{metrics['latent_distance']:.1f}",
                f"{metrics['ll_diff']}" if isinstance(metrics['ll_diff'], str) else f"{metrics['ll_diff']:.1f}",
                f"{metrics['im1']}" if isinstance(metrics['im1'], str) else f"{metrics['im1']:.3f}"
            ])
        
        col_labels = ['Method', 'Before', 'After', 'Increase', 'Distance', 'LL-Diff', 'IM1']
        table_ax = plt.subplot(4, 1, 4)
        table_ax.axis('off')
        table = table_ax.table(
            cellText=metrics_table,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        table_ax.set_title("Comparison Metrics", pad=20)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.0)
        
        if show_plot:
            plt.show()
    else:
        fig = None
    
    return all_results, fig
    