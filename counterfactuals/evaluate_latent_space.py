import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsRegressor
import os
from tqdm import tqdm
import torchvision

def determine_latent_range(model, dataloader, n_samples=1000, device='cuda'):
    """
    Determine the typical range of latent representations in the model.
    
    Args:
        model: The classifier model with an encoder component
        dataloader: DataLoader providing images
        n_samples: Maximum number of samples to analyze
        device: Device to perform computation on
        
    Returns:
        Dictionary with statistics about the latent space dimensions
    """
    model.eval()
    latent_vectors = []
    count = 0
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Extracting latent vectors"):
            features = model.extract_features(images.to(device))
            latent_vectors.append(features.cpu())
            
            count += len(features)
            if count >= n_samples:
                break
    
    # Concatenate all vectors
    all_latents = torch.cat(latent_vectors)[:n_samples]
    
    # Compute statistics per dimension
    mins = all_latents.min(dim=0)[0]
    maxs = all_latents.max(dim=0)[0]
    means = all_latents.mean(dim=0)
    stds = all_latents.std(dim=0)
    
    # For most scenarios, mean ± 3*std covers most values
    safe_mins = means - 3 * stds
    safe_maxs = means + 3 * stds
    
    return {
        'min': mins,
        'max': maxs,
        'mean': means,
        'std': stds,
        'safe_min': safe_mins,
        'safe_max': safe_maxs,
        'all_latents': all_latents  # Return all vectors for potential further analysis
    }

def create_2d_latent_visualization(model, dataloader, n_samples=1000, method='pca', 
                                  random_state=42, device='cuda', perplexity=30):
    """
    Create a 2D representation of the latent space using dimensionality reduction.
    
    Args:
        model: The classifier model with an encoder component
        dataloader: DataLoader providing images and labels
        n_samples: Maximum number of samples to analyze
        method: Dimensionality reduction technique ('pca' or 'tsne')
        random_state: Random seed for reproducibility
        device: Device to perform computation on
        perplexity: Perplexity parameter for t-SNE
        
    Returns:
        Dictionary with reduced latent representations and mapping functions
    """
    # Extract latent vectors
    latent_vectors = []
    labels = []
    count = 0
    
    model.eval()
    with torch.no_grad():
        for images, batch_labels in tqdm(dataloader, desc="Extracting features for visualization"):
            features = model.extract_features(images.to(device))
            latent_vectors.append(features.cpu())
            labels.append(batch_labels)
            
            count += len(features)
            if count >= n_samples:
                break
    
    latent_vectors = torch.cat(latent_vectors)[:n_samples]
    labels = torch.cat(labels)[:n_samples]
    
    print(f"Applying {method.upper()} to reduce dimensions...")
    # Reduce dimensionality
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
    else:  # Default to t-SNE
        reducer = TSNE(n_components=2, perplexity=perplexity, 
                      learning_rate='auto', init='pca', random_state=random_state)
    
    reduced_latents = reducer.fit_transform(latent_vectors.numpy())
    
    # Create an inverse mapping from 2D to original space for grid evaluation
    if method.lower() == 'pca':
        # For PCA, we can directly use inverse_transform
        inverse_transform = lambda points: torch.tensor(
            reducer.inverse_transform(points), 
            dtype=torch.float32
        ).to(device)
    else:
        # For t-SNE, we need to create an approximation using KNN
        print("Training KNN for inverse mapping...")
        knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
        knn.fit(reduced_latents, latent_vectors.numpy())
        inverse_transform = lambda points: torch.tensor(
            knn.predict(points), 
            dtype=torch.float32
        ).to(device)
    
    return {
        'reduced_latents': reduced_latents,
        'labels': labels.numpy(),
        'reducer': reducer,
        'inverse_transform': inverse_transform,
        'original_latents': latent_vectors.numpy()
    }

def create_grid_in_latent_space(reduced_latents, inverse_transform, grid_size=25, margin=0.2):
    """
    Create a grid in the reduced latent space and map back to the original latent space.
    
    Args:
        reduced_latents: 2D reduced latent representations
        inverse_transform: Function to map from reduced space to original space
        grid_size: Size of the grid (grid_size × grid_size points)
        margin: Margin around the data points in reduced space
        
    Returns:
        Dictionary with grid data in both reduced and original latent space
    """
    # Determine the bounds of the reduced data
    x_min, x_max = reduced_latents[:, 0].min(), reduced_latents[:, 0].max()
    y_min, y_max = reduced_latents[:, 1].min(), reduced_latents[:, 1].max()
    
    # Add margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range
    
    # Create a grid in 2D space
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_2d = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Map back to original latent space
    print("Mapping grid points back to original latent space...")
    grid_latent = inverse_transform(grid_2d)
    
    return {
        'grid_2d': grid_2d.reshape(grid_size, grid_size, 2),
        'grid_latent': grid_latent.reshape(grid_size, grid_size, -1),
        'x': x,
        'y': y
    }

def compute_gradients_for_model(model, grid_latent, target_class, batch_size=32, bayesian=False, num_samples=200):
    """
    Compute gradients at grid points toward a target class for a specific model.
    
    Args:
        model: The classifier model
        grid_latent: Grid of points in latent space
        target_class: Target class index for gradient computation
        batch_size: Batch size for gradient computation
        bayesian: Whether this is a Bayesian model
        num_samples: Number of samples for Bayesian model
        
    Returns:
        Dictionary with gradients and magnitudes
    """
    grid_size = grid_latent.shape[0]
    
    # Flatten grid for batch processing
    flat_grid = grid_latent.reshape(-1, grid_latent.shape[-1])
    
    # Arrays to store results
    all_gradients = []
    all_magnitudes = []
    all_probs = []
    
    model.eval()
    
    for i in range(0, len(flat_grid), batch_size):
        batch = flat_grid[i:i+batch_size].clone().requires_grad_(True)
        
        # Forward pass and compute loss based on model type
        if bayesian:
            # For Bayesian model
            if hasattr(model, 'sample_predict_z'):
                probs = model.sample_predict_z(batch, num_samples)
                mean_probs = probs.mean(dim=0)
            else:
                # Fallback if the specific method isn't available
                logits = model.classifier(batch)
                mean_probs = torch.softmax(logits, dim=1)
                
            target_prob = mean_probs[:, target_class]
        else:
            # For deterministic model
            logits = model.classifier(batch)
            probs = torch.softmax(logits, dim=1)
            target_prob = probs[:, target_class]
            mean_probs = probs
        
        # Store probabilities for later analysis
        all_probs.append(mean_probs.detach().cpu().numpy())
        
        # Compute loss (negative log probability to maximize the target class)
        loss = -torch.sum(torch.log(target_prob + 1e-10))
        
        # Compute gradients
        model.zero_grad()
        loss.backward()
        
        # Store gradient and magnitude
        grad = batch.grad.detach().cpu().numpy()
        all_gradients.append(grad)
        all_magnitudes.append(np.linalg.norm(grad, axis=1))
    
    # Concatenate results
    all_gradients = np.vstack(all_gradients)
    all_magnitudes = np.concatenate(all_magnitudes)
    all_probs = np.vstack(all_probs)
    
    # Reshape for plotting
    gradients_reshaped = all_gradients.reshape(grid_size, grid_size, -1)
    magnitudes_reshaped = all_magnitudes.reshape(grid_size, grid_size)
    probs_reshaped = all_probs.reshape(grid_size, grid_size, -1)
    
    return {
        'gradients': all_gradients,
        'magnitudes': all_magnitudes,
        'magnitudes_reshaped': magnitudes_reshaped,
        'gradients_reshaped': gradients_reshaped,
        'probs': all_probs,
        'probs_reshaped': probs_reshaped,
        'target_probs': all_probs[:, target_class],
        'target_probs_reshaped': probs_reshaped[:, :, target_class]
    }

def project_gradients_to_2d(gradients, reducer=None):
    """
    Project gradients from original latent space to 2D visualization space.
    
    Args:
        gradients: Gradients in original latent space
        reducer: Dimensionality reduction object (PCA or t-SNE)
        
    Returns:
        Gradients projected to 2D space
    """
    # For PCA, we can directly use the components to project
    if hasattr(reducer, 'components_'):
        gradients_2d = np.dot(gradients, reducer.components_.T)
    else:
        # Otherwise use a simple approach - project to first 2 dimensions
        # This is a simplification and may not be accurate for t-SNE
        gradients_2d = gradients[:, :2]
    
    return gradients_2d

def visualize_class_gradients(model, grid_data, target_class, reduced_latents=None, 
                            labels=None, bayesian=False, num_samples=30, figsize=(12, 10),
                            title_suffix="", save_path=None):
    """
    Visualize gradients toward a target class in latent space.
    
    Args:
        model: The classifier model
        grid_data: Grid data from create_grid_in_latent_space
        target_class: Target class index
        reduced_latents: 2D reduced latent representations
        labels: Class labels for data points
        bayesian: Whether this is a Bayesian model
        num_samples: Number of samples for Bayesian model
        figsize: Figure size
        title_suffix: Additional text for plot title
        save_path: Path to save the figure
        
    Returns:
        Figure and gradient data
    """
    grid_latent = grid_data['grid_latent']
    grid_size = grid_latent.shape[0]
    
    # Compute gradients
    print(f"Computing gradients for {'Bayesian' if bayesian else 'Deterministic'} model...")
    grad_data = compute_gradients_for_model(
        model, grid_latent, target_class, bayesian=bayesian, num_samples=num_samples
    )
    
    # Project gradients to 2D if reducer is available
    if 'reducer' in grid_data:
        gradients_2d = project_gradients_to_2d(grad_data['gradients'], grid_data['reducer'])
        grad_x = gradients_2d[:, 0].reshape(grid_size, grid_size)
        grad_y = gradients_2d[:, 1].reshape(grid_size, grid_size)
    else:
        # Fallback - use first two dimensions
        grad_x = grad_data['gradients'][:, 0].reshape(grid_size, grid_size)
        grad_y = grad_data['gradients'][:, 1].reshape(grid_size, grid_size)
    
    magnitudes = grad_data['magnitudes_reshaped']
    target_probs = grad_data['target_probs_reshaped']
    
    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Gradient magnitude + vector field
    ax = axes[0]
    
    # Plot data points if provided
    if reduced_latents is not None and labels is not None:
        scatter = ax.scatter(reduced_latents[:, 0], reduced_latents[:, 1], 
                            c=labels, cmap='tab10', alpha=0.5, s=30)
    
    # Plot the gradient field with log scale
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    log_magnitudes = np.log10(magnitudes + epsilon)
    contour = ax.contourf(grid_data['x'], grid_data['y'], log_magnitudes, 
                         alpha=0.4, cmap='Blues', levels=20)
    cbar = plt.colorbar(contour, ax=ax, label='Log10 Gradient Magnitude')
    
    # Show gradient directions
    # Scale vectors for better visualization
    scale_factor = 25 / np.percentile(magnitudes[magnitudes > 0], 90)
    quiver = ax.quiver(
        grid_data['grid_2d'][::2, ::2, 0], grid_data['grid_2d'][::2, ::2, 1],
        grad_x[::2, ::2], grad_y[::2, ::2],
        np.log10(magnitudes[::2, ::2] + epsilon), cmap='Blues', scale=30*scale_factor, width=0.002
    )
    
    ax.set_title(f"Gradient Field toward Class {target_class}\n{title_suffix}")
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    
    # Plot 2: Target class probability
    ax = axes[1]
    
    # Plot data points if provided
    if reduced_latents is not None and labels is not None:
        scatter = ax.scatter(reduced_latents[:, 0], reduced_latents[:, 1], 
                            c=labels, cmap='tab10', alpha=0.5, s=30)
    
    # Plot the target class probability with log scale
    # Transform probabilities to log scale, handling values that may be 0
    log_probs = np.log10(target_probs + epsilon)
    contour = ax.contourf(grid_data['x'], grid_data['y'], log_probs, 
                         alpha=0.8, cmap='plasma', levels=20)
    cbar = plt.colorbar(contour, ax=ax, label=f'Log10 P(class={target_class})')
    
    ax.set_title(f"Class {target_class} Probability (Log Scale)\n{title_suffix}")
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, grad_data

def compare_bayesian_vs_deterministic_gradients(bayesian_model, deterministic_model, 
                                              grid_data, target_class, num_samples=30,
                                              figsize=(16, 14), save_path=None):
    """
    Compare gradient magnitudes between Bayesian and deterministic models.
    
    Args:
        bayesian_model: The Bayesian classifier model
        deterministic_model: The deterministic classifier model
        grid_data: Grid data from create_grid_in_latent_space
        target_class: Target class index
        num_samples: Number of samples for Bayesian model
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Figure and comparison data
    """
    grid_latent = grid_data['grid_latent']
    
    # Compute gradients for both models
    print("Computing gradients for Bayesian model...")
    bayes_results = compute_gradients_for_model(
        bayesian_model, grid_latent, target_class, bayesian=True, num_samples=num_samples
    )
    
    print("Computing gradients for Deterministic model...")
    det_results = compute_gradients_for_model(
        deterministic_model, grid_latent, target_class, bayesian=False
    )
    
    # Get reshaped data
    bayes_magnitudes = bayes_results['magnitudes_reshaped']
    det_magnitudes = det_results['magnitudes_reshaped']
    bayes_probs = bayes_results['target_probs_reshaped']
    det_probs = det_results['target_probs_reshaped']
    
    # Calculate ratio of magnitudes (avoiding division by zero)
    epsilon = 1e-10
    magnitude_ratio = bayes_magnitudes / (det_magnitudes + epsilon)
    probability_diff = bayes_probs - det_probs
    
    # Find plateau regions (where deterministic gradient is small but Bayesian isn't)
    det_threshold = np.percentile(det_magnitudes, 10)  # Bottom 10% as threshold
    bayes_threshold = np.percentile(bayes_magnitudes, 50)  # Median as threshold 
    plateau_mask = (det_magnitudes < det_threshold) & (bayes_magnitudes > bayes_threshold)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Bayesian gradient magnitudes
    contour1 = axes[0, 0].contourf(grid_data['x'], grid_data['y'], bayes_magnitudes, 
                                  cmap='viridis', levels=20)
    axes[0, 0].set_title('Bayesian Model Gradient Magnitudes')
    plt.colorbar(contour1, ax=axes[0, 0])
    
    # Deterministic gradient magnitudes
    contour2 = axes[0, 1].contourf(grid_data['x'], grid_data['y'], det_magnitudes, 
                                  cmap='viridis', levels=20)
    axes[0, 1].set_title('Deterministic Model Gradient Magnitudes')
    plt.colorbar(contour2, ax=axes[0, 1])
    
    # Ratio of magnitudes - capped for better visualization
    ratio_capped = np.clip(magnitude_ratio, 0, 10)  # Cap at 10x difference
    contour3 = axes[1, 0].contourf(grid_data['x'], grid_data['y'], ratio_capped, 
                                  cmap='coolwarm', levels=20)
    axes[1, 0].set_title('Ratio of Bayesian to Deterministic Gradient Magnitudes')
    plt.colorbar(contour3, ax=axes[1, 0])
    
    # Identified plateau regions
    contour4 = axes[1, 1].contourf(grid_data['x'], grid_data['y'], plateau_mask.astype(float), 
                                 cmap='Reds', levels=2)
    axes[1, 1].set_title('Identified Gradient Plateaus\n(Det. gradient small, Bayes. gradient large)')
    plt.colorbar(contour4, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison figure to {save_path}")
    
    # Calculate statistics
    plateau_percentage = plateau_mask.mean() * 100
    print(f"Percentage of latent space with gradient plateaus: {plateau_percentage:.2f}%")
    
    return fig, {
        'bayes_magnitudes': bayes_magnitudes,
        'det_magnitudes': det_magnitudes,
        'magnitude_ratio': magnitude_ratio,
        'plateau_mask': plateau_mask,
        'plateau_percentage': plateau_percentage,
        'bayes_results': bayes_results,
        'det_results': det_results,
        'probability_diff': probability_diff
    }

def analyze_multiple_target_classes(bayesian_model, deterministic_model, grid_data, 
                                   target_classes, num_samples=30, output_dir=None):
    """
    Analyze gradients for multiple target classes.
    
    Args:
        bayesian_model: The Bayesian classifier model
        deterministic_model: The deterministic classifier model
        grid_data: Grid data from create_grid_in_latent_space
        target_classes: List of target class indices
        num_samples: Number of samples for Bayesian model
        output_dir: Directory to save output figures
        
    Returns:
        Summary statistics for all classes
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    summary = []
    for target_class in target_classes:
        print(f"\nAnalyzing target class {target_class}")
        
        # Create paths for saving figures
        save_paths = None
        if output_dir:
            bayes_path = os.path.join(output_dir, f"bayes_gradients_class_{target_class}.png")
            det_path = os.path.join(output_dir, f"det_gradients_class_{target_class}.png")
            comparison_path = os.path.join(output_dir, f"comparison_class_{target_class}.png")
            save_paths = (bayes_path, det_path, comparison_path)
        
        # Visualize individual model gradients
        _, bayes_data = visualize_class_gradients(
            bayesian_model, grid_data, target_class, 
            bayesian=True, num_samples=num_samples,
            title_suffix="Bayesian Model",
            save_path=save_paths[0] if save_paths else None
        )
        
        _, det_data = visualize_class_gradients(
            deterministic_model, grid_data, target_class,
            bayesian=False,
            title_suffix="Deterministic Model",
            save_path=save_paths[1] if save_paths else None
        )
        
        # Compare models
        _, comparison_data = compare_bayesian_vs_deterministic_gradients(
            bayesian_model, deterministic_model, grid_data, target_class,
            num_samples=num_samples,
            save_path=save_paths[2] if save_paths else None
        )
        
        # Collect summary statistics
        class_summary = {
            'target_class': target_class,
            'plateau_percentage': comparison_data['plateau_percentage'],
            'bayes_magnitude_mean': np.mean(comparison_data['bayes_magnitudes']),
            'det_magnitude_mean': np.mean(comparison_data['det_magnitudes']),
            'magnitude_ratio_mean': np.mean(comparison_data['magnitude_ratio']),
            'bayes_prob_mean': np.mean(bayes_data['target_probs']),
            'det_prob_mean': np.mean(det_data['target_probs'])
        }
        summary.append(class_summary)
        
        print(f"Class {target_class} summary:")
        print(f"  Plateau percentage: {class_summary['plateau_percentage']:.2f}%")
        print(f"  Mean Bayesian gradient magnitude: {class_summary['bayes_magnitude_mean']:.6f}")
        print(f"  Mean Deterministic gradient magnitude: {class_summary['det_magnitude_mean']:.6f}")
        print(f"  Mean magnitude ratio: {class_summary['magnitude_ratio_mean']:.2f}")
        
    return summary

def compute_class_centroids(model, dataloader, classes):
    """
    Compute centroids for specified classes in latent space.
    
    Args:
        model: The model to extract features
        dataloader: DataLoader containing the dataset
        classes: List of class indices to compute centroids for
        
    Returns:
        Dictionary mapping class indices to their centroids
    """
    centroids = {}
    for cls in classes:
        features = []
        for images, labels in dataloader:
            idx = (labels == cls)
            if idx.any():
                batch_features = model.extract_features(images[idx].to(model.device))
                features.append(batch_features)
            if len(features) > 10:  # Limit sample size
                break
        centroids[cls] = torch.cat(features).mean(0)
    
    return centroids

def analyze_class_to_class_path(bayes_model, det_model, centroids, start_class=0, target_class=1, n_points=100, 
                               save_path=None, num_samples=200, ensemble_model=None):
    """
    Analyze gradients along a straight line between class centroids.
    
    Args:
        bayes_model: The Bayesian model
        det_model: The deterministic model
        centroids: Dictionary mapping class indices to their centroids
        start_class: Starting class index
        target_class: Target class index
        n_points: Number of interpolation points
        save_path: Path to save the figure
        num_samples: Number of Monte Carlo samples for Bayesian model
        ensemble_model: Optional ensemble model for comparison
        
    Returns:
        Dictionary with analysis results
    """
    # Create interpolation points
    z_start = centroids[start_class]
    z_end = centroids[target_class]
    alphas = torch.linspace(0, 1, n_points)
    points = [(1-a)*z_start + a*z_end for a in alphas]
    
    # Store all results for comprehensive analysis
    bayes_grads = []
    det_grads = []
    ensemble_grads = []
    bayes_probs = []
    det_probs = []
    ensemble_probs = []
    
    print(f"Analyzing path from class {start_class} to class {target_class}...")
    for i, z in enumerate(points):
        if i % 10 == 0:
            print(f"Processing point {i+1}/{n_points}")
            
        # Bayesian model gradients and probabilities
        z_bayes = z.clone().to(bayes_model.device).unsqueeze(0).requires_grad_(True)
        with torch.no_grad():
            probs_bayes = bayes_model.sample_predict_z(z_bayes, Nsamples=num_samples)
            mean_probs_bayes = probs_bayes.mean(dim=0)
            bayes_probs.append(mean_probs_bayes[0, target_class].item())
        
        # Compute gradient with fresh forward pass
        z_bayes = z.clone().to(bayes_model.device).unsqueeze(0).requires_grad_(True)
        probs_bayes = bayes_model.sample_predict_z(z_bayes, Nsamples=num_samples)
        mean_probs_bayes = probs_bayes.mean(dim=0)
        target_prob_bayes = mean_probs_bayes[0, target_class]
        loss_bayes = -torch.log(target_prob_bayes + 1e-10)
        bayes_model.zero_grad()
        loss_bayes.backward()
        bayes_grad = z_bayes.grad.detach().cpu()
        bayes_grads.append(bayes_grad.norm().item())
        
        # Ensemble model gradients and probabilities (if provided)
        if ensemble_model is not None:
            z_ensemble = z.clone().to(ensemble_model.device).unsqueeze(0).requires_grad_(True)
            with torch.no_grad():
                probs_ensemble = ensemble_model.sample_predict_z(z_ensemble, Nsamples=num_samples)
                mean_probs_ensemble = probs_ensemble.mean(dim=0)
                ensemble_probs.append(mean_probs_ensemble[0, target_class].item())
            
            # Compute gradient with fresh forward pass
            z_ensemble = z.clone().to(ensemble_model.device).unsqueeze(0).requires_grad_(True)
            probs_ensemble = ensemble_model.sample_predict_z(z_ensemble, Nsamples=num_samples)
            mean_probs_ensemble = probs_ensemble.mean(dim=0)
            target_prob_ensemble = mean_probs_ensemble[0, target_class]
            loss_ensemble = -torch.log(target_prob_ensemble + 1e-10)
            ensemble_model.zero_grad()
            loss_ensemble.backward()
            ensemble_grad = z_ensemble.grad.detach().cpu()
            ensemble_grads.append(ensemble_grad.norm().item())
        
        # Deterministic model gradients and probabilities
        z_det = z.clone().to(det_model.device).unsqueeze(0)
        with torch.no_grad():
            logits_det = det_model.classifier(z_det)
            probs_det = torch.nn.functional.softmax(logits_det, dim=1)
            det_probs.append(probs_det[0, target_class].item())
        
        # Compute gradient with fresh forward pass
        z_det = z.clone().to(det_model.device).unsqueeze(0).requires_grad_(True)
        logits_det = det_model.classifier(z_det)
        probs_det = torch.nn.functional.softmax(logits_det, dim=1)
        target_prob_det = probs_det[0, target_class]
        loss_det = -torch.log(target_prob_det + 1e-10)
        det_model.zero_grad()
        loss_det.backward()
        det_grad = z_det.grad.detach().cpu()
        det_grads.append(det_grad.norm().item())
    
    # Convert to numpy arrays
    alphas_np = alphas.numpy()
    bayes_grads_np = np.array(bayes_grads)
    det_grads_np = np.array(det_grads)
    bayes_probs_np = np.array(bayes_probs)
    det_probs_np = np.array(det_probs)
    
    if ensemble_model is not None:
        ensemble_grads_np = np.array(ensemble_grads)
        ensemble_probs_np = np.array(ensemble_probs)
    
    # Create a combined visualization to better analyze the relationship
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Plot the gradient magnitudes
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    ax1.plot(alphas_np, bayes_grads_np, 'b-', linewidth=2, label='Bayesian')
    ax1.plot(alphas_np, det_grads_np, 'r--', linewidth=2, label='Deterministic')
    if ensemble_model is not None:
        ax1.plot(alphas_np, ensemble_grads_np, 'g-', linewidth=2, label='Ensemble')
    ax1.set_title(f'Loss Function Gradient Magnitude: Class {start_class} → Class {target_class}')
    ax1.set_xlabel('Interpolation Parameter (α)')
    ax1.set_ylabel('Gradient Norm')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Plot the ratios using log scale
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    epsilon = 1e-10  # Better threshold for division
    ratios = np.array([b/(d + epsilon) for b, d in zip(bayes_grads, det_grads)])
    ax2.semilogy(alphas_np, ratios, 'g-', linewidth=2, label='Bayesian/Deterministic')
    if ensemble_model is not None:
        ratios_ensemble = np.array([b/(e + epsilon) for b, e in zip(bayes_grads, ensemble_grads)])
        ax2.semilogy(alphas_np, ratios_ensemble, 'm-', linewidth=2, label='Bayesian/Ensemble')
        ratios_det_ensemble = np.array([d/(e + epsilon) for d, e in zip(det_grads, ensemble_grads)])
        ax2.semilogy(alphas_np, ratios_det_ensemble, 'c-', linewidth=2, label='Deterministic/Ensemble')
    ax2.set_title('Ratio of Gradient Norms (Log Scale)')
    ax2.set_xlabel('Interpolation Parameter (α)')
    ax2.set_ylabel('Ratio')
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, which='both')  # Grid lines for both major and minor ticks
    ax2.legend()
    
    # 3. Plot the probabilities
    ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
    ax3.plot(alphas_np, bayes_probs_np, 'b-', linewidth=2, label='Bayesian')
    ax3.plot(alphas_np, det_probs_np, 'r--', linewidth=2, label='Deterministic')
    if ensemble_model is not None:
        ax3.plot(alphas_np, ensemble_probs_np, 'g-', linewidth=2, label='Ensemble')
    ax3.set_title(f'Target Class Probability: Class {start_class} → Class {target_class}')
    ax3.set_xlabel('Interpolation Parameter (α)')
    ax3.set_ylabel(f'P(class={target_class})')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Plot the rate of change of probabilities (numerical derivative)
    ax4 = plt.subplot2grid((2, 3), (1, 2))
    # Compute numerical derivatives (central difference)
    bayes_prob_deriv = np.gradient(bayes_probs_np, alphas_np)
    det_prob_deriv = np.gradient(det_probs_np, alphas_np)
    ax4.plot(alphas_np, np.abs(bayes_prob_deriv), 'b-', linewidth=2, label='Bayesian')
    ax4.plot(alphas_np, np.abs(det_prob_deriv), 'r--', linewidth=2, label='Deterministic')
    if ensemble_model is not None:
        ensemble_prob_deriv = np.gradient(ensemble_probs_np, alphas_np)
        ax4.plot(alphas_np, np.abs(ensemble_prob_deriv), 'g-', linewidth=2, label='Ensemble')
    ax4.set_title('Rate of Change of Probability')
    ax4.set_xlabel('Interpolation Parameter (α)')
    ax4.set_ylabel('|dP/dα|')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    
    # Calculate correlation between gradient magnitudes and probability derivatives
    bayes_corr = np.corrcoef(bayes_grads_np, np.abs(bayes_prob_deriv))[0, 1]
    det_corr = np.corrcoef(det_grads_np, np.abs(det_prob_deriv))[0, 1]
    
    result = {
        'alphas': alphas_np,
        'bayes_grads': bayes_grads,
        'det_grads': det_grads,
        'bayes_probs': bayes_probs,
        'det_probs': det_probs,
        'ratios': ratios.tolist(),
        'bayes_prob_deriv': bayes_prob_deriv.tolist(),
        'det_prob_deriv': det_prob_deriv.tolist(),
        'bayes_corr': bayes_corr,
        'det_corr': det_corr
    }
    
    print(f"Correlation between gradient magnitude and probability rate of change:")
    print(f"  Bayesian: {bayes_corr:.4f}")
    print(f"  Deterministic: {det_corr:.4f}")
    
    if ensemble_model is not None:
        ensemble_corr = np.corrcoef(ensemble_grads_np, np.abs(ensemble_prob_deriv))[0, 1]
        print(f"  Ensemble: {ensemble_corr:.4f}")
        
        result.update({
            'ensemble_grads': ensemble_grads,
            'ensemble_probs': ensemble_probs,
            'ensemble_prob_deriv': ensemble_prob_deriv.tolist(),
            'ensemble_corr': ensemble_corr
        })
    
    return result

def analyze_class_to_class_path_probs(bayes_model, det_model, centroids, start_class=0, target_class=1, n_points=50, 
                                     num_samples=200, save_path=None, ensemble_model=None):
    """
    Analyze probabilities along a straight line between class centroids.
    
    Args:
        bayes_model: The Bayesian model
        det_model: The deterministic model
        centroids: Dictionary mapping class indices to their centroids
        start_class: Starting class index
        target_class: Target class index
        n_points: Number of interpolation points
        num_samples: Number of Monte Carlo samples for Bayesian model
        save_path: Path to save the figure
        ensemble_model: Optional ensemble model for comparison
        
    Returns:
        Dictionary with analysis results
    """
    # Create interpolation points
    z_start = centroids[start_class]
    z_end = centroids[target_class]
    alphas = torch.linspace(0, 1, n_points)
    points = [(1-a)*z_start + a*z_end for a in alphas]
    
    # Compute probabilities at each point for all models
    bayes_probs = []
    det_probs = []
    ensemble_probs = []
    
    print(f"Analyzing probabilities from class {start_class} to class {target_class}...")
    for i, z in enumerate(points):
        if i % 10 == 0:
            print(f"Processing point {i+1}/{n_points}")
            
        # Bayesian model probabilities
        z_bayes = z.clone().to(bayes_model.device).unsqueeze(0)
        probs_bayes = bayes_model.sample_predict_z(z_bayes, Nsamples=num_samples)
        mean_probs_bayes = probs_bayes.mean(dim=0)
        target_prob_bayes = mean_probs_bayes[0, target_class].item()
        bayes_probs.append(target_prob_bayes)
        
        # Deterministic model probabilities
        z_det = z.clone().to(det_model.device).unsqueeze(0)
        logits_det = det_model.classifier(z_det)
        probs_det = torch.nn.functional.softmax(logits_det, dim=1)
        target_prob_det = probs_det[0, target_class].item()
        det_probs.append(target_prob_det)
        
        # Ensemble model probabilities (if provided)
        if ensemble_model is not None:
            z_ensemble = z.clone().to(ensemble_model.device).unsqueeze(0)
            probs_ensemble = ensemble_model.sample_predict_z(z_ensemble, Nsamples=num_samples)
            mean_probs_ensemble = probs_ensemble.mean(dim=0)
            target_prob_ensemble = mean_probs_ensemble[0, target_class].item()
            ensemble_probs.append(target_prob_ensemble)
    
    # Convert to numpy arrays
    alphas_np = alphas.numpy()
    bayes_probs_np = np.array(bayes_probs)
    det_probs_np = np.array(det_probs)
    
    if ensemble_model is not None:
        ensemble_probs_np = np.array(ensemble_probs)
    
    # Visualize results
    fig = plt.figure(figsize=(15, 8))
    
    # 1. Plot the probabilities
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.plot(alphas_np, bayes_probs_np, 'b-', linewidth=2, label='Bayesian')
    ax1.plot(alphas_np, det_probs_np, 'r--', linewidth=2, label='Deterministic')
    if ensemble_model is not None:
        ax1.plot(alphas_np, ensemble_probs_np, 'g-', linewidth=2, label='Ensemble')
    ax1.set_title(f'Target Class Probability: Class {start_class} → Class {target_class}')
    ax1.set_xlabel('Interpolation Parameter (α)')
    ax1.set_ylabel(f'P(class={target_class})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Plot the probability ratios
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    epsilon = 1e-10  # Better threshold for division
    ratios_bayes_det = np.array([b/(d + epsilon) for b, d in zip(bayes_probs, det_probs)])
    ax2.plot(alphas_np, ratios_bayes_det, 'g-', linewidth=2, label='Bayesian/Deterministic')
    if ensemble_model is not None:
        ratios_bayes_ensemble = np.array([b/(e + epsilon) for b, e in zip(bayes_probs, ensemble_probs)])
        ax2.plot(alphas_np, ratios_bayes_ensemble, 'm-', linewidth=2, label='Bayesian/Ensemble')
        ratios_det_ensemble = np.array([d/(e + epsilon) for d, e in zip(det_probs, ensemble_probs)])
        ax2.plot(alphas_np, ratios_det_ensemble, 'c-', linewidth=2, label='Deterministic/Ensemble')
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax2.set_title('Ratio of Probabilities')
    ax2.set_xlabel('Interpolation Parameter (α)')
    ax2.set_ylabel('Ratio')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Plot the rate of change of probabilities (numerical derivative)
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    # Compute numerical derivatives (central difference)
    bayes_prob_deriv = np.gradient(bayes_probs_np, alphas_np)
    det_prob_deriv = np.gradient(det_probs_np, alphas_np)
    ax3.plot(alphas_np, np.abs(bayes_prob_deriv), 'b-', linewidth=2, label='Bayesian')
    ax3.plot(alphas_np, np.abs(det_prob_deriv), 'r--', linewidth=2, label='Deterministic')
    if ensemble_model is not None:
        ensemble_prob_deriv = np.gradient(ensemble_probs_np, alphas_np)
        ax3.plot(alphas_np, np.abs(ensemble_prob_deriv), 'g-', linewidth=2, label='Ensemble')
    ax3.set_title('Rate of Change of Probability')
    ax3.set_xlabel('Interpolation Parameter (α)')
    ax3.set_ylabel('|dP/dα|')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    
    # Prepare result dictionary
    result = {
        'alphas': alphas_np,
        'bayes_probs': bayes_probs,
        'det_probs': det_probs,
        'ratios_bayes_det': ratios_bayes_det.tolist(),
        'bayes_prob_deriv': bayes_prob_deriv.tolist(),
        'det_prob_deriv': det_prob_deriv.tolist()
    }
    
    if ensemble_model is not None:
        result.update({
            'ensemble_probs': ensemble_probs,
            'ratios_bayes_ensemble': ratios_bayes_ensemble.tolist(),
            'ratios_det_ensemble': ratios_det_ensemble.tolist(),
            'ensemble_prob_deriv': ensemble_prob_deriv.tolist()
        })
    
    return result

def analyze_probability_gradients(bayes_model, det_model, centroids, start_class, target_class, 
                                 n_points=100, num_samples=200, 
                                 figsize=(14, 8), save_path=None, ensemble_model=None):
    """
    Analyze gradients of raw probability (not log-probability) along interpolation path.
    
    Args:
        bayes_model: The Bayesian model
        det_model: The deterministic model
        centroids: Dictionary mapping class indices to centroid vectors
        start_class: Starting class index
        target_class: Target class index
        n_points: Number of interpolation points
        num_samples: Number of samples for Bayesian model
        figsize: Figure size
        save_path: Path to save the figure
        ensemble_model: Optional ensemble model for comparison
        
    Returns:
        Dictionary with analysis results
    """
    # Get centroids for the specified classes
    centroid_a = centroids[start_class]
    centroid_b = centroids[target_class]
    
    # Create interpolation points
    alphas = torch.linspace(0, 1, n_points)
    points = [(1-a)*centroid_a + a*centroid_b for a in alphas]
    
    # Store results
    bayes_probs = []
    det_probs = []
    ensemble_probs = []
    bayes_grad_norms = []
    det_grad_norms = []
    ensemble_grad_norms = []
    
    print(f"Analyzing probability gradients from class {start_class} to class {target_class}...")
    for i, z in enumerate(points):
        if i % 10 == 0:
            print(f"Processing point {i+1}/{n_points}")
        
        # Bayesian model raw probability gradient
        z_bayes = z.clone().to(bayes_model.device).unsqueeze(0).requires_grad_(True)
        probs_bayes = bayes_model.sample_predict_z(z_bayes, Nsamples=num_samples)
        mean_probs_bayes = probs_bayes.mean(dim=0)
        target_prob_bayes = mean_probs_bayes[0, target_class]
        bayes_probs.append(target_prob_bayes.item())
        
        # Compute gradient of raw probability (not loss)
        bayes_model.zero_grad()
        target_prob_bayes.backward()
        grad_norm = z_bayes.grad.norm().item()
        bayes_grad_norms.append(grad_norm)
        
        # Deterministic model raw probability gradient
        z_det = z.clone().to(det_model.device).unsqueeze(0).requires_grad_(True)
        logits_det = det_model.classifier(z_det)
        probs_det = torch.nn.functional.softmax(logits_det, dim=1)
        target_prob_det = probs_det[0, target_class]
        det_probs.append(target_prob_det.item())
        
        # Compute gradient of raw probability
        det_model.zero_grad()
        target_prob_det.backward()
        grad_norm = z_det.grad.norm().item()
        det_grad_norms.append(grad_norm)
        
        # Ensemble model raw probability gradient (if provided)
        if ensemble_model is not None:
            z_ensemble = z.clone().to(ensemble_model.device).unsqueeze(0).requires_grad_(True)
            probs_ensemble = ensemble_model.sample_predict_z(z_ensemble, Nsamples=num_samples)
            mean_probs_ensemble = probs_ensemble.mean(dim=0)
            target_prob_ensemble = mean_probs_ensemble[0, target_class]
            ensemble_probs.append(target_prob_ensemble.item())
            
            # Compute gradient of raw probability
            ensemble_model.zero_grad()
            target_prob_ensemble.backward()
            grad_norm = z_ensemble.grad.norm().item()
            ensemble_grad_norms.append(grad_norm)
    
    # Convert to numpy arrays
    alphas_np = alphas.numpy()
    bayes_probs_np = np.array(bayes_probs)
    det_probs_np = np.array(det_probs)
    bayes_grad_norms_np = np.array(bayes_grad_norms)
    det_grad_norms_np = np.array(det_grad_norms)
    
    # Create visualization
    if ensemble_model is not None:
        ensemble_probs_np = np.array(ensemble_probs)
        ensemble_grad_norms_np = np.array(ensemble_grad_norms)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]))
    
    # Plot probabilities
    ax1.plot(alphas_np, bayes_probs_np, 'b-', linewidth=2, label='Bayesian')
    ax1.plot(alphas_np, det_probs_np, 'r--', linewidth=2, label='Deterministic')
    if ensemble_model is not None:
        ax1.plot(alphas_np, ensemble_probs_np, 'g-', linewidth=2, label='Ensemble')
    ax1.set_title(f'Target Class Probability: Class {start_class} → Class {target_class}')
    ax1.set_xlabel('Interpolation Parameter (α)')
    ax1.set_ylabel('Probability')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot gradients of probability
    ax2.plot(alphas_np, bayes_grad_norms_np, 'b-', linewidth=2, label='Bayesian')
    ax2.plot(alphas_np, det_grad_norms_np, 'r--', linewidth=2, label='Deterministic')
    if ensemble_model is not None:
        ax2.plot(alphas_np, ensemble_grad_norms_np, 'g-', linewidth=2, label='Ensemble')
    ax2.set_title(f'Gradient of Raw Probability (||∇P(class={target_class})||)')
    ax2.set_xlabel('Interpolation Parameter (α)')
    ax2.set_ylabel('Gradient Norm')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    
    # Calculate correlation as a sanity check
    bayes_prob_deriv = np.gradient(bayes_probs_np, alphas_np)
    det_prob_deriv = np.gradient(det_probs_np, alphas_np)
    bayes_corr = np.corrcoef(bayes_grad_norms_np, np.abs(bayes_prob_deriv))[0, 1]
    det_corr = np.corrcoef(det_grad_norms_np, np.abs(det_prob_deriv))[0, 1]
    
    print(f"Correlation between probability gradient and numerical probability derivative:")
    print(f"  Bayesian: {bayes_corr:.4f}")
    print(f"  Deterministic: {det_corr:.4f}")
    
    result = {
        'alphas': alphas_np,
        'bayes_probs': bayes_probs,
        'det_probs': det_probs,
        'bayes_grad_norms': bayes_grad_norms,
        'det_grad_norms': det_grad_norms,
        'bayes_prob_deriv': bayes_prob_deriv.tolist(),
        'det_prob_deriv': det_prob_deriv.tolist(),
        'bayes_corr': bayes_corr,
        'det_corr': det_corr
    }
    
    if ensemble_model is not None:
        ensemble_prob_deriv = np.gradient(ensemble_probs_np, alphas_np)
        ensemble_corr = np.corrcoef(ensemble_grad_norms_np, np.abs(ensemble_prob_deriv))[0, 1]
        print(f"  Ensemble: {ensemble_corr:.4f}")
        
        result.update({
            'ensemble_probs': ensemble_probs,
            'ensemble_grad_norms': ensemble_grad_norms,
            'ensemble_prob_deriv': ensemble_prob_deriv.tolist(),
            'ensemble_corr': ensemble_corr
        })
    
    return result

def main():
    """
    Main function to demonstrate the full analysis pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate latent space gradients')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory with dataset')
    parser.add_argument('--bayes_model_path', type=str, required=True, help='Path to Bayesian model')
    parser.add_argument('--det_model_path', type=str, required=True, help='Path to deterministic model')
    parser.add_argument('--output_dir', type=str, default='./latent_space_analysis', help='Output directory')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of data samples')
    parser.add_argument('--grid_size', type=int, default=25, help='Size of evaluation grid')
    parser.add_argument('--target_classes', type=str, default='0,1,2', help='Target classes (comma-separated)')
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'tsne'], help='Dimension reduction method')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models (placeholder - actual implementation will depend on your model structure)
    # This would need to be adapted to your specific model loading approach
    print(f"Loading models from {args.bayes_model_path} and {args.det_model_path}")
    
    # ... model loading code here ...
    bayesian_model = torch.load(args.bayes_model_path)
    deterministic_model = torch.load(args.det_model_path)
    
    # Load dataset (placeholder)
    # ... dataset loading code here ...
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=100, shuffle=False, num_workers=2
    )
    
    # Determine target classes
    target_classes = [int(c) for c in args.target_classes.split(',')]
    print(f"Analyzing target classes: {target_classes}")
    
    # Create latent space visualization
    latent_vis = create_2d_latent_visualization(
        model=bayesian_model,  # This would be your actual loaded model
        dataloader=test_loader,  # This would be your actual test loader
        n_samples=args.n_samples,
        method=args.method,
        device=args.device
    )
    
    # Create grid in latent space
    grid_data = create_grid_in_latent_space(
        reduced_latents=latent_vis['reduced_latents'],
        inverse_transform=latent_vis['inverse_transform'],
        grid_size=args.grid_size
    )
    
    # Analyze multiple target classes
    summary = analyze_multiple_target_classes(
        bayesian_model=bayesian_model,  # This would be your actual loaded model
        deterministic_model=deterministic_model,  # This would be your actual loaded model
        grid_data=grid_data,
        target_classes=target_classes,
        output_dir=args.output_dir
    )
    
    # Save summary to CSV
    import pandas as pd
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(args.output_dir, 'gradient_analysis_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

if __name__ == "__main__":
    main()