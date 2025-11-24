#!/usr/bin/env python3
"""
Geometric metrics for analyzing learned representations.

Includes:
- CircularityScore: measures how well activations form a circular structure
- TrajectoryLogger: tracks activation trajectories during training
- RobustnessProbe: tests geometry stability under noise
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


def compute_circularity_score(
    activations: torch.Tensor,
    modulus: int,
) -> float:
    """
    Compute how well activations form a circular structure for mod-p addition.

    For modular arithmetic (a + b) mod p, we expect:
    - Activations to cluster by equivalence class (result mod p)
    - Clusters arranged in a circle

    Args:
        activations: [n_examples, d_model] tensor of layer activations
        modulus: modulus p (e.g., 113)

    Returns:
        circularity_score in [0, 1], where:
        - 1.0 = perfect circular arrangement
        - 0.0 = random/linear arrangement
    """
    # Convert to numpy for easier manipulation
    acts = activations.detach().cpu().numpy()
    n_examples, d_model = acts.shape

    # Project to 2D using PCA for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    acts_2d = pca.fit_transform(acts)

    # Expected: activations should cluster by (result mod p)
    # We need labels for each activation - this requires access to data
    # For now, compute a simpler metric: check if points lie on a circle

    # Center the points
    center = acts_2d.mean(axis=0)
    centered = acts_2d - center

    # Compute radius for each point
    radii = np.linalg.norm(centered, axis=1)

    # Circularity = 1 - coefficient of variation of radii
    # Perfect circle has constant radius (CV=0), random has high CV
    mean_radius = radii.mean()
    std_radius = radii.std()

    if mean_radius < 1e-6:
        return 0.0

    cv = std_radius / mean_radius
    # Map CV to [0, 1] score: lower CV = higher circularity
    # CV of 0.0 -> score 1.0
    # CV of 1.0 -> score 0.0
    circularity = max(0.0, 1.0 - cv)

    return float(circularity)


def compute_circularity_score_labeled(
    activations: torch.Tensor,
    labels: torch.Tensor,
    modulus: int,
) -> Dict[str, float]:
    """
    Compute circularity metrics with ground-truth labels.

    This version clusters activations by their true result (mod p)
    and checks if cluster centers form a circle.

    Args:
        activations: [n_examples, d_model]
        labels: [n_examples] ground-truth results (0 to p-1)
        modulus: modulus p

    Returns:
        Dictionary with:
        - 'circularity': overall circularity score
        - 'cluster_separation': how well-separated clusters are
        - 'explained_variance': variance explained by top 2 PCs
    """
    from sklearn.decomposition import PCA

    acts = activations.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # PCA to 2D
    pca = PCA(n_components=2)
    acts_2d = pca.fit_transform(acts)

    # Compute cluster centers for each equivalence class
    cluster_centers = []
    for i in range(modulus):
        mask = labels_np == i
        if mask.sum() > 0:
            center = acts_2d[mask].mean(axis=0)
            cluster_centers.append(center)

    cluster_centers = np.array(cluster_centers)  # [p, 2]

    # Check if cluster centers lie on a circle
    mean_center = cluster_centers.mean(axis=0)
    centered = cluster_centers - mean_center
    radii = np.linalg.norm(centered, axis=1)

    mean_radius = radii.mean()
    std_radius = radii.std()
    cv = std_radius / mean_radius if mean_radius > 1e-6 else 1.0
    circularity = max(0.0, 1.0 - cv)

    # Cluster separation: ratio of between-cluster to within-cluster variance
    # Higher = better separated
    between_var = np.var(cluster_centers, axis=0).sum()
    within_var = 0.0
    for i in range(modulus):
        mask = labels_np == i
        if mask.sum() > 1:
            within_var += np.var(acts_2d[mask], axis=0).sum()
    within_var /= modulus

    separation = between_var / (within_var + 1e-6)

    return {
        'circularity': float(circularity),
        'cluster_separation': float(separation),
        'explained_variance': float(pca.explained_variance_ratio_[:2].sum()),
    }


def compute_trajectory_curvature(
    activations_seq: list[torch.Tensor],
    layer_idx: int = 0,
) -> float:
    """
    Compute discrete curvature of activation trajectory during training.

    High curvature = trajectory bends a lot (exploring)
    Low curvature = trajectory is straight (committed)

    Args:
        activations_seq: List of activation tensors at different checkpoints
            Each tensor: [n_examples, d_model]
        layer_idx: Which layer's activations to analyze

    Returns:
        Mean curvature (angle between consecutive trajectory segments)
    """
    if len(activations_seq) < 3:
        return 0.0

    # Use PCA to project all checkpoints to same 2D space
    from sklearn.decomposition import PCA

    # Stack all activations
    all_acts = torch.cat(activations_seq, dim=0).detach().cpu().numpy()
    pca = PCA(n_components=2)
    pca.fit(all_acts)

    # Project each checkpoint to 2D and compute centroid
    centroids = []
    n_per_checkpoint = activations_seq[0].shape[0]
    for acts in activations_seq:
        acts_2d = pca.transform(acts.detach().cpu().numpy())
        centroid = acts_2d.mean(axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)  # [n_checkpoints, 2]

    # Compute angles between consecutive segments
    angles = []
    for i in range(1, len(centroids) - 1):
        v1 = centroids[i] - centroids[i-1]
        v2 = centroids[i+1] - centroids[i]

        # Angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(angle)

    mean_curvature = np.mean(angles) if angles else 0.0
    return float(mean_curvature)


def test_geometry_robustness(
    model: nn.Module,
    test_data: list[Dict],
    layer_idx: int,
    modulus: int,
    noise_levels: list[float] = [0.0, 0.1, 0.2, 0.5],
    device: torch.device = torch.device('cpu'),
) -> Dict[str, float]:
    """
    Test how robust the learned circular geometry is to noise injection.

    Injects Gaussian noise at specified layer and measures circularity degradation.

    Args:
        model: Trained model
        test_data: Test examples with 'a', 'b', 'result' fields
        layer_idx: Which layer to perturb
        modulus: Modulus p
        noise_levels: Standard deviations of Gaussian noise to inject
        device: torch device

    Returns:
        Dictionary mapping noise_level -> circularity_score
        Also includes 'auc' (area under curve) as overall robustness metric
    """
    model.eval()
    results = {}

    # Prepare test batch
    batch_size = min(256, len(test_data))
    test_batch = test_data[:batch_size]

    with torch.no_grad():
        for noise_std in noise_levels:
            # Get activations with noise injection
            activations, labels = get_layer_activations_with_noise(
                model=model,
                data=test_batch,
                layer_idx=layer_idx,
                noise_std=noise_std,
                device=device,
            )

            # Compute circularity
            metrics = compute_circularity_score_labeled(
                activations=activations,
                labels=labels,
                modulus=modulus,
            )

            results[f'noise_{noise_std}'] = metrics['circularity']

    # Compute area under curve as overall robustness metric
    noise_vals = sorted([k for k in noise_levels])
    circ_vals = [results[f'noise_{n}'] for n in noise_vals]
    auc = float(np.trapz(circ_vals, noise_vals))
    results['auc'] = auc

    return results


def get_layer_activations_with_noise(
    model: nn.Module,
    data: list[Dict],
    layer_idx: int,
    noise_std: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract activations at specified layer with optional noise injection.

    Args:
        model: Model with forward() method
        data: List of examples with 'a', 'b', 'result' fields
        layer_idx: Which layer to extract from
        noise_std: Standard deviation of Gaussian noise to add
        device: torch device

    Returns:
        (activations, labels) where:
        - activations: [n_examples, d_model]
        - labels: [n_examples] ground-truth results
    """
    # This is a placeholder - actual implementation depends on model architecture
    # You'll need to modify this based on your specific model

    activations_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for ex in data:
            # Convert to input format
            # This depends on your model - adjust as needed
            # For now, just collect labels
            labels_list.append(ex['result'])

    # Placeholder return
    # In real implementation, you'd do a forward pass with hooks
    # to extract intermediate activations
    raise NotImplementedError(
        "get_layer_activations_with_noise needs to be implemented "
        "based on your specific model architecture"
    )
