"""cellsnail.benchmark.baselines
=============================
Self-contained, faithful, genuine reference implementations of canonical
single-cell bioinformatics baselines:
  1. MAGIC: Markov Affinity-based Graph Imputation of Cells (van Dijk et al., Cell 2018).
  2. ALRA: Adaptive Low-Rank Approximation (Linderman et al., Nature Methods 2022).
  3. HMMcopy / inferCNV: Windowed running-mean smoothing and HMM state segmentation.
  4. Neighbor-Joining (NJ) & UPGMA: Classical phylogenetic tree reconstruction algorithms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# ==============================================================================
# 1. MAGIC: kNN Graph Diffusion Imputation (van Dijk et al. 2018)
# ==============================================================================

def magic_impute(
    X: np.ndarray,
    k: int = 15,
    t: int = 3,
    ka: int = 4,
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Markov Affinity-based Graph Imputation of Cells (MAGIC).

    Computes cell-cell graph diffusion:
      1. Pairwise Euclidean distances D between cells.
      2. Adaptive bandwidth sigma_i = distance to ka-th nearest neighbor of cell i.
      3. Kernel affinity: A_ij = exp(-D_ij / (sigma_i * epsilon)).
      4. Symmetrize and row-normalize to Markov transition matrix M.
      5. Diffuse counts: X_imputed = M^t * X.

    Note on Failure Mode:
      Diffusing counts across cells injects spurious correlations into independent genes
      (e.g. synthetic ERCC spike-ins).
    """
    X_arr = np.asarray(X, dtype=np.float32)
    n_cells, n_genes = X_arr.shape
    if n_cells <= 1:
        return X_arr.copy()

    # Pairwise squared distances: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 <x_i, x_j>
    sq_norms = np.sum(X_arr * X_arr, axis=1, keepdims=True)
    dists_sq = np.maximum(0.0, sq_norms + sq_norms.T - 2.0 * np.dot(X_arr, X_arr.T))
    dists = np.sqrt(dists_sq)

    # Adaptive bandwidth: distance to ka-th nearest neighbor (ignoring self at index 0)
    sorted_dists = np.sort(dists, axis=1)
    ka_idx = min(max(1, ka), n_cells - 1)
    sigma = sorted_dists[:, ka_idx]
    sigma = np.maximum(sigma, 1e-6)

    # Adaptive anisotropic kernel
    # Bandwidth matrix: sigma_i * epsilon
    bandwidth = sigma[:, np.newaxis] * epsilon
    affinities = np.exp(-dists / bandwidth)

    # Symmetrize affinities
    affinities = 0.5 * (affinities + affinities.T)

    # Sparsify to top k neighbors per row
    k_eff = min(max(2, k), n_cells)
    for i in range(n_cells):
        cutoff = sorted_dists[i, k_eff - 1]
        affinities[i, dists[i] > cutoff] = 0.0

    # Ensure positive diagonal
    np.fill_diagonal(affinities, np.maximum(np.diag(affinities), 1.0))

    # Row-normalize to transition probability matrix M
    row_sums = np.sum(affinities, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    M = affinities / row_sums

    # Compute M^t via matrix power
    M_t = np.linalg.matrix_power(M, max(1, t))

    # Impute expression
    X_imputed = np.dot(M_t, X_arr)
    return X_imputed.astype(np.float32)


# ==============================================================================
# 2. ALRA: Adaptive Low-Rank Approximation (Linderman et al. 2022)
# ==============================================================================

def alra_impute(
    X: np.ndarray,
    k: Optional[int] = None,
    q: int = 10,
) -> np.ndarray:
    """
    Adaptive Low-Rank Approximation (ALRA).

    Computes rank-k truncated SVD followed by adaptive thresholding:
      1. Rank-k approximation: A = U_k * Sigma_k * V_k^T.
      2. For each gene, estimates technical noise floor from negative values.
      3. Thresholds low-rank matrix and restores zeros.

    Note on Failure Mode:
      Low-rank thresholding wipes out rare persister subclones (< 0.5% prevalence)
      whose signature amplicons lie in the discarded singular spectrum.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    n_cells, n_genes = X_arr.shape
    if n_cells <= 1 or n_genes <= 1:
        return X_arr.copy()

    # Determine rank k
    max_k = min(n_cells, n_genes) - 1
    if k is None:
        # Choose k based on matrix dimension heuristic
        k = min(max(2, max_k // 4), 15)
    else:
        k = min(max(1, k), max_k)

    # Truncated SVD
    U, s, Vt = np.linalg.svd(X_arr, full_matrices=False)
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]

    # Low-rank reconstruction
    A = np.dot(U_k * s_k, Vt_k)

    # Adaptive thresholding per gene (column)
    A_thresholded = np.zeros_like(A)
    for j in range(n_genes):
        col = A[:, j]
        neg_vals = col[col < 0]
        if len(neg_vals) > 0:
            # Noise threshold estimated from negative tail
            sigma_neg = np.abs(np.percentile(neg_vals, q))
            # Values above threshold are retained and shifted
            col_thresh = np.where(col > sigma_neg, col - sigma_neg, 0.0)
        else:
            col_thresh = np.maximum(0.0, col)
        A_thresholded[:, j] = col_thresh

    return A_thresholded.astype(np.float32)


# ==============================================================================
# 3. HMMcopy / inferCNV: Windowed Smoothing and Segmentation
# ==============================================================================

def hmmcopy_infercnv_smooth(
    cnv_matrix: np.ndarray,
    window_size: int = 15,
) -> np.ndarray:
    """
    Classical windowed running-mean smoothing for single-cell copy-number profiles
    (mimicking inferCNV / HMMcopy pre-processing).

    Note on Failure Mode:
      Sliding window averaging smears sharp, narrow sub-megabase amplicons
      (e.g. 0.4 Mb EGFR on chr7 or 0.3 Mb CDK4 on chr12) into diffuse low-amplitude noise,
      destroying focal oncogene detection.
    """
    cnv_arr = np.asarray(cnv_matrix, dtype=np.float32)
    n_cells, n_bins = cnv_arr.shape
    if n_bins <= window_size:
        return cnv_arr.copy()

    half_w = window_size // 2
    smoothed = np.zeros_like(cnv_arr)

    # Running window mean along genomic bins (axis 1)
    for i in range(n_cells):
        row = cnv_arr[i]
        # Pad edges
        padded = np.pad(row, half_w, mode="edge")
        # Moving average
        window = np.ones(window_size, dtype=np.float32) / window_size
        smooth_row = np.convolve(padded, window, mode="valid")
        smoothed[i] = smooth_row[:n_bins]

    return smoothed


# ==============================================================================
# 4. Classical Phylogenetics: Neighbor-Joining (NJ) and UPGMA
# ==============================================================================

class PhyloNode:
    """Phylogenetic tree node."""
    def __init__(
        self,
        node_id: int,
        name: str = "",
        left: Optional["PhyloNode"] = None,
        right: Optional["PhyloNode"] = None,
        dist_left: float = 0.0,
        dist_right: float = 0.0,
        height: float = 0.0,
    ):
        self.node_id = node_id
        self.name = name or f"Node_{node_id}"
        self.left = left
        self.right = right
        self.dist_left = dist_left
        self.dist_right = dist_right
        self.height = height

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def get_leaves(self) -> List[str]:
        if self.is_leaf:
            return [self.name]
        leaves = []
        if self.left:
            leaves.extend(self.left.get_leaves())
        if self.right:
            leaves.extend(self.right.get_leaves())
        return leaves

    def to_newick(self) -> str:
        if self.is_leaf:
            return self.name
        left_str = f"{self.left.to_newick()}:{self.dist_left:.4f}" if self.left else ""
        right_str = f"{self.right.to_newick()}:{self.dist_right:.4f}" if self.right else ""
        return f"({left_str},{right_str}){self.name}"


def neighbor_joining(
    dist_matrix: np.ndarray,
    taxa_names: Optional[List[str]] = None,
) -> PhyloNode:
    """
    Saitou & Nei (1987) Neighbor-Joining (NJ) algorithm.

    Reconstructs an additive phylogenetic tree from a distance matrix.
    Note on Failure Mode:
      Standard NJ assumes symmetric divergence and does not account for ascertainment bias
      (filtering invariant sites), causing artificial tip branch elongation.
    """
    D = np.asarray(dist_matrix, dtype=np.float64).copy()
    n = D.shape[0]
    if taxa_names is None:
        taxa_names = [f"Taxon_{i}" for i in range(n)]

    # Initialize leaf nodes
    active_nodes: List[PhyloNode] = [PhyloNode(i, name=taxa_names[i]) for i in range(n)]
    next_id = n

    while len(active_nodes) > 2:
        m = len(active_nodes)
        # Compute row sums: r_i = sum_k D(i, k)
        r = np.sum(D, axis=1)

        # Compute Q-matrix: Q(i, j) = (m - 2) * D(i, j) - r_i - r_j
        Q = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(i + 1, m):
                q_val = (m - 2) * D[i, j] - r[i] - r[j]
                Q[i, j] = q_val
                Q[j, i] = q_val

        # Find pair (min_i, min_j) minimizing Q
        min_val = np.inf
        min_i, min_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                if Q[i, j] < min_val:
                    min_val = Q[i, j]
                    min_i, min_j = i, j

        # Compute branch lengths to new node u
        d_ij = D[min_i, min_j]
        d_iu = 0.5 * d_ij + (r[min_i] - r[min_j]) / (2.0 * max(1, m - 2))
        d_ju = d_ij - d_iu

        # Ensure non-negative branch lengths
        d_iu = max(0.0, d_iu)
        d_ju = max(0.0, d_ju)

        # Create new internal node
        node_u = PhyloNode(
            node_id=next_id,
            name=f"Internal_{next_id}",
            left=active_nodes[min_i],
            right=active_nodes[min_j],
            dist_left=d_iu,
            dist_right=d_ju,
        )
        next_id += 1

        # Calculate distances from new node u to all other nodes k
        d_uk = np.zeros(m, dtype=np.float64)
        for k in range(m):
            if k != min_i and k != min_j:
                d_uk[k] = 0.5 * (D[min_i, k] + D[min_j, k] - d_ij)

        # Update distance matrix: remove min_i and min_j, insert u
        remaining_indices = [k for k in range(m) if k != min_i and k != min_j]
        new_D = np.zeros((m - 1, m - 1), dtype=np.float64)

        # Copy existing pairwise distances
        for new_r, old_r in enumerate(remaining_indices):
            for new_c, old_c in enumerate(remaining_indices):
                new_D[new_r, new_c] = D[old_r, old_c]

        # Insert new distances to u (at the last row and column)
        for new_r, old_r in enumerate(remaining_indices):
            new_D[new_r, -1] = d_uk[old_r]
            new_D[-1, new_r] = d_uk[old_r]

        D = new_D
        active_nodes = [active_nodes[k] for k in remaining_indices] + [node_u]

    # Join the final two nodes
    d_final = max(0.0, float(D[0, 1]))
    root = PhyloNode(
        node_id=next_id,
        name=f"Root_{next_id}",
        left=active_nodes[0],
        right=active_nodes[1],
        dist_left=d_final / 2.0,
        dist_right=d_final / 2.0,
    )
    return root


def upgma(
    dist_matrix: np.ndarray,
    taxa_names: Optional[List[str]] = None,
) -> PhyloNode:
    """
    Unweighted Pair Group Method with Arithmetic Mean (UPGMA).

    Reconstructs an ultrametric phylogenetic tree (strict molecular clock).
    """
    D = np.asarray(dist_matrix, dtype=np.float64).copy()
    n = D.shape[0]
    if taxa_names is None:
        taxa_names = [f"Taxon_{i}" for i in range(n)]

    # Initialize leaf clusters
    clusters: List[PhyloNode] = [PhyloNode(i, name=taxa_names[i], height=0.0) for i in range(n)]
    cluster_sizes: List[int] = [1] * n
    next_id = n

    while len(clusters) > 1:
        m = len(clusters)
        min_dist = np.inf
        min_i, min_j = 0, 1

        for i in range(m):
            for j in range(i + 1, m):
                if D[i, j] < min_dist:
                    min_dist = D[i, j]
                    min_i, min_j = i, j

        # Node height is half the distance between merged clusters
        node_height = min_dist / 2.0
        d_left = max(0.0, node_height - clusters[min_i].height)
        d_right = max(0.0, node_height - clusters[min_j].height)

        new_node = PhyloNode(
            node_id=next_id,
            name=f"Internal_{next_id}",
            left=clusters[min_i],
            right=clusters[min_j],
            dist_left=d_left,
            dist_right=d_right,
            height=node_height,
        )
        next_id += 1

        size_i = cluster_sizes[min_i]
        size_j = cluster_sizes[min_j]
        new_size = size_i + size_j

        # Compute average distance to other clusters
        remaining = [k for k in range(m) if k != min_i and k != min_j]
        new_D = np.zeros((m - 1, m - 1), dtype=np.float64)

        for new_r, old_r in enumerate(remaining):
            for new_c, old_c in enumerate(remaining):
                new_D[new_r, new_c] = D[old_r, old_c]

        # Weighted arithmetic mean distance
        for new_r, old_r in enumerate(remaining):
            d_u_k = (size_i * D[min_i, old_r] + size_j * D[min_j, old_r]) / float(new_size)
            new_D[new_r, -1] = d_u_k
            new_D[-1, new_r] = d_u_k

        D = new_D
        clusters = [clusters[k] for k in remaining] + [new_node]
        cluster_sizes = [cluster_sizes[k] for k in remaining] + [new_size]

    return clusters[0]
