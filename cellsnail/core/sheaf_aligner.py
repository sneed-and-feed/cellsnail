"""cellsnail.core.sheaf_aligner
==============================
SE(2) Cellular Sheaf contact-graph coboundary solver with diagnostic
spatial registration jitter estimation and Huber directional soft-clamping (v0.2.0).

Grounded in:
  - Theorem 2.10b: Contact Sheaf Normal Coboundary Non-Penetration Condition
  - Theorem 2.18: Jitter-Adaptive Sheaf Energy & Bounded Normal Bleed
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# ==============================================================================
# Diagnostic Jitter Estimator
# ==============================================================================

def estimate_spatial_jitter(
    coords: np.ndarray,
    neighbor_dist: float = 55.0,
    expected_dist: Optional[float] = None,
) -> float:
    """
    Diagnostic estimator for spatial coordinate mis-segmentation jitter Delta_x.
    Evaluates variance in inter-spot nearest-neighbor distances relative to expected
    regular grid spacing.
    """
    xy = np.asarray(coords, dtype=np.float32)
    n = len(xy)
    if n <= 3:
        return 0.0

    # Sample pairwise nearest-neighbor distances
    min_dists = []
    sample_size = min(n, 200)
    indices = np.linspace(0, n - 1, sample_size, dtype=int)
    for i in indices:
        diffs = xy - xy[i]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        nbrs = dists[(dists > 1e-4) & (dists <= neighbor_dist)]
        if len(nbrs) > 0:
            min_dists.append(float(np.min(nbrs)))

    if len(min_dists) == 0:
        return 0.0

    min_dists_arr = np.array(min_dists)
    if expected_dist is None:
        expected_dist = float(np.median(min_dists_arr))

    # Variance from expected regular lattice distance
    delta_x_est = float(np.sqrt(np.mean((min_dists_arr - expected_dist) ** 2)))
    return delta_x_est


# ==============================================================================
# Contact Graph Primitives
# ==============================================================================

class ContactEdge:
    def __init__(
        self,
        u: int,
        v: int,
        normal_x: float,
        normal_y: float,
        is_active: bool = True,
        edge_type: int = 0,
    ):
        self.u = u
        self.v = v
        self.normal_x = normal_x
        self.normal_y = normal_y
        self.is_active = is_active
        self.edge_type = edge_type


class ContactNode:
    def __init__(self, node_id: int, is_fixed: bool = False):
        self.id = node_id
        self.is_fixed = is_fixed


# ==============================================================================
# Core Operator: Huber Directional Soft-Clamping (v0.2.0)
# ==============================================================================

def contact_sheaf_se2_step(
    *args,
    **kwargs,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    SE(2) Cellular Sheaf contact-graph coboundary solver with Huber soft-clamping (v0.2.0).
    
    Supports:
      1. Interface vector mode: (dx, dy, nx, ny, delta_x_jitter=0.0, huber_delta=10.0)
      2. Graph mode: (ego_id, dx, dy, nodes=None, edges=None, delta_x_jitter=0.0)
    
    Mathematical Formulation:
      - Normal projection: u = dx * nx + dy * ny.
      - Leakage coefficient:
          gamma(Delta_x) = min(1.0, Delta_x^2 / (Delta_x^2 + Delta_x_0^2)), with Delta_x_0 = 15 um.
      - Adaptive Huber threshold:
          delta_H = delta_0 * exp(-Delta_x / 20.0), with delta_0 = 10.0.
      - Clamping penalty:
          psi(u) = 0 if u <= 0
                 = u if 0 < u <= delta_H
                 = delta_H + gamma * (u - delta_H) if u > delta_H.
      - Clamped displacement: v_clamped = v - psi(u) * n.
    """
    delta_x_jitter = float(kwargs.get("delta_x_jitter", 0.0))
    huber_delta = float(kwargs.get("huber_delta", 10.0))

    if len(args) >= 4 and isinstance(args[2], (int, float)) and isinstance(args[3], (int, float)):
        dx, dy, nx, ny = float(args[0]), float(args[1]), float(args[2]), float(args[3])
        if len(args) > 4:
            delta_x_jitter = float(args[4])
        if len(args) > 5:
            huber_delta = float(args[5])

        if nx == 0.0 and ny == 0.0:
            return dx, dy, {"hit_wall": False, "penetration_residual": 0.0, "leakage": 0.0}

        dot = dx * nx + dy * ny
        if dot <= 0.0:
            # Separating or tangential -> no resistance
            return dx, dy, {"hit_wall": False, "penetration_residual": 0.0, "leakage": 0.0}

        # Penetration detected: evaluate Huber soft-clamping
        if delta_x_jitter <= 0.0:
            # Pure hard clamping (v0.1.0 exact parity)
            return 0.0, 0.0, {"hit_wall": True, "penetration_residual": 0.0, "leakage": 0.0}

        # Jitter compensation
        delta_x0 = 15.0
        gamma = float(min(1.0, (delta_x_jitter ** 2) / (delta_x_jitter ** 2 + delta_x0 ** 2)))
        delta_h = float(huber_delta * np.exp(-min(50.0, delta_x_jitter / 20.0)))

        if dot <= delta_h:
            psi = dot
        else:
            psi = delta_h + gamma * (dot - delta_h)

        out_dx = dx - psi * nx
        out_dy = dy - psi * ny
        pen_residual = float(max(0.0, out_dx * nx + out_dy * ny))

        return out_dx, out_dy, {
            "hit_wall": True,
            "penetration_residual": pen_residual,
            "leakage": gamma,
            "huber_threshold": delta_h,
        }

    # Graph mode
    if len(args) >= 3:
        ego_id = int(args[0])
        dx = float(args[1])
        dy = float(args[2])
        nodes = args[3] if len(args) > 3 else kwargs.get("nodes", None)
        edges = args[4] if len(args) > 4 else kwargs.get("edges", None)
    else:
        ego_id = int(kwargs.get("ego_id", 0))
        dx = float(kwargs.get("dx", 0.0))
        dy = float(kwargs.get("dy", 0.0))
        nodes = kwargs.get("nodes", None)
        edges = kwargs.get("edges", None)

    if nodes is None or len(nodes) == 0:
        return dx, dy, {"hit_wall": False, "penetration_residual": 0.0, "leakage": 0.0}

    node_fixed: Dict[int, bool] = {}
    for n in nodes:
        if isinstance(n, dict):
            node_fixed[int(n.get("id", 0))] = bool(n.get("is_fixed", False))
        else:
            node_fixed[int(n.id)] = bool(n.is_fixed)

    visited = {ego_id}
    queue = deque([ego_id])
    hit_wall = False

    while queue:
        cur = queue.popleft()
        if node_fixed.get(cur, False):
            hit_wall = True
            break

        if edges:
            for edge in edges:
                if isinstance(edge, dict):
                    is_active = bool(edge.get("is_active", True))
                    u_id = int(edge.get("u", 0))
                    v_id = int(edge.get("v", 0))
                    nx = float(edge.get("normal_x", 0.0))
                    ny = float(edge.get("normal_y", 0.0))
                    etype = int(edge.get("edge_type", 0))
                else:
                    is_active = edge.is_active
                    u_id = edge.u
                    v_id = edge.v
                    nx = edge.normal_x
                    ny = edge.normal_y
                    etype = edge.edge_type

                if is_active and u_id == cur:
                    dot = nx * dx + ny * dy
                    if dot > 0 or etype == 1:
                        if v_id not in visited:
                            visited.add(v_id)
                            queue.append(v_id)

    if hit_wall:
        if delta_x_jitter <= 0.0:
            return 0.0, 0.0, {"hit_wall": True, "penetration_residual": 0.0, "leakage": 0.0}
        else:
            delta_x0 = 15.0
            gamma = float(min(1.0, (delta_x_jitter ** 2) / (delta_x_jitter ** 2 + delta_x0 ** 2)))
            return gamma * dx, gamma * dy, {"hit_wall": True, "penetration_residual": 0.0, "leakage": gamma}

    return dx, dy, {"hit_wall": False, "penetration_residual": 0.0, "leakage": 0.0}


# Stealth Single-Cell Domain Alias
spatial_tissue_sheaf_solver = contact_sheaf_se2_step


# ==============================================================================
# Object-Oriented SE(2) Cellular Sheaf Class
# ==============================================================================

class SE2CellularSheaf:
    """
    High-level SE(2) Cellular Sheaf manager for spatial transcriptomics alignment
    and tissue compartment smoothing under registration jitter.
    """

    def __init__(self, delta_x_jitter: float = 0.0, huber_delta: float = 10.0):
        self.delta_x_jitter = float(delta_x_jitter)
        self.huber_delta = float(huber_delta)

    def align_with_jitter_compensation(
        self,
        coords: np.ndarray,
        marker_scores: np.ndarray,
        apparent_layers: Optional[np.ndarray] = None,
        neighbor_dist: float = 55.0,
        delta_x_est: Optional[float] = None,
        mode: str = "adaptive",
    ) -> Dict[str, Any]:
        """
        Smooth spatial tissue marker scores across contact graph with adaptive Huber soft-clamping.
        
        At low jitter (<= 5 um): enforces sharp compartmental isolation.
        At high jitter (>= 15 um): applies soft leakage, eliminating the double penalty.
        """
        xy = np.asarray(coords, dtype=np.float32)
        X = np.asarray(marker_scores, dtype=np.float32)
        n_cells = len(xy)

        if delta_x_est is None:
            if self.delta_x_jitter > 0:
                delta_x_est = self.delta_x_jitter
            else:
                delta_x_est = estimate_spatial_jitter(xy, neighbor_dist=neighbor_dist)

        if apparent_layers is None:
            apparent_layers = np.clip((xy[:, 1] // 100.0).astype(int), 0, 3)

        delta_x0 = self.huber_delta
        gamma = float(min(1.0, (delta_x_est ** 2) / (delta_x_est ** 2 + delta_x0 ** 2))) if mode == "adaptive" else 0.0

        smoothed = np.zeros_like(X)
        for i in range(n_cells):
            diffs = xy - xy[i]
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            nbrs = np.where((dists > 0) & (dists <= neighbor_dist))[0]
            if len(nbrs) == 0:
                smoothed[i] = X[i]
                continue

            same_app = nbrs[apparent_layers[nbrs] == apparent_layers[i]]
            within_comp = np.mean(X[same_app], axis=0) if len(same_app) > 0 else X[i]
            all_nbrs = np.mean(X[nbrs], axis=0)

            if mode == "adaptive":
                # Continuous Huber transition: gamma=0 recovers hard sheaf, gamma->1 recovers isotropic
                nbr_blend = (1.0 - gamma) * within_comp + gamma * all_nbrs
                self_w = 0.15 + 0.10 * gamma
                smoothed[i] = self_w * X[i] + (1.0 - self_w) * nbr_blend
            else:
                if len(same_app) > 0:
                    smoothed[i] = 0.15 * X[i] + 0.85 * within_comp
                else:
                    smoothed[i] = X[i]

        return {
            "smoothed_scores": smoothed,
            "delta_x_est": float(delta_x_est),
            "leakage": float(gamma),
        }
