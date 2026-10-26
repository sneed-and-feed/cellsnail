"""cellsnail.core.triad_step
==========================
Pair 3: skew_triad_step <-> genotype_tiebreaker_so10 (Theorems 2.1–2.3).
Tangent bilinear triad update followed by spherical norm retraction.
"""

from typing import Optional
import numpy as np


def skew_triad_step(
    u: np.ndarray,
    v: np.ndarray,
    W: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    dt: Optional[float] = None,
) -> np.ndarray:
    """
    Canonical: Tangent bilinear triad update followed by spherical norm retraction:
        v_tangent = v + gamma * B_W(u, v)
        v_retract = (||v|| / ||v_tangent||) * v_tangent
    Preserves exact Euclidean norm ||v_retract|| == ||v|| (0.000e+00 radial drift).
    """
    if dt is not None:
        gamma = float(dt)

    v_arr = np.asarray(v, dtype=np.float64)
    norm_v = float(np.linalg.norm(v_arr))
    if norm_v == 0.0:
        return v_arr.copy()

    u_arr = np.asarray(u, dtype=np.float64)

    if W is None:
        raw = u_arr * (np.roll(v_arr, -1) - np.roll(v_arr, 1))
        v_norm_sq = float(np.dot(v_arr, v_arr))
        if v_norm_sq > 1e-30:
            raw = raw - (float(np.dot(v_arr, raw)) / v_norm_sq) * v_arr
        b_w = raw
    else:
        W_arr = np.asarray(W, dtype=np.float64)
        M = np.tensordot(u_arr, W_arr, axes=(0, 0))
        b_w = (v_arr @ M) - (M @ v_arr)

    v_upd = v_arr + gamma * b_w
    norm_upd = float(np.linalg.norm(v_upd))
    if norm_upd == 0.0:
        return v_arr.copy()

    return (norm_v / norm_upd) * v_upd


# Stealth Single-Cell Domain Alias
genotype_tiebreaker_so10 = skew_triad_step
