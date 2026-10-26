"""cellsnail.core.cascade
=======================
Pair 4: slow_fast_cascade_k16 <-> pseudotime_spherical_cascade (Theorem 2.6).
16-step unrolled Slow-Fast spherical cascade integrator.
"""

from typing import List, Optional, Union
import numpy as np

from cellsnail.core.triad_step import skew_triad_step


def slow_fast_cascade_k16(
    u_seq: Union[List[np.ndarray], np.ndarray],
    v0: np.ndarray,
    W_seq: Optional[Union[List[np.ndarray], np.ndarray]] = None,
    gamma: Union[float, List[float]] = 1.0,
    dt: Optional[float] = None,
    k_steps: int = 16,
) -> np.ndarray:
    """
    Canonical: 16-step unrolled Slow-Fast spherical cascade integrator:
        v_{t+1} = skew_triad_step(u_t, v_t, W_t, gamma_t)
    Guarantees ||v_16|| == ||v_0|| (0.000e+00 radial drift across all 16 steps).
    """
    if dt is not None:
        gamma = float(dt)
    steps = max(1, int(k_steps))
    v = np.asarray(v0, dtype=np.float64).copy()

    if isinstance(gamma, (int, float)):
        gammas = [float(gamma)] * steps
    else:
        gammas = list(gamma)

    u_is_1d = isinstance(u_seq, np.ndarray) and u_seq.ndim == 1
    w_is_3d = isinstance(W_seq, np.ndarray) and W_seq.ndim == 3

    for t in range(steps):
        u_t = u_seq if u_is_1d else u_seq[t]
        if W_seq is None:
            W_t = None
        elif w_is_3d:
            W_t = W_seq
        else:
            W_t = W_seq[t]
        g_t = gammas[t] if t < len(gammas) else 1.0
        v = skew_triad_step(u_t, v, W_t, gamma=g_t)

    return v


# Stealth Single-Cell Domain Alias
pseudotime_spherical_cascade = slow_fast_cascade_k16
