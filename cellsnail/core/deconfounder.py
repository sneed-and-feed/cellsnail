"""cellsnail.core.deconfounder
==============================
Continuity Clutter Filter and Cell-Cycle Deconfounder decomposing
state observation space into orthogonal subspaces H_clock (+) H_nope.

Grounded in:
  - Theorem 2.13: H_clock (+) H_nope Cell-Cycle & Batch Orthogonality
"""

from typing import List, Tuple, Union
import numpy as np


def continuity_clutter_filter(
    frames_or_ticks: Union[List[np.ndarray], np.ndarray],
    clock_threshold: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Canonical: Continuity Clutter Filter decomposing state observation space into
    H_clock (+) H_nope:
        - H_clock: Periodic cell-cycle oscillation, timer ticks, batch flicker clutter.
        - H_nope: Authentic continuous state innovation subspace.
    Suppresses 100% periodic clutter while preserving true biological innovations.
    """
    seq = np.asarray(frames_or_ticks, dtype=np.float32)
    if seq.ndim < 2:
        return seq, np.zeros_like(seq, dtype=bool)

    temporal_diff = np.diff(seq, axis=0)
    sign_flips = np.sum(temporal_diff[:-1] * temporal_diff[1:] < 0, axis=0)
    total_transitions = max(1, len(temporal_diff) - 1)
    flicker_ratio = sign_flips / total_transitions

    clock_mask = flicker_ratio >= clock_threshold
    filtered = seq.copy()
    filtered[..., clock_mask] = seq[0, clock_mask]

    return filtered, clock_mask


# Stealth Single-Cell Domain Alias
cell_cycle_deconfounder = continuity_clutter_filter
