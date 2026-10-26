"""cellsnail.operators
===================
Mathematical oracle and SIMD-accelerated implementations of the 6 core primitives
for single-cell data science in the DarkSnail / triad-core ecosystem, exposing both
canonical DarkSnail symbols and stealth domain-native single-cell bioinformatics aliases.

Dual Symbol Pairs:
  1. beltrami_morse_gate <-> morse_dropout_filter (Theorems 2.12, 2.16, 2.17)
  2. continuity_clutter_filter <-> cell_cycle_deconfounder (Theorem 2.13)
  3. skew_triad_step <-> genotype_tiebreaker_so10 (Theorems 2.1–2.3)
  4. slow_fast_cascade_k16 <-> pseudotime_spherical_cascade (Theorem 2.6)
  5. contact_sheaf_se2_step <-> spatial_tissue_sheaf_solver (Theorems 2.10b, 2.18)
  6. kelvin_mandel_so6_rotor <-> multiomic_covariance_rotor (Theorems 2.7, 2.15, 2.19)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Re-export from modular core
from cellsnail.core import (
    ContactEdge,
    ContactNode,
    SE2CellularSheaf,
    SO6MultiModalBundle,
    beltrami_morse_gate,
    build_kelvin_mandel_so6_rotor,
    cell_cycle_deconfounder,
    compute_spatial_mad_floor_256,
    contact_sheaf_se2_step,
    continuity_clutter_filter,
    estimate_missingness,
    estimate_segment_widths,
    estimate_spatial_jitter,
    estimate_umi_depth,
    genotype_tiebreaker_so10,
    kelvin_mandel_so6_rotor,
    ledoit_wolf_shrinkage_so6,
    morse_cnv_filter,
    morse_dropout_filter,
    multiomic_covariance_rotor,
    pseudotime_spherical_cascade,
    rotate_stiffness_so6,
    sigmoidal_shrinkage_weight,
    skew_triad_step,
    slow_fast_cascade_k16,
    spatial_tissue_sheaf_solver,
)

try:
    from cellsnail._native import (
        is_native_available,
        native_cell_cycle_deconfounder,
        native_genotype_tiebreaker_so10,
        native_morse_dropout_filter,
        native_multiomic_covariance_rotor,
        native_pseudotime_spherical_cascade,
        native_spatial_tissue_sheaf_solver,
    )
    _NATIVE_READY = is_native_available()
except Exception:
    _NATIVE_READY = False


# ==============================================================================
# Unified CellSnail Functor Engine
# ==============================================================================

class CellSnailEngine:
    """
    Unified Single-Cell Functor Engine for sneed-and-feed/cellsnail v0.2.0.
    Unifies all 6 core mathematical operators with canonical DarkSnail names
    and stealth single-cell bioinformatics aliases.
    """

    def __init__(self, mode: str = "adaptive"):
        self.step_count = 0
        self.mode = mode

    def morse_dropout_filter(
        self,
        signal: np.ndarray,
        mad_floor: float = 0.0,
        tau: float = 0.0,
        u_hat: Optional[float] = None,
        prior_matrix: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        self.step_count += 1
        return morse_dropout_filter(
            signal,
            tau_mad=tau if tau > 0 else (mad_floor if mad_floor > 0 else None),
            u_hat=u_hat,
            prior_matrix=prior_matrix,
            mode=kwargs.get("mode", self.mode),
            **kwargs,
        )

    def morse_cnv_filter(
        self,
        profile: np.ndarray,
        tau_mad: float = 3.2,
        tau_step: Optional[float] = None,
        w_min_plateau: int = 10,
        max_window: int = 15,
    ) -> Dict[str, Any]:
        self.step_count += 1
        return morse_cnv_filter(
            profile,
            tau_mad=tau_mad,
            tau_step=tau_step,
            w_min_plateau=w_min_plateau,
            max_window=max_window,
        )

    def cell_cycle_deconfounder(
        self,
        matrix: np.ndarray,
        threshold: float = 0.85,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.step_count += 1
        return cell_cycle_deconfounder(matrix, clock_threshold=threshold)

    def genotype_tiebreaker_so10(
        self,
        u: np.ndarray,
        v: np.ndarray,
        W: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.step_count += 1
        return genotype_tiebreaker_so10(u, v, W)

    def pseudotime_spherical_cascade(
        self,
        v0: np.ndarray,
        u_seq: np.ndarray,
        gamma: float = 1.0,
    ) -> np.ndarray:
        self.step_count += 16
        return pseudotime_spherical_cascade(u_seq, v0, gamma=gamma)

    def spatial_tissue_sheaf_solver(
        self,
        dx: float,
        dy: float,
        nx: float,
        ny: float,
        delta_x_jitter: float = 0.0,
        huber_delta: float = 10.0,
    ) -> Tuple[float, float, Dict[str, Any]]:
        self.step_count += 1
        return spatial_tissue_sheaf_solver(
            dx, dy, nx, ny,
            delta_x_jitter=delta_x_jitter,
            huber_delta=huber_delta,
        )

    def multiomic_covariance_rotor(
        self,
        rot3: np.ndarray,
        cov6: Optional[np.ndarray] = None,
        m_miss: Optional[float] = None,
        shrink_alpha: Optional[float] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        self.step_count += 1
        return multiomic_covariance_rotor(
            rot3,
            cov6,
            m_miss=m_miss,
            shrink_alpha=shrink_alpha,
            mode=self.mode,
        )


__all__ = [
    "CellSnailEngine",
    # Canonical symbols
    "beltrami_morse_gate",
    "compute_spatial_mad_floor_256",
    "continuity_clutter_filter",
    "skew_triad_step",
    "slow_fast_cascade_k16",
    "contact_sheaf_se2_step",
    "kelvin_mandel_so6_rotor",
    "build_kelvin_mandel_so6_rotor",
    "rotate_stiffness_so6",
    # Stealth bioinformatics aliases
    "morse_dropout_filter",
    "morse_cnv_filter",
    "cell_cycle_deconfounder",
    "genotype_tiebreaker_so10",
    "pseudotime_spherical_cascade",
    "spatial_tissue_sheaf_solver",
    "multiomic_covariance_rotor",
    # Adaptive estimators & high-level wrappers
    "estimate_umi_depth",
    "sigmoidal_shrinkage_weight",
    "estimate_segment_widths",
    "estimate_spatial_jitter",
    "SE2CellularSheaf",
    "ContactEdge",
    "ContactNode",
    "estimate_missingness",
    "ledoit_wolf_shrinkage_so6",
    "SO6MultiModalBundle",
]
