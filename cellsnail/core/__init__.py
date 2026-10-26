"""cellsnail.core
==================
Core mathematical primitives, functors, and adaptive hybrid operators
for single-cell data science in the DarkSnail / triad-core ecosystem (v0.2.0).
"""

from cellsnail.core.morse_filter import (
    beltrami_morse_gate,
    compute_spatial_mad_floor_256,
    estimate_segment_widths,
    estimate_umi_depth,
    morse_cnv_filter,
    morse_dropout_filter,
    sigmoidal_shrinkage_weight,
)
from cellsnail.core.deconfounder import (
    cell_cycle_deconfounder,
    continuity_clutter_filter,
)
from cellsnail.core.triad_step import (
    genotype_tiebreaker_so10,
    skew_triad_step,
)
from cellsnail.core.cascade import (
    pseudotime_spherical_cascade,
    slow_fast_cascade_k16,
)
from cellsnail.core.sheaf_aligner import (
    ContactEdge,
    ContactNode,
    SE2CellularSheaf,
    contact_sheaf_se2_step,
    estimate_spatial_jitter,
    spatial_tissue_sheaf_solver,
)
from cellsnail.core.so6_bundle import (
    SO6MultiModalBundle,
    build_kelvin_mandel_so6_rotor,
    estimate_missingness,
    kelvin_mandel_so6_rotor,
    ledoit_wolf_shrinkage_so6,
    multiomic_covariance_rotor,
    rotate_stiffness_so6,
)

__all__ = [
    # Operator 1: Morse Dropout Filter & Beltrami Gate
    "beltrami_morse_gate",
    "morse_dropout_filter",
    "morse_cnv_filter",
    "estimate_umi_depth",
    "estimate_segment_widths",
    "sigmoidal_shrinkage_weight",
    "compute_spatial_mad_floor_256",
    # Operator 2: Continuity Clutter & Cell Cycle
    "continuity_clutter_filter",
    "cell_cycle_deconfounder",
    # Operator 3: Skew Triad Step & Genotype Tiebreaker
    "skew_triad_step",
    "genotype_tiebreaker_so10",
    # Operator 4: Slow-Fast Cascade & Pseudotime Spherical Cascade
    "slow_fast_cascade_k16",
    "pseudotime_spherical_cascade",
    # Operator 5: Sheaf Aligner
    "contact_sheaf_se2_step",
    "spatial_tissue_sheaf_solver",
    "SE2CellularSheaf",
    "ContactEdge",
    "ContactNode",
    "estimate_spatial_jitter",
    # Operator 6: SO(6) Bundle & Rotor
    "build_kelvin_mandel_so6_rotor",
    "rotate_stiffness_so6",
    "kelvin_mandel_so6_rotor",
    "multiomic_covariance_rotor",
    "SO6MultiModalBundle",
    "ledoit_wolf_shrinkage_so6",
    "estimate_missingness",
]
