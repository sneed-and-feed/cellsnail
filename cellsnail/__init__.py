"""cellsnail: Standalone, zero-dependency, proof-carrying multi-omic single-cell data science spoke repository.

Exposes canonical DarkSnail mathematical symbols and stealth single-cell bioinformatics aliases.
"""

from cellsnail.operators import (
    CellSnailEngine,
    # Canonical DarkSnail names
    beltrami_morse_gate,
    compute_spatial_mad_floor_256,
    continuity_clutter_filter,
    skew_triad_step,
    slow_fast_cascade_k16,
    contact_sheaf_se2_step,
    kelvin_mandel_so6_rotor,
    build_kelvin_mandel_so6_rotor,
    rotate_stiffness_so6,
    # Stealth single-cell bioinformatics aliases
    morse_dropout_filter,
    morse_cnv_filter,
    cell_cycle_deconfounder,
    genotype_tiebreaker_so10,
    pseudotime_spherical_cascade,
    spatial_tissue_sheaf_solver,
    multiomic_covariance_rotor,
    # Adaptive estimators & classes
    SE2CellularSheaf,
    SO6MultiModalBundle,
    estimate_umi_depth,
    estimate_segment_widths,
    estimate_spatial_jitter,
    estimate_missingness,
    ledoit_wolf_shrinkage_so6,
    sigmoidal_shrinkage_weight,
)
from cellsnail.io import (
    SingleCellContainer,
    read_mtx,
    write_mtx,
    read_vcf,
    write_vcf,
    read_csv,
    write_csv,
    read_h5ad,
    write_h5ad,
)
from cellsnail.discovery import (
    DiscoveryEngine,
    run_discovery_pipeline,
)
from cellsnail.benchmark import (
    binomial_count_split,
    run_mode_a_oracle_audit,
    run_mode_b_blind_cross_validation,
)

__version__ = "0.2.0"
__author__ = "Noah Cohen"
__license__ = "CC0-1.0"

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
    "SE2CellularSheaf",
    "SO6MultiModalBundle",
    "estimate_umi_depth",
    "estimate_segment_widths",
    "estimate_spatial_jitter",
    "estimate_missingness",
    "ledoit_wolf_shrinkage_so6",
    "sigmoidal_shrinkage_weight",
    # I/O
    "SingleCellContainer",
    "read_mtx",
    "write_mtx",
    "read_vcf",
    "write_vcf",
    "read_csv",
    "write_csv",
    "read_h5ad",
    "write_h5ad",
    # Discovery
    "DiscoveryEngine",
    "run_discovery_pipeline",
    # Benchmarks
    "binomial_count_split",
    "run_mode_a_oracle_audit",
    "run_mode_b_blind_cross_validation",
    "__version__",
]
