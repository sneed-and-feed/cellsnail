"""cellsnail.benchmark
===================
Benchmark baselines, Mode A oracle verification, and Mode B blind cross-validation.
"""

from cellsnail.benchmark.baselines import (
    magic_impute,
    alra_impute,
    hmmcopy_infercnv_smooth,
    neighbor_joining,
    upgma,
)
from cellsnail.benchmark.mode_a_oracle import (
    run_mode_a_oracle_audit,
    verify_triad_orthogonality,
    verify_skew_antisymmetry,
    verify_step0_homotopy,
    verify_discrete_energy_identity,
    verify_rope_isometric_invariance,
    verify_cascade_norm_conservation,
    verify_kelvin_mandel_trace,
    verify_cayley_transform_unitarity,
    verify_ideal_mhd_transverse_invariance,
    verify_cellular_sheaf_coboundary,
    verify_bcm2712_arena_geometry,
    verify_morse_dropout_shielding,
    verify_h_clock_h_nope_orthogonality,
    verify_ultrametric_site_condensation,
    verify_multiomic_dosage_linkage,
)
from cellsnail.benchmark.mode_b_blind import (
    binomial_count_split,
    run_mode_b_blind_cross_validation,
    benchmark_ercc_spurious_correlation,
    benchmark_rare_persister_subclone_recovery,
    benchmark_cnv_edge_fidelity,
    benchmark_spatial_boundary_penetration,
)

__all__ = [
    # Baselines
    "magic_impute",
    "alra_impute",
    "hmmcopy_infercnv_smooth",
    "neighbor_joining",
    "upgma",
    # Mode A
    "run_mode_a_oracle_audit",
    "verify_triad_orthogonality",
    "verify_skew_antisymmetry",
    "verify_step0_homotopy",
    "verify_discrete_energy_identity",
    "verify_rope_isometric_invariance",
    "verify_cascade_norm_conservation",
    "verify_kelvin_mandel_trace",
    "verify_cayley_transform_unitarity",
    "verify_ideal_mhd_transverse_invariance",
    "verify_cellular_sheaf_coboundary",
    "verify_bcm2712_arena_geometry",
    "verify_morse_dropout_shielding",
    "verify_h_clock_h_nope_orthogonality",
    "verify_ultrametric_site_condensation",
    "verify_multiomic_dosage_linkage",
    # Mode B
    "binomial_count_split",
    "run_mode_b_blind_cross_validation",
    "benchmark_ercc_spurious_correlation",
    "benchmark_rare_persister_subclone_recovery",
    "benchmark_cnv_edge_fidelity",
    "benchmark_spatial_boundary_penetration",
]
