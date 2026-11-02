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
    verify_transverse_sheaf_curvature_invariance,
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
from cellsnail.benchmark.mode_b_structural import (
    compute_classification_metrics,
    compute_steric_clash_residual,
    compute_cryoem_recovery_metrics,
    euclidean_distance_to_pocket,
    isotropic_gnm_coupling,
    naive_bfactor_cutoff,
    global_bfactor_sharpening,
    gaussian_lowpass_smoothing,
    local_wiener_filter,
    generate_mode_b_cryoem_blind_volume,
    run_mode_b_structural_benchmarks,
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
    "verify_transverse_sheaf_curvature_invariance",
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
    # Mode B Structural
    "compute_classification_metrics",
    "compute_steric_clash_residual",
    "compute_cryoem_recovery_metrics",
    "euclidean_distance_to_pocket",
    "isotropic_gnm_coupling",
    "naive_bfactor_cutoff",
    "global_bfactor_sharpening",
    "gaussian_lowpass_smoothing",
    "local_wiener_filter",
    "generate_mode_b_cryoem_blind_volume",
    "run_mode_b_structural_benchmarks",
]

