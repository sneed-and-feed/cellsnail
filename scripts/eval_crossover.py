import sys
sys.path.insert(0, ".")
from cellsnail.benchmark.mode_b_blind import (
    benchmark_umi_depth_crossover_sweep,
    benchmark_cnv_width_resolution_crossover,
    benchmark_spatial_jitter_crossover_sweep,
    benchmark_permutation_and_missing_modality_crossover,
)

u = benchmark_umi_depth_crossover_sweep(random_state=42)
print("--- UMI SWEEP ---")
for row in u:
    print(f"p={row['p_thin']:.2f}: ALRA R2={row['broad_module_r2']['alra_svd']:.4f}, v0.1.0 R2={row['broad_module_r2']['cellsnail_morse']:.4f}, v0.2.0 R2={row['broad_module_r2']['cellsnail_v020_adaptive']:.4f}, ERCC spurious: ALRA={row['ercc_spurious_pairs']['alra_svd']}, v0.2.0={row['ercc_spurious_pairs']['cellsnail_v020_adaptive']}")

c = benchmark_cnv_width_resolution_crossover(random_state=42)
print("--- CNV SWEEP ---")
for row in c:
    print(f"W={row['amplicon_width_bins']}: EdgeFid HMM={row['bounded_edge_fidelity']['hmmcopy_infercnv']:.4f}, v0.1={row['bounded_edge_fidelity']['cellsnail_morse']:.4f}, v0.2={row['bounded_edge_fidelity']['cellsnail_v020_adaptive']:.4f} -- RMSE HMM={row['single_cell_plateau_rmse']['hmmcopy_infercnv']:.4f}, v0.1={row['single_cell_plateau_rmse']['cellsnail_unsmoothed_interior']:.4f}, v0.2={row['single_cell_plateau_rmse']['cellsnail_v020_adaptive']:.4f}")

j = benchmark_spatial_jitter_crossover_sweep(random_state=42)
print("--- JITTER SWEEP ---")
for row in j:
    print(f"jit={row['coordinate_jitter_um']:.1f}: Iso={row['compartment_marker_accuracy']['isotropic_smoothing']:.4f}, Sheaf_v0.1={row['compartment_marker_accuracy']['cellsnail_sheaf']:.4f}, Sheaf_v0.2={row['compartment_marker_accuracy']['cellsnail_v020_adaptive']:.4f}")

m = benchmark_permutation_and_missing_modality_crossover(random_state=42)['missing_modality_ambient_soup_sweep']
print("--- MISSINGNESS SWEEP ---")
for row in m:
    print(f"m={row['missing_modality_block_rate']:.2f}: Unreg={row['driver_subspace_cosine_similarity']['unregularized_so6_rotor']}, Shrunk={row['driver_subspace_cosine_similarity']['shrinkage_regularized_rotor']}, v0.2={row['driver_subspace_cosine_similarity']['cellsnail_v020_adaptive']}, Tr_res={row['relative_trace_residual']:.2e}")
