"""tests/test_mode_b_blind.py
===========================
Mode B: Zero-Leakage Blind Cross-Validation & Baseline Comparison Tests.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from cellsnail.benchmark.mode_b_blind import (
    apply_snr_noise,
    apply_technical_dropout,
    benchmark_cnv_edge_fidelity,
    benchmark_cnv_width_resolution_crossover,
    benchmark_ercc_spurious_correlation,
    benchmark_loco_lodo_lofo_cross_validation,
    benchmark_permutation_and_missing_modality_crossover,
    benchmark_rare_persister_subclone_recovery,
    benchmark_spatial_boundary_penetration,
    benchmark_spatial_jitter_crossover_sweep,
    benchmark_umi_depth_crossover_sweep,
    binomial_count_split,
    run_mode_b_blind_cross_validation,
)


def test_binomial_count_split_properties():
    rng = np.random.default_rng(42)
    X = rng.poisson(lam=10.0, size=(100, 50)).astype(np.float32)

    X1, X2 = binomial_count_split(X, p=0.5, random_state=42)

    # 1. Exact sum preservation
    np.testing.assert_allclose(X1 + X2, X, atol=1e-5)

    # 2. Non-negativity
    assert np.all(X1 >= 0)
    assert np.all(X2 >= 0)

    # 3. Independent splits: correlation between independent Poisson splits given mean is ~0
    X1_cent = X1 - np.mean(X1, axis=0)
    X2_cent = X2 - np.mean(X2, axis=0)
    cov = np.mean(X1_cent * X2_cent)
    assert np.abs(cov) < 1.0


def test_ercc_spurious_correlation_zero_guarantee():
    res = benchmark_ercc_spurious_correlation(random_state=42)

    # cellsnail injects 0 spurious pairs
    assert res["spurious_pairs"]["cellsnail"] == 0
    assert res["spurious_rate"]["cellsnail"] == 0.0

    # MAGIC graph diffusion injects spurious correlations
    assert res["spurious_pairs"]["magic"] > 0
    assert res["spurious_rate"]["magic"] > 0.0
    assert "unpermuted_capture_efficiency_audit" in res


def test_rare_persister_subclone_recovery_f1():
    res = benchmark_rare_persister_subclone_recovery(random_state=42)

    # Persister prevalence < 0.5%
    assert res["persister_fraction"] < 0.005

    # cellsnail achieves F1 > 0.90
    assert res["f1_scores"]["cellsnail"] >= 0.90
    assert res["cellsnail_metrics"]["recall"] == 1.0

    # Baseline clustering and ALRA fail to isolate the rare 0.4% persisters
    assert res["f1_scores"]["baseline_clustering"] < 0.50
    assert res["f1_scores"]["alra"] < 0.50


def test_cnv_edge_fidelity_focal_amplicons():
    res = benchmark_cnv_edge_fidelity(random_state=42)

    # cellsnail maintains ~100% peak amplitude retention and bounded fidelity in [0, 1]
    assert res["amplitude_retention_ratio"]["cellsnail"] >= 0.90
    assert 0.85 <= res["bounded_edge_gradient_fidelity"]["cellsnail"] <= 1.0
    assert res["adjacent_boundary_leakage"]["cellsnail"] <= 0.15

    # HMMcopy/inferCNV windowed smoothing smears focal peak into adjacent bins
    assert res["amplitude_retention_ratio"]["hmmcopy_infercnv"] < 0.60
    assert res["bounded_edge_gradient_fidelity"]["hmmcopy_infercnv"] < 0.45
    assert res["adjacent_boundary_leakage"]["hmmcopy_infercnv"] > 0.30


def test_spatial_boundary_penetration_residual():
    res = benchmark_spatial_boundary_penetration(random_state=42)

    # cellsnail sheaf non-penetration enforces 0 boundary penetration
    assert res["cellsnail_sheaf"]["normal_penetration_residual"] == 0.0
    assert res["cellsnail_sheaf"]["bleed_rate"] == 0.0

    # Isotropic smoothing bleeds across cell borders
    assert res["isotropic_smoothing"]["normal_penetration_residual"] > 0.0
    assert res["isotropic_smoothing"]["bleed_rate"] > 0.0


def test_failure_mode_crossover_sweeps():
    umi_sweep = benchmark_umi_depth_crossover_sweep(random_state=42)
    assert len(umi_sweep) == 5
    # At full depth (p=1.0), single-cell focal recall is high; at p=0.05 it degrades
    assert umi_sweep[0]["single_cell_focal_egfr_recall"] > umi_sweep[-1]["single_cell_focal_egfr_recall"]
    # At ultra-shallow depth (p=0.05), ALRA pools across 20 genes for higher broad module R^2
    assert umi_sweep[-1]["broad_module_r2"]["alra_svd"] >= umi_sweep[-1]["broad_module_r2"]["raw_thinned"]

    cnv_sweep = benchmark_cnv_width_resolution_crossover(random_state=42)
    assert len(cnv_sweep) == 6
    # Narrow 1-bin focal stringer: cellsnail dominates HMMcopy
    assert cnv_sweep[0]["bounded_edge_fidelity"]["cellsnail_morse"] > cnv_sweep[0]["bounded_edge_fidelity"]["hmmcopy_infercnv"]
    # Broad 15-bin plateau: HMMcopy reduces interior plateau RMSE
    assert cnv_sweep[-1]["single_cell_plateau_rmse"]["hmmcopy_infercnv"] < cnv_sweep[-1]["single_cell_plateau_rmse"]["cellsnail_unsmoothed_interior"]

    jit_sweep = benchmark_spatial_jitter_crossover_sweep(random_state=42)
    assert len(jit_sweep) == 5
    # At 0 um jitter, sheaf clamping beats isotropic smoothing
    assert jit_sweep[0]["compartment_marker_accuracy"]["cellsnail_sheaf"] >= jit_sweep[0]["compartment_marker_accuracy"]["isotropic_smoothing"]
    # At 50 um severe mis-segmentation jitter, double-penalty degrades sheaf below 0 um accuracy
    assert jit_sweep[-1]["compartment_marker_accuracy"]["cellsnail_sheaf"] < jit_sweep[0]["compartment_marker_accuracy"]["cellsnail_sheaf"]


def test_loco_lodo_lofo_cross_validation():
    cv = benchmark_loco_lodo_lofo_cross_validation(random_state=42)
    assert len(cv["loco_sc_mixology_folds"]) == 3
    assert len(cv["lodo_gbm_patient_folds"]) == 5
    assert len(cv["lofo_cscc_spatial_folds"]) == 4
    for fold in cv["loco_sc_mixology_folds"]:
        assert fold["cellsnail_spurious_ercc_pairs"] == 0
    for fold in cv["lodo_gbm_patient_folds"]:
        assert fold["max_orthogonality_residual"] < 1e-12
    for fold in cv["lofo_cscc_spatial_folds"]:
        assert fold["cellsnail_max_penetration_residual"] == 0.0


def test_permutation_and_missing_modality_crossover():
    res = benchmark_permutation_and_missing_modality_crossover(random_state=42)
    perm_audit = res["column_permutation_collision_audit"]
    assert perm_audit["adversarial_adjacent_amplicons_recall"]["unsorted_1d_morse_gate"] <= 0.25
    assert perm_audit["adversarial_adjacent_amplicons_recall"]["chromosomal_sorted_morse_gate"] == 1.0
    assert perm_audit["random_10_permutations_mean_recall"]["chromosomal_sorted_morse_gate"] == 1.0

    m_sweep = res["missing_modality_ambient_soup_sweep"]
    assert len(m_sweep) == 5
    for row in m_sweep:
        assert row["relative_trace_residual"] < 1e-13
    # At 85% missing modality + heavy log-normal soup, shrinkage regularization beats unregularized rotor
    assert m_sweep[-1]["driver_subspace_cosine_similarity"]["shrinkage_regularized_rotor"] > m_sweep[-1]["driver_subspace_cosine_similarity"]["unregularized_so6_rotor"]


def test_empirical_wetlab_umi_integer_and_json_integrity():
    root = Path(__file__).resolve().parent.parent
    npz_path = root / "cellsnail" / "data" / "empirical" / "wetlab_cohorts.npz"
    assert npz_path.exists()
    with np.load(npz_path, allow_pickle=False) as z:
        for key in ("mix_X_human", "mix_X_ercc_perm", "cnv_sc_rna_expr", "st_X"):
            arr = z[key]
            assert np.all(arr >= 0.0)
            np.testing.assert_allclose(arr, np.round(arr), atol=1e-6)
        assert len(z["gbm_cells"]) == 543
        assert "" not in list(z["gbm_cells"])
        # Verify ENSG00000210082 is mapped to its true Ensembl GRCh38 symbol MT-RNR2
        ens_ids = list(z["mix_human_gene_ids"])
        syms = list(z["mix_human_gene_names"])
        idx = ens_ids.index("ENSG00000210082")
        assert syms[idx] == "MT-RNR2"

    summary_path = root / "results" / "cellsnail_evaluation_summary.json"
    if summary_path.exists():
        raw_text = summary_path.read_text(encoding="utf-8")
        parsed = json.loads(
            raw_text,
            parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Non-standard JSON constant: {c}")),
        )
        assert parsed["mode_b_blind_cross_validation"]["status"] == "MODE_B_BENCHMARKS_COMPLETE"


def test_run_mode_b_blind_cross_validation_sweeps():
    sweep_res = run_mode_b_blind_cross_validation(
        snr_levels=[np.inf, 20.0, 10.0],
        dropout_levels=[0.0, 0.10],
        random_state=42,
    )
    assert sweep_res["status"] == "MODE_B_BENCHMARKS_COMPLETE"
    assert len(sweep_res["ercc_spurious_correlation_sweep"]) == 6
    assert len(sweep_res["rare_persister_subclone_sweep"]) == 6
    assert len(sweep_res["cnv_edge_fidelity_sweep"]) == 6
    assert "loco_lodo_lofo_cross_validation" in sweep_res
    assert "failure_mode_crossover_sweeps" in sweep_res


