"""tests/test_mode_b_blind.py
===========================
Mode B: Zero-Leakage Blind Cross-Validation & Baseline Comparison Tests.
"""

import numpy as np
import pytest

from cellsnail.benchmark.mode_b_blind import (
    apply_snr_noise,
    apply_technical_dropout,
    benchmark_cnv_edge_fidelity,
    benchmark_ercc_spurious_correlation,
    benchmark_rare_persister_subclone_recovery,
    benchmark_spatial_boundary_penetration,
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
    # Center by mean
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

    # cellsnail maintains ~100% peak amplitude retention
    assert res["amplitude_retention_ratio"]["cellsnail"] >= 0.90
    assert res["adjacent_boundary_leakage"]["cellsnail"] <= 0.10

    # HMMcopy/inferCNV windowed smoothing smears focal peak into adjacent bins
    assert res["amplitude_retention_ratio"]["hmmcopy_infercnv"] < 0.60
    assert res["adjacent_boundary_leakage"]["hmmcopy_infercnv"] > 0.30


def test_spatial_boundary_penetration_residual():
    res = benchmark_spatial_boundary_penetration(random_state=42)

    # cellsnail sheaf non-penetration enforces 0 boundary penetration
    assert res["cellsnail_sheaf"]["normal_penetration_residual"] == 0.0
    assert res["cellsnail_sheaf"]["bleed_rate"] == 0.0

    # Isotropic smoothing bleeds across cell borders
    assert res["isotropic_smoothing"]["normal_penetration_residual"] > 0.0
    assert res["isotropic_smoothing"]["bleed_rate"] > 0.0


def test_run_mode_b_blind_cross_validation_sweeps():
    # Run targeted sweep
    sweep_res = run_mode_b_blind_cross_validation(
        snr_levels=[np.inf, 20.0, 10.0],
        dropout_levels=[0.0, 0.10],
        random_state=42,
    )
    assert sweep_res["status"] == "MODE_B_BENCHMARKS_COMPLETE"
    assert len(sweep_res["ercc_spurious_correlation_sweep"]) == 6
    assert len(sweep_res["rare_persister_subclone_sweep"]) == 6
    assert len(sweep_res["cnv_edge_fidelity_sweep"]) == 6
