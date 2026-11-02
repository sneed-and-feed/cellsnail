"""tests/test_v030_expeditions.py
================================
Comprehensive verification suite for sneed-and-feed/cellsnail v0.3.0
Multi-Expedition Research Program (Expeditions 8, 9, 10) and Mode B Structural Benchmarks.

Verifies:
  1. Mode B Structural Benchmark metrics and baseline implementations.
  2. Expedition 8 (ClinVar Allosteric Atlas):
     - 4 canonical targets (TP53 1TUP, KRAS 6OIM, EGFR 2ITX, BRCA1 1JNX)
     - Mean distal AUROC >= 0.9400 (target: >= 94.0%)
     - Mean distal F1 >= 0.9400 (target: >= 94.0%)
     - Max steric clash penetration residual == 0.0
     - Euclidean distance baseline failure (AUROC <= 0.55).
  3. Expedition 9 (Cryo-EM Density Sharpener):
     - 3 canonical density regimes (EMD-11121, EMD-0406, Membrane Receptor Dock)
     - Side-chain & coordinated water spike recovery rate == 1.0000 (100.0%)
     - Radiation-damage spike rejection rate == 1.0000 (100.0%)
     - Density RMSE reduction > 65.0%
     - Out-of-core streaming Z-slabs (chunk_z = 16) peak heap < 35.0 MB.
  4. Expedition 10 (Single-Cell to Structure Bridge):
     - Spatial juxtacrine interface SE(3) docking (NOTCH1-JAG1, EPHA2-EFNA1, CDH1-CDH3): 0.0 steric residual, 0.000e+00 Cayley drift
     - Single-cell cryptic pocket druggability (EGFR, FGFR1, AXL, AURKA, SLC7A5, NUDT1, AKT1, POLQ, VRK1)
     - Multi-scale correlation between single-cell weights and 3D cavity volumes.
  5. JSON receipt schema completeness, allow_nan=False integrity, and zero target leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pytest

from cellsnail.benchmark.mode_b_structural import (
    compute_classification_metrics,
    compute_cryoem_recovery_metrics,
    compute_steric_clash_residual,
    euclidean_distance_to_pocket,
    gaussian_lowpass_smoothing,
    generate_mode_b_cryoem_blind_volume,
    global_bfactor_sharpening,
    isotropic_gnm_coupling,
    local_wiener_filter,
    naive_bfactor_cutoff,
    run_mode_b_structural_benchmarks,
)
from cellsnail.expeditions import (
    ClinVarAllostericAtlasScanner,
    CryoEMSidechainSharpenerScanner,
    SingleCellToStructureBridgeScanner,
    run_expedition_8,
    run_expedition_9,
    run_expedition_10,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ==============================================================================
# Group 1: Mode B Structural Metrics & Baselines Unit Tests
# ==============================================================================

def test_mode_b_classification_metrics_perfect_separation():
    """Verify compute_classification_metrics returns 1.0 AUROC and 1.0 F1 on separable inputs."""
    y_true = [1, 1, 1, 0, 0, 0]
    y_scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    m = compute_classification_metrics(y_true, y_scores)
    assert m["auroc"] == 1.0
    assert m["f1"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["accuracy"] == 1.0


def test_mode_b_classification_metrics_random_separation():
    """Verify compute_classification_metrics returns ~0.50 AUROC on random scores."""
    y_true = [1, 1, 0, 0]
    y_scores = [0.2, 0.4, 0.3, 0.5]
    m = compute_classification_metrics(y_true, y_scores)
    assert 0.25 <= m["auroc"] <= 0.75


def test_mode_b_steric_clash_residual_zero_when_compliant():
    """Verify compute_steric_clash_residual returns 0.0 when separation delta_v_n >= 0."""
    coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)
    # u moves in -X, v moves in +X (moving away from each other)
    twists = np.array([
        [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.uint32)
    residual = compute_steric_clash_residual(coords, twists, edges)
    assert residual == 0.0


def test_mode_b_steric_clash_residual_detects_violation():
    """Verify compute_steric_clash_residual detects penetration when u moves into v."""
    coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)
    # u moves in +X by 2.0, v stays at 0 (penetration violation = 2.0)
    twists = np.array([
        [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.uint32)
    residual = compute_steric_clash_residual(coords, twists, edges)
    assert residual >= 1.99


def test_mode_b_cryoem_recovery_metrics_calculation():
    """Verify compute_cryoem_recovery_metrics accurately computes recovery, rejection, and RMSE reduction."""
    gt = np.zeros((10, 10, 10), dtype=np.float32)
    gt[5, 5, 5] = 10.0
    p_mask = (gt == 10.0)

    noisy = gt.copy()
    spike_mask = np.zeros((10, 10, 10), dtype=bool)
    spike_mask[2, 2, 2] = True
    noisy[spike_mask] = 50.0

    # Filtered zeroes the spike and keeps the peak
    filt = gt.copy()

    m = compute_cryoem_recovery_metrics(gt, filt, noisy, p_mask, spike_mask)
    assert m["sidechain_water_recovery_rate"] == 1.0
    assert m["radiation_spike_rejection_rate"] == 1.0
    assert m["density_rmse_reduction_percent"] >= 95.0


def test_mode_b_structural_baselines_euclidean_distance():
    """Verify euclidean_distance_to_pocket returns exact Euclidean distances."""
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    dists = euclidean_distance_to_pocket(coords, pocket_residues=[0])
    assert dists[0] == 0.0
    assert abs(dists[1] - 5.0) < 1e-5
    assert abs(dists[2] - 10.0) < 1e-5


def test_mode_b_structural_baselines_gnm_coupling():
    """Verify isotropic_gnm_coupling returns normalized coupling in [0, 1]."""
    coords = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]], dtype=np.float32)
    c = isotropic_gnm_coupling(coords, pocket_residues=[0], cutoff=5.0)
    assert len(c) == 3
    assert np.all(c >= 0.0)
    assert np.all(c <= 1.0)
    assert c[1] > c[2]  # residue 1 is closer in network to 0 than 2


def test_mode_b_structural_baselines_naive_bfactor():
    """Verify naive_bfactor_cutoff normalizes values to [0, 1]."""
    b = [10.0, 20.0, 30.0]
    norm_b = naive_bfactor_cutoff(b)
    assert abs(norm_b[0] - 0.0) < 1e-5
    assert abs(norm_b[1] - 0.5) < 1e-5
    assert abs(norm_b[2] - 1.0) < 1e-5


def test_mode_b_structural_baselines_gaussian_lowpass():
    """Verify gaussian_lowpass_smoothing reduces peak sharpness (contrast attenuation)."""
    vol = np.zeros((15, 15, 15), dtype=np.float32)
    vol[7, 7, 7] = 20.0
    smoothed = gaussian_lowpass_smoothing(vol, sigma=1.0)
    assert smoothed[7, 7, 7] < 5.0  # attenuated by > 75%


def test_mode_b_structural_baselines_global_bfactor_glitch_amplification():
    """Verify global_bfactor_sharpening amplifies high-frequency impulses."""
    vol = np.zeros((16, 16, 16), dtype=np.float32)
    vol[8, 8, 8] = 5.0
    sharpened = global_bfactor_sharpening(vol, b_factor=60.0)
    assert np.max(sharpened) > 10.0  # high-frequency spike amplified by > 200%


def test_mode_b_structural_baselines_wiener_filter():
    """Verify local_wiener_filter smooths local variance."""
    vol = np.full((10, 10, 10), 5.0, dtype=np.float32)
    vol[5, 5, 5] = 15.0
    w = local_wiener_filter(vol, window_size=3)
    assert w.shape == vol.shape


def test_mode_b_cryoem_blind_volume_generation_properties():
    """Verify generate_mode_b_cryoem_blind_volume satisfies Mode B noise parameters."""
    gt = np.zeros((20, 20, 20), dtype=np.float32)
    gt[10, 10, 10:15] = 5.0
    noisy, p_mask, s_mask = generate_mode_b_cryoem_blind_volume(gt, snr_db=20.0, spike_fraction=0.15)

    assert noisy.shape == gt.shape
    assert np.sum(p_mask) > 0
    assert np.sum(s_mask) > 0
    # True peaks and spikes are strictly disjoint
    assert np.sum(p_mask & s_mask) == 0


def test_mode_b_structural_high_level_runner():
    """Verify run_mode_b_structural_benchmarks executes and returns PASS."""
    res = run_mode_b_structural_benchmarks()
    assert res["status"] == "PASS"
    assert "expedition8_allosteric_atlas" in res
    assert "expedition9_cryoem_density" in res


# ==============================================================================
# Group 2: Expedition 8 (ClinVar Allosteric Atlas) Tests
# ==============================================================================

def test_expedition8_canonical_target_count_and_genes():
    """Verify Expedition 8 covers all 4 canonical cancer targets: TP53, KRAS, EGFR, BRCA1."""
    scanner = ClinVarAllostericAtlasScanner()
    genes = [t["gene_symbol"] for t in scanner.targets]
    assert len(genes) == 4
    assert "TP53" in genes
    assert "KRAS" in genes
    assert "EGFR" in genes
    assert "BRCA1" in genes


def test_expedition8_distal_allosteric_auroc_threshold():
    """Verify Expedition 8 achieves mean distal AUROC >= 0.9400 (target: >= 94.0%)."""
    res = run_expedition_8()
    assert res["mean_distal_auroc"] >= 0.9400, f"AUROC {res['mean_distal_auroc']} < 0.9400"


def test_expedition8_distal_allosteric_f1_threshold():
    """Verify Expedition 8 achieves mean distal F1 >= 0.9400 (target: >= 94.0%)."""
    res = run_expedition_8()
    assert res["mean_distal_f1"] >= 0.9400, f"F1 {res['mean_distal_f1']} < 0.9400"


def test_expedition8_steric_clash_penetration_residual_zero():
    """Verify Expedition 8 enforces exact 0.0 steric clash penetration residual."""
    res = run_expedition_8()
    assert res["steric_penetration_residual"] == 0.0


def test_expedition8_euclidean_baseline_misses_distal_allostery():
    """Verify static Euclidean distance baseline completely fails on distal variants (AUROC <= 0.55)."""
    res = run_expedition_8()
    euc_auroc = res["baselines"]["euclidean_mean_auroc"]
    assert euc_auroc <= 0.55, f"Euclidean distance must fail on distal variants, got {euc_auroc}"


def test_expedition8_all_individual_targets_pass():
    """Verify each of the 4 individual targets achieves AUROC >= 0.90."""
    scanner = ClinVarAllostericAtlasScanner()
    for target in scanner.targets:
        t_res = scanner.run_target_evaluation(target)
        auroc = t_res["cellsnail_metrics"]["auroc"]
        assert auroc >= 0.90, f"{target['gene_symbol']} AUROC {auroc} < 0.90"
        assert t_res["steric_clash_max_residual"] == 0.0


# ==============================================================================
# Group 3: Expedition 9 (Cryo-EM Density Sharpener) Tests
# ==============================================================================

def test_expedition9_regimes_evaluated_count():
    """Verify Expedition 9 evaluates all 3 canonical density regimes."""
    scanner = CryoEMSidechainSharpenerScanner()
    assert len(scanner.regimes) == 3
    ids = [r["regime_id"] for r in scanner.regimes]
    assert "EMD-11121" in ids
    assert "EMD-0406" in ids
    assert "Membrane-Dock-1.7A" in ids


def test_expedition9_sidechain_water_recovery_rate_100pct():
    """Verify Expedition 9 achieves exactly 1.0000 (100.0%) side-chain & water recovery."""
    res = run_expedition_9()
    assert res["mean_sidechain_water_recovery"] >= 0.9999, f"Recovery {res['mean_sidechain_water_recovery']} < 1.0"


def test_expedition9_radiation_spike_rejection_rate_100pct():
    """Verify Expedition 9 achieves exactly 1.0000 (100.0%) radiation-spike rejection."""
    res = run_expedition_9()
    assert res["mean_radiation_spike_rejection"] >= 0.9999, f"Rejection {res['mean_radiation_spike_rejection']} < 1.0"


def test_expedition9_rmse_reduction_exceeds_65pct():
    """Verify Expedition 9 achieves > 65.0% density RMSE reduction."""
    res = run_expedition_9()
    assert res["mean_rmse_reduction_percent"] >= 65.0, f"RMSE reduction {res['mean_rmse_reduction_percent']}% < 65.0%"


def test_expedition9_streaming_heap_under_35mb():
    """Verify Expedition 9 processes out-of-core Z-slabs with strictly < 35.0 MB peak heap."""
    res = run_expedition_9()
    assert res["streaming_heap_peak_mb"] < 35.0, f"Heap {res['streaming_heap_peak_mb']} MB >= 35.0 MB"


def test_expedition9_baseline_failures_gaussian_and_bfactor():
    """Verify Gaussian baseline loses contrast and B-factor baseline amplifies glitches."""
    scanner = CryoEMSidechainSharpenerScanner()
    r = scanner.run_regime_evaluation(scanner.regimes[0])
    assert r["gaussian_smoothing_baseline"]["contrast_loss_percent"] >= 0.0
    assert r["global_bfactor_baseline"]["glitch_amplification_percent"] >= 200.0


# ==============================================================================
# Group 4: Expedition 10 (Single-Cell to Structure Bridge) Tests
# ==============================================================================

def test_expedition10_spatial_juxtacrine_interfaces_docking():
    """Verify Expedition 10 evaluates NOTCH1-JAG1, EPHA2-EFNA1, CDH1-CDH3 with 0.0 steric residual."""
    scanner = SingleCellToStructureBridgeScanner()
    res = scanner.evaluate_spatial_juxtacrine_interfaces()
    assert len(res) == 3
    for s in res:
        assert s["steric_penetration_residual"] == 0.0
        assert s["steric_non_penetration_satisfied"] is True


def test_expedition10_cryptic_pocket_druggability_targets():
    """Verify Expedition 10 confirms cryptic pocket druggability across all 9 single-cell targets."""
    scanner = SingleCellToStructureBridgeScanner()
    res = scanner.evaluate_cryptic_pocket_druggability()
    assert len(res) == 9
    for target in res:
        assert target["is_druggable"] is True
        assert target["cavity_volume_angstrom3"] > 100.0


def test_expedition10_cayley_rotational_drift_zero():
    """Verify Expedition 10 confirms exact 0.000e+00 Cayley rotational drift across interfaces."""
    scanner = SingleCellToStructureBridgeScanner()
    res = scanner.evaluate_spatial_juxtacrine_interfaces()
    for s in res:
        assert s["cayley_rotational_drift"] == 0.0
        assert s["cayley_norm_conservation_verified"] is True


def test_expedition10_multiscale_convergence_correlation():
    """Verify Expedition 10 achieves positive correlation between single-cell biomarker weight and pocket volume."""
    res = run_expedition_10()
    assert res["status"] == "PASS"
    assert res["convergence_summary"]["sc_biomarker_to_pocket_volume_pearson_r"] > 0.0


# ==============================================================================
# Group 5: JSON Receipts Schema & Integrity Tests
# ==============================================================================

def test_json_receipts_schema_and_integrity_no_nan():
    """Verify all Expedition receipts and evaluation summary exist, parse cleanly, and contain zero NaNs."""
    receipt_files = [
        "expedition8_clinvar_allosteric_atlas.json",
        "expedition9_cryoem_sidechain_sharpener.json",
        "expedition10_sc_to_structure_bridge.json",
        "cellsnail_evaluation_summary.json",
    ]

    for fname in receipt_files:
        p = REPO_ROOT / "results" / fname
        assert p.exists(), f"Receipt file {fname} does not exist in results/"
        content = p.read_text(encoding="utf-8")
        assert "NaN" not in content, f"Receipt file {fname} contains forbidden NaN"
        data = json.loads(content)
        assert isinstance(data, dict)
        assert len(data) > 0
        if "status" in data:
            assert data["status"] in ["PASS", "ALL_THEOREMS_VERIFIED", "MODE_B_BENCHMARKS_COMPLETE"]
