"""
Unit, Mathematical & Adversarial Falsification Tests for Expedition 3:
Spatial Contact-Boundary & Multi-Modal Sheaf Audit (`exp3_spatial_border_audit.py`).

Tests 16 granular empirical, mathematical, and adversarial invariants:
  1. Strict JSON schema & allow_nan=False serialization integrity.
  2. Empirical GSE144239 (GSM4284316_P2_ST_rep1) 400-spot UMI integrity & layer profiles.
  3. Zero classical cadherin switch falsification (CDH2 = 0.0 UMI vs CDH1 = 1393.0, CDH3 = 2107.0).
  4. Multi-scale spatial contact-graph topology (d <= 35 um and d <= 55 um).
  5. 0-cochain 35 um isotropic boundary bleed inflation (+319.2% MMP10+COL1A1, +241.2% LAMC2+COL1A1, +417.0% KRT14+FAP) & STRING v12.0 null PPI verification.
  6. 0-cochain 55 um compartment-deconfounded 5-operator comparison (0.0% B_intra under SE2CellularSheaf vs up to 47.17% dual-positivity under isotropic smoothing).
  7. Theorem 2.18 registration jitter sweep (Delta_x in [0, 30] um) & elimination of the high-jitter double penalty.
  8. 1-cochain 28-pair boundary-clamped juxtacrine SBR sharpening (16.96x mean, up to 50.59x at eps=1e-2 and 4916.54x at eps=1e-4).
  9. 1-cochain 35 um raw-UMI directional trans-membrane polarity gain (2.03x to 38.76x across AREG->EGFR, COL1A1->CD44, TGFA->EGFR, JAG1->NOTCH3, TGFB1->TGFBR2, COL1A1->ITGB1, MMP10->ITGA6, LAMC2->ITGA6).
 10. Controlled synthetic ground-truth benchmark with lateral capture bleed (eta = 0.18, 1018.04x sharpening over isotropic).
 11. Adversarial Test 1: 200-permutation spatial spot-label null (z = 8.96 for J_sheaf, z = 175.40 for SBR_sheaf, p = 0.00498 < 0.01).
 12. Adversarial Test 2: 90-deg and 180-deg normal orientation rotation & graph clamping swap.
 13. Adversarial Test 3: Countsplit binomial thinning holdout (p = 0.50, Pearson r = 0.9918 / 0.9995 > 0.90 across 28 pairs).
 14. Adversarial Test 4: 4-fold Leave-One-FOV-Out (LOFO) spatial quadrant cross-validation (0.0% blind pseudo-EMT rate, 37.13x mean LAMC2->ITGA6 SBR sharpening).
 15. Theorem 2.17 SO(6) Kelvin-Mandel bundle trace conservation (<= 1.14e-13 across m_miss in [0.0, 0.85]) & Tri-Modal Cross-Expedition Convergence (Exp 1 cap Exp 2 cap Exp 3).
 16. Deterministic bit-exact reproducibility across independent auditor runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cellsnail.expeditions.exp3_spatial_border_audit import (
    SpatialBorderAuditor,
    run_expedition3_spatial_border_audit,
)


@pytest.fixture(scope="module")
def auditor_instance() -> SpatialBorderAuditor:
    """Instantiate SpatialBorderAuditor once per module for fast array-level inspection."""
    return SpatialBorderAuditor(random_state=42)


@pytest.fixture(scope="module")
def exp3_results() -> dict:
    """Run the full Expedition 3 audit once per test module and verify canonical JSON output."""
    repo_root = Path(__file__).resolve().parents[1]
    canonical_json = repo_root / "results" / "expedition3_spatial_border_audit.json"
    res = run_expedition3_spatial_border_audit(random_state=42, output_path=canonical_json)
    assert canonical_json.exists(), "Canonical JSON result file was not created."
    return res


def test_01_json_schema_and_strict_nan_free_serialization(exp3_results: dict) -> None:
    """Verify strict JSON schema, top-level metadata, and absence of NaN/Infinity tokens."""
    repo_root = Path(__file__).resolve().parents[1]
    canonical_json = repo_root / "results" / "expedition3_spatial_border_audit.json"
    raw_text = canonical_json.read_text(encoding="utf-8")

    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text
    assert "-Infinity" not in raw_text

    loaded = json.loads(raw_text)
    assert loaded["expedition_id"] == "Expedition_3_Spatial_Contact_Boundary_And_MultiModal_Sheaf_Audit"
    assert loaded["version"] == "0.2.0"
    assert loaded["random_state"] == 42
    assert loaded["all_adversarial_tests_passed"] is True

    required_sections = [
        "executive_summary",
        "section1_empirical_cohort_characterization",
        "section2_zero_cochain_pseudo_emt_audit",
        "section3_one_cochain_juxtacrine_signaling_audit",
        "section4_multimodal_so6_and_cross_expedition_bridge",
        "section5_adversarial_falsification_suite",
    ]
    for sec_key in required_sections:
        assert sec_key in loaded, f"Missing required top-level section: {sec_key}"

    # Verify strict re-serialization with allow_nan=False
    reserialized = json.dumps(exp3_results, allow_nan=False, sort_keys=True)
    assert len(reserialized) > 10000


def test_02_gse144239_empirical_umi_integrity_and_compartment_stratification(
    exp3_results: dict, auditor_instance: SpatialBorderAuditor
) -> None:
    """Verify GSE144239 (GSM4284316_P2_ST_rep1) 400-spot UMI integrity and layer marker profiles."""
    sec1 = exp3_results["section1_empirical_cohort_characterization"]
    assert sec1["cohort_id"] == "GSE144239_GSM4284316_P2_ST_rep1"
    assert sec1["n_spots"] == 400
    assert sec1["n_primary_genes"] == 50
    assert sec1["n_total_audited_genes"] == 78
    assert sec1["raw_umi_discrepancy_vs_geo_tsv"] == 0.0
    assert sec1["estimated_spatial_lattice_jitter_um"] == pytest.approx(5.6365, abs=1e-3)

    # Verify raw UMI matrix non-negativity and exact integer count properties
    X_raw = auditor_instance.X_all_raw
    assert X_raw.shape == (400, 78)
    assert np.all(np.isfinite(X_raw))
    assert np.all(X_raw >= 0.0)
    assert np.allclose(X_raw, np.round(X_raw), atol=1e-5)

    # Verify histopathologically stratified compartment enrichments across Layer_1..Layer_4
    layers = sec1["layer_compartment_profiles"]
    for l_name in ["Layer_1", "Layer_2", "Layer_3", "Layer_4"]:
        assert layers[l_name]["n_spots"] == 100

    assert layers["Layer_1"]["mean_raw_umi"]["KRT14"] == pytest.approx(316.39, abs=0.05)
    assert layers["Layer_1"]["mean_raw_umi"]["KRT5"] == pytest.approx(206.52, abs=0.05)
    assert layers["Layer_2"]["mean_raw_umi"]["KRT1"] == pytest.approx(78.06, abs=0.05)
    assert layers["Layer_2"]["mean_raw_umi"]["KRT10"] == pytest.approx(81.58, abs=0.05)
    assert layers["Layer_3"]["mean_raw_umi"]["MMP10"] == pytest.approx(1.39, abs=0.02)
    assert layers["Layer_4"]["mean_raw_umi"]["MMP10"] == pytest.approx(0.09, abs=0.02)
    assert layers["Layer_3"]["mean_raw_umi"]["LAMC2"] == pytest.approx(5.36, abs=0.02)
    assert layers["Layer_4"]["mean_raw_umi"]["LAMC2"] == pytest.approx(1.74, abs=0.02)
    assert layers["Layer_4"]["mean_raw_umi"]["COL1A1"] == pytest.approx(128.75, abs=0.05)
    assert layers["Layer_3"]["mean_raw_umi"]["COL1A1"] == pytest.approx(45.98, abs=0.05)
    assert layers["Layer_4"]["mean_raw_umi"]["VIM"] == pytest.approx(29.99, abs=0.05)
    assert layers["Layer_4"]["mean_raw_umi"]["CD74"] == pytest.approx(23.71, abs=0.05)


def test_03_zero_classical_cadherin_switch_falsification(exp3_results: dict) -> None:
    """Verify CDH2 = 0.0 UMI across 400/400 spots vs CDH1 = 1393.0 UMI and CDH3 = 2107.0 UMI."""
    cad_block = exp3_results["section1_empirical_cohort_characterization"]["cadherin_switch_falsification"]
    assert "FALSIFIED" in cad_block["verdict"]
    cad = cad_block["markers"]

    # Classical mesenchymal N-cadherin (CDH2) is strictly 0.0 UMI across all 400 spots
    assert cad["CDH2"]["total_umi_400_spots"] == 0.0
    assert cad["CDH2"]["nonzero_spots"] == 0
    assert cad["CDH2"]["nonzero_fraction"] == 0.0
    for l_name in ["Layer_1", "Layer_2", "Layer_3", "Layer_4"]:
        assert cad["CDH2"]["mean_by_layer"][l_name] == 0.0

    # Epithelial E-cadherin (CDH1) and P-cadherin (CDH3) remain strongly expressed in Layer_1..Layer_3
    assert cad["CDH1"]["total_umi_400_spots"] == pytest.approx(1393.0, abs=0.1)
    assert cad["CDH1"]["nonzero_spots"] == 284
    assert cad["CDH1"]["mean_by_layer"]["Layer_1"] == pytest.approx(5.25, abs=0.02)
    assert cad["CDH1"]["mean_by_layer"]["Layer_3"] == pytest.approx(3.74, abs=0.02)

    assert cad["CDH3"]["total_umi_400_spots"] == pytest.approx(2107.0, abs=0.1)
    assert cad["CDH3"]["nonzero_spots"] == 299
    assert cad["CDH3"]["mean_by_layer"]["Layer_1"] == pytest.approx(9.62, abs=0.02)
    assert cad["CDH3"]["mean_by_layer"]["Layer_3"] == pytest.approx(4.92, abs=0.02)

    # Canonical EMT transcription factor SNAI1 is near-absent (14.0 UMI across 12/400 spots)
    assert cad["SNAI1"]["total_umi_400_spots"] == pytest.approx(14.0, abs=0.1)
    assert cad["SNAI1"]["nonzero_spots"] == 12


def test_04_multiscale_contact_graph_topology_35um_and_55um(exp3_results: dict) -> None:
    """Verify multi-scale spatial contact graph topology at d <= 35 um and d <= 55 um."""
    topo = exp3_results["section1_empirical_cohort_characterization"]["contact_graph_topology"]

    r35 = topo["radius_35um"]
    assert r35["max_dist_um"] == 35.0
    assert r35["n_total_directed_edges"] == 4764
    assert r35["n_tsk_to_stroma_edges"] == 20
    assert r35["n_stroma_to_tsk_edges"] == 20
    assert r35["n_border_tsk_spots"] == 9
    assert r35["n_border_str_spots"] == 7
    assert r35["n_border_l3_l4_spots"] == 16

    r55 = topo["radius_55um"]
    assert r55["max_dist_um"] == 55.0
    assert r55["n_total_directed_edges"] == 9906
    assert r55["n_tsk_to_stroma_edges"] == 184
    assert r55["n_stroma_to_tsk_edges"] == 184
    assert r55["n_border_tsk_spots"] == 24
    assert r55["n_border_str_spots"] == 29
    assert r55["n_border_l3_l4_spots"] == 53


def test_05_zero_cochain_35um_isotropic_bleed_inflation_and_string_null(exp3_results: dict) -> None:
    """Verify 35 um 0-cochain boundary bleed inflation (+319.2% MMP10+COL1A1, +241.2% LAMC2+COL1A1, +417.0% KRT14+FAP)."""
    sec2 = exp3_results["section2_zero_cochain_pseudo_emt_audit"]
    b35 = sec2["isotropic_bleed_35um_audit"]
    assert b35["boundary_radius_um"] == 35.0
    assert b35["n_boundary_spots"] == 16
    assert b35["n_cross_compartment_edges"] == 40

    pairs_35 = {r["pair_id"]: r for r in b35["pairs"]}
    assert pairs_35["MMP10+COL1A1"]["isotropic_bleed_inflation_pct"] == pytest.approx(319.20, abs=0.1)
    assert pairs_35["MMP10+COL1A1"]["isotropic_dual_positive_rate"] == pytest.approx(0.9375, abs=1e-4)
    assert pairs_35["MMP10+COL1A1"]["sheaf_dual_positive_rate"] == pytest.approx(0.1250, abs=1e-4)

    assert pairs_35["LAMC2+COL1A1"]["isotropic_bleed_inflation_pct"] == pytest.approx(241.21, abs=0.1)
    assert pairs_35["LAMC2+COL1A1"]["isotropic_dual_positive_rate"] == pytest.approx(1.0000, abs=1e-4)
    assert pairs_35["LAMC2+COL1A1"]["sheaf_dual_positive_rate"] == pytest.approx(0.1250, abs=1e-4)

    assert pairs_35["KRT14+FAP"]["isotropic_bleed_inflation_pct"] == pytest.approx(416.99, abs=0.1)
    assert pairs_35["KRT14+FAP"]["string_ppi_score"] == 0.0

    assert pairs_35["MMP10+VIM"]["isotropic_bleed_inflation_pct"] == pytest.approx(286.38, abs=0.1)
    assert pairs_35["MMP10+VIM"]["string_ppi_score"] == 0.0

    assert pairs_35["KRT5+ACTA2"]["isotropic_bleed_inflation_pct"] == pytest.approx(67.36, abs=0.1)
    assert pairs_35["KRT16+TAGLN"]["isotropic_bleed_inflation_pct"] == pytest.approx(56.35, abs=0.1)
    assert pairs_35["KRT17+COL3A1"]["isotropic_bleed_inflation_pct"] == pytest.approx(30.93, abs=0.1)

    # Verify STRING v12.0 null PPI and gnomAD v4 constraint table
    prot = sec2["proteomic_and_constraint_falsification"]
    assert all(p["string_score"] == 0.0 for p in prot["string_v12_null_ppi_pairs"])
    assert prot["gnomad_v4_constraint"]["COL1A1"]["pLI"] == 1.0
    assert prot["gnomad_v4_constraint"]["EGFR"]["pLI"] == pytest.approx(0.996, abs=1e-3)


def test_06_zero_cochain_55um_compartment_deconfounded_five_operator_comparison(
    exp3_results: dict,
) -> None:
    """Verify 55 um 5-operator stalk alignment: 0.0% stromal B_intra under SE2CellularSheaf across all 10 pairs."""
    b55 = exp3_results["section2_zero_cochain_pseudo_emt_audit"]["compartment_deconfounded_55um_audit"]
    assert b55["boundary_radius_um"] == 55.0
    assert b55["n_stromal_border_spots"] == 29
    assert b55["n_all_border_spots"] == 53
    assert len(b55["pairs"]) == 10

    pairs_55 = {r["pair_id"]: r for r in b55["pairs"]}
    for pid, row in pairs_55.items():
        # Both v0.1.0 hard-clamped and v0.2.0 Huber soft-clamped sheaf achieve 0.0% stromal border bleed
        assert row["se2_sheaf_v010_hard"]["stromal_border_dual_positive_rate"] == 0.0, pid
        assert row["se2_sheaf_v010_hard"]["stromal_border_b_intra"] == 0.0, pid
        assert row["se2_sheaf_v020_huber"]["stromal_border_dual_positive_rate"] == 0.0, pid
        assert row["se2_sheaf_v020_huber"]["stromal_border_b_intra"] == 0.0, pid
        # Isotropic smoothing induces severe stromal border bleed across every pair
        assert row["isotropic_smoothing"]["all_border_dual_positive_rate"] >= 0.245, pid
        assert row["isotropic_smoothing"]["stromal_border_b_intra"] >= 0.052, pid

    assert pairs_55["LAMC2+COL1A1"]["isotropic_smoothing"]["all_border_dual_positive_rate"] == pytest.approx(
        0.4717, abs=1e-3
    )
    assert pairs_55["LAMC2+COL1A1"]["isotropic_smoothing"]["stromal_border_b_intra"] == pytest.approx(
        0.2117, abs=1e-3
    )
    assert pairs_55["MMP10+COL1A1"]["isotropic_smoothing"]["all_border_dual_positive_rate"] == pytest.approx(
        0.3396, abs=1e-3
    )


def test_07_theorem_2_18_registration_jitter_sweep_and_double_penalty_elimination(
    exp3_results: dict,
) -> None:
    """Verify Theorem 2.18 registration jitter sweep across Delta_x in [0.0, 30.0] um (5 seeds)."""
    js = exp3_results["section2_zero_cochain_pseudo_emt_audit"]["registration_jitter_sweep"]
    assert js["n_seeds"] == 5
    sweep = {r["delta_x_jitter_um"]: r for r in js["sweep"]}
    assert set(sweep.keys()) == {0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0}

    # Exact v0.1.0 parity and zero penetration at Delta_x = 0.0 um
    assert sweep[0.0]["se2_v020_adaptive_accuracy"] == pytest.approx(0.8300, abs=1e-4)
    assert sweep[0.0]["se2_v010_hard_accuracy"] == pytest.approx(0.8300, abs=1e-4)
    assert sweep[0.0]["isotropic_unclamped_accuracy"] == pytest.approx(0.8125, abs=1e-4)
    assert sweep[0.0]["v020_gain_over_v010"] == 0.0
    assert sweep[0.0]["v020_gain_over_isotropic"] == pytest.approx(0.0175, abs=1e-4)
    assert sweep[0.0]["v020_penetration_rate"] == 0.0
    assert sweep[0.0]["v020_theoretical_leakage_gamma"] == 0.0

    # Elimination of the high-jitter double penalty at Delta_x in [10, 30] um
    for j_um, expected_gain in [(10.0, 0.0055), (15.0, 0.0090), (20.0, 0.0125), (30.0, 0.0190)]:
        row = sweep[j_um]
        assert row["se2_v020_adaptive_accuracy"] > row["se2_v010_hard_accuracy"]
        assert row["se2_v020_adaptive_accuracy"] > row["isotropic_unclamped_accuracy"]
        assert row["v020_gain_over_v010"] == pytest.approx(expected_gain, abs=1e-3)


def test_08_one_cochain_28_pair_juxtacrine_sbr_sharpening(exp3_results: dict) -> None:
    """Verify 1-cochain directed juxtacrine SBR sharpening across all 28 ligand-receptor pairs."""
    sbr_block = exp3_results["section3_one_cochain_juxtacrine_signaling_audit"][
        "deconfounded_sbr_sharpening_28_pairs"
    ]
    assert len(sbr_block["pairs"]) == 28
    assert sbr_block["regularization_epsilon"] == 0.01
    assert sbr_block["strict_epsilon"] == 0.0001
    assert sbr_block["mean_sbr_sheaf"] == pytest.approx(67.03, abs=0.1)
    assert sbr_block["mean_sbr_isotropic"] == pytest.approx(3.55, abs=0.1)
    assert sbr_block["mean_sharpening_factor"] == pytest.approx(16.96, abs=0.1)
    assert sbr_block["max_sharpening_factor"] == pytest.approx(50.59, abs=0.1)
    assert sbr_block["max_sharpening_factor_eps1e4"] == pytest.approx(4916.54, abs=0.5)

    by_pair = {r["pair_id"]: r for r in sbr_block["pairs"]}
    expected_sharpening = {
        "MMP10->COL1A1": (50.59, 4916.54, 0.0),
        "HLA-DRA->CD74": (47.13, 4619.07, 0.0),
        "MMP9->TIMP1": (35.35, 3420.60, 0.0),
        "LAMC2->ITGA6": (33.90, 3283.14, 0.0),
        "B2M->CD74": (31.61, 3058.72, 0.0),
        "FN1->ITGA6": (24.94, 38.44, 0.0174),
        "LAMC2->ITGB1": (22.90, 2182.85, 0.0),
        "MMP10->FN1": (19.86, 1836.74, 0.0),
        "S100A9->CD74": (18.95, 1799.34, 0.0),
        "S100A8->CD74": (18.53, 1753.61, 0.0),
        "LAMC2->ITGA5": (17.48, 1638.71, 0.0),
        "INHBA->TGFBR2": (15.76, 1473.90, 0.0),
        "MMP10->ITGA6": (15.54, 1420.51, 0.0),
        "SPARC->EGFR": (13.45, 18.19, 0.0251),
        "COL1A1->ITGA6": (12.57, 14.81, 0.0501),
        "JAG1->NOTCH3": (10.62, 961.24, 0.0),
    }
    for pid, (sharp_e2, sharp_e4, b_shf) in expected_sharpening.items():
        assert by_pair[pid]["sbr_sharpening_factor"] == pytest.approx(sharp_e2, abs=0.05), pid
        assert by_pair[pid]["sbr_sharpening_factor_eps1e4"] == pytest.approx(sharp_e4, abs=0.5), pid
        assert by_pair[pid]["b_intra_sheaf"] == pytest.approx(b_shf, abs=1e-3), pid


def test_09_one_cochain_35um_raw_umi_directional_polarity_gain(exp3_results: dict) -> None:
    """Verify 35 um raw-UMI 1-cochain directional polarity gain (2.03x to 38.76x) across contact edges."""
    pol_block = exp3_results["section3_one_cochain_juxtacrine_signaling_audit"][
        "raw_umi_directional_polarity_35um"
    ]
    assert pol_block["boundary_radius_um"] == 35.0
    assert pol_block["n_directed_cross_edges"] == 20
    assert pol_block["mean_polarity_gain_x"] == pytest.approx(7.68, abs=0.05)
    assert pol_block["max_polarity_gain_x"] == pytest.approx(38.76, abs=0.05)

    pol_by_pair = {r["pair_id"]: r for r in pol_block["pairs"]}
    expected_gains = {
        "AREG->EGFR": 38.76,
        "COL1A1->CD44": 8.06,
        "TGFA->EGFR": 7.62,
        "JAG1->NOTCH3": 7.29,
        "TGFB1->TGFBR2": 6.10,
        "COL1A1->ITGB1": 5.75,
        "MMP10->ITGA6": 5.09,
        "FN1->ITGB1": 4.07,
        "FN1->ITGA5": 2.63,
        "LAMC2->ITGB1": 2.57,
        "LAMC2->ITGA6": 2.47,
        "SPP1->CD44": 1.75,
    }
    for pid, gain in expected_gains.items():
        assert pol_by_pair[pid]["sheaf_over_isotropic_polarity_gain_x"] == pytest.approx(gain, abs=0.05), pid
        assert pol_by_pair[pid]["sheaf_active_boundary_edge_fraction"] == 1.0, pid


def test_10_synthetic_ground_truth_lateral_bleed_benchmark(exp3_results: dict) -> None:
    """Verify controlled synthetic ground-truth benchmark with lateral capture bleed eta = 0.18."""
    syn = exp3_results["section3_one_cochain_juxtacrine_signaling_audit"][
        "synthetic_ground_truth_benchmark"
    ]
    assert syn["lateral_capture_bleed_eta"] == 0.18
    assert syn["ground_truth_b_intra"] == 0.0
    assert syn["sheaf_b_intra"] == 0.0
    assert syn["isotropic_b_intra"] == pytest.approx(0.6574, abs=1e-3)
    assert syn["magic_b_intra"] == pytest.approx(0.0624, abs=1e-3)
    assert syn["sheaf_over_isotropic_sharpening_x"] == pytest.approx(1018.04, abs=1.0)
    assert syn["sheaf_over_magic_sharpening_x"] == pytest.approx(61.94, abs=0.5)


def test_11_adversarial_test1_spatial_spot_label_permutation_null(exp3_results: dict) -> None:
    """Verify Adversarial Test 1: N = 200 spatial spot-label permutation null (z = 8.96 / 175.40, p = 0.00498)."""
    t1 = exp3_results["section5_adversarial_falsification_suite"]["test1_spot_label_permutation_null"]
    assert t1["passed"] is True
    assert t1["n_permutations"] == 200
    assert t1["observed_mean_j_sheaf"] == pytest.approx(1.3470, abs=1e-3)
    assert t1["null_mean_j_sheaf"] == pytest.approx(0.4410, abs=1e-3)
    assert t1["j_sheaf_z_score"] == pytest.approx(8.96, abs=0.05)
    assert t1["j_sheaf_empirical_p_value"] == pytest.approx(0.00498, abs=1e-5)

    assert t1["observed_mean_sbr_sheaf"] == pytest.approx(92.58, abs=0.1)
    assert t1["null_mean_sbr_sheaf"] == pytest.approx(1.25, abs=0.05)
    assert t1["sbr_sheaf_z_score"] == pytest.approx(175.40, abs=0.2)
    assert t1["sbr_sheaf_empirical_p_value"] == pytest.approx(0.00498, abs=1e-5)

    per_pair = {r["pair_id"]: r for r in t1["per_pair_permutation_audit"]}
    assert len(per_pair) == 15
    assert per_pair["MMP10->COL1A1"]["z_score"] == pytest.approx(574.08, abs=0.5)
    assert per_pair["HLA-DRA->CD74"]["z_score"] == pytest.approx(349.19, abs=0.5)
    assert per_pair["B2M->CD74"]["z_score"] == pytest.approx(304.46, abs=0.5)
    assert per_pair["LAMC2->ITGA6"]["z_score"] == pytest.approx(202.85, abs=0.5)
    assert all(r["z_score"] > 10.0 and r["empirical_p_value"] < 0.01 for r in per_pair.values())


def test_12_adversarial_test2_normal_tangential_orientation_swap(exp3_results: dict) -> None:
    """Verify Adversarial Test 2: 90-deg and 180-deg normal orientation rotation & graph clamping swap."""
    t2 = exp3_results["section5_adversarial_falsification_suite"][
        "test2_normal_tangential_orientation_swap"
    ]
    assert t2["passed"] is True
    assert t2["true_orientation_normal_penetration_residual"] == 0.0
    assert t2["true_orientation_tangential_preservation_fraction"] == pytest.approx(1.0, abs=1e-4)
    assert t2["rotated_90deg_normal_leakage_fraction"] == pytest.approx(1.0, abs=1e-4)
    assert t2["rotated_90deg_tangential_remaining_fraction"] == 0.0
    assert t2["flipped_180deg_normal_leakage_fraction"] == pytest.approx(1.0, abs=1e-4)
    assert t2["graph_swap_stromal_bleed_mmp10_col1a1"] == pytest.approx(0.4242, abs=1e-3)
    assert t2["graph_swap_stromal_bleed_lamc2_itga6"] == pytest.approx(0.4363, abs=1e-3)
    assert t2["true_sheaf_stromal_bleed_mmp10_col1a1"] == 0.0
    assert t2["true_sheaf_stromal_bleed_lamc2_itga6"] == 0.0


def test_13_adversarial_test3_countsplit_binomial_thinning_holdout(exp3_results: dict) -> None:
    """Verify Adversarial Test 3: countsplit binomial holdout (p = 0.50, r = 0.9918 / 0.9995 > 0.90)."""
    t3 = exp3_results["section5_adversarial_falsification_suite"]["test3_countsplit_binomial_holdout"]
    assert t3["passed"] is True
    assert t3["split_probability_p"] == 0.5
    assert t3["target_threshold_r"] == 0.90
    assert t3["pearson_r_contrast_deconfounded_28_pairs"] == pytest.approx(0.9918, abs=1e-3)
    assert t3["pearson_r_log1p_sheaf_28_pairs"] == pytest.approx(0.9995, abs=1e-3)
    assert len(t3["pair_replications"]) == 28


def test_14_adversarial_test4_leave_one_fov_out_quadrant_cross_validation(
    exp3_results: dict,
) -> None:
    """Verify Adversarial Test 4: 4-fold Leave-One-FOV-Out (LOFO) spatial quadrant cross-validation."""
    t4 = exp3_results["section5_adversarial_falsification_suite"][
        "test4_leave_one_fov_out_spatial_holdout"
    ]
    assert t4["passed"] is True
    assert t4["n_folds"] == 4
    assert t4["mean_held_out_sheaf_pseudo_emt_rate"] == 0.0
    assert t4["mean_held_out_lamc2_itga6_sbr_sheaf"] == pytest.approx(81.92, abs=0.1)
    assert t4["mean_held_out_lamc2_itga6_sharpening_x"] == pytest.approx(37.13, abs=0.1)

    folds = t4["folds"]
    assert len(folds) == 4
    expected_sharpening_by_fold = [30.76, 27.94, 40.74, 49.07]
    for f_row, exp_sh in zip(folds, expected_sharpening_by_fold):
        assert f_row["n_train_spots"] + f_row["n_test_spots"] == 400
        assert f_row["sheaf_pseudo_emt_dual_pos_rate"] == 0.0
        assert f_row["lamc2_itga6_sharpening_x"] == pytest.approx(exp_sh, abs=0.1)


def test_15_so6_kelvin_mandel_bundle_and_trimodal_cross_expedition_convergence(
    exp3_results: dict,
) -> None:
    """Verify Theorem 2.17 SO(6) Kelvin-Mandel trace conservation (<= 1.14e-13) and Tri-Modal Convergence."""
    sec4 = exp3_results["section4_multimodal_so6_and_cross_expedition_bridge"]
    so6 = sec4["so6_kelvin_mandel_missing_modality_audit"]
    assert so6["max_relative_trace_residual_across_all_tests"] <= 1.14e-13

    assert len(so6["tnbc3_gse148673_profiles"]) == 7
    assert len(so6["gbm_gse57872_profiles"]) == 3

    for profile in so6["tnbc3_gse148673_profiles"] + so6["gbm_gse57872_profiles"]:
        assert profile["max_relative_trace_residual"] <= 1.14e-13
        m_levels = [step["m_missing"] for step in profile["missing_modality_sweep"]]
        assert m_levels == [0.0, 0.25, 0.5, 0.7, 0.85]
        for step in profile["missing_modality_sweep"]:
            assert step["relative_trace_residual"] <= 1.14e-13
            assert step["min_eigenvalue"] > 0.0
            assert step["max_eigenvalue"] > step["min_eigenvalue"]
            assert step["trace_before"] == pytest.approx(step["trace_after"], rel=1e-12)

    # Verify GSE144239 cell-cycle erasure check & Compound Tri-Modal Catastrophe Table
    cc_check = sec4["cross_expedition_convergence"]["gse144239_cscc_cell_cycle_erasure_check"]
    assert cc_check["KRT14"]["linear_regression_variance_erasure_fraction"] == pytest.approx(0.9686, abs=1e-3)
    assert cc_check["EGFR"]["linear_regression_variance_erasure_fraction"] == pytest.approx(0.5433, abs=1e-3)
    for g_sym, g_row in cc_check.items():
        assert g_row["cellsnail_variance_retained_fraction"] == 1.0, g_sym

    conv = sec4["cross_expedition_convergence"]["compound_tri_modal_catastrophe_table"]
    assert len(conv) == 5


def test_16_deterministic_bit_exact_reproducibility() -> None:
    """Verify two independent instantiations with random_state=42 produce bit-identical JSON output."""
    a1 = SpatialBorderAuditor(random_state=42).run_full_audit()
    a2 = SpatialBorderAuditor(random_state=42).run_full_audit()
    assert json.dumps(a1, sort_keys=True, allow_nan=False) == json.dumps(
        a2, sort_keys=True, allow_nan=False
    )
