"""tests/test_expedition1_delta_hunt.py
=======================================
Comprehensive unit and falsification test suite for Expedition 1:
The "Trash-Can" Delta-Hunt (`cellsnail.expeditions.exp1_delta_hunt`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pytest

from cellsnail.expeditions.exp1_delta_hunt import (
    COHORT_SPECS,
    DeltaErasureDiscoveryPipeline,
    build_cross_expedition_convergence,
    compute_cellsnail_trajectory_deconfounding,
    compute_delta_erasure_profile,
    compute_ols_cell_cycle_regression,
    load_exp1_db_annotations,
    load_exp1_delta_cohorts,
    run_countsplit_binomial_holdout,
    run_expedition1_delta_hunt,
    run_lodo_and_out_of_group_generalization,
    run_permutation_null_and_housekeeping_falsification,
    summarize_multi_database_annotations,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON_PATH = REPO_ROOT / "results" / "expedition1_delta_hunt.json"


@pytest.fixture(scope="module")
def cohorts_bundle():
    return load_exp1_delta_cohorts()


@pytest.fixture(scope="module")
def db_annotations():
    return load_exp1_db_annotations()


@pytest.fixture(scope="module")
def delta_profile(cohorts_bundle, db_annotations):
    return compute_delta_erasure_profile(cohorts_bundle, db_annotations)


@pytest.fixture(scope="module")
def lodo_results(cohorts_bundle):
    return run_lodo_and_out_of_group_generalization(cohorts_bundle)


@pytest.fixture(scope="module")
def countsplit_results(cohorts_bundle):
    return run_countsplit_binomial_holdout(cohorts_bundle)


@pytest.fixture(scope="module")
def permutation_results(cohorts_bundle):
    return run_permutation_null_and_housekeeping_falsification(
        cohorts_bundle, n_permutations=200, random_state=42
    )


@pytest.fixture(scope="module")
def cross_exp_results(delta_profile, cohorts_bundle, db_annotations):
    return build_cross_expedition_convergence(delta_profile, cohorts_bundle, db_annotations)


# ==============================================================================
# 1. Empirical Bundle & Multi-Database Schema Integrity Tests
# ==============================================================================

def test_01_empirical_npz_bundle_structure(cohorts_bundle):
    """Verify the 77-array empirical NPZ bundle contains all 4 GEO cohorts with exact dimensions."""
    assert len(cohorts_bundle) == 77
    assert cohorts_bundle["gbm_X_expanded"].shape == (543, 43)
    assert cohorts_bundle["tnbc3_X_expanded_log1p"].shape == (500, 146)
    assert cohorts_bundle["tnbc3_X_expanded_umi"].shape == (500, 146)
    assert cohorts_bundle["mix_X_expanded_log1p"].shape == (300, 252)
    assert cohorts_bundle["mix_X_expanded_umi"].shape == (300, 252)
    assert cohorts_bundle["cscc_X_expanded_log1p"].shape == (400, 118)
    assert cohorts_bundle["cscc_X_expanded_umi"].shape == (400, 118)


def test_02_db_annotations_11_databases_completeness(db_annotations):
    """Verify all 42 genes (37 candidate Delta-genes + 5 core clock controls) and 11 databases are present."""
    meta = db_annotations["metadata"]
    assert meta["total_genes_annotated"] == 42
    assert meta["candidate_delta_genes_count"] == 37
    assert meta["core_mitotic_clock_controls_count"] == 5
    assert len(meta["databases_integrated"]) == 11

    genes = db_annotations["genes"]
    assert len(genes) == 42
    for clock_g in ["CCNB1", "TOP2A", "MKI67", "PCNA", "CDK1"]:
        assert clock_g in genes
        assert "ensembl_grch38" in genes[clock_g]
        assert "druggability_opentargets_chembl" in genes[clock_g]

    for cand_g in ["EGFR", "MYC", "CDK4", "PTMA", "HMGB2", "DEK", "EZH2", "DNMT1", "PARP1", "HSP90AA1"]:
        assert cand_g in genes
        ens = genes[cand_g]["ensembl_grch38"]
        assert ens["ensembl_gene_id"].startswith("ENSG")
        assert int(ens["canonical_tss"]) > 0


# ==============================================================================
# 2. Mathematical Invariants of OLS Projection & H_clock (+) H_nope
# ==============================================================================

def test_03_ols_projection_orthogonality_and_pythagorean_identity(cohorts_bundle):
    """Verify centered OLS residuals are strictly orthogonal to clock covariates and obey Pythagoras."""
    X = cohorts_bundle["tnbc3_X_expanded_log1p"].astype(np.float64)
    genes = [str(g) for g in cohorts_bundle["tnbc3_expanded_genes"]]
    core_idx = [genes.index(str(g)) for g in cohorts_bundle["tnbc3_clock_core_genes"]]
    C = X[:, core_idx]

    res = compute_ols_cell_cycle_regression(X, C)
    C_cent = C - np.mean(C, axis=0, keepdims=True)
    R = res["residuals"]
    F = res["fitted_clock_component"]

    # Residuals must be orthogonal to all clock covariates: C_cent^T @ R == 0
    cross_cov = C_cent.T @ R
    assert float(np.max(np.abs(cross_cov))) < 1e-9

    # Pythagorean variance decomposition: Var(X_cent) == Var(F) + Var(R)
    var_sum = np.var(F, axis=0) + res["residual_variance"]
    assert float(np.max(np.abs(var_sum - res["orig_variance"]))) < 1e-9

    # Clock genes regressed onto themselves must have zero residual variance
    assert float(np.max(res["residual_variance"][core_idx])) < 1e-12


def test_04_cellsnail_subspace_orthogonality_all_four_cohorts(delta_profile):
    """Verify `<P_clock x, P_nope x> = 0` and Frobenius energy conservation = 1.0 across all 4 cohorts."""
    for cid, cdata in delta_profile["cohorts"].items():
        orth = cdata["orthogonality_verification"]
        assert orth["max_inner_product_residual"] == 0.0, f"Non-zero inner product in {cid}"
        assert abs(orth["total_norm_conservation_ratio"] - 1.0) < 1e-7, f"Norm drift in {cid}"
        assert abs(orth["cellsnail_nope_candidate_variance_retained"] - 1.0) < 1e-6
        assert abs(orth["clock_core_recall"] - 1.0) < 1e-6
        assert orth["false_positive_clock_leakage_into_nope"] == 0


def test_05_calibrated_trajectory_thresholds_match_empirical_cohorts(delta_profile):
    """Verify trajectory-calibrated flicker thresholds across the 4 cohorts."""
    expected_thresholds = {
        "GSE57872_GBM": 0.6292,
        "GSE148673_TNBC3": 0.6285,
        "GSE118767_sc_mixology": 0.6543,
        "GSE144239_cSCC_spatial": 0.5753,
    }
    for cid, exp_tau in expected_thresholds.items():
        obs_tau = delta_profile["cohorts"][cid]["orthogonality_verification"][
            "trajectory_calibrated_flicker_threshold"
        ]
        assert abs(obs_tau - exp_tau) <= 1e-4


def test_06_nested_clock_subspace_monotonicity_across_all_cohorts(delta_profile):
    """
    Because span(X_{5-clock}) is a linear subspace of span(X_{20-clock}),
    Delta_erasure^(5-clock)(g) <= Delta_erasure^(20-clock)(g) must hold for every gene g.
    """
    for cid, cdata in delta_profile["cohorts"].items():
        viol = cdata["orthogonality_verification"]["nested_subspace_monotonicity_max_violation"]
        assert viol <= 1e-7, f"Monotonicity violated in {cid}: {viol}"
        for rec in cdata["all_candidate_delta_erasure_records"]:
            assert 0.0 <= rec["delta_erasure_2score"] <= 1.0
            assert 0.0 <= rec["delta_erasure_5clock"] <= 1.0
            assert 0.0 <= rec["delta_erasure_extclock"] <= 1.0
            assert rec["delta_erasure_5clock"] <= rec["delta_erasure_extclock"] + 1e-4


# ==============================================================================
# 3. Empirical Cohort Delta_erasure Discovery Verification Tests
# ==============================================================================

def test_07_cohort_wide_delta_erasure_summaries(delta_profile):
    """Verify cohort-wide mean and maximum Delta_erasure across all 4 GEO cohorts."""
    gbm_s = delta_profile["cohorts"]["GSE57872_GBM"]["cohort_wide_candidate_delta_erasure_summary"]
    assert abs(gbm_s["mean_delta_erasure_5clock"] - 0.1242) < 1e-3
    assert abs(gbm_s["max_delta_erasure_5clock"] - 0.6175) < 1e-3
    assert abs(gbm_s["max_delta_erasure_extclock"] - 0.6817) < 1e-3

    tnbc_s = delta_profile["cohorts"]["GSE148673_TNBC3"][
        "cohort_wide_candidate_delta_erasure_summary"
    ]
    assert abs(tnbc_s["mean_delta_erasure_5clock"] - 0.3020) < 1e-3
    assert abs(tnbc_s["max_delta_erasure_5clock"] - 0.7965) < 1e-3
    assert abs(tnbc_s["max_delta_erasure_extclock"] - 0.8491) < 1e-3
    assert tnbc_s["genes_with_delta_5clock_above_0_25"] == 26

    mix_s = delta_profile["cohorts"]["GSE118767_sc_mixology"][
        "cohort_wide_candidate_delta_erasure_summary"
    ]
    assert abs(mix_s["mean_delta_erasure_5clock"] - 0.2910) < 1e-3
    assert abs(mix_s["max_delta_erasure_5clock"] - 0.6697) < 1e-3
    assert abs(mix_s["max_delta_erasure_extclock"] - 0.7382) < 1e-3

    cscc_s = delta_profile["cohorts"]["GSE144239_cSCC_spatial"][
        "cohort_wide_candidate_delta_erasure_summary"
    ]
    assert abs(cscc_s["mean_delta_erasure_5clock"] - 0.2211) < 1e-3
    assert abs(cscc_s["max_delta_erasure_5clock"] - 0.6213) < 1e-3
    assert abs(cscc_s["max_delta_erasure_extclock"] - 0.7215) < 1e-3


def test_08_chromatin_and_epigenetic_inheritance_drivers_erasure(delta_profile):
    """Verify severe OLS variance erasure on replication-coupled chromatin & epigenetic factors."""
    meta = {r["gene"]: r for r in delta_profile["all_cross_cohort_gene_meta_records"]}
    assert abs(meta["PTMA"]["mean_delta_erasure_5clock"] - 0.5588) < 1e-3
    assert abs(meta["PTMA"]["mean_delta_erasure_extclock"] - 0.6778) < 1e-3
    assert abs(meta["HMGB2"]["mean_delta_erasure_5clock"] - 0.5453) < 1e-3
    assert abs(meta["HMGB2"]["max_delta_erasure_5clock"] - 0.7533) < 1e-3
    assert abs(meta["DEK"]["max_delta_erasure_5clock"] - 0.6408) < 1e-3
    assert abs(meta["EZH2"]["delta_5clock_by_cohort"]["GSE148673_TNBC3"] - 0.5316) < 1e-3
    assert abs(meta["DNMT1"]["delta_5clock_by_cohort"]["GSE118767_sc_mixology"] - 0.5528) < 1e-3
    assert abs(meta["UHRF1"]["delta_5clock_by_cohort"]["GSE148673_TNBC3"] - 0.5420) < 1e-3
    assert abs(meta["HELLS"]["delta_5clock_by_cohort"]["GSE118767_sc_mixology"] - 0.5513) < 1e-3


def test_09_homologous_recombination_and_fanconi_anemia_repair_erasure(delta_profile):
    """Verify OLS variance erasure on HR, Fanconi Anemia, and fork protection genes."""
    meta = {r["gene"]: r for r in delta_profile["all_cross_cohort_gene_meta_records"]}
    assert abs(meta["PBK"]["delta_5clock_by_cohort"]["GSE57872_GBM"] - 0.6175) < 1e-3
    assert abs(meta["PBK"]["delta_5clock_by_cohort"]["GSE148673_TNBC3"] - 0.7965) < 1e-3
    assert abs(meta["UBE2T"]["delta_5clock_by_cohort"]["GSE57872_GBM"] - 0.5103) < 1e-3
    assert abs(meta["UBE2T"]["delta_5clock_by_cohort"]["GSE148673_TNBC3"] - 0.7721) < 1e-3
    assert abs(meta["KIAA0101"]["delta_5clock_by_cohort"]["GSE148673_TNBC3"] - 0.7052) < 1e-3
    assert abs(meta["PARP1"]["delta_5clock_by_cohort"]["GSE148673_TNBC3"] - 0.5594) < 1e-3
    assert abs(meta["BRCA1"]["delta_5clock_by_cohort"]["GSE118767_sc_mixology"] - 0.5489) < 1e-3
    assert abs(meta["FANCI"]["delta_5clock_by_cohort"]["GSE148673_TNBC3"] - 0.4775) < 1e-3
    assert abs(meta["FANCD2"]["delta_5clock_by_cohort"]["GSE148673_TNBC3"] - 0.4219) < 1e-3


def test_10_subgroup_stratified_erasure_across_cohorts(delta_profile):
    """Verify subgroup-stratified Delta_erasure across GBM donors, TNBC3 subclones, mixology lines, and cSCC layers."""
    gbm_sg = delta_profile["cohorts"]["GSE57872_GBM"]["subgroup_stratified_erasure"]
    assert set(gbm_sg.keys()) == {"MGH26", "MGH28", "MGH29", "MGH30", "MGH31", "CSC6", "CSC8"}
    assert gbm_sg["CSC6"]["mean_candidate_delta_5clock"] > 0.25

    tnbc_sg = delta_profile["cohorts"]["GSE148673_TNBC3"]["subgroup_stratified_erasure"]
    assert set(tnbc_sg.keys()) == {"Subclone_A", "Subclone_B"}
    assert abs(tnbc_sg["Subclone_A"]["mean_candidate_delta_5clock"] - 0.2714) < 1e-3
    assert abs(tnbc_sg["Subclone_B"]["mean_candidate_delta_5clock"] - 0.3676) < 1e-3

    mix_sg = delta_profile["cohorts"]["GSE118767_sc_mixology"]["subgroup_stratified_erasure"]
    assert set(mix_sg.keys()) == {"HCC827", "H1975", "H2228"}

    cscc_sg = delta_profile["cohorts"]["GSE144239_cSCC_spatial"]["subgroup_stratified_erasure"]
    assert set(cscc_sg.keys()) == {"Layer_1", "Layer_2", "Layer_3", "Layer_4"}


# ==============================================================================
# 4. Out-of-Donor (LODO), Out-of-Subclone (LOCO) & Out-of-Layer (LOLO) Tests
# ==============================================================================

def test_11_lodo_gbm_seven_folds_orthogonality_and_retention(lodo_results):
    """Verify all 7 GBM LODO folds (MGH26-MGH31, CSC6, CSC8) achieve 0.0 orthogonality residual and 100% H_nope retention."""
    gbm_lodo = lodo_results["cohorts"]["GSE57872_GBM"]
    assert gbm_lodo["n_folds"] == 7
    assert gbm_lodo["max_orthogonality_residual_across_folds"] == 0.0
    assert gbm_lodo["min_nope_variance_retained_across_folds"] == 1.0
    assert gbm_lodo["total_false_positive_clock_leakage_across_folds"] == 0
    assert gbm_lodo["mean_pearson_r"] > 0.80
    assert gbm_lodo["mean_top10_gene_recall"] >= 0.85


def test_12_loco_and_lolo_generalization_across_tnbc3_mixology_cscc(lodo_results):
    """Verify out-of-subclone (TNBC3), out-of-cell-line (sc_mixology), and out-of-layer (cSCC) generalization."""
    assert lodo_results["total_out_of_group_folds_evaluated"] == 16
    assert lodo_results["global_max_orthogonality_residual"] == 0.0
    assert lodo_results["global_min_nope_variance_retained"] == 1.0
    assert lodo_results["global_false_positive_clock_leakage"] == 0
    assert lodo_results["global_mean_out_of_fold_pearson_r"] > 0.80
    assert lodo_results["global_mean_out_of_fold_spearman_rho"] > 0.75

    tnbc_loco = lodo_results["cohorts"]["GSE148673_TNBC3"]
    assert tnbc_loco["n_folds"] == 2
    assert tnbc_loco["mean_pearson_r"] > 0.94
    assert tnbc_loco["mean_spearman_rho"] > 0.94


# ==============================================================================
# 5. Zero-Leakage Binomial Molecule Thinning (`countsplit`) Holdout Tests
# ==============================================================================

def test_13_countsplit_five_seeds_reproducibility_and_zero_leakage(countsplit_results):
    """Verify countsplit binomial holdout across 5 seeds [42, 1337, 2026, 7, 99] achieves r > 0.85 on all 15 runs."""
    assert countsplit_results["seeds_evaluated"] == [42, 1337, 2026, 7, 99]
    assert countsplit_results["total_split_evaluations"] == 15
    assert countsplit_results["global_min_pearson_r"] > 0.85
    assert countsplit_results["global_mean_pearson_r"] > 0.95
    assert countsplit_results["global_mean_spearman_rho"] > 0.93
    assert countsplit_results["global_mean_top20_recall"] >= 0.90
    assert countsplit_results["global_false_positive_clock_leakage"] == 0

    for cid, c_res in countsplit_results["cohorts"].items():
        assert c_res["min_pearson_r"] > 0.85, f"countsplit r <= 0.85 in {cid}"
        assert c_res["max_orthogonality_residual"] == 0.0
        for s_run in c_res["seed_runs"]:
            assert s_run["umi_additivity_max_error"] == 0.0
            assert s_run["split2_nope_variance_retained"] == 1.0


# ==============================================================================
# 6. Permutation Null & Pure Mitotic Clock Falsification Tests
# ==============================================================================

def test_14_permutation_null_matches_theoretical_beta_expectation(permutation_results):
    """Verify empirical permutation null mean closely matches theoretical k/(n-1) Beta expectation."""
    for cid, p_data in permutation_results["cohorts"].items():
        theo = p_data["theoretical_beta_null_mean"]
        emp = p_data["empirical_global_null_mean"]
        assert abs(theo - emp) < 0.002, f"Null mean mismatch in {cid}: {theo} vs {emp}"
        assert p_data["max_observed_z_score"] > 50.0


def test_15_top_delta_genes_significant_under_permutation_null(permutation_results):
    """Verify top Delta-genes achieve empirical p < 0.01 and BH-FDR q < 0.01 across all 4 cohorts."""
    for cid, p_data in permutation_results["cohorts"].items():
        assert p_data["candidates_with_p_lt_0_01"] >= 15
        assert p_data["candidates_with_fdr_lt_0_01"] >= 15
        for top_g in p_data["top_10_permutation_validated_genes"]:
            assert top_g["empirical_p_value"] < 0.01
            assert top_g["bh_fdr_q_value"] < 0.01
            assert top_g["z_score"] > 10.0


def test_16_zero_false_positive_clock_and_housekeeping_leakage(permutation_results):
    """
    Falsification test: verify that 100% of pure mitotic clock genes (CCNB1, TOP2A,
    MKI67, PCNA, CDK1) and 100% of synthetic phase-locked mitotic housekeeping
    oscillators are assigned to H_clock with 0 false-positive leakage into H_nope.
    """
    assert permutation_results["total_pure_clock_controls_evaluated"] == 19
    assert permutation_results["total_pure_clock_false_positives_in_h_nope"] == 0
    assert permutation_results["total_synthetic_housekeeping_oscillators_evaluated"] == 40
    assert permutation_results["total_synthetic_oscillator_false_positives_in_h_nope"] == 0
    assert permutation_results["false_positive_clock_leakage_rate"] == 0.0

    for cid, p_data in permutation_results["cohorts"].items():
        fals = p_data["pure_mitotic_clock_falsification"]
        assert fals["clock_core_false_positive_in_h_nope"] == 0
        assert fals["clock_core_max_delta_erasure"] == 0.0
        assert fals["synthetic_oscillators_false_positive_in_h_nope"] == 0


# ==============================================================================
# 7. Multi-Database & Cross-Expedition Convergence Tests
# ==============================================================================

def test_17_multi_database_summary_metrics(db_annotations, delta_profile):
    """Verify multi-database summary counts (gnomAD LOEUF < 0.60, Phase 4 drugs, STRING bridges)."""
    summary = summarize_multi_database_annotations(db_annotations, delta_profile)
    assert summary["total_genes_annotated"] == 42
    assert summary["candidate_delta_genes_count"] == 37
    assert summary["core_mitotic_clock_controls_count"] == 5
    assert summary["candidates_lof_intolerant_loeuf_lt_0_60"] == 26
    assert summary["candidates_with_phase4_approved_drugs"] >= 10
    assert summary["string_network_summary"]["total_ppi_edges"] == 481
    assert summary["string_network_summary"]["cross_subspace_clock_to_delta_bridges_count"] == 111
    assert len(summary["string_network_summary"]["top_cross_subspace_bridges"]) == 25


def test_18_cross_expedition_exp1_exp2_dual_blind_spot_table(cross_exp_results):
    """Verify the 15-gene Exp 1 <-> Exp 2 Dual Blind-Spot table and compound signal survival."""
    s2 = cross_exp_results["exp1_exp2_dual_blind_spot_summary"]
    assert s2["total_shared_targets"] == 15
    assert s2["cellsnail_joint_survival_percent"] == 100.0
    assert s2["mean_dna_window15_washout_percent"] > 75.0
    assert s2["mean_rna_5clock_erasure_percent"] > 30.0
    assert s2["mean_standard_pipeline_joint_survival_5clock_percent"] < 20.0

    genes_in_table = {r["gene"] for r in s2["dual_blind_spot_table"]}
    expected_15 = {
        "EGFR", "MYC", "CDK4", "ABCB1", "PTEN",
        "HSP90AA1", "CDKN2A", "LY6E", "MDM2", "S100A6",
        "LGALS1", "KRAS", "AXL", "PROM1", "ERBB2",
    }
    assert genes_in_table == expected_15


def test_19_cross_expedition_exp1_exp3_spatial_catastrophe_table(cross_exp_results):
    """Verify the 21-gene Exp 1 <-> Exp 3 Spatial Boundary Catastrophe table on GSE144239 cSCC."""
    s3 = cross_exp_results["exp1_exp3_spatial_boundary_catastrophe_summary"]
    assert s3["total_spatial_interface_genes_audited"] == 21
    assert s3["cdh2_mesenchymal_n_cadherin_umi_across_400_spots"] == 0.0
    assert s3["cdh1_epithelial_e_cadherin_umi_across_400_spots"] == 1393.0
    assert s3["isotropic_smoothing_pseudo_emt_inflation_percent"]["KRT14_FAP"] == 417.0
    assert s3["isotropic_smoothing_pseudo_emt_inflation_percent"]["MMP10_COL1A1"] == 319.2
    assert s3["se2_cellular_sheaf_juxtacrine_polarity_gain_range_x"] == [2.03, 38.76]

    table_by_gene = {r["gene"]: r for r in s3["spatial_boundary_catastrophe_table"]}
    assert abs(table_by_gene["PTMA"]["exp1_cscc_delta_5clock_percent"] - 62.13) < 0.1
    assert abs(table_by_gene["HSP90AA1"]["exp1_cscc_delta_5clock_percent"] - 52.96) < 0.1
    assert abs(table_by_gene["CDK4"]["exp1_cscc_delta_5clock_percent"] - 47.70) < 0.1
    assert abs(table_by_gene["KRT14"]["exp1_cscc_delta_5clock_percent"] - 37.77) < 0.1
    assert abs(table_by_gene["EGFR"]["exp1_cscc_delta_5clock_percent"] - 31.90) < 0.1
    assert abs(table_by_gene["LAMC2"]["exp1_cscc_delta_5clock_percent"] - 20.19) < 0.1
    assert abs(table_by_gene["MMP10"]["exp1_cscc_delta_5clock_percent"] - 4.37) < 0.1
    assert abs(table_by_gene["MMP10"]["exp1_cscc_delta_20clock_percent"] - 10.68) < 0.1
    assert abs(table_by_gene["MMP10"]["exp1_layer3_tsk_delta_5clock_percent"] - 22.50) < 0.1
    assert abs(table_by_gene["MMP10"]["exp1_layer3_tsk_delta_20clock_percent"] - 49.92) < 0.1


# ==============================================================================
# 8. Deterministic JSON Receipt & Pipeline Execution Tests
# ==============================================================================

def _assert_no_nan_or_inf(obj):
    if isinstance(obj, float):
        assert not math.isnan(obj) and not math.isinf(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _assert_no_nan_or_inf(v)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_nan_or_inf(item)


def test_20_deterministic_json_receipt_exists_and_has_zero_nan():
    """Verify `results/expedition1_delta_hunt.json` exists, parses cleanly, and has zero NaN/Inf."""
    assert RESULTS_JSON_PATH.exists(), f"Missing JSON receipt: {RESULTS_JSON_PATH}"
    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    _assert_no_nan_or_inf(data)
    required_keys = [
        "expedition",
        "version",
        "mathematical_foundation",
        "empirical_cohorts_evaluated",
        "cross_cohort_top_25_delta_erased_genes",
        "cohorts_detailed_delta_erasure",
        "all_cross_cohort_gene_meta_records",
        "lodo_and_out_of_group_generalization",
        "countsplit_binomial_holdout_validation",
        "permutation_null_and_housekeeping_falsification",
        "multi_database_11_source_cross_validation",
        "cross_expedition_convergence",
    ]
    for k in required_keys:
        assert k in data, f"Missing required top-level key: {k}"


def test_21_pipeline_reexecution_determinism(tmp_path):
    """Verify two consecutive runs of `run_expedition1_delta_hunt` produce bit-identical JSON output."""
    temp_json_1 = tmp_path / "exp1_test_receipt_1.json"
    temp_json_2 = tmp_path / "exp1_test_receipt_2.json"
    receipt = run_expedition1_delta_hunt(save_json=True, output_path=temp_json_1)
    run_expedition1_delta_hunt(save_json=True, output_path=temp_json_2)
    assert temp_json_1.exists() and temp_json_2.exists()

    with open(temp_json_1, "r", encoding="utf-8") as f1, open(
        temp_json_2, "r", encoding="utf-8"
    ) as f2:
        assert f1.read() == f2.read()
    assert len(receipt["cross_cohort_top_25_delta_erased_genes"]) == 25


def test_22_reactome_and_string_subspace_separation(db_annotations):
    """Verify Reactome pathway separation and STRING v12.0 cross-subspace bridges in annotations."""
    reactome = db_annotations["reactome_pathway_separation"]
    assert "pathways" in reactome
    pathways = reactome["pathways"]
    assert len(pathways) >= 20
    assert "R-HSA-2559585" in pathways
    ois = pathways["R-HSA-2559585"]
    assert len(ois["clock_genes_present"]) == 0
    assert "CDKN2A" in ois["delta_genes_present"]

    string_net = db_annotations["string_subspace_coupling_network"]
    assert db_annotations["metadata"]["ncbi_taxon_id"] == 9606
    assert string_net["total_edges_in_panel"] == 481
    assert string_net["clock_to_delta_bridge_count"] == 111


def test_23_jaspar_and_unibind_transcriptional_coupling(db_annotations):
    """Verify JASPAR 2024 PWMs and UniBind ChIP-seq datasets for master proliferation TFs."""
    tf_reg = db_annotations["transcriptional_regulation_jaspar_unibind"]
    pwms = tf_reg["jaspar_pwm_profiles"]
    unibind = tf_reg["unibind_human_chipseq_datasets"]
    for tf in ["E2F1", "E2F4", "FOXM1", "MYC", "SOX2", "TP53", "STAT3"]:
        assert tf in pwms
        assert pwms[tf][0]["matrix_id"].startswith("MA")
    assert unibind["MYC"]["total_human_datasets"] == 137
    assert unibind["TP53"]["total_human_datasets"] == 123
    assert unibind["E2F1"]["total_human_datasets"] == 33


def test_24_tri_modal_convergence_headline_completeness(cross_exp_results):
    """Verify Tri-Modal Convergence headline linking Expeditions 1, 2, and 3."""
    tri = cross_exp_results["tri_modal_convergence_headline"]
    assert len(tri["anchor_genes"]) == 10
    assert "EGFR" in tri["anchor_genes"]
    assert "LAMC2" in tri["anchor_genes"]
    assert tri["exp2_dna_window15_washout_range_x"] == [15.02, 16.63]
    assert tri["exp3_se2_sheaf_polarity_gain_range_x"] == [2.03, 38.76]


# ==============================================================================
# 9. Adversarial Falsification & Manuscript-JSON Synchronization Audit Tests
# ==============================================================================

def test_25_extended_mitotic_markers_segregated_from_h_nope(cohorts_bundle, permutation_results):
    """
    Adversarial falsification: verify that no pure mitotic clock gene (CCNB1, TOP2A,
    MKI67, PCNA, CDK1) or extended mitotic marker (AURKA, AURKB, BIRC5, UBE2C,
    NUSAP1, CDC20, MCM2..7, RRM2, TYMS, DHFR, FEN1) leaks into the H_nope candidate
    set or is misclassified into H_nope across all 4 cohorts.
    """
    assert permutation_results["total_extended_clock_markers_evaluated"] == 54
    assert permutation_results["total_extended_clock_false_positives_in_h_nope"] == 0
    pure_and_ext_clock = {
        "CCNB1", "TOP2A", "MKI67", "PCNA", "CDK1",
        "AURKA", "AURKB", "BIRC5", "UBE2C", "NUSAP1", "CDC20",
        "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
        "RRM2", "TYMS", "DHFR", "FEN1",
    }
    for cid, spec in COHORT_SPECS.items():
        nope_set = {str(g) for g in cohorts_bundle[spec["nope_key"]]}
        assert len(nope_set & pure_and_ext_clock) == 0, (
            f"Clock/extended marker leaked into H_nope candidate set in {cid}: {nope_set & pure_and_ext_clock}"
        )
        fals = permutation_results["cohorts"][cid]["pure_mitotic_clock_falsification"]
        assert fals["clock_ext_false_positive_in_h_nope"] == 0
        assert fals["clock_and_ext_overlap_with_h_nope_candidates"] == 0


def test_26_strict_json_parse_constant_no_nan_or_infinity():
    """
    Adversarial audit: load `results/expedition1_delta_hunt.json` with strict
    `parse_constant` rejecting NaN/Infinity, and verify every top-25 gene has an Ensembl ID.
    """
    raw_text = RESULTS_JSON_PATH.read_text(encoding="utf-8")

    def _forbid_constant(val: str):
        raise ValueError(f"Forbidden non-standard JSON constant encountered: {val}")

    parsed = json.loads(raw_text, parse_constant=_forbid_constant)
    for rec in parsed["cross_cohort_top_25_delta_erased_genes"]:
        assert rec["ensembl_id"].startswith("ENSG"), f"Missing Ensembl ID for {rec['gene']}"


def test_27_spatial_interface_non_zero_slide_erasure_for_all_21_genes(cross_exp_results):
    """
    Adversarial audit: verify all 21 cSCC spatial interface genes (including KRT5,
    PKM, LDHA, and ITGA6) have strictly positive slide-wide and Layer_3 TSK erasure.
    """
    s3 = cross_exp_results["exp1_exp3_spatial_boundary_catastrophe_summary"]
    assert abs(s3["mean_cscc_5clock_erasure_on_interface_genes_percent"] - 37.90) < 0.05
    assert abs(s3["mean_layer3_tsk_5clock_erasure_percent"] - 46.33) < 0.05
    for row in s3["spatial_boundary_catastrophe_table"]:
        assert row["exp1_cscc_delta_2score_percent"] > 0.0, f"Zero 2-score erasure for {row['gene']}"
        assert row["exp1_cscc_delta_5clock_percent"] > 0.0, f"Zero 5-clock erasure for {row['gene']}"
        assert row["exp1_cscc_delta_20clock_percent"] > 0.0, f"Zero 20-clock erasure for {row['gene']}"
        assert row["exp1_layer3_tsk_delta_5clock_percent"] > 0.0
        assert row["exp1_layer3_tsk_delta_20clock_percent"] > 0.0


def test_28_manuscript_tables_match_json_receipt_zero_discrepancy():
    """
    Adversarial audit: verify that all numbers cited in `docs/EXPEDITION_1_DELTA_HUNT.md`
    match `results/expedition1_delta_hunt.json` with zero fabricated or mismatched statistics.
    """
    doc_path = REPO_ROOT / "docs" / "EXPEDITION_1_DELTA_HUNT.md"
    doc_text = doc_path.read_text(encoding="utf-8")
    data = json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))

    # 1. Verify all 25 rows of Table 1.1 appear with exact mean 5-clock and 20-clock values
    for idx, rec in enumerate(data["cross_cohort_top_25_delta_erased_genes"], 1):
        g = rec["gene"]
        ens = rec["ensembl_id"]
        m5 = f"{rec['mean_delta_erasure_5clock']:.4f}"
        m20 = f"{rec['mean_delta_erasure_extclock']:.4f}"
        max20 = f"{rec['max_delta_erasure_extclock']:.4f}"
        assert f"| **{idx}** | **`{g}`** | `{ens}` |" in doc_text, f"Table 1.1 mismatch at rank {idx} ({g})"
        assert f"**`{m5}`** | **`{m20}`** | `{max20}`" in doc_text or f"**`{m5}`** | **`{m20}`** | **`{max20}`**" in doc_text

    # 2. Verify all 16 LODO/LOCO/LOLO folds in Table 4.1
    for cid, c_lodo in data["lodo_and_out_of_group_generalization"]["cohorts"].items():
        for fold in c_lodo["folds"]:
            tau_s = f"`{fold['train_calibrated_threshold']:.4f}`"
            r_s = f"{fold['out_of_fold_delta_pearson_r']:.4f}"
            rho_s = f"{fold['out_of_fold_delta_spearman_rho']:.4f}"
            assert tau_s in doc_text, f"Missing LODO threshold {tau_s} for {fold['held_out_group']}"
            assert r_s in doc_text, f"Missing LODO Pearson r {r_s} for {fold['held_out_group']}"
            assert rho_s in doc_text, f"Missing LODO Spearman rho {rho_s} for {fold['held_out_group']}"

    # 3. Verify Dual Blind-Spot summary metrics in Section 6.1
    s2 = data["cross_expedition_convergence"]["exp1_exp2_dual_blind_spot_summary"]
    assert f"{s2['mean_dna_window15_washout_percent']:.2f}%" in doc_text
    assert f"{s2['mean_rna_5clock_erasure_percent']:.2f}%" in doc_text
    assert f"{s2['mean_standard_pipeline_joint_survival_5clock_percent']:.2f}%" in doc_text


