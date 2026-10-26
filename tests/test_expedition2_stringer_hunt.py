"""tests/test_expedition2_stringer_hunt.py
=========================================
Comprehensive unit, cross-expedition convergence, and adversarial falsification
test suite for Expedition 2: The Blind-Spot 1-3 Gene Stringer & Relapse Persister Hunt.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from cellsnail.expeditions.exp2_stringer_hunt import (
    BlindSpotStringerHunter,
    run_expedition2_stringer_hunt,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_JSON = _REPO_ROOT / "results" / "expedition2_stringer_hunt.json"


@pytest.fixture(scope="module")
def hunter() -> BlindSpotStringerHunter:
    return BlindSpotStringerHunter(random_state=42)


@pytest.fixture(scope="module")
def exp2_receipt(hunter: BlindSpotStringerHunter) -> Dict[str, Any]:
    return hunter.run_all(output_json_path=_RESULTS_JSON)


# ==============================================================================
# 1. Pipeline Execution & JSON Serialization Tests
# ==============================================================================


def test_expedition2_pipeline_status_and_top_level_schema(exp2_receipt: Dict[str, Any]):
    assert exp2_receipt["expedition_id"] == "EXPEDITION_2_BLIND_SPOT_STRINGER_AND_PERSISTER_HUNT"
    assert exp2_receipt["status"] == "EXPEDITION_2_COMPLETE"
    assert exp2_receipt["version"] == "0.2.0"
    for key in [
        "window_washout_transfer_function",
        "window_invisible_focal_stringers",
        "persister_collateral_vulnerabilities",
        "cross_expedition_convergence",
        "adversarial_falsification_audit",
    ]:
        assert key in exp2_receipt


def test_expedition2_json_artifact_written_and_nan_free(exp2_receipt: Dict[str, Any]):
    assert _RESULTS_JSON.exists(), "results/expedition2_stringer_hunt.json must exist"
    raw_text = _RESULTS_JSON.read_text(encoding="utf-8")
    assert "NaN" not in raw_text and "Infinity" not in raw_text
    loaded = json.loads(raw_text)
    assert loaded["status"] == "EXPEDITION_2_COMPLETE"
    # Verify strict allow_nan=False round-trip
    dumped = json.dumps(loaded, allow_nan=False)
    assert len(dumped) > 5000


def test_research_dossier_loaded_with_targets(hunter: BlindSpotStringerHunter):
    assert "targets" in hunter.dossier
    for sym in ["EGFR", "MYC", "CDK4", "ABCB1", "PTEN", "HSP90AA1", "AXL", "ERBB2", "LY6E", "LGALS1", "PROM1"]:
        assert sym in hunter.dossier["targets"]


# ==============================================================================
# 2. Moving-Window Attenuation Law & Transfer Function Tests
# ==============================================================================


def test_window_washout_analytical_attenuation_law(exp2_receipt: Dict[str, Any]):
    tf = exp2_receipt["window_washout_transfer_function"]
    assert tf["amplicon_widths_L"] == [1, 2, 3, 5, 10]
    assert tf["window_sizes_W"] == [3, 5, 11, 15, 25, 51, 101]
    assert tf["attenuation_law_verified"] is True
    assert tf["max_theory_vs_empirical_error"] < 1e-4

    for entry in tf["transfer_grid"]:
        L = entry["amplicon_width_L"]
        W = entry["window_size_W"]
        expected_r = min(1.0, float(L) / float(W))
        assert math.isclose(entry["theoretical_retention_R_theory"], expected_r, abs_tol=1e-5)
        assert math.isclose(entry["excess_retention_window"], expected_r, abs_tol=1e-4)


def test_window_washout_ratios_and_bounded_edge_fidelity(exp2_receipt: Dict[str, Any]):
    tf = exp2_receipt["window_washout_transfer_function"]
    by_lw = tf["by_L_and_W"]

    # L=1 under HMMcopy (W=15), CopyKAT (W=25), inferCNV (W=51, 101)
    e_1_15 = by_lw["L_1"]["W_15"]
    assert math.isclose(e_1_15["window_washout_ratio_W_over_L"], 15.0, rel_tol=1e-3)
    assert math.isclose(e_1_15["empirical_peak_cn_window"], 2.4, abs_tol=1e-3)
    assert math.isclose(e_1_15["empirical_peak_cn_cellsnail"], 8.0, abs_tol=1e-3)
    assert e_1_15["adjacent_boundary_leakage_cellsnail"] == 0.0
    assert e_1_15["bounded_edge_fidelity_cellsnail"] >= 0.99
    assert 0.35 <= e_1_15["bounded_edge_fidelity_window"] <= 0.38

    e_1_25 = by_lw["L_1"]["W_25"]
    assert math.isclose(e_1_25["window_washout_ratio_W_over_L"], 25.0, rel_tol=1e-3)

    e_1_51 = by_lw["L_1"]["W_51"]
    assert math.isclose(e_1_51["window_washout_ratio_W_over_L"], 51.0, rel_tol=1e-3)

    e_1_101 = by_lw["L_1"]["W_101"]
    assert math.isclose(e_1_101["window_washout_ratio_W_over_L"], 101.0, rel_tol=1e-3)


def test_bounded_edge_fidelity_strictly_in_unit_interval(exp2_receipt: Dict[str, Any]):
    tf = exp2_receipt["window_washout_transfer_function"]
    for entry in tf["transfer_grid"]:
        assert 0.0 <= entry["bounded_edge_fidelity_window"] <= 1.0
        assert 0.0 <= entry["bounded_edge_fidelity_cellsnail"] <= 1.0


# ==============================================================================
# 3. Window-Invisible Focal Stringers in GSE148673 TNBC3 & GSE118767 sc_mixology
# ==============================================================================


def test_tnbc3_egfr_focal_stringer_recovery(exp2_receipt: Dict[str, Any]):
    s = exp2_receipt["window_invisible_focal_stringers"]["tnbc3_focal_stringers"]["EGFR"]
    assert s["bin_index"] == 25
    assert s["locus_tag"] == "chr7:55019017"
    assert s["ensembl_gene_id"] == "ENSG00000146648"
    assert s["mane_select_transcript"] == "ENST00000275493"
    assert math.isclose(s["cn_cellsnail_subclone_a"], 8.024, abs_tol=1e-3)
    assert math.isclose(s["cn_cellsnail_rare_persister"], 8.755, abs_tol=1e-2)
    assert math.isclose(s["cn_window_15"], 2.401, abs_tol=1e-3)
    assert math.isclose(s["bounded_edge_fidelity_cellsnail"], 0.9960, abs_tol=1e-3)
    assert math.isclose(s["bounded_edge_fidelity_window_15"], 0.3688, abs_tol=1e-3)
    assert math.isclose(s["excess_cn_washout_ratio_w15"], 15.02, abs_tol=1e-2)
    assert s["confirmed_morse_innovation"] is True


def test_tnbc3_myc_ecdna_spire_recovery(exp2_receipt: Dict[str, Any]):
    s = exp2_receipt["window_invisible_focal_stringers"]["tnbc3_focal_stringers"]["MYC"]
    assert s["bin_index"] == 35
    assert s["locus_tag"] == "chr8:127735434"
    assert s["ensembl_gene_id"] == "ENSG00000136997"
    assert s["mane_select_transcript"] == "ENST00000621592"
    assert math.isclose(s["cn_cellsnail_subclone_a"], 9.824, abs_tol=1e-2)
    assert math.isclose(s["cn_window_15"], 3.164, abs_tol=1e-3)
    assert math.isclose(s["bounded_edge_fidelity_cellsnail"], 0.9765, abs_tol=1e-3)
    assert math.isclose(s["bounded_edge_fidelity_window_15"], 0.3694, abs_tol=1e-3)
    assert s["plateau_bins"] == list(range(30, 40))


def test_tnbc3_cdk4_and_12q_plateau_co_amplicon_recovery(exp2_receipt: Dict[str, Any]):
    s = exp2_receipt["window_invisible_focal_stringers"]["tnbc3_focal_stringers"]["CDK4"]
    assert s["bin_index"] == 55
    assert s["locus_tag"] == "chr12:57747727"
    assert s["ensembl_gene_id"] == "ENSG00000135446"
    assert s["mane_select_transcript"] == "ENST00000257904"
    assert math.isclose(s["cn_cellsnail_subclone_b"], 6.357, abs_tol=1e-2)
    assert math.isclose(s["cn_window_15"], 3.692, abs_tol=1e-3)
    assert math.isclose(s["bounded_edge_fidelity_cellsnail"], 0.9221, abs_tol=1e-3)
    co = s["co_amplified_12q_plateau_genes"]
    assert co["HSP90AA1"]["bin"] == 53 and co["HSP90AA1"]["cn_subclone_b"] > 4.0
    assert co["LY6E"]["bin"] == 57 and co["LY6E"]["cn_subclone_b"] > 4.0
    assert co["MDM2"]["bin"] == 58 and co["MDM2"]["cn_subclone_b"] > 4.0


def test_tnbc3_abcb1_ultra_rare_persister_stringer_recovery(exp2_receipt: Dict[str, Any]):
    s = exp2_receipt["window_invisible_focal_stringers"]["tnbc3_focal_stringers"]["ABCB1"]
    assert s["bin_index"] == 28
    assert s["locus_tag"] == "chr7:87503863"
    assert s["ensembl_gene_id"] == "ENSG00000085563"
    assert s["mane_select_transcript"] == "ENST00000622132"
    assert s["persister_cell_indices"] == [498, 499]
    assert math.isclose(s["persister_prevalence_fraction"], 0.0040, abs_tol=1e-5)
    assert math.isclose(s["cn_cellsnail_rare_persister"], 12.262, abs_tol=1e-2)
    assert math.isclose(s["cn_window_15"], 2.617, abs_tol=1e-3)
    assert math.isclose(s["cn_cohort_mean"], 1.969, abs_tol=1e-2)
    assert math.isclose(s["bounded_edge_fidelity_cellsnail"], 0.9748, abs_tol=1e-3)
    assert math.isclose(s["bounded_edge_fidelity_window_15"], 0.3705, abs_tol=1e-3)
    assert math.isclose(s["excess_cn_washout_ratio_w15"], 16.63, abs_tol=1e-2)


def test_tnbc3_pten_homozygous_deletion_stringer_recovery(exp2_receipt: Dict[str, Any]):
    s = exp2_receipt["window_invisible_focal_stringers"]["tnbc3_focal_stringers"]["PTEN"]
    assert s["bin_index"] == 45
    assert s["locus_tag"] == "chr10:87863113"
    assert s["ensembl_gene_id"] == "ENSG00000171862"
    assert s["mane_select_transcript"] == "ENST00000371953"
    assert math.isclose(s["cn_obs_subclone_a"], 0.123, abs_tol=1e-2)
    assert math.isclose(s["cn_obs_rare_persister"], 0.100, abs_tol=1e-3)
    assert math.isclose(s["cn_window_15"], 1.792, abs_tol=1e-3)
    assert math.isclose(s["deletion_washout_percent_w15"], 89.6, abs_tol=0.1)
    assert s["false_negative_diploid_call_under_w15"] is True


def test_sc_mixology_chromosomal_stringers(exp2_receipt: Dict[str, Any]):
    mix = exp2_receipt["window_invisible_focal_stringers"]["sc_mixology_stringers"]
    assert mix["n_cells"] == 300
    assert mix["n_human_genes"] == 200

    hcc = mix["HCC827_focal_amplicon_stringers"]
    assert math.isclose(hcc["EGFR"]["mean_umi_in_lineage"], 125.83, abs_tol=0.05)
    assert math.isclose(hcc["EGFR"]["fold_enrichment"], 8.56, abs_tol=0.05)
    assert math.isclose(hcc["CDK4"]["mean_umi_in_lineage"], 94.95, abs_tol=0.05)
    assert math.isclose(hcc["CDK4"]["fold_enrichment"], 6.86, abs_tol=0.05)
    assert math.isclose(hcc["CDKN2A"]["mean_umi_in_lineage"], 37.73, abs_tol=0.05)
    assert math.isclose(hcc["CDKN2A"]["fold_enrichment"], 6.10, abs_tol=0.05)

    h1975 = mix["H1975_tki_resistance_stringers"]
    assert math.isclose(h1975["SERPINE1"]["mean_umi_in_lineage"], 34.39, abs_tol=0.05)
    assert math.isclose(h1975["SERPINE1"]["fold_enrichment"], 8.47, abs_tol=0.05)
    assert math.isclose(h1975["DHRS2"]["mean_umi_in_lineage"], 23.83, abs_tol=0.05)
    assert math.isclose(h1975["DHRS2"]["fold_enrichment"], 23.14, abs_tol=0.05)

    h2228 = mix["H2228_oxidative_stress_stringers"]
    assert math.isclose(h2228["NQO1"]["mean_umi_in_lineage"], 84.98, abs_tol=0.05)
    assert math.isclose(h2228["ALDH3A1"]["mean_umi_in_lineage"], 49.41, abs_tol=0.05)
    assert math.isclose(h2228["CD74"]["mean_umi_in_lineage"], 61.12, abs_tol=0.05)
    assert math.isclose(h2228["IGFBP3"]["mean_umi_in_lineage"], 40.35, abs_tol=0.05)


# ==============================================================================
# 4. Persister Subclone & Collateral Vulnerability Discovery Tests
# ==============================================================================


def test_cohort1_tnbc3_rare_persister_mitotic_arrest_and_collateral_targets(exp2_receipt: Dict[str, Any]):
    c1 = exp2_receipt["persister_collateral_vulnerabilities"]["cohort_1_tnbc3_rare_persister"]
    assert c1["n_persister_cells"] == 2
    assert c1["n_total_cells"] == 500
    assert math.isclose(c1["prevalence_percent"], 0.40, abs_tol=1e-3)
    assert c1["persister_barcodes"] == ["TNBC3_AAACCTGCAGACGCAA", "TNBC3_AAACCTGTCACCCGAG"]

    mit = c1["quiescent_mitotic_arrest"]
    assert math.isclose(mit["persister_mean_clock_umi"], 0.30, abs_tol=0.02)
    assert math.isclose(mit["cycling_non_persister_mean_clock_umi"], 1.99, abs_tol=0.02)
    assert math.isclose(mit["mitotic_suppression_fold"], 6.65, abs_tol=0.05)

    col = c1["collateral_co_activated_targets"]
    assert math.isclose(col["ABCB1"]["persister_mean_cn"], 12.263, abs_tol=0.02)
    assert math.isclose(col["ABCB1"]["rna_fold_enrichment"], 12.45, abs_tol=0.05)
    assert math.isclose(col["EGFR"]["persister_mean_cn"], 8.753, abs_tol=0.02)
    assert math.isclose(col["EGFR"]["rna_fold_enrichment"], 14.69, abs_tol=0.05)
    assert math.isclose(col["HSP90AA1"]["persister_mean_umi"], 283.0, abs_tol=0.1)
    assert math.isclose(col["HSP90AA1"]["rna_fold_enrichment"], 3.34, abs_tol=0.05)
    assert math.isclose(col["LY6E"]["persister_mean_umi"], 272.5, abs_tol=0.1)
    assert math.isclose(col["LY6E"]["rna_fold_enrichment"], 3.38, abs_tol=0.05)
    assert math.isclose(col["S100A6"]["persister_mean_umi"], 972.0, abs_tol=0.1)
    assert math.isclose(col["S100A6"]["rna_fold_enrichment"], 15.29, abs_tol=0.05)
    assert math.isclose(col["CDKN2A"]["persister_mean_umi"], 31.5, abs_tol=0.1)
    assert math.isclose(col["CDKN2A"]["rna_fold_enrichment"], 3.11, abs_tol=0.05)
    assert math.isclose(col["LGALS1"]["persister_mean_umi"], 178.5, abs_tol=0.1)
    assert math.isclose(col["LGALS1"]["rna_fold_enrichment"], 2.83, abs_tol=0.05)

    f1 = c1["persister_recovery_f1_comparison"]
    assert f1["cellsnail_morse"] == 1.000
    assert f1["alra_rank5_svd"] < 0.05
    assert f1["magic_graph_diffusion"] < 0.05
    assert f1["hmmcopy_w15_tau5"] == 0.000


def test_cohort2a_gbm_dtp1_quiescent_mesenchymal_stem_persisters(exp2_receipt: Dict[str, Any]):
    c2a = exp2_receipt["persister_collateral_vulnerabilities"]["cohort_2a_gbm_dtp1_quiescent_mesenchymal"]
    assert c2a["geo_accession"] == "GSE57872"
    assert c2a["n_persister_cells"] == 4
    assert c2a["n_total_cells"] == 543
    assert math.isclose(c2a["prevalence_percent"], 0.737, abs_tol=1e-3)
    assert c2a["persister_barcodes"] == ["MGH31_H05", "MGH31_C09", "MGH31_B04", "MGH31_B09"]

    sig = c2a["signature_co_activation"]
    assert math.isclose(sig["AXL_mean_sigma"], 4.80, abs_tol=0.02)
    assert math.isclose(sig["PROM1_CD133_mean_sigma"], 6.98, abs_tol=0.02)
    assert math.isclose(sig["EGFR_mean_sigma"], 5.05, abs_tol=0.02)
    assert math.isclose(sig["CDK4_max_sigma_MGH31_H05"], 4.55, abs_tol=0.02)
    assert math.isclose(sig["mitotic_clock_mean_sigma"], -0.42, abs_tol=0.02)
    assert math.isclose(sig["mitotic_clock_min_sigma_MGH31_H05"], -1.82, abs_tol=0.02)


def test_cohort2b_gbm_dtp2_mosaic_erbb2_axl_persisters(exp2_receipt: Dict[str, Any]):
    c2b = exp2_receipt["persister_collateral_vulnerabilities"]["cohort_2b_gbm_dtp2_mosaic_erbb2_axl"]
    assert c2b["geo_accession"] == "GSE57872"
    assert c2b["n_persister_cells"] == 4
    assert c2b["n_total_cells"] == 543
    assert math.isclose(c2b["prevalence_percent"], 0.737, abs_tol=1e-3)
    assert c2b["persister_barcodes"] == ["MGH30_F12", "MGH30_H03", "MGH30_H11", "CSC6_D03"]

    sig = c2b["signature_co_activation"]
    assert math.isclose(sig["ERBB2_mean_sigma"], 6.62, abs_tol=0.02)
    assert math.isclose(sig["ERBB2_max_sigma"], 8.19, abs_tol=0.02)
    assert math.isclose(sig["AXL_mean_sigma"], 4.51, abs_tol=0.02)
    assert sig["bulk_tumor_average_erbb2"] == 0.0


def test_verified_chembl_nct_and_pdb_structural_linkages(exp2_receipt: Dict[str, Any]):
    c1 = exp2_receipt["persister_collateral_vulnerabilities"]["cohort_1_tnbc3_rare_persister"]
    abcb1_pharm = c1["collateral_co_activated_targets"]["ABCB1"]["pharmacology"]
    assert abcb1_pharm["chembl_target_id"] == "CHEMBL4302"
    assert any(d["chembl_id"] == "CHEMBL348475" for d in abcb1_pharm["drugs"])
    assert any(t["nct_id"] == "NCT00048633" for t in abcb1_pharm["clinical_trials"])
    assert "6QEX" in abcb1_pharm["structural_models"]["pdb_ids"]

    hsp90_pharm = c1["collateral_co_activated_targets"]["HSP90AA1"]["pharmacology"]
    assert hsp90_pharm["chembl_target_id"] == "CHEMBL3880"
    assert "1YET" in hsp90_pharm["structural_models"]["pdb_ids"]

    c2a = exp2_receipt["persister_collateral_vulnerabilities"]["cohort_2a_gbm_dtp1_quiescent_mesenchymal"]
    axl_pharm = c2a["pharmacology"]["AXL"]
    assert axl_pharm["chembl_target_id"] == "CHEMBL4895"
    assert any(d["chembl_id"] == "CHEMBL3545184" for d in axl_pharm["drugs"])
    assert any(t["nct_id"] == "NCT03965002" for t in axl_pharm["clinical_trials"])
    assert "5U6B" in axl_pharm["structural_models"]["pdb_ids"]

    c2b = exp2_receipt["persister_collateral_vulnerabilities"]["cohort_2b_gbm_dtp2_mosaic_erbb2_axl"]
    erbb2_pharm = c2b["pharmacology"]["ERBB2"]
    assert erbb2_pharm["chembl_target_id"] == "CHEMBL1824"
    assert any(d["chembl_id"] == "CHEMBL3989928" for d in erbb2_pharm["drugs"])
    assert "3PP0" in erbb2_pharm["structural_models"]["pdb_ids"]


# ==============================================================================
# 5. Cross-Expedition Convergence Tests (Exp 2 <-> Exp 1 & Exp 2 <-> Exp 3)
# ==============================================================================


def test_cross_expedition_exp1_double_erasure_and_orthogonality(exp2_receipt: Dict[str, Any]):
    conv1 = exp2_receipt["cross_expedition_convergence"]["exp2_to_exp1_double_erasure"]
    assert conv1["shared_genes_evaluated"] == ["EGFR", "MYC", "CDK4", "ERBB2", "CDKN2A", "MDM2", "AXL", "PROM1"]
    assert conv1["tnbc3_orthogonality_residual"] == 0.0
    assert conv1["gbm_orthogonality_residual"] == 0.0
    assert conv1["tnbc3_cellsnail_h_nope_variance_retained"] == 1.0
    assert conv1["gbm_cellsnail_h_nope_variance_retained"] == 1.0

    ols_mat = conv1["exp1_exact_ols_over_subtraction_matrix"]
    assert math.isclose(ols_mat["HSP90AA1"]["tnbc3_delta_5clock"], 0.5612, abs_tol=1e-4)
    assert math.isclose(ols_mat["PTMA"]["tnbc3_delta_5clock"], 0.5452, abs_tol=1e-4)
    assert math.isclose(ols_mat["CDKN2A"]["tnbc3_delta_5clock"], 0.5117, abs_tol=1e-4)
    assert math.isclose(ols_mat["LY6E"]["tnbc3_delta_5clock"], 0.4648, abs_tol=1e-4)
    assert math.isclose(ols_mat["MYC"]["tnbc3_delta_5clock"], 0.2786, abs_tol=1e-4)
    assert math.isclose(ols_mat["CDK4"]["tnbc3_delta_5clock"], 0.2436, abs_tol=1e-4)
    assert math.isclose(ols_mat["EGFR"]["tnbc3_delta_5clock"], 0.1045, abs_tol=1e-4)
    assert "PARP1" in conv1["exp1_replication_coupled_dna_repair_and_chromatin_targets"]


def test_cross_expedition_exp3_spatial_border_sheaf_and_so6_bundle(exp2_receipt: Dict[str, Any]):
    conv3 = exp2_receipt["cross_expedition_convergence"]["exp2_to_exp3_spatial_border_coupling"]
    assert conv3["geo_accession"] == "GSE144239"
    for g in ["EGFR", "MYC", "CDKN2A", "COL1A1", "COL3A1", "B2M", "MMP10", "LAMC2", "ITGA6", "FAP"]:
        assert g in conv3["compartment_expression_profiles"]

    sheaf = conv3["se2_cellular_sheaf_boundary_audit"]
    assert sheaf["se2_sheaf_max_normal_penetration_residual"] == 0.0
    assert sheaf["se2_sheaf_bleed_rate"] == 0.0
    assert sheaf["isotropic_bleed_rate"] > 0.30

    so6 = conv3["so6_multimodal_bundle_audit"]
    assert so6["trace_invariant_verified"] is True
    assert so6["relative_trace_residual"] <= 1.14e-13
    assert so6["orthogonality_error_frobenius"] <= 1e-13

    jux = conv3["exp3_juxtacrine_and_pseudo_emt_border_audit"]
    assert jux["cadherin_switch_audit"]["CDH2_total_umi"] == 0.0
    assert jux["cadherin_switch_audit"]["CDH1_total_umi"] == 1393.0
    assert jux["cadherin_switch_audit"]["CDH3_total_umi"] == 2107.0
    assert jux["layer3_layer4_boundary_pseudo_emt_inflation"]["MMP10_COL1A1"]["isotropic_inflation_percent"] == 319.2
    assert jux["layer3_layer4_boundary_pseudo_emt_inflation"]["LAMC2_COL1A1"]["isotropic_inflation_percent"] == 241.2
    assert jux["layer3_layer4_boundary_pseudo_emt_inflation"]["KRT14_FAP"]["isotropic_inflation_percent"] == 417.0
    assert jux["layer3_layer4_boundary_pseudo_emt_inflation"]["MMP10_VIM"]["isotropic_inflation_percent"] == 286.4
    axes = jux["focal_stringer_to_juxtacrine_wall_axes"]
    assert axes["EGFR_MYC_CDKN2A_invasive_tsk_axis"]["AREG_TGFA_EREG_to_EGFR_polarity_sharpening_range"] == [4.50, 38.76]
    assert axes["COL1A1_COL3A1_FN1_stromal_ecm_wall_axis"]["se2_sheaf_polarity_sharpening_range"] == [2.63, 8.30]
    assert axes["B2M_CD74_HLA_DRA_MIF_immune_checkpoint_axis"]["se2_sheaf_polarity_sharpening_range"] == [2.03, 4.37]


# ==============================================================================
# 6. Adversarial Falsification & Negative-Control Suite Tests
# ==============================================================================


def test_adversarial_control1_100_trial_permutation_invariance(exp2_receipt: Dict[str, Any]):
    c1 = exp2_receipt["adversarial_falsification_audit"]["control_1_column_permutation_invariance"]
    assert c1["n_permutation_trials"] == 100
    assert c1["sort_by_genomic_coords_true"]["mean_jaccard_concordance"] == 1.0000
    assert c1["sort_by_genomic_coords_true"]["min_jaccard_concordance"] == 1.0000
    assert c1["sort_by_genomic_coords_true"]["perfect_concordance_trials"] == 100
    assert c1["sort_by_genomic_coords_true"]["pten_homdel_bin45_invariance_rate"] == 1.0000
    assert c1["sort_by_genomic_coords_true"]["subclone_b_12q_coamplicon_invariance_rate"] == 1.0000
    assert c1["sort_by_genomic_coords_true"]["gbm_dtp1_dtp2_and_mixology_invariance_rate"] == 1.0000
    assert c1["sort_by_genomic_coords_true"]["exact_permutation_invariance_verified"] is True
    assert c1["sort_by_genomic_coords_false"]["degradation_demonstrated"] is True
    assert c1["sort_by_genomic_coords_false"]["mean_jaccard_concordance"] < 0.50


def test_adversarial_control2_diploid_stromal_zero_fpr(exp2_receipt: Dict[str, Any]):
    c2 = exp2_receipt["adversarial_falsification_audit"]["control_2_diploid_stromal_negative_control"]
    assert c2["n_diploid_normal_cells_evaluated"] == 320
    assert c2["false_positive_focal_stringers_mean_profile"] == 0
    assert c2["false_positive_single_cells"] == 0
    assert c2["false_positive_homozygous_deletions"] == 0
    assert c2["false_positive_rate_fpr"] == 0.0000
    assert c2["ercc92_spike_in_negative_control"]["cellsnail_morse_spurious_pairs_r070"] == 0
    assert c2["ercc92_spike_in_negative_control"]["fpr_ercc_ghost_regulons"] == 0.0
    assert c2["zero_false_positive_verified"] is True


def test_adversarial_control3_20_seed_countsplit_holdout_replication(exp2_receipt: Dict[str, Any]):
    c3 = exp2_receipt["adversarial_falsification_audit"]["control_3_countsplit_molecular_thinning_holdout"]
    assert c3["n_independent_seeds"] == 20
    assert c3["tnbc3_persister_collateral_panel"]["out_of_sample_replication_rate"] == 1.0000
    assert c3["tnbc3_persister_collateral_panel"]["min_split_correlation_r"] > 0.98
    assert c3["sc_mixology_stringer_panel"]["out_of_sample_replication_rate"] == 1.0000
    assert c3["sc_mixology_stringer_panel"]["min_split_correlation_r"] > 0.98
    assert c3["sc_mixology_full_9gene_stringer_panel"]["out_of_sample_replication_rate"] == 1.0000
    assert c3["sc_mixology_full_9gene_stringer_panel"]["min_split_correlation_r"] > 0.98
    assert c3["gbm_dtp1_dtp2_persister_holdout_panel"]["out_of_sample_replication_rate"] == 1.0000
    assert c3["gbm_dtp1_dtp2_persister_holdout_panel"]["min_split_correlation_r"] > 0.98
    assert c3["zero_poisson_spike_artifacts_verified"] is True
    assert exp2_receipt["adversarial_falsification_audit"]["all_adversarial_controls_passed"] is True


def test_cli_entrypoint_run_expedition2_stringer_hunt(tmp_path: Path):
    out_file = tmp_path / "exp2_test.json"
    res = run_expedition2_stringer_hunt(output_json_path=out_file, random_state=42)
    assert out_file.exists()
    assert res["status"] == "EXPEDITION_2_COMPLETE"
