"""
Comprehensive Unit, Cross-Orchestrator Convergence & Adversarial Falsification Test Suite
for Expedition 6: Pan-Autosome GRCh38 Multi-Patient Uncharted 1-3 Gene Stringer & Persister Atlas
==================================================================================================

Author: Noah Cohen
Repository: CellSnail (v0.2.0)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cellsnail.expeditions.exp6_pan_autosome_stringer_atlas import (
    PanAutosomeStringerAtlas,
    run_expedition6_pan_autosome_stringer_atlas,
)


@pytest.fixture(scope="module")
def exp6_atlas() -> PanAutosomeStringerAtlas:
    return PanAutosomeStringerAtlas(random_state=42)


@pytest.fixture(scope="module")
def exp6_results(exp6_atlas: PanAutosomeStringerAtlas) -> dict:
    return exp6_atlas.run_all()


# ==============================================================================
# 1. Pipeline Execution, Determinism & Schema Integrity Tests
# ==============================================================================


def test_01_atlas_instantiation_and_fast_execution(exp6_results: dict) -> None:
    """Verify PanAutosomeStringerAtlas executes all 5 stages in < 5.0 seconds and credits Noah Cohen."""
    assert exp6_results["author"] == "Noah Cohen"
    assert exp6_results["version"] == "0.2.0"
    assert exp6_results["execution_time_seconds"] < 5.0
    for stage_key in (
        "stage1_pan_autosome_window_washout_and_hvg_census",
        "stage2_two_tier_stringers_and_microdeletions",
        "stage3_multi_donor_persister_collateral_atlas",
        "stage4_cross_orchestrator_scaleup_convergence",
        "stage5_adversarial_falsification_and_lodo_audit",
    ):
        assert stage_key in exp6_results


def test_02_strict_json_serialization_and_results_artifact(tmp_path: Path) -> None:
    """Verify run_expedition6_pan_autosome_stringer_atlas serializes with allow_nan=False."""
    custom_json = tmp_path / "exp6_test_out.json"
    res = run_expedition6_pan_autosome_stringer_atlas(output_json_path=custom_json, random_state=42)
    assert custom_json.exists()
    raw_text = custom_json.read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text
    parsed = json.loads(raw_text)
    assert parsed["scale_up_summary"]["total_single_cells_analyzed"] == 10058
    assert res["scale_up_summary"]["total_autosomes_covered"] == 22

    repo_results_json = (
        Path(__file__).resolve().parents[1] / "results" / "expedition6_pan_autosome_stringer_atlas.json"
    )
    assert repo_results_json.exists()
    assert repo_results_json.stat().st_size > 500_000


def test_03_deterministic_reproducibility_across_runs() -> None:
    """Verify two independent runs with random_state=42 produce identical numerical metrics."""
    r1 = PanAutosomeStringerAtlas(random_state=42).run_all()
    r2 = PanAutosomeStringerAtlas(random_state=42).run_all()
    r1_clean = {k: v for k, v in r1.items() if k != "execution_time_seconds"}
    r2_clean = {k: v for k, v in r2.items() if k != "execution_time_seconds"}
    assert json.dumps(r1_clean, sort_keys=True) == json.dumps(r2_clean, sort_keys=True)


# ==============================================================================
# 2. Multi-Cohort & 22-Autosome Scale-Up Census Tests (Stage 1)
# ==============================================================================


def test_04_scale_up_summary_10058_cells_and_17_groups(exp6_results: dict) -> None:
    """Verify the scale-up covers 10,058 single cells across 17 patient/lineage groups and 22 autosomes."""
    summary = exp6_results["scale_up_summary"]
    assert summary["total_single_cells_analyzed"] == 10058
    assert summary["total_patient_and_lineage_groups"] == 17
    assert summary["total_autosomes_covered"] == 22
    assert summary["grch38_autosomal_reference_genes"] == 44544
    assert summary["tnbc_multi_patient_cells"] == 8944
    assert summary["tnbc_tumor_cells"] == 5459
    assert summary["tnbc_diploid_normal_cells"] == 3485
    assert summary["tnbc_tumor_cells"] + summary["tnbc_diploid_normal_cells"] == 8944
    assert summary["tnbc_expressed_autosomal_genes"] == 19589
    assert summary["gbm_multi_patient_cells"] == 543
    assert summary["gbm_expressed_autosomal_genes"] == 5543
    assert summary["sc_mixology_5cl_cells"] == 571
    assert summary["sc_mixology_shared_autosomal_genes"] == 12409
    assert 8944 + 543 + 571 == 10058


def test_05_cohort_a_tnbc_five_donor_breakdown(exp6_results: dict) -> None:
    """Verify Cohort A (GSE148673) 5-patient TNBC1-TNBC5 cell and tumor/normal counts."""
    tnbc = exp6_results["cohorts"]["Cohort_A_GSE148673_Multi_Patient_TNBC"]
    donors = tnbc["donors"]
    assert set(donors.keys()) == {"TNBC1", "TNBC2", "TNBC3", "TNBC4", "TNBC5"}
    assert donors["TNBC1"]["n_cells"] == 1097
    assert donors["TNBC1"]["n_tumor_cells"] == 797
    assert donors["TNBC1"]["n_diploid_normal_cells"] == 300
    assert donors["TNBC2"]["n_cells"] == 1034
    assert donors["TNBC3"]["n_cells"] == 532
    assert donors["TNBC4"]["n_cells"] == 3056
    assert donors["TNBC5"]["n_cells"] == 3225
    assert sum(d["n_cells"] for d in donors.values()) == 8944
    assert sum(d["n_tumor_cells"] for d in donors.values()) == 5459
    assert sum(d["n_diploid_normal_cells"] for d in donors.values()) == 3485


def test_06_cohort_b_gbm_and_cohort_c_sc_mixology_breakdown(exp6_results: dict) -> None:
    """Verify Cohort B (GSE57872 7-group GBM) and Cohort C (GSE118767 5-line sc_mixology) counts."""
    gbm = exp6_results["cohorts"]["Cohort_B_GSE57872_Multi_Patient_GBM"]
    assert sum(gbm["donors"].values()) == 543
    assert set(gbm["donors"].keys()) == {
        "MGH26", "MGH28", "MGH29", "MGH30", "MGH31", "CSC6", "CSC8"
    }

    mix = exp6_results["cohorts"]["Cohort_C_GSE118767_sc_mixology_5_Line_Lung"]
    assert sum(mix["cell_lines"].values()) == 571
    assert set(mix["cell_lines"].keys()) == {"HCC827", "H1975", "H2228", "H838", "A549"}


def test_07_pan_autosome_22_chromosome_atlas_completeness(exp6_results: dict) -> None:
    """Verify all 22 human autosomes (chr1-chr22) are profiled with lead stringers and >55% W15 washout."""
    chrom_atlas = exp6_results["pan_autosome_22_chromosome_atlas"]
    assert len(chrom_atlas) == 22
    expected_chroms = [f"chr{i}" for i in range(1, 23)]
    assert [c["chrom"] for c in chrom_atlas] == expected_chroms
    assert sum(c["grch38_reference_genes"] for c in chrom_atlas) == 44544
    assert sum(c["tnbc_expressed_genes"] for c in chrom_atlas) == 19589
    for c in chrom_atlas:
        assert c["multi_donor_uncharted_stringers"] > 0
        assert abs(c["lead_peak_cn_cellsnail"] - 2.0) > abs(c["lead_cn_w15"] - 2.0)
        assert abs(c["lead_peak_cn_cellsnail"] - 2.0) > abs(c["lead_cn_w51"] - 2.0)
        if c["lead_peak_cn_cellsnail"] > 2.0:
            assert c["lead_attenuation_w15_pct"] >= 55.0
            assert c["lead_attenuation_w51_pct"] >= 73.0


def test_08_live_morse_stencil_vs_boxcar_window_attenuation_law(exp6_results: dict) -> None:
    """Verify live 3-point Morse-Smale CNV filter vs W in {15, 51, 101} analytical L/W law."""
    stage1 = exp6_results["stage1_pan_autosome_window_washout_and_hvg_census"]
    ver = stage1["theoretical_and_live_stencil_vs_boxcar_verification"]
    for width_l in (1, 2, 3):
        entry = ver[f"L{width_l}"]
        assert entry["cellsnail_morse_detected"] is True
        assert entry["cellsnail_retained_fraction"] == pytest.approx(1.0, abs=1e-3)
        assert entry["w15_retained_fraction"] == pytest.approx(width_l / 15.0, abs=1e-3)
        assert entry["w51_retained_fraction"] == pytest.approx(width_l / 51.0, abs=1e-3)
        assert entry["w101_retained_fraction"] == pytest.approx(width_l / 101.0, abs=1e-3)

    del_ver = stage1["micro_deletion_false_roh_verification"]
    assert del_ver["cellsnail_morse_detected"] is True
    assert del_ver["false_roh_under_w15"] is True
    assert del_ver["w15_smoothed_cn"] > 1.80


# ==============================================================================
# 3. Two-Tier Autosomal Stringer, L=2-3 Amplicon & Micro-Deletion Tests (Stage 2)
# ==============================================================================


def test_09_tier_a_18_non_hvg_uncharted_stringers(exp6_results: dict) -> None:
    """Verify all 18 Tier A Non-HVG Uncharted Autosomal Stringers and their pharmacology linkages."""
    tier_a = exp6_results["tier_a_non_hvg_uncharted_stringers"]
    assert len(tier_a) == 18
    expected_tier_a = {
        "YES1", "PBK", "PTK2", "PVR", "WEE1", "IGF1R", "BRD4", "ROCK2", "AURKB",
        "FOLR1", "WWTR1", "CDK6", "BCL2L1", "CRKL", "EPHB4", "SLC2A1", "PFKP", "KLF5",
    }
    assert {g["gene_symbol"] for g in tier_a} == expected_tier_a
    for g in tier_a:
        assert g["is_non_hvg_top2000"] is True
        assert g["excess_cn_attenuation_w15_pct"] >= 50.0
        assert g["excess_cn_attenuation_w51_pct"] >= 40.0
        assert g["ensembl_gene_id"].startswith("ENSG")
        assert g["mane_select_transcript"].startswith("ENST")
        assert g["chembl_pharmacology"]["chembl_target_id"].startswith("CHEMBL")
        assert len(g["chembl_pharmacology"]["curated_approved_and_clinical_inhibitors"]) >= 1


def test_10_tier_b_18_actionable_repurposing_stringers(exp6_results: dict) -> None:
    """Verify all 18 Tier B Actionable Repurposing Stringers and their FDA/clinical inhibitors."""
    tier_b = exp6_results["tier_b_actionable_repurposing_stringers"]
    assert len(tier_b) == 18
    expected_tier_b = {
        "NECTIN4", "TACSTD2", "TOP1", "KDM5B", "RB1CC1", "PARP1", "BIRC5", "PHGDH", "PROM1",
        "HSP90AA1", "HSP90AB1", "HDAC2", "GPX4", "IDH2", "GLS", "AURKA", "MCL1", "SMARCA4",
    }
    assert {g["gene_symbol"] for g in tier_b} == expected_tier_b
    for g in tier_b:
        assert g["excess_cn_attenuation_w15_pct"] >= 75.0
        assert g["excess_cn_attenuation_w51_pct"] >= 70.0
        assert g["ensembl_gene_id"].startswith("ENSG")
        assert g["chembl_pharmacology"]["chembl_target_id"].startswith("CHEMBL")
        assert len(g["chembl_pharmacology"]["curated_approved_and_clinical_inhibitors"]) >= 1


def test_11_contiguous_l2_l3_multi_gene_stringers_and_pkmyt1_triplet(exp6_results: dict) -> None:
    """Verify all 6 contiguous L=2-3 multi-gene stringers, including the 16.17 kb chr16p13.3 PKMYT1 triplet."""
    l2_l3 = exp6_results["contiguous_multi_gene_stringers_L2_L3"]
    assert len(l2_l3) == 6
    by_id = {x["amplicon_id"]: x for x in l2_l3}
    assert "chr16p13.3_KREMEN2_PAQR4_PKMYT1_L3" in by_id
    pkmyt1_triplet = by_id["chr16p13.3_KREMEN2_PAQR4_PKMYT1_L3"]
    assert pkmyt1_triplet["genes"] == ["KREMEN2", "PAQR4", "PKMYT1"]
    assert pkmyt1_triplet["width_genes_L"] == 3
    assert pkmyt1_triplet["span_kb"] == pytest.approx(16.17, abs=0.1)
    assert pkmyt1_triplet["n_active_donors"] == 3
    assert pkmyt1_triplet["mean_triplet_cn_cellsnail"] == pytest.approx(8.74, abs=0.05)
    assert pkmyt1_triplet["mean_triplet_cn_w15"] == pytest.approx(4.10, abs=0.05)


def test_12_uncharted_focal_micro_deletions_and_synthetic_lethality(exp6_results: dict) -> None:
    """Verify all 8 uncharted focal micro-deletions (TGFBR2, CD74, ALDH1A1, BCL2, KDR, ARID1A, MTAP, STK11)."""
    micro_dels = exp6_results["uncharted_focal_micro_deletions"]
    assert len(micro_dels) == 8
    del_syms = {m["gene_symbol"] for m in micro_dels}
    assert del_syms == {"TGFBR2", "CD74", "ALDH1A1", "BCL2", "KDR", "ARID1A", "MTAP", "STK11"}
    by_sym = {m["gene_symbol"]: m for m in micro_dels}
    assert by_sym["TGFBR2"]["min_tumor_cn_cellsnail"] < 0.10
    assert by_sym["TGFBR2"]["window_15_smoothed_cn"] > 1.25
    assert by_sym["BCL2"]["min_tumor_cn_cellsnail"] < 0.05
    assert by_sym["BCL2"]["window_15_smoothed_cn"] > 1.80
    assert by_sym["CD74"]["n_deleted_donors"] == 5
    assert "EZH2" in " ".join(by_sym["ARID1A"]["synthetic_lethal_partners"])
    assert "PRMT5" in " ".join(by_sym["MTAP"]["synthetic_lethal_partners"])


def test_13_twenty_three_autosomal_axes_and_86_dossier_targets(exp6_results: dict) -> None:
    """Verify all 23 autosomal cytoband axes and 86 verified targets in pan_autosome_target_dictionary."""
    axes = exp6_results["twenty_three_autosomal_axis_groups"]
    assert len(axes) == 23
    target_dict = exp6_results["pan_autosome_target_dictionary"]
    assert len(target_dict) == 86
    for key_gene in (
        "MCL1", "ARID1A", "XPO1", "NFE2L2", "TGFBR2", "PDGFRA", "FGFR1", "FGFR2",
        "MTAP", "WEE1", "BRCA2", "AKT1", "IGF1R", "ABCC1", "PKMYT1", "BIRC5", "YES1",
        "BCL2", "STK11", "GPX4", "BRD4", "CCNE1", "AXL", "AURKA", "BCL2L1", "CRKL",
    ):
        assert key_gene in target_dict
        entry = target_dict[key_gene]
        assert entry["ensembl_gene_id"].startswith("ENSG")
        assert entry["structural_biology"]["alphafold_global_plddt"] > 40.0


def test_14_gbm_and_sc_mixology_multi_lineage_stringers(exp6_results: dict) -> None:
    """Verify multi-patient GBM (GSE57872) and 5-lineage sc_mixology (GSE118767) stringer recovery."""
    stage2 = exp6_results["stage2_two_tier_stringers_and_microdeletions"]
    gbm_str = stage2["gbm_multi_patient_uncharted_stringers"]
    mix_str = stage2["sc_mixology_5_line_uncharted_stringers"]
    assert len(gbm_str) >= 10
    assert len(mix_str) >= 12
    gbm_genes = {g["gene_symbol"] for g in gbm_str}
    mix_genes = {g["gene_symbol"] for g in mix_str}
    assert {"BIRC5", "PROM1", "SERPINE1", "AURKB", "CD74", "PBK", "AXL", "WWTR1", "IGF1R", "WEE1"}.issubset(
        gbm_genes
    )
    assert {"DHRS2", "SERPINE1", "ALDH1A1", "NQO1", "CD74", "NT5E", "SLC7A11", "LGALS1", "CA12", "TACSTD2"}.issubset(
        mix_genes
    )


# ==============================================================================
# 4. Multi-Donor Rare Drug-Tolerant Persister (DTP) Atlas Tests (Stage 3)
# ==============================================================================


def test_15_tnbc_five_donor_dtp_prevalence_and_mitotic_silence(exp6_results: dict) -> None:
    """Verify rare zero-mitotic DTP persisters are isolated across all 5 TNBC donors (0.26%-0.56%)."""
    pers = exp6_results["multi_patient_rare_persister_atlas"]
    assert pers["total_tnbc_persister_cells_all_tiers"] == 35
    assert pers["overall_tnbc_persister_prevalence_pct_all_tiers"] == pytest.approx(0.3913, abs=1e-3)
    assert pers["total_tnbc_persister_cells_strict_core"] == 28
    assert pers["overall_tnbc_persister_prevalence_pct_strict_core"] == pytest.approx(0.3131, abs=1e-3)

    per_donor = pers["tnbc_per_donor_summary"]
    assert set(per_donor.keys()) == {"TNBC1", "TNBC2", "TNBC3", "TNBC4", "TNBC5"}
    assert sum(d["n_dtp"] for d in per_donor.values()) == 35
    assert sum(d["n_dtp_strict_core"] for d in per_donor.values()) == 28
    for donor_id, d in per_donor.items():
        assert 0.20 <= d["prevalence_pct"] <= 0.60, f"{donor_id} prevalence out of range"
        assert d["mitotic_suppression_fold"] > 1.30


def test_16_tnbc_and_gbm_dtp_collateral_druggable_targets(exp6_results: dict) -> None:
    """Verify enriched collateral vulnerabilities across TNBC1-TNBC5 DTPs and GBM-DTP1/DTP2."""
    pers = exp6_results["multi_patient_rare_persister_atlas"]
    tnbc_dtps = pers["tnbc_5_donor_persisters"]
    assert tnbc_dtps["TNBC1"]["enriched_targets"]["WEE1"]["dtp_vs_bulk_fold"] >= 5.0
    assert tnbc_dtps["TNBC1"]["enriched_targets"]["MCL1"]["dtp_vs_bulk_fold"] >= 4.5
    assert tnbc_dtps["TNBC1"]["enriched_targets"]["PROM1"]["dtp_vs_bulk_fold"] >= 4.4
    assert tnbc_dtps["TNBC2"]["enriched_targets"]["FOLR1"]["dtp_vs_bulk_fold"] >= 8.0
    assert tnbc_dtps["TNBC2"]["enriched_targets"]["CD74"]["dtp_vs_bulk_fold"] >= 14.0

    gbm_dtps = pers["gbm_multi_patient_persisters"]
    assert gbm_dtps["GBM_DTP1_Proneural_Stem"]["n_dtp_cells"] == 4
    assert gbm_dtps["GBM_DTP1_Proneural_Stem"]["prevalence_pct"] == pytest.approx(0.7366, abs=1e-3)
    assert gbm_dtps["GBM_DTP1_Proneural_Stem"]["enriched_collateral_targets"]["PROM1"]["z_score"] == pytest.approx(
        6.98, abs=0.05
    )
    assert gbm_dtps["GBM_DTP2_Mesenchymal_Bypass"]["n_dtp_cells"] == 4
    assert gbm_dtps["GBM_DTP2_Mesenchymal_Bypass"]["enriched_collateral_targets"]["AXL"]["z_score"] > 3.0


def test_17_live_deconfounder_persister_collateral_preservation(exp6_results: dict) -> None:
    """Verify live cell_cycle_deconfounder preserves collateral target expression in zero-mitotic DTPs."""
    live_ver = exp6_results["multi_patient_rare_persister_atlas"]["live_operator_persister_verification"]
    assert live_ver["dtp_mitotic_clock_umi"] == 0.0
    assert live_ver["deconfounded_collateral_target_retention_ratio"] == pytest.approx(1.0, abs=1e-3)
    assert live_ver["clock_columns_flagged"] == 4
    assert live_ver["morse_confirmed_collateral_innovations"] >= 1


# ==============================================================================
# 5. Cross-Orchestrator Scale-Up Convergence Tests (Stage 4: Exp 4, Exp 5, Exp 7)
# ==============================================================================


def test_18_exp4_whole_genome_streamer_tile_boundary_invariance(exp6_results: dict) -> None:
    """Verify Expedition 4 WholeGenomeArenaStreamer achieves 100% stringer recall and 0.0 seam drift."""
    exp4 = exp6_results["stage4_cross_orchestrator_scaleup_convergence"]["exp4_streaming_integration"]
    assert exp4["arena_bytes_allocated"] == 983872
    assert exp4["rss_growth_bytes"] == 0
    assert exp4["max_tile_seam_laplacian_drift"] == 0.0
    assert exp4["streamer_boundary_audit_passed"] is True
    for b_key in ("B128", "B256", "B512", "B1024"):
        assert exp4["flagship_36_stringer_recall_by_tile_size"][b_key] == 1.0


def test_19_exp5_triple_erased_dark_genome_convergence(exp6_results: dict) -> None:
    """Verify Expedition 5 convergence on triple-erased stringers (PKMYT1, WEE1, YES1, AURKA, BIRC5, PARP1, TOP1, HSP90AB1)."""
    exp5 = exp6_results["stage4_cross_orchestrator_scaleup_convergence"]["exp5_20_clock_deconfounding_convergence"]
    assert exp5["convergent_stringer_count"] == 8
    assert exp5["mean_ols_20clock_oversubtraction_pct"] > 65.0
    assert exp5["mean_cellsnail_retention_pct"] > 94.0
    by_sym = {g["gene_symbol"]: g for g in exp5["convergent_genes"]}
    assert by_sym["PKMYT1"]["hvg_variance_rank"] == 6146
    assert by_sym["PKMYT1"]["exp5_ols_20clock_oversubtraction_pct"] == pytest.approx(59.0, abs=0.5)
    assert by_sym["WEE1"]["hvg_variance_rank"] == 3200
    assert by_sym["WEE1"]["exp5_ols_20clock_oversubtraction_pct"] == pytest.approx(82.7, abs=0.5)
    assert by_sym["BIRC5"]["exp5_ols_20clock_oversubtraction_pct"] == pytest.approx(88.2, abs=0.5)


def test_20_exp7_spatial_se2_sheaf_1cochain_barrier_convergence(exp6_results: dict) -> None:
    """Verify Expedition 7 spatial SE(2) sheaf 1-cochain barriers couple to intracellular Exp 6 stringers."""
    exp7 = exp6_results["stage4_cross_orchestrator_scaleup_convergence"]["exp7_spatial_sheaf_1cochain_convergence"]
    assert exp7["convergent_spatial_axes_count"] == 6
    assert exp7["live_se2_sheaf_max_edge_energy"] > 0.0
    assert exp7["live_so6_bundle_stalk_shape"] == [6, 6]
    pairs = [a["spatial_lr_pair"] for a in exp7["convergent_spatial_axes"]]
    assert any("FN1 -> ITGA5" in p for p in pairs)
    assert any("EFNB2 -> EPHB4" in p for p in pairs)
    assert any("ENTPD1 -> NT5E" in p for p in pairs)
    assert any("GAS6 -> AXL" in p for p in pairs)
    assert any("LGALS1 -> PTPRC" in p for p in pairs)
    assert any("MIF -> CD74" in p for p in pairs)


# ==============================================================================
# 6. Four-Way Adversarial Falsification & Generalization Tests (Stage 5)
# ==============================================================================


def test_21_control_1_lodo_cross_patient_generalization(exp6_results: dict) -> None:
    """Verify Control 1: 5-fold TNBC LODO (88.79% replication, r=0.7431) and 7-fold GBM LODO."""
    c1 = exp6_results["falsification_and_lodo_metrics"]["control_1_lodo_cross_patient_replication"]
    assert len(c1["tnbc_5_fold_lodo"]) == 5
    assert c1["mean_tnbc_lodo_replication_rate"] == pytest.approx(0.8879, abs=1e-3)
    assert c1["mean_tnbc_lodo_stencil_corr_r"] == pytest.approx(0.7431, abs=1e-3)
    for fold in c1["tnbc_5_fold_lodo"]:
        assert fold["lodo_replication_rate"] >= 0.80
        assert fold["cross_donor_stencil_profile_corr_r"] >= 0.52

    assert len(c1["gbm_7_fold_lodo"]) == 7
    assert c1["mean_gbm_lodo_stencil_corr_r"] == pytest.approx(0.4277, abs=1e-3)


def test_22_control_2_diploid_normal_3485_cells_zero_fpr(exp6_results: dict) -> None:
    """Verify Control 2: 3,485 diploid normal cells across TNBC1-TNBC5 and 92 ERCC92 spike-ins yield 0.0000 FPR."""
    c2 = exp6_results["falsification_and_lodo_metrics"]["control_2_diploid_normal_negative_control"]
    assert c2["total_diploid_normal_cells_evaluated"] == 3485
    assert c2["false_positive_donors"] == 0
    assert c2["false_positive_bins_across_97945_tests"] == 0
    assert c2["empirical_fpr"] == 0.0
    assert c2["live_operator_fp_count"] == 0
    assert c2["ercc92_spike_in_bins_evaluated"] == 92
    assert c2["ercc92_false_positive_bins"] == 0
    assert c2["ercc92_empirical_fpr"] == 0.0
    for donor_id in ("TNBC1", "TNBC2", "TNBC3", "TNBC4", "TNBC5"):
        assert c2["per_donor_audit"][donor_id]["false_positive_stringer_bins"] == 0
        assert c2["per_donor_audit"][donor_id]["fpr"] == 0.0


def test_23_control_3_coordinate_permutation_invariance_100_trials(exp6_results: dict) -> None:
    """Verify Control 3: 100 random coordinate permutations yield Jaccard = 1.0000 when sorted."""
    c3 = exp6_results["falsification_and_lodo_metrics"]["control_3_coordinate_permutation_invariance"]
    assert c3["n_trials"] == 100
    assert c3["n_autosomal_genes_permuted"] == 19589
    assert c3["jaccard_concordance_sorted_mean"] == 1.0
    assert c3["jaccard_concordance_sorted_min"] == 1.0
    assert c3["live_sorted_jaccard"] == 1.0
    assert c3["permutation_fdr"] < 0.001


def test_24_control_4_countsplit_20_seed_binomial_holdout(exp6_results: dict) -> None:
    """Verify Control 4: 20-seed binomial count-splitting holdout achieves r >= 0.9975 and 20/20 replication."""
    c4 = exp6_results["falsification_and_lodo_metrics"]["control_4_countsplit_20seed_holdout"]
    assert c4["n_seeds"] == 20
    assert c4["tnbc_multi_patient_mean_r"] == pytest.approx(0.9975, abs=1e-3)
    assert c4["tnbc_multi_patient_min_r"] >= 0.994
    assert c4["sc_mixology_5line_mean_r"] == pytest.approx(0.9979, abs=1e-3)
    assert c4["replication_rate_20_of_20"] == 1.0
    assert c4["live_operator_split_corr_r"] > 0.95
    assert c4["poisson_spike_artifacts_detected"] == 0


def test_25_chembl_clinical_molecules_catalog_completeness(exp6_results: dict) -> None:
    """Verify chembl_clinical_molecules_catalog contains >= 55 verified clinical molecules with ChEMBL IDs."""
    cat = exp6_results["stage2_two_tier_stringers_and_microdeletions"]["chembl_clinical_molecules_catalog"]
    assert len(cat) >= 55
    for drug_key in (
        "tapotoclax", "tazemetostat", "ceralasertib", "selinexor", "adavosertib",
        "lunresertib", "rp-6306", "dasatinib", "olaparib", "talazoparib",
        "alisertib", "venetoclax", "sotorasib", "adagrasib",
    ):
        assert drug_key in cat
        assert str(cat[drug_key]["molecule_chembl_id"]).startswith("CHEMBL")


def test_26_hunter_alias_and_expeditions_package_export() -> None:
    """Verify PanAutosomeStringerAtlasHunter alias is exported cleanly by cellsnail.expeditions."""
    from cellsnail.expeditions import (
        PanAutosomeStringerAtlasHunter,
        run_expedition6_pan_autosome_stringer_atlas as run_exp6,
    )

    assert PanAutosomeStringerAtlasHunter is PanAutosomeStringerAtlas
    hunter = PanAutosomeStringerAtlasHunter(random_state=42)
    stage1 = hunter.compute_pan_autosome_window_washout_and_hvg_census()
    assert stage1["total_autosomes_profiled"] == 22
    assert callable(run_exp6)


def test_27_cd74_compartment_resolved_bulk_deletion_vs_persister_surge(exp6_results: dict) -> None:
    """Verify CD74 (chr5q33.1) bulk tumor micro-deletion across 5/5 TNBC donors vs 14.59x DTP and 13.36x H2228 surge."""
    micro_dels = {m["gene_symbol"]: m for m in exp6_results["uncharted_focal_micro_deletions"]}
    assert "CD74" in micro_dels
    cd74_del = micro_dels["CD74"]
    assert cd74_del["n_deleted_donors"] == 5
    assert cd74_del["min_tumor_cn_cellsnail"] == pytest.approx(0.019, abs=1e-3)

    tnbc2_dtp_cd74 = exp6_results["multi_patient_rare_persister_atlas"]["tnbc_5_donor_persisters"]["TNBC2"][
        "enriched_targets"
    ]["CD74"]
    assert tnbc2_dtp_cd74["dtp_vs_bulk_fold"] == pytest.approx(14.59, abs=0.05)


def test_28_bcl2_chr18q21_deletion_to_mcl1_bcl2l1_synthetic_lethality(exp6_results: dict) -> None:
    """Verify BCL2 (chr18q21.33) focal deletion creates collateral dependency on amplified MCL1 and BCL2L1."""
    micro_dels = {m["gene_symbol"]: m for m in exp6_results["uncharted_focal_micro_deletions"]}
    bcl2_del = micro_dels["BCL2"]
    assert bcl2_del["chrom"] == "chr18"
    assert bcl2_del["n_deleted_donors"] == 4
    assert bcl2_del["false_negative_diploid_call_under_w15"] is True

    tier_a = {g["gene_symbol"]: g for g in exp6_results["tier_a_non_hvg_uncharted_stringers"]}
    tier_b = {g["gene_symbol"]: g for g in exp6_results["tier_b_actionable_repurposing_stringers"]}
    assert tier_a["BCL2L1"]["peak_cn_cellsnail"] > 5.5
    assert tier_b["MCL1"]["peak_cn_cellsnail"] >= 6.0


def test_29_markdown_and_json_numerical_and_author_parity(exp6_results: dict) -> None:
    """Verify EXPEDITION_6_PAN_AUTOSOME_STRINGER_ATLAS.md matches JSON metrics and credits Noah Cohen."""
    repo_root = Path(__file__).resolve().parents[1]
    md_path = repo_root / "docs" / "EXPEDITION_6_PAN_AUTOSOME_STRINGER_ATLAS.md"
    assert md_path.exists()
    md_text = md_path.read_text(encoding="utf-8")

    # Author attribution in header and BibTeX block
    assert "**Lead Author & System Architect:** Noah Cohen" in md_text
    assert "author       = {Cohen, Noah}" in md_text

    # Exact parity on 22-autosome counts and stringer washouts
    for chrom_row in exp6_results["pan_autosome_22_chromosome_atlas"]:
        ref_str = f"{chrom_row['grch38_reference_genes']:,}"
        tnbc_str = f"{chrom_row['tnbc_expressed_genes']:,}"
        assert f"| `{chrom_row['chrom']}` | {ref_str} | {tnbc_str} |" in md_text

    for row in exp6_results["tier_a_non_hvg_uncharted_stringers"]:
        assert f"| `{row['gene_symbol']}` | `chr{row['cytoband']}` `{row['grch38_locus']}` |" in md_text
        assert f"| {row['peak_cn_cellsnail']:.2f} | {row['cn_hmmcopy_w15']:.2f} / {row['cn_infercnv_w51']:.2f} | {row['excess_cn_attenuation_w15_pct']:.2f}% / {row['excess_cn_attenuation_w51_pct']:.2f}% |" in md_text

    for row in exp6_results["tier_b_actionable_repurposing_stringers"]:
        assert f"| `{row['gene_symbol']}` | `chr{row['cytoband']}` `{row['grch38_locus']}` |" in md_text
        assert f"| {row['peak_cn_cellsnail']:.2f} | {row['cn_hmmcopy_w15']:.2f} / {row['cn_infercnv_w51']:.2f} | {row['excess_cn_attenuation_w15_pct']:.2f}% / {row['excess_cn_attenuation_w51_pct']:.2f}% |" in md_text

