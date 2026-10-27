"""
Comprehensive Verification & Adversarial Falsification Test Suite for Expedition 7:
Whole-Slide 17,138-Gene Spatial Secretome & Multi-Slide cSCC Replicate Discovery
================================================================================

Tests every empirical, mathematical, biological-database, multi-slide,
cross-orchestrator, and adversarial invariant of:
  - `cellsnail/expeditions/exp7_whole_slide_spatial_secretome.py`
  - `results/expedition7_whole_slide_spatial_secretome.json`
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from cellsnail.expeditions.exp7_whole_slide_spatial_secretome import (
    WholeSlideSpatialSecretomeScanner,
    run_expedition7_whole_slide_spatial_secretome,
)


@pytest.fixture(scope="module")
def exp7_scanner_and_results(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """Execute the full Expedition 7 scan once per test module and measure wall-clock time."""
    out_dir = tmp_path_factory.mktemp("exp7_results")
    out_json = out_dir / "expedition7_whole_slide_spatial_secretome.json"

    t0 = time.perf_counter()
    scanner = WholeSlideSpatialSecretomeScanner(random_state=42)
    results = run_expedition7_whole_slide_spatial_secretome(
        random_state=42, output_path=out_json
    )
    elapsed_s = time.perf_counter() - t0
    return {
        "scanner": scanner,
        "results": results,
        "out_json": out_json,
        "elapsed_s": elapsed_s,
    }


# ------------------------------------------------------------------------------
# 1. Execution Speed, Offline Bundle Integrity & Strict JSON Schema Validation
# ------------------------------------------------------------------------------
def test_01_offline_execution_speed_and_raw_umi_matrix_integrity(
    exp7_scanner_and_results: Dict[str, Any],
) -> None:
    """Verify offline execution completes in < 5.0s and loads 400 spots x 354 integer UMI profiles."""
    scanner: WholeSlideSpatialSecretomeScanner = exp7_scanner_and_results["scanner"]
    elapsed_s: float = exp7_scanner_and_results["elapsed_s"]

    assert elapsed_s < 5.0, f"Expected offline scan in < 5.0s, took {elapsed_s:.3f}s"
    assert scanner.n_cells == 400
    assert scanner.X_all_raw.shape == (400, 354)
    assert scanner.coords.shape == (400, 2)
    assert np.all(np.isfinite(scanner.X_all_raw))
    assert np.all(scanner.X_all_raw >= 0.0)
    # Integer UMI counts check
    assert np.allclose(scanner.X_all_raw, np.round(scanner.X_all_raw), atol=1e-5)

    # Spatial contact graphs at d <= 35 um and d <= 55 um
    assert scanner.graph_35["n_tsk_to_stroma_edges"] == 20
    assert scanner.graph_35["n_any_cross_layer_spots"] == 28
    assert scanner.graph_55["n_tsk_to_stroma_edges"] == 184
    assert scanner.graph_55["n_border_tsk_spots"] == 24
    assert scanner.graph_55["n_border_str_spots"] == 29
    assert scanner.graph_55["n_border_l3_l4_spots"] == 53
    assert scanner.graph_55["n_any_cross_layer_spots"] == 142


def test_02_strict_json_serialization_and_artifact_on_disk(
    exp7_scanner_and_results: Dict[str, Any],
) -> None:
    """Verify strict JSON compliance (no NaN/Infinity tokens) both in tmp output and repo results/."""
    out_json: Path = exp7_scanner_and_results["out_json"]
    raw_text = out_json.read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text
    assert "-Infinity" not in raw_text

    parsed = json.loads(raw_text)
    assert parsed["expedition_id"] == (
        "Expedition_7_Whole_Slide_17138_Gene_Spatial_Secretome_And_Multi_Slide_Replication"
    )
    assert parsed["version"] == "0.2.0"
    assert parsed["random_state"] == 42
    assert parsed["all_adversarial_tests_passed"] is True

    required_sections = [
        "executive_summary",
        "section1_whole_transcriptome_17138_gene_hvg_blindspot_census",
        "section2_two_tier_secretome_receptome_discovery_catalog",
        "section3_expanded_34_pair_pseudo_emt_and_pan_patient_cadherin_audit",
        "section4_multi_slide_4_patient_gse144239_replication",
        "section5_quad_engine_cross_orchestrator_convergence",
        "section6_adversarial_falsification_suite",
    ]
    for sec in required_sections:
        assert sec in parsed, f"Missing required section: {sec}"

    repo_json = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "expedition7_whole_slide_spatial_secretome.json"
    )
    assert repo_json.exists(), f"Expected artifact at {repo_json}"
    repo_text = repo_json.read_text(encoding="utf-8")
    assert "NaN" not in repo_text
    assert "Infinity" not in repo_text
    repo_parsed = json.loads(repo_text)
    assert repo_parsed["all_adversarial_tests_passed"] is True


# ------------------------------------------------------------------------------
# 2. Section 1: Whole-Transcriptome 17,138-Gene HVG-2000 Blind-Spot Census
# ------------------------------------------------------------------------------
def test_03_section1_whole_transcriptome_17138_gene_census(
    exp7_scanner_and_results: Dict[str, Any],
) -> None:
    """Verify the 17,138-gene transcriptome census, artifact slots, and L3/L4 Non-HVG fractions."""
    sec1 = exp7_scanner_and_results["results"][
        "section1_whole_transcriptome_17138_gene_hvg_blindspot_census"
    ]
    c = sec1["transcriptome_census"]

    assert sec1["primary_slide"] == "GSM4284316_P2_ST_rep1"
    assert sec1["geo_accession"] == "GSE144239"
    assert c["total_genes_on_array"] == 17138
    assert c["ribosomal_genes_filtered"] == 177
    assert c["mitochondrial_genes_filtered"] == 23
    assert c["housekeeping_artifacts_filtered"] == 9
    assert c["total_artifacts_filtered"] == 209
    assert c["clean_biological_genes_total"] == 16929
    assert c["clean_expressed_400_spots"] == 16120
    assert c["clean_expressed_666_spots"] == 16565
    assert c["artifact_slots_in_top2000_hvg"] == 123
    assert c["clean_in_top2000_hvg"] == 1877
    assert c["clean_expressed_non_hvg_dark_genome_400"] == 14243
    assert c["pct_clean_expressed_dropped_by_hvg2000"] == pytest.approx(88.36, abs=0.01)

    # Layer 3 TSK & Layer 4 Stroma HVG-2000 blindspot
    assert c["sig_l3_tsk_enriched_genes_fdr01"] == 152
    assert c["sig_l3_tsk_non_hvg_genes_fdr01"] == 69
    assert c["pct_l3_tsk_genes_dropped_by_hvg2000"] == pytest.approx(45.39, abs=0.02)
    assert c["pct_l3_tsk_genes_dropped_rounded_1dp"] == pytest.approx(45.4, abs=0.05)

    assert c["sig_l4_stroma_enriched_genes_fdr01"] == 498
    assert c["sig_l4_stroma_non_hvg_genes_fdr01"] == 350
    assert c["pct_l4_stroma_genes_dropped_by_hvg2000"] == pytest.approx(70.28, abs=0.02)
    assert c["pct_l4_stroma_genes_dropped_rounded_1dp"] == pytest.approx(70.3, abs=0.05)

    # Canonical MMP10 blindspot (rank 3,321 on 400 spots; dropped on 5/6 slides)
    assert c["mmp10_hvg_rank_400"] == 3321
    assert c["mmp10_slides_dropped_by_hvg2000"] == 5
    assert c["mmp10_slides_evaluated"] == 6


def test_04_section1_14890_pair_17_family_whole_transcriptome_screen(
    exp7_scanner_and_results: Dict[str, Any],
) -> None:
    """Verify the 14,890-pair 17-family genome-wide screen and 79.33% Non-HVG loss."""
    sec1 = exp7_scanner_and_results["results"][
        "section1_whole_transcriptome_17138_gene_hvg_blindspot_census"
    ]
    lr = sec1["whole_transcriptome_14890_pair_screen"]

    assert lr["total_cognate_pairs_in_17138_genes"] == 14890
    assert lr["active_expressed_cognate_pairs"] == 13513
    assert lr["significant_cross_wall_pairs_fdr01"] == 9084
    assert lr["significant_non_hvg_dark_genome_pairs_fdr01"] == 7206
    assert lr["pct_significant_pairs_dropped_by_hvg2000"] == pytest.approx(79.33, abs=0.01)
    assert lr["significant_both_non_hvg_pairs_fdr01"] == 2807
    assert lr["pct_significant_both_non_hvg_dropped_by_hvg2000"] == pytest.approx(30.90, abs=0.02)
    assert lr["n_signaling_families"] == 17

    fams = lr["signaling_family_breakdown"]
    assert len(fams) == 17
    assert fams["FGF_FGFR_HSPG"]["pct_non_hvg_of_significant"] == pytest.approx(100.0, abs=0.02)
    assert fams["Ephrin_Eph_Juxtacrine"]["pct_non_hvg_of_significant"] == pytest.approx(100.0, abs=0.02)
    assert fams["GrowthFactor_Angiogenic_RTK"]["pct_non_hvg_of_significant"] == pytest.approx(99.00, abs=0.02)
    assert fams["Peptide_GPCR_Complement_Coag"]["pct_non_hvg_of_significant"] == pytest.approx(98.53, abs=0.02)
    assert fams["Wnt_RSPO_Frizzled_LRP"]["pct_non_hvg_of_significant"] == pytest.approx(98.31, abs=0.02)
    assert fams["Semaphorin_Plexin_Guidance"]["pct_non_hvg_of_significant"] == pytest.approx(98.00, abs=0.02)
    assert fams["Chemokine_GPCR_ACKR"]["pct_non_hvg_of_significant"] == pytest.approx(97.62, abs=0.02)
    assert fams["TGFB_Activin_BMP_Superfamily"]["pct_non_hvg_of_significant"] == pytest.approx(95.81, abs=0.02)
    assert fams["EGF_ERBB_RTK"]["pct_non_hvg_of_significant"] == pytest.approx(91.49, abs=0.02)
    assert fams["Laminin_Integrin_ECM"]["pct_non_hvg_of_significant"] == pytest.approx(90.37, abs=0.02)

    # Verify sum of family counts equals screen totals
    assert sum(v["total_active_pairs"] for v in fams.values()) == 13513
    assert sum(v["significant_pairs_fdr01"] for v in fams.values()) == 9084
    assert sum(v["significant_non_hvg_pairs"] for v in fams.values()) == 7206


# ------------------------------------------------------------------------------
# 3. Section 2: Two-Tier Secretome-Receptome Discovery Catalog (251 / 225 Pairs)
# ------------------------------------------------------------------------------
def test_05_section2_two_tier_discovery_catalog_counts_and_classes(
    exp7_scanner_and_results: Dict[str, Any],
) -> None:
    """Verify 251 quantified pairs, 225 new active pairs, 141 Tier A, 153 Tier B, 87 Combined across 7 classes."""
    sec2 = exp7_scanner_and_results["results"][
        "section2_two_tier_secretome_receptome_discovery_catalog"
    ]
    sm = sec2["summary_metrics"]

    assert sm["n_expressed_pairs_quantified"] == 251
    assert sm["n_newly_discovered_active_pairs_beyond_exp3"] == 225
    assert sm["n_tier_a_non_hvg2000_active_pairs"] == 141
    assert sm["pct_tier_a_of_newly_discovered_active_pairs"] == pytest.approx(62.67, abs=0.02)
    assert sm["n_tier_b_actionable_repurposing_active_pairs"] == 153
    assert sm["n_tier_a_and_b_combined_pairs"] == 87
    assert sm["n_functional_classes"] == 7

    assert len(sec2["all_quantified_pairs_251"]) == 251
    assert len(sec2["tier_a_non_hvg2000_dark_spatial_genome_pairs"]) == 141
    assert len(sec2["tier_b_actionable_repurposing_pairs"]) == 153
    assert len(sec2["tier_a_and_b_combined_pairs"]) == 87

    fc = sec2["functional_class_breakdown"]
    expected_classes = {
        "Protease-ECM": (40, 40, 16, 34),
        "ECM-Integrin": (75, 75, 27, 48),
        "GrowthFactor-Cytokine": (68, 60, 58, 37),
        "Immune-Checkpoint": (32, 21, 16, 16),
        "Semaphorin-Plexin": (19, 17, 14, 8),
        "Metabolic-Shuttle": (7, 6, 4, 6),
        "Ephrin-Eph": (10, 6, 6, 4),
    }
    for cls_name, (n_q, n_new, n_ta, n_tb) in expected_classes.items():
        assert cls_name in fc
        assert fc[cls_name]["n_quantified_pairs"] == n_q
        assert fc[cls_name]["n_newly_discovered_active_pairs"] == n_new
        assert fc[cls_name]["n_tier_a_non_hvg2000_pairs"] == n_ta
        assert fc[cls_name]["n_tier_b_actionable_pairs"] == n_tb


def test_06_section2_headline_newly_discovered_tier_a_and_tier_b_pairs(
    exp7_scanner_and_results: Dict[str, Any],
) -> None:
    """Verify exact SBR_shf, sharpness_gain, and HVG ranks for headline newly discovered pairs."""
    sec2 = exp7_scanner_and_results["results"][
        "section2_two_tier_secretome_receptome_discovery_catalog"
    ]
    by_pair = {p["pair"]: p for p in sec2["all_quantified_pairs_251"]}

    headline_expectations = {
        "LOXL2->COL1A1": (403.38, 150.59, False, True),
        "MMP13->COL1A1": (218.54, 62.69, True, True),
        "THBS1->SDC1": (208.35, 87.67, False, True),
        "THBS1->CD47": (104.97, 44.19, False, True),
        "FAP->COL1A1": (95.23, 30.72, True, True),
        "LOX->COL1A1": (88.26, 61.85, True, True),
        "PTN->SDC1": (68.30, 45.06, True, True),
        "PDGFA->PDGFRB": (43.44, 24.29, True, True),
        "SEMA4D->PLXNB2": (37.57, 10.27, True, True),
        "SEMA3A->PLXNA1": (34.66, 9.10, True, False),
        "SLC1A5->SLC3A2": (33.11, 11.32, True, True),
        "MIF->CD74": (32.52, 36.56, True, True),
        "COL1A1->DDR1": (29.10, 11.80, True, True),
        "SEMA3A->NRP1": (27.49, 5.61, True, True),
        "IGF2->IGF1R": (26.10, 8.22, True, True),
        "GAS6->AXL": (24.51, 6.81, True, True),
        "EFNB2->EPHB4": (11.09, 3.82, True, True),
        "ENTPD1->NT5E": (4.21, 3.42, True, True),
    }
    for pid, (exp_sbr, exp_sharp, exp_ta, exp_tb) in headline_expectations.items():
        assert pid in by_pair, f"Missing headline pair {pid}"
        p = by_pair[pid]
        assert p["in_exp3_28"] is False
        assert p["B_shf"] < 0.05
        assert p["B_shf"] < p["B_iso"]
        assert p["SBR_shf"] == pytest.approx(exp_sbr, abs=0.05)
        assert p["sharpness_gain"] == pytest.approx(exp_sharp, abs=0.05)
        assert p["is_tier_a_non_hvg2000"] is exp_ta
        assert p["is_tier_b_actionable_repurposing"] is exp_tb

    # Check specific Non-HVG ranks
    assert by_pair["MMP13->COL1A1"]["ligand_hvg_rank_400"] == 7552
    assert by_pair["FAP->COL1A1"]["ligand_hvg_rank_400"] == 12303
    assert by_pair["LOX->COL1A1"]["ligand_hvg_rank_400"] == 5193
    assert by_pair["PDGFA->PDGFRB"]["ligand_hvg_rank_400"] == 5184
    assert by_pair["SEMA4D->PLXNB2"]["ligand_hvg_rank_400"] == 6252
    assert by_pair["SEMA3A->PLXNA1"]["ligand_hvg_rank_400"] == 10949
    assert by_pair["SEMA3A->PLXNA1"]["receptor_hvg_rank_400"] == 3500
    assert by_pair["MIF->CD74"]["ligand_hvg_rank_400"] == 7706
    assert by_pair["MIF->CD74"]["ligand_hvg_rank_666"] == 8517
    assert by_pair["COL1A1->DDR1"]["receptor_hvg_rank_400"] == 2576
    assert by_pair["COL1A1->DDR1"]["receptor_hvg_rank_666"] == 2702
    assert by_pair["IGF2->IGF1R"]["ligand_hvg_rank_400"] == 12353
    assert by_pair["IGF2->IGF1R"]["receptor_hvg_rank_400"] == 3304
    assert by_pair["GAS6->AXL"]["receptor_hvg_rank_400"] == 3232


def test_07_section2_biological_database_verification_catalogs(
    exp7_scanner_and_results: Dict[str, Any],
) -> None:
    """Verify STRING v12.0, Human Protein Atlas, Reactome, OpenTargets, and ChEMBL cross-references."""
    sec2 = exp7_scanner_and_results["results"][
        "section2_two_tier_secretome_receptome_discovery_catalog"
    ]
    sm = sec2["summary_metrics"]

    assert sm["n_hpa_genes_annotated"] == 151
    assert sm["n_reactome_enriched_pathways"] == 65
    assert sm["n_opentargets_targets_annotated"] == 60
    assert sm["n_chembl_drugs_verified"] == 33

    assert len(sec2["hpa_subcellular_and_secretome_catalog"]) == 151
    assert len(sec2["opentargets_tractability_catalog"]) == 60
    assert len(sec2["chembl_clinical_repurposing_drugs"]) == 33
    assert len(sec2["reactome_pathway_enrichment"]) == 45

    # Check key ChEMBL drugs
    chembl_by_name = sec2["chembl_clinical_repurposing_drugs"]
    for drug_name, chembl_id in [
        ("simtuzumab", "CHEMBL2109667"),
        ("pepinemab", "CHEMBL4297884"),
        ("magrolimab", "CHEMBL4297985"),
        ("bintrafusp", "CHEMBL4298024"),
        ("oleclumab", "CHEMBL3990016"),
        ("bemcentinib", "CHEMBL3809489"),
        ("linsitinib", "CHEMBL1091644"),
        ("dasatinib", "CHEMBL1421"),
        ("imatinib", "CHEMBL941"),
        ("cetuximab", "CHEMBL1201577"),
    ]:
        assert drug_name in chembl_by_name
        assert chembl_by_name[drug_name]["chembl_id"] == chembl_id


# ------------------------------------------------------------------------------
# 4. Section 3: Expanded 34-Pair Pseudo-EMT & Pan-Patient Cadherin Switch Audit
# ------------------------------------------------------------------------------
def test_08_section3_expanded_34_pair_pseudo_emt_falsification(
    exp7_scanner_and_results: Dict[str, Any],
) -> None:
    """Verify 34/34 cross-wall epithelial+stromal pairs have 0.0% Sheaf bleed and 31/34 have 0.0 STRING physical score."""
    sec3 = exp7_scanner_and_results["results"][
        "section3_expanded_34_pair_pseudo_emt_and_pan_patient_cadherin_audit"
    ]

    assert sec3["n_pseudo_emt_pairs_tested"] == 34
    assert sec3["n_pseudo_emt_pairs_falsified_zero_sheaf_bleed"] == 34
    assert sec3["sheaf_dual_positive_spots_all_34_pairs"] == 0
    assert sec3["sheaf_b_intra_all_34_pairs"] == 0.0
    assert sec3["string_v12_zero_physical_complex_score_count"] == 31
    assert sec3["string_v12_zero_functional_score_count"] == 11

    by_pair = {p["pair"]: p for p in sec3["pseudo_emt_34_pairs_catalog"]}
    assert len(by_pair) == 34

    # Check Non-HVG pseudo-EMT pairs and classical epithelial+stromal pairs
    for pid, min_coexpr_iso, exp_phys in [
        ("MMP10+COL1A1", 0.05, 0.0),
        ("LAMC2+COL1A1", 0.15, 0.0),
        ("KRT14+VIM", 0.05, 0.317),
        ("DSP+VIM", 0.05, 0.422),
        ("TP63+VIM", 0.05, 0.0),
        ("GRHL2+VIM", 0.10, 0.0),
        ("OVOL1+VIM", 0.08, 0.0),
        ("OCLN+ACTA2", 0.10, 0.0),
        ("EVPL+COL1A1", 0.05, 0.0),
    ]:
        assert pid in by_pair
        r = by_pair[pid]
        assert r["coexpr_iso"] >= min_coexpr_iso
        assert r["dual_pos_shf"] == 0
        assert r["dual_pos_rate_shf"] == 0.0
        assert r["coexpr_shf"] == 0.0
        assert r["b_intra_sheaf"] == 0.0
        assert r["string_v12_verification"]["physical_complex_score"] == pytest.approx(exp_phys, abs=1e-3)


def test_09_section3_pan_patient_6_slide_cadherin_switch_falsification(
    exp7_scanner_and_results: Dict[str, Any],
) -> None:
    """Verify pan-patient CDH1 (6,540) + CDH3 (10,463) = 17,003 UMI vs CDH2 = 7.0 UMI (2,429.0x ratio)."""
    cad = exp7_scanner_and_results["results"][
        "section3_expanded_34_pair_pseudo_emt_and_pan_patient_cadherin_audit"
    ]["pan_patient_cadherin_switch_audit"]

    assert cad["n_slides_evaluated"] == 6
    assert cad["n_patients_evaluated"] == 4
    assert cad["total_tissue_spots_across_6_slides"] == 4293
    assert cad["total_cdh1_umi_6_slides"] == pytest.approx(6540.0, abs=0.1)
    assert cad["total_cdh3_umi_6_slides"] == pytest.approx(10463.0, abs=0.1)
    assert cad["total_cdh1_plus_cdh3_umi_6_slides"] == pytest.approx(17003.0, abs=0.1)
    assert cad["total_cdh2_umi_6_slides"] == pytest.approx(7.0, abs=0.1)
    assert cad["cdh1_cdh3_to_cdh2_umi_ratio_6_slides"] == pytest.approx(2429.0, abs=0.2)

    zero_cdh2_slides = set(cad["slides_with_exact_zero_cdh2_umi"])
    assert zero_cdh2_slides == {"P2_ST_rep1", "P5_ST_rep1", "P10_ST_rep1"}

    # Primary 400-spot live check
    p400 = cad["primary_400_spot_live_cadherin_audit"]
    assert p400["CDH2"]["total_umi_400_spots"] == 0.0
    assert p400["CDH2"]["nonzero_spots"] == 0
    assert p400["CDH1"]["total_umi_400_spots"] == pytest.approx(1393.0, abs=0.1)
    assert p400["CDH3"]["total_umi_400_spots"] == pytest.approx(2107.0, abs=0.1)


# ------------------------------------------------------------------------------
# 5. Section 4: Multi-Slide 4-Patient GSE144239 Replication (6 Slides, 4,293 Spots)
# ------------------------------------------------------------------------------
def test_10_section4_multi_slide_4_patient_gse144239_replication(
    exp7_scanner_and_results: Dict[str, Any],
) -> None:
    """Verify multi-slide replication across 6 slides (P2, P5, P9, P10; 4,293 spots, 1,003 edges)."""
    sec4 = exp7_scanner_and_results["results"][
        "section4_multi_slide_4_patient_gse144239_replication"
    ]

    assert sec4["geo_accession"] == "GSE144239"
    assert sec4["n_slides_evaluated"] == 6
    assert sec4["n_patients_evaluated"] == 4
    assert sec4["patients"] == ["P2", "P5", "P9", "P10"]
    assert sec4["total_tissue_spots_across_6_slides"] == 4293
    assert sec4["total_native_2d_boundary_edges_across_6_slides"] == 1003
    assert sec4["primary_400_zero_sheaf_pseudo_emt_bleed"] is True
    assert sec4["all_6_slides_sheaf_suppresses_pseudo_emt_bleed"] is True

    hr = sec4["headline_replicated_pairs"]
    expected_replications = {
        "MMP10->ITGA6": (6, 14.62),
        "COL1A1->DDR1": (6, 8.82),
        "LAMC2->ITGA6": (6, 13.01),
        "LAMC2->ITGB4": (6, 11.83),
        "MMP10->TIMP1": (5, 13.50),
        "PLAU->F2R": (5, 12.09),
        "MIF->CD74": (5, 6.74),
        "FN1->ITGA5": (5, 4.74),
        "LAMA4->ITGA6": (5, 3.94),
        "LAMA4->SDC1": (5, 4.23),
        "COL1A1->ITGA1": (5, 5.23),
        "KLK12->SPINK5": (5, 4.08),
        "JAG1->NOTCH3": (5, 8.81),
        "INHBA->TGFBR2": (4, 3.34),
        "MMP1->F2R": (4, 10.23),
        "SLC7A5->SLC3A2": (4, 8.84),
    }
    for pid, (exp_n_slides, exp_mean_sbr) in expected_replications.items():
        assert pid in hr, f"Missing headline replicated pair {pid}"
        assert hr[pid]["n_slides_replicated_fdr01"] == exp_n_slides
        assert hr[pid]["multi_slide_mean_sbr_sheaf"] == pytest.approx(exp_mean_sbr, abs=0.05)
        assert hr[pid]["multi_slide_mean_sbr_sheaf"] > hr[pid]["multi_slide_mean_sbr_iso"]


# ------------------------------------------------------------------------------
# 6. Section 5: Quad-Engine Cross-Orchestrator Convergence (Exp 4, 5, 6, 7)
# ------------------------------------------------------------------------------
def test_11_section5_quad_engine_cross_orchestrator_convergence(
    exp7_scanner_and_results: Dict[str, Any],
) -> None:
    """Verify cross-orchestrator bridges to Exp 4 (SRAM streaming), Exp 5 (Delta_20clock), and Exp 6 (GRCh38 stringers)."""
    sec5 = exp7_scanner_and_results["results"][
        "section5_quad_engine_cross_orchestrator_convergence"
    ]

    # Exp 4 systems & streaming bridge
    e4 = sec5["exp4_streaming_and_hvg_metacell_blindspot_bridge"]
    assert e4["streamer_operator"] == "WholeGenomeArenaStreamer.stream_spatial_sheaf_aligner"
    assert e4["native_c_abi_kernel"] == "spatial_tissue_sheaf_solver_huber_f32"
    assert e4["static_sram_arena_bytes"] == 983872
    assert e4["static_sram_arena_mb"] < 2.0
    assert e4["ring_buffer_bytes"] == 524288
    assert e4["column_tile_size_bg_n400"] == 163
    assert e4["n_column_tiles_17138_genes"] == 106
    assert e4["max_abs_diff_vs_full_slide_sheaf"] == 0.0
    assert e4["dynamic_heap_allocations"] == 0
    assert e4["pseudobulk50_cscc_spatial_polarity_washout_mean_pct"] == pytest.approx(84.17, abs=0.02)
    assert e4["pseudobulk50_cscc_spatial_polarity_washout_top15_pct"] == pytest.approx(91.99, abs=0.02)

    # Exp 5 temporal Delta_20clock bridge
    e5 = sec5["exp5_whole_genome_delta_erasure_bridge"]
    expected_deltas = {
        "SLC7A5": 0.689,
        "ITGB4": 0.630,
        "TP63": 0.651,
        "AKT1": 0.403,
        "NUDT1": 0.444,
        "DNAJC9": 0.436,
        "VRK1": 0.323,
        "PKMYT1": 0.384,
        "RACGAP1": 0.509,
    }
    for gene, exp_delta in expected_deltas.items():
        assert gene in e5
        assert e5[gene]["cscc_delta_20clock"] == pytest.approx(exp_delta, abs=0.002)
        assert 0.0 < e5[gene]["cellsnail_400_variance_retained_fraction"] <= 1.0

    # Exp 6 pan-autosome GRCh38 stringer bridge
    e6_by_sym = {x["gene_symbol"]: x for x in sec5["exp6_pan_autosome_grch38_stringer_and_persister_bridge"]}
    for sym in ["IGF1R", "EPHB4", "TGFBR2", "NT5E", "AXL", "ERBB2", "KDR_PDGFRB_DDR1", "CD74_LGALS1"]:
        assert sym in e6_by_sym
    assert "chr15:98648538-98964530" in e6_by_sym["IGF1R"]["grch38_locus"]
    assert "chr7:100802564-100825996" in e6_by_sym["EPHB4"]["grch38_locus"]
    assert "chr3:30606601-30694142" in e6_by_sym["TGFBR2"]["grch38_locus"]


# ------------------------------------------------------------------------------
# 7. Section 6: 4-Test Adversarial Falsification Suite on Newly Discovered Pairs
# ------------------------------------------------------------------------------
def test_12_section6_test1_spot_label_permutation_null(
    exp7_scanner_and_results: Dict[str, Any],
) -> None:
    """Verify 200-permutation spatial spot-label null on 30 newly discovered Tier A + Tier B pairs."""
    t1 = exp7_scanner_and_results["results"]["section6_adversarial_falsification_suite"][
        "test1_spot_label_permutation_null"
    ]
    assert t1["passed"] is True
    assert t1["n_permutations"] == 200
    assert t1["n_newly_discovered_pairs_tested"] == 30
    assert t1["observed_mean_j_sheaf"] > 2.5 * t1["null_mean_j_sheaf"]
    assert t1["j_sheaf_z_score"] > 5.0
    assert t1["j_sheaf_empirical_p_value"] < 0.01
    assert t1["observed_mean_sbr_sheaf"] > 50.0 * t1["null_mean_sbr_sheaf"]
    assert t1["sbr_sheaf_z_score"] > 50.0
    assert t1["sbr_sheaf_empirical_p_value"] < 0.01
    assert len(t1["per_pair_permutation_audit"]) == 30


def test_13_section6_test2_normal_tangential_orientation_swap(
    exp7_scanner_and_results: Dict[str, Any],
) -> None:
    """Verify 90-deg & 180-deg normal-vs-tangential orientation swap and cross-wall graph-clamping swap."""
    t2 = exp7_scanner_and_results["results"]["section6_adversarial_falsification_suite"][
        "test2_normal_tangential_orientation_swap"
    ]
    assert t2["passed"] is True
    assert t2["true_orientation_normal_penetration_residual"] == pytest.approx(0.0, abs=1e-5)
    assert t2["true_orientation_tangential_preservation_fraction"] == pytest.approx(1.0, abs=1e-4)
    assert t2["rotated_90deg_normal_leakage_fraction"] == pytest.approx(1.0, abs=1e-4)
    assert t2["rotated_90deg_tangential_remaining_fraction"] == pytest.approx(0.0, abs=1e-5)
    assert t2["flipped_180deg_normal_leakage_fraction"] == pytest.approx(1.0, abs=1e-4)

    # Graph-clamping swap induces severe cross-wall stromal bleed whereas true Sheaf has 0.0 bleed
    assert t2["true_sheaf_stromal_bleed_mmp13_col1a1"] == 0.0
    assert t2["true_sheaf_stromal_bleed_loxl2_col1a1"] == 0.0
    assert t2["graph_swap_stromal_bleed_mmp13_col1a1"] > 0.50
    assert t2["graph_swap_stromal_bleed_loxl2_col1a1"] > 1.00
    assert t2["graph_swap_stromal_bleed_col1a1_ddr1"] > 0.25


def test_14_section6_test3_countsplit_binomial_holdout(
    exp7_scanner_and_results: Dict[str, Any],
) -> None:
    """Verify countsplit binomial thinning holdout (p = 0.50, Pearson r > 0.90 across all subsets)."""
    t3 = exp7_scanner_and_results["results"]["section6_adversarial_falsification_suite"][
        "test3_countsplit_binomial_holdout"
    ]
    assert t3["passed"] is True
    assert t3["split_probability_p"] == 0.5
    assert t3["n_all_quantified_pairs"] == 251
    assert t3["n_newly_discovered_active_pairs"] == 225
    assert t3["n_tier_a_non_hvg2000_pairs"] == 141
    assert t3["n_tier_ab_combined_pairs"] == 87

    assert t3["pearson_r_all_251_pairs"] == pytest.approx(0.9773, abs=0.001)
    assert t3["pearson_r_newly_discovered_225_pairs"] == pytest.approx(0.9762, abs=0.001)
    assert t3["pearson_r_tier_a_141_non_hvg_pairs"] == pytest.approx(0.9315, abs=0.001)
    assert t3["pearson_r_tier_ab_87_combined_pairs"] == pytest.approx(0.9682, abs=0.001)
    assert t3["pearson_r_log1p_sheaf_225_pairs"] > 0.99


def test_15_section6_test4_lofo_and_lopo_spatial_holdouts(
    exp7_scanner_and_results: Dict[str, Any],
) -> None:
    """Verify 4-fold Leave-One-FOV-Out (LOFO) spatial quadrant holdout & 4-patient LOPO holdout."""
    t4 = exp7_scanner_and_results["results"]["section6_adversarial_falsification_suite"][
        "test4_multi_slide_lopo_and_lofo_spatial_holdout"
    ]
    assert t4["passed"] is True
    assert t4["n_lofo_spatial_folds"] == 4
    assert t4["n_lopo_patient_folds"] == 4
    assert t4["all_4_lofo_folds_zero_sheaf_pseudo_emt_bleed"] is True
    assert t4["all_4_patients_sheaf_suppresses_pseudo_emt_bleed"] is True
    assert t4["mean_held_out_sheaf_pseudo_emt_rate"] == 0.0
    assert t4["mean_held_out_sheaf_non_hvg_grhl2_vim_rate"] == 0.0
    assert t4["mean_held_out_mmp13_col1a1_sharpening_x"] > 50.0
    assert t4["mean_held_out_loxl2_col1a1_sharpening_x"] > 100.0

    assert len(t4["lofo_spatial_quadrant_folds"]) == 4
    assert len(t4["lopo_multi_patient_folds"]) == 4
    assert sum(f["held_out_spots"] for f in t4["lopo_multi_patient_folds"]) == 4293
    assert sum(f["held_out_boundary_edges"] for f in t4["lopo_multi_patient_folds"]) == 1003


def test_16_deterministic_bit_exact_reproducibility(
    exp7_scanner_and_results: Dict[str, Any],
) -> None:
    """Verify two independent runs with random_state=42 produce bit-identical JSON payloads."""
    res1 = exp7_scanner_and_results["results"]
    scanner2 = WholeSlideSpatialSecretomeScanner(random_state=42)
    res2 = scanner2.run_full_scan()

    s1 = json.dumps(res1, sort_keys=True, allow_nan=False)
    s2 = json.dumps(res2, sort_keys=True, allow_nan=False)
    assert s1 == s2
