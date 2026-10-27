"""tests/test_expedition5_whole_genome_delta_census.py
======================================================
Comprehensive unit and statistical falsification test suite for Expedition 5:
Unbiased Whole-Transcriptome 33,694-Gene "Dark Genome" Delta-Erasure Census
(`cellsnail.expeditions.exp5_whole_genome_delta_census`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pytest

from cellsnail.expeditions.exp5_whole_genome_delta_census import (
    CLOCK_CORE_5,
    CLOCK_EXTENDED_20,
     DEFAULT_RESULTS_PATH,
    EXCLUDE_EXACT,
    EXCLUDE_PREFIXES,
    INITIAL_PANEL_50,
    TIER_A_GENES_30,
    TIER_B_GENES_30,
    TIROSH_AND_MITOTIC_HOUSEKEEPING,
    classify_gene_category,
    compute_whole_genome_qr_deltas,
    run_expedition5_whole_genome_delta_census,
    verify_nested_subspace_monotonicity,
    verify_orthogonal_decomposition_on_panel,
)


@pytest.fixture(scope="module")
def exp5_results():
    assert DEFAULT_RESULTS_PATH.exists(), f"Missing {DEFAULT_RESULTS_PATH}"
    with open(DEFAULT_RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_01_results_json_exists_and_strict_no_nan(exp5_results):
    """Verify expedition5_whole_genome_delta_census.json exists and has zero NaN/Inf values."""
    raw_text = DEFAULT_RESULTS_PATH.read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text
    assert exp5_results["expedition_id"] == "exp5_whole_genome_delta_census"
    assert exp5_results["author"] == "Noah Cohen"


def test_02_whole_genome_census_counts_tnbc_5patients(exp5_results):
    """Verify all 5 GSE148673 TNBC patients (33,694 GRCh38 genes, 8,944 cells) are accounted for."""
    tnbc = exp5_results["whole_genome_census_counts"]["GSE148673_TNBC_Series"]
    assert tnbc["total_patients"] == 5
    assert tnbc["total_cells_across_5_patients"] == 8944
    assert tnbc["reference_annotated_genes_grch38_v2"] == 33694
    assert tnbc["reference_annotated_genes_grch38_v3"] == 33538
    pats = tnbc["patients"]
    assert set(pats.keys()) == {"TNBC1", "TNBC2", "TNBC3", "TNBC4", "TNBC5"}
    assert pats["TNBC3"]["n_cells"] == 532
    assert pats["TNBC3"]["total_annotated_genes"] == 33694
    assert pats["TNBC3"]["expressed_ge5_cells"] == 17744
    assert pats["TNBC3"]["non_hvg_dark_genome_genes_evaluated"] == 15734


def test_03_whole_genome_census_counts_gbm_and_cscc(exp5_results):
    """Verify GSE57872 GBM (5,948 genes, 543 cells) and GSE144239 cSCC (17,138 / 17,344 genes)."""
    gbm = exp5_results["whole_genome_census_counts"]["GSE57872_GBM_Cohort"]
    assert gbm["total_patients_and_stem_lines"] == 7
    assert gbm["n_cells"] == 543
    assert gbm["expressed_genes"] == 5948
    assert gbm["non_hvg_dark_genome_genes_evaluated"] == 3941

    cscc = exp5_results["whole_genome_census_counts"]["GSE144239_cSCC_Spatial_Series"]
    assert cscc["P2_ST_rep1"]["n_tissue_spots"] == 666
    assert cscc["P2_ST_rep1"]["total_annotated_genes"] == 17138
    assert cscc["P2_ST_rep1"]["expressed_ge5_spots"] == 14129
    assert cscc["P2_ST_rep2"]["n_tissue_spots"] == 646
    assert cscc["P2_ST_rep2"]["total_annotated_genes"] == 17344
    assert cscc["P2_ST_rep2"]["expressed_ge5_spots"] == 14279


def test_04_nested_subspace_monotonicity_theorem_live():
    """Verify 0 <= Delta_5clock(g) <= Delta_20clock(g) <= 1 on live empirical matrices."""
    ortho = verify_orthogonal_decomposition_on_panel()
    for cohort_key in ["GSE148673_TNBC3", "GSE57872_GBM", "GSE144239_cSCC"]:
        mono = ortho[cohort_key]["nested_subspace_monotonicity"]
        assert mono["theorem_holds_strictly"] is True
        assert mono["monotonicity_violations_count"] == 0
        assert mono["min_subspace_expansion_gain"] >= -1e-12
        assert mono["mean_subspace_expansion_gain"] > 0.01


def test_05_hilbert_space_orthogonality_and_pythagoras_live():
    """Verify <P_clock X, P_nope X>_F == 0.0 and ||P_clock X||_F^2 + ||P_nope X||_F^2 == ||X||_F^2."""
    ortho = verify_orthogonal_decomposition_on_panel()
    for cohort_key, metrics in ortho.items():
        assert metrics["max_inner_product_residual"] == 0.0
        assert abs(metrics["total_norm_conservation_ratio"] - 1.0) < 1e-12
        assert metrics["h_nope_variance_retained"] == 1.0
        assert metrics["clock_core_recall"] == 1.0
        assert metrics["false_positive_clock_leakage_into_nope"] == 0.0


def test_06_hvg2000_blind_spot_quantification(exp5_results):
    """Verify >80% of FDR<0.01 20-clock erased genes in TNBC3 lie outside top-2000 HVG."""
    tnbc3_q = exp5_results["genome_wide_distribution_quantiles_by_cohort"]["GSE148673_TNBC3"]
    sig = tnbc3_q["significant_fdr_0_01_counts"]
    assert sig["all_expressed_20clock_fdr_lt_0_01"] == 10599
    assert sig["non_hvg_dark_genome_20clock_fdr_lt_0_01"] == 8680
    frac_outside_hvg = sig["non_hvg_dark_genome_20clock_fdr_lt_0_01"] / sig["all_expressed_20clock_fdr_lt_0_01"]
    assert frac_outside_hvg > 0.80

    dark_q = tnbc3_q["quantiles"]["non_hvg_dark_genome_rank_gt_2000"]["delta_20clock"]
    assert dark_q["count_gt_0_15"] == 4985
    assert dark_q["count_gt_0_25"] == 2254
    assert dark_q["max"] > 0.75


def test_07_tier_a_30_genes_strict_disjointness_and_non_hvg_status(exp5_results):
    """Verify all 30 Tier A genes are disjoint from Initial 50 & Housekeeping and have hvg_rank > 2000."""
    tier_a = exp5_results["tier_a_top30_non_hvg_dark_genome_hits"]
    assert len(tier_a) == 30
    genes_a = [r["gene"] for r in tier_a]
    assert set(genes_a) == set(TIER_A_GENES_30)
    assert len(set(genes_a) & INITIAL_PANEL_50) == 0
    assert len(set(genes_a) & TIROSH_AND_MITOTIC_HOUSEKEEPING) == 0
    assert len(set(genes_a) & EXCLUDE_EXACT) == 0

    for rec in tier_a:
        g = rec["gene"]
        assert classify_gene_category(g) == "Tier_A_NonHVG_Dark_Genome"
        assert not g.startswith(EXCLUDE_PREFIXES)
        assert rec["n_cohorts_dropped_by_hvg2000"] >= 1
        assert max(rec["hvg_ranks_by_cohort"].values()) > 2000
        assert rec["cellsnail_h_nope_variance_retained"] == 1.0


def test_08_tier_a_flagship_dark_genome_discoveries(exp5_results):
    """Verify flagship Tier A Non-HVG discoveries (DNAJC9, VRK1, PKMYT1, TRIP13, CHAF1A, ASF1B, RACGAP1)."""
    by_gene = {r["gene"]: r for r in exp5_results["tier_a_top30_non_hvg_dark_genome_hits"]}

    dnajc9 = by_gene["DNAJC9"]
    assert dnajc9["per_cohort_metrics"]["GSE148673_TNBC3"]["hvg_rank"] == 2351
    assert dnajc9["per_cohort_metrics"]["GSE148673_TNBC3"]["delta_20clock"] > 0.65
    assert dnajc9["per_cohort_metrics"]["GSE144239_cSCC_P2_rep1"]["hvg_rank"] == 3837

    vrk1 = by_gene["VRK1"]
    assert vrk1["per_cohort_metrics"]["GSE148673_TNBC3"]["hvg_rank"] == 2856
    assert vrk1["per_cohort_metrics"]["GSE148673_TNBC3"]["delta_20clock"] > 0.65

    pkmyt1 = by_gene["PKMYT1"]
    assert pkmyt1["per_cohort_metrics"]["GSE148673_TNBC3"]["hvg_rank"] == 6146
    assert pkmyt1["per_cohort_metrics"]["GSE148673_TNBC3"]["delta_20clock"] > 0.58
    assert pkmyt1["per_cohort_metrics"]["GSE144239_cSCC_P2_rep1"]["hvg_rank"] == 6182

    trip13 = by_gene["TRIP13"]
    assert trip13["per_cohort_metrics"]["GSE148673_TNBC3"]["hvg_rank"] == 3583
    assert trip13["per_cohort_metrics"]["GSE148673_TNBC3"]["delta_20clock"] > 0.70

    racgap1 = by_gene["RACGAP1"]
    for c in ["GSE148673_TNBC3", "GSE57872_GBM", "GSE144239_cSCC_P2_rep1"]:
        assert racgap1["per_cohort_metrics"][c]["hvg_rank"] > 2000


def test_09_tier_b_30_genes_strict_disjointness_and_repurposing(exp5_results):
    """Verify all 30 Tier B actionable repurposing genes are disjoint from Initial 50 & Housekeeping."""
    tier_b = exp5_results["tier_b_top30_actionable_repurposing_hits"]
    assert len(tier_b) == 30
    genes_b = [r["gene"] for r in tier_b]
    assert set(genes_b) == set(TIER_B_GENES_30)
    assert len(set(genes_b) & set(TIER_A_GENES_30)) == 0
    assert len(set(genes_b) & INITIAL_PANEL_50) == 0
    assert len(set(genes_b) & TIROSH_AND_MITOTIC_HOUSEKEEPING) == 0

    for rec in tier_b:
        g = rec["gene"]
        assert classify_gene_category(g) == "Tier_B_Actionable_Repurposing"
        assert rec["max_delta_20clock_across_cohorts"] > 0.40
        assert rec["cellsnail_h_nope_variance_retained"] == 1.0


def test_10_tier_b_flagship_repurposing_targets(exp5_results):
    """Verify flagship Tier B targets (NUDT1, SLC7A5, AKT1, HRAS, VCP, HSP90AB1, CDC37, DUT, ITGB4, TP63)."""
    by_gene = {r["gene"]: r for r in exp5_results["tier_b_top30_actionable_repurposing_hits"]}
    assert by_gene["NUDT1"]["per_cohort_metrics"]["GSE148673_TNBC3"]["delta_20clock"] > 0.65
    assert by_gene["SLC7A5"]["per_cohort_metrics"]["GSE144239_cSCC_P2_rep1"]["delta_20clock"] > 0.68
    assert by_gene["AKT1"]["per_cohort_metrics"]["GSE148673_TNBC3"]["delta_20clock"] > 0.50
    assert by_gene["HRAS"]["per_cohort_metrics"]["GSE144239_cSCC_P2_rep1"]["delta_20clock"] > 0.55
    assert by_gene["HSP90AB1"]["per_cohort_metrics"]["GSE148673_TNBC3"]["delta_20clock"] > 0.65
    assert by_gene["HSP90AB1"]["per_cohort_metrics"]["GSE144239_cSCC_P2_rep1"]["delta_20clock"] > 0.65
    assert by_gene["ITGB4"]["per_cohort_metrics"]["GSE144239_cSCC_P2_rep1"]["delta_20clock"] > 0.62
    assert by_gene["TP63"]["per_cohort_metrics"]["GSE144239_cSCC_P2_rep1"]["delta_20clock"] > 0.64


def test_11_multi_database_11_sources_completeness(exp5_results):
    """Verify all 60 genes carry complete 11-database dossiers (Ensembl, gnomAD, ChEMBL, Reactome, QuickGO, STRING, HPA, JASPAR/UniBind)."""
    meta = exp5_results["multi_database_cross_validation_summary"]
    assert meta["total_genes_annotated"] == 60
    assert len(meta["databases_integrated"]) == 11
    assert meta["summary_metrics"]["total_lof_intolerant_count"] >= 35
    assert meta["summary_metrics"]["tier_b_clinical_stage_count"] >= 25
    assert meta["summary_metrics"]["tier_b_approved_drug_targets_count"] >= 7

    for rec in exp5_results["tier_a_top30_non_hvg_dark_genome_hits"] + exp5_results["tier_b_top30_actionable_repurposing_hits"]:
        dos = rec["multi_database_dossier"]
        assert dos["ensembl_grch38"]["ensembl_gene_id"].startswith("ENSG")
        assert 0.0 <= dos["gnomad_constraint"]["pLI"] <= 1.0
        assert dos["druggability_opentargets_chembl"]["uniprot_id"] != ""
        assert dos["reactome_pathway"]["pathway_id"].startswith("R-HSA-")
        assert dos["quickgo_primary_annotation"]["go_id"].startswith("GO:")
        assert dos["string_clock_bridge"]["combined_score"] > 0.60
        assert len(dos["hpa_subcellular_localization"]["main_locations"]) >= 1
        assert len(dos["jaspar_unibind_tf_regulators"]) >= 2


def test_12_pillar1_permutation_null_b200_fdr(exp5_results):
    """Verify 30/30 Tier A and 30/30 Tier B genes pass B=200 permutation null at BH-FDR < 0.01."""
    perm = exp5_results["statistical_falsification_summary"]["permutation_null_b200"]
    assert perm["n_permutations"] == 200
    assert perm["fdr_threshold"] == 0.01
    assert perm["tier_a_genes_passing_fdr_0_01_in_at_least_one_cohort"] == 30
    assert perm["tier_b_genes_passing_fdr_0_01_in_at_least_one_cohort"] == 30

    for rec in exp5_results["tier_a_top30_non_hvg_dark_genome_hits"] + exp5_results["tier_b_top30_actionable_repurposing_hits"]:
        assert rec["min_permutation_fdr_20clock"] < 0.01
        assert rec["max_permutation_z_20clock"] > 4.0


def test_13_pillar2_countsplit_binomial_holdout_5seeds(exp5_results):
    """Verify 5-seed Binomial countsplit UMI holdout achieves r > 0.92 with 0.0 UMI additivity error."""
    cs = exp5_results["statistical_falsification_summary"]["countsplit_binomial_holdout"]
    for cname, min_r in [
        ("GSE148673_TNBC3", 0.94),
        ("GSE148673_TNBC1", 0.94),
        ("GSE144239_cSCC_P2_rep1", 0.92),
    ]:
        cohort_cs = cs[cname]
        assert cohort_cs["n_seeds"] == 5
        assert cohort_cs["mean_whole_genome_pearson_r_20clock"] >= min_r
        assert cohort_cs["mean_tier_ab_pearson_r_20clock"] >= min_r
        for run in cohort_cs["seed_runs"]:
            assert run["umi_additivity_max_error"] == 0.0


def test_14_pillar3_multi_patient_lodo_and_cscc_replicate_generalization(exp5_results):
    """Verify 5-patient TNBC LODO, 7-donor GBM LODO, and cSCC P2_ST_rep1 vs P2_ST_rep2 generalization."""
    gen = exp5_results["statistical_falsification_summary"]["multi_patient_and_replicate_generalization"]
    assert gen["tnbc_5patient_lodo_mean_pearson_r"] > 0.75
    assert gen["tnbc_5patient_lodo_mean_spearman_rho"] > 0.75
    assert len(gen["tnbc_5patient_lodo_folds"]) == 5

    assert gen["gbm_7donor_lodo_mean_pearson_r"] > 0.75
    assert len(gen["gbm_7donor_lodo_folds"]) == 7

    assert gen["cscc_p2_rep1_vs_rep2_pearson_r"] > 0.95
    assert gen["cscc_p2_rep1_vs_rep2_spearman_rho"] > 0.94


def test_15_cross_engine_convergence_exp4_exp6_exp7(exp5_results):
    """Verify Cross-Engine Convergence across Expeditions 4, 5, 6, and 7."""
    ce = exp5_results["cross_engine_convergence"]

    exp4 = ce["exp5_exp4_streaming_and_hvg_blind_spot"]
    assert exp4["max_abs_diff_f32_rust_vs_python"] == 0.0
    assert exp4["mask_mismatches_across_33694_genes"] == 0
    assert exp4["bcm2712_l3_cache_hit_rate_pct"] == 99.53

    exp6 = ce["exp5_exp6_pan_autosome_focal_stringers"]
    assert exp6["triple_erasure_flagship_discovery"]["gene"] == "PKMYT1"
    assert exp6["double_erasure_pan_patient_chaperone_hub"]["gene"] == "HSP90AB1"
    assert len(exp6["pan_autosome_stringer_candidates_evaluated"]) == 25

    exp7 = ce["exp5_exp7_spatial_boundary_secretome"]
    assert exp7["total_cscc_slides_integrated_in_exp7"] == 6
    assert exp7["fraction_of_significant_juxtacrine_pairs_dropped_by_hvg2000"] == 0.7933
    assert len(exp7["boundary_secretome_receptome_60gene_table"]) == 60


def test_16_qr_deltas_match_exact_ols_regression():
    """Verify compute_whole_genome_qr_deltas matches explicit Moore-Penrose OLS R^2 to machine precision."""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(120, 40))
    core_idx = [0, 1, 2, 3, 4]
    ext_idx = list(range(15))
    qr_res = compute_whole_genome_qr_deltas(X, core_indices=core_idx, ext_indices=ext_idx)

    Xc = X - np.mean(X, axis=0, keepdims=True)
    Cc = Xc[:, core_idx]
    beta, _, _, _ = np.linalg.lstsq(Cc, Xc, rcond=None)
    pred = Cc @ beta
    r2_ols = np.var(pred, axis=0) / np.var(Xc, axis=0)
    assert np.max(np.abs(qr_res["delta_5clock"] - r2_ols)) < 1e-13


def test_17_nested_subspace_chain_d2_le_d5_le_d20_on_tier_ab(exp5_results):
    """Verify nested subspace ordering Delta_5clock <= Delta_20clock holds across every Tier A and Tier B gene in every cohort."""
    for rec in exp5_results["tier_a_top30_non_hvg_dark_genome_hits"] + exp5_results["tier_b_top30_actionable_repurposing_hits"]:
        for cname, m in rec["per_cohort_metrics"].items():
            assert m["delta_5clock"] <= m["delta_20clock"] + 1e-6, f"Monotonicity violated for {rec['gene']} in {cname}"


def test_18_classify_gene_category_exhaustive():
    """Verify classify_gene_category correctly partitions Initial 50, Housekeeping, Tier A, Tier B, and Other."""
    assert classify_gene_category("EGFR") == "Initial_50_Gene_Panel"
    assert classify_gene_category("CCNB1") == "Excluded_Housekeeping_or_Artifact"
    assert classify_gene_category("RPL13A") == "Excluded_Housekeeping_or_Artifact"
    assert classify_gene_category("MT-CO1") == "Excluded_Housekeeping_or_Artifact"
    assert classify_gene_category("HIST1H4C") == "Excluded_Housekeeping_or_Artifact"
    assert classify_gene_category("DNAJC9") == "Tier_A_NonHVG_Dark_Genome"
    assert classify_gene_category("PKMYT1") == "Tier_A_NonHVG_Dark_Genome"
    assert classify_gene_category("NUDT1") == "Tier_B_Actionable_Repurposing"
    assert classify_gene_category("SLC7A5") == "Tier_B_Actionable_Repurposing"
    assert classify_gene_category("TP63") == "Tier_B_Actionable_Repurposing"
    assert classify_gene_category("UNCHARACTERIZED_GENE_XYZ") == "Other_Expressed_H_nope"


def test_19_tier_a_gnomad_constraint_and_string_clock_bridges(exp5_results):
    """Verify Tier A genes include strongly constrained chromatin/replication factors with direct STRING bridges to the mitotic clock."""
    tier_a = exp5_results["tier_a_top30_non_hvg_dark_genome_hits"]
    lof_intolerant = [
        r["gene"] for r in tier_a
        if r["multi_database_dossier"]["gnomad_constraint"]["haploinsufficient_lof_intolerant"]
    ]
    assert len(lof_intolerant) >= 14
    for must_have in ["SSRP1", "NELFCD", "TFDP1", "MORF4L1", "LMNB1", "CHAF1A", "BOP1", "RACGAP1", "SEPHS1", "VRK1"]:
        assert must_have in lof_intolerant


def test_20_tier_b_clinical_tractability_and_potency(exp5_results):
    """Verify Tier B genes include approved and Phase II/III targets with nanomolar ChEMBL bioactivities."""
    by_gene = {r["gene"]: r["multi_database_dossier"] for r in exp5_results["tier_b_top30_actionable_repurposing_hits"]}
    assert by_gene["AKT1"]["druggability_opentargets_chembl"]["max_clinical_phase"] == 4
    assert by_gene["HSP90AB1"]["druggability_opentargets_chembl"]["max_clinical_phase"] == 4
    assert by_gene["PSMB6"]["druggability_opentargets_chembl"]["max_clinical_phase"] == 4
    assert by_gene["SLC7A5"]["druggability_opentargets_chembl"]["max_clinical_phase"] == 3
    assert by_gene["HRAS"]["druggability_opentargets_chembl"]["max_clinical_phase"] == 3
    assert by_gene["NUDT1"]["druggability_opentargets_chembl"]["potency_ic50_or_ki_nM"] <= 10.0


def test_21_cscc_layer3_tsk_niche_secretome_erasure(exp5_results):
    """Verify Layer-3 TSK leading-edge OLS erasure on boundary secretome/receptome genes (SLC7A5, MMP13, DDR1, LOXL2, ADAM17)."""
    sec = exp5_results["cross_engine_convergence"]["exp5_exp7_spatial_boundary_secretome"]
    table = {
        r["gene"]: r
        for r in sec["boundary_secretome_receptome_60gene_table"]
    }
    assert table["SLC7A5"]["cSCC_P2_rep1"]["layer3_tsk_delta_20clock"] > 0.55
    assert table["SLC3A2"]["cSCC_P2_rep1"]["delta_20clock"] > 0.55
    assert table["MMP13"]["cSCC_P2_rep1"]["layer3_tsk_delta_20clock"] > 0.18
    assert table["DDR1"]["TNBC3"]["delta_20clock"] > 0.45
    assert table["DDR1"]["cSCC_P2_rep1"]["layer3_tsk_delta_20clock"] > 0.40
    assert table["LOXL2"]["TNBC3"]["hvg_rank"] > 2000
    assert table["ADAM17"]["TNBC3"]["hvg_rank"] > 2000
    axes = {a["axis"]: a for a in sec["flagship_cross_wall_axes_erased_by_ols"]}
    assert axes["LAMC2 -> ITGB4"]["itgb4_cscc_rep1_tsk_layer3_delta_20clock"] > 0.45


def test_22_pan_autosome_stringer_candidates_cross_validation(exp5_results):
    """Verify pan-autosome stringer candidates (PKMYT1, HSP90AB1, MCL1, XPO1, BRD4, GPX4, TOP1) in Exp 5 <-> Exp 6 table."""
    table = {
        r["gene"]: r
        for r in exp5_results["cross_engine_convergence"]["exp5_exp6_pan_autosome_focal_stringers"]["pan_autosome_stringer_candidates_evaluated"]
    }
    assert table["PKMYT1"]["max_delta_20clock"] > 0.65
    assert table["HSP90AB1"]["max_delta_20clock"] > 0.65
    assert table["TOP1"]["max_delta_20clock"] > 0.50
    assert table["GPX4"]["max_delta_20clock"] > 0.40
    assert table["BRD4"]["max_delta_20clock"] > 0.35


def test_23_tnbc_5patient_orthogonality_across_all_donors(exp5_results):
    """Verify CellSnail orthogonality and 100% H_nope retention across all 5 TNBC donors (TNBC1..TNBC5)."""
    q_by_cohort = exp5_results["genome_wide_distribution_quantiles_by_cohort"]
    for pid in ["TNBC1", "TNBC2", "TNBC3", "TNBC4", "TNBC5"]:
        ov = q_by_cohort[f"GSE148673_{pid}"]["orthogonality_verification"]
        assert ov["max_inner_product_residual"] == 0.0
        assert abs(ov["total_norm_conservation_ratio"] - 1.0) < 1e-12
        assert ov["h_nope_variance_retained_ratio"] == 1.0
        assert ov["false_positive_clock_leakage_into_nope"] == 0.0


def test_24_countsplit_per_gene_consistency_scores(exp5_results):
    """Verify per-gene Binomial countsplit consistency scores exceed 0.90 for top Tier A/B genes in TNBC3."""
    per_gene = exp5_results["statistical_falsification_summary"]["countsplit_binomial_holdout"]["GSE148673_TNBC3"]["per_gene_split_summary"]
    for g in ["DNAJC9", "VRK1", "PKMYT1", "TRIP13", "ASF1B", "RNASEH2A", "NUDT1", "SLC7A5", "HSP90AB1", "RAD21"]:
        assert per_gene[g]["split_consistency_score"] > 0.95


def test_25_pipeline_runner_idempotency(tmp_path):
    """Verify run_expedition5_whole_genome_delta_census writes identical deterministic JSON output."""
    out_file = tmp_path / "exp5_test_out.json"
    res = run_expedition5_whole_genome_delta_census(output_path=out_file)
    assert out_file.exists()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["expedition_id"] == res["expedition_id"]
    assert len(loaded["tier_a_top30_non_hvg_dark_genome_hits"]) == 30
    assert len(loaded["tier_b_top30_actionable_repurposing_hits"]) == 30
