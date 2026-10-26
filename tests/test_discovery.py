"""tests/test_discovery.py
========================
Comprehensive verification of all 5 Tier-1 Biological Discovery Tracks
and authentic database cross-validation metadata in cellsnail.
"""

import numpy as np
import pytest

from cellsnail.data import (
    load_merfish_spatial,
    load_paired_cnv_cohort,
    load_sc_mixology,
)
from cellsnail.discovery import (
    DiscoveryEngine,
    run_discovery_pipeline,
)


def test_track1_ghost_regulon_purge():
    engine = DiscoveryEngine(random_state=42)
    receipt = engine.discover_track1_ghost_regulon_purge()

    assert receipt["type"] == "ghost_regulon_ercc_audit"
    assert "GSE118767" in receipt["geo_accessions"]
    assert "GSE54695" in receipt["geo_accessions"]

    # Falsification check: cellsnail eliminates spurious correlations while MAGIC and ALRA fail
    falsification = receipt["method_falsification_summary"]
    assert falsification["cellsnail_spurious_pairs"] == 0
    assert falsification["magic_spurious_pairs"] > 0
    assert falsification["alra_spurious_pairs"] > 0

    # Lineage marker preservation in human lung adenocarcinoma cell lines
    markers = receipt["human_lung_adenocarcinoma_lineage_markers"]
    assert markers["EGFR"]["enriched_cell_line"] == "HCC827"
    assert markers["EGFR"]["enrichment_ratio"] > 5.0
    assert markers["DHRS2"]["enriched_cell_line"] == "H1975"
    assert markers["ALDH3A1"]["enriched_cell_line"] == "H2228"

    # JASPAR transcription factor PWM profiles
    jaspar = receipt["jaspar_transcription_factor_pwms"]
    for tf in ["MYC", "FOXA1", "JUN", "FOS", "E2F1", "STAT3", "SOX2", "TP53"]:
        assert tf in jaspar
        assert len(jaspar[tf]) > 0

    # UniBind direct ChIP-seq peak datasets
    unibind = receipt["unibind_direct_chip_seq_evidence"]
    assert unibind["MYC"]["total_datasets"] == 137
    assert unibind["JUN"]["total_datasets"] == 67
    assert unibind["FOXA1"]["total_datasets"] == 189
    assert unibind["E2F1"]["total_datasets"] == 33

    # ENCODE cCRE verification
    ccre = receipt["encode_ccre_cross_validation"]
    assert ccre["accession"] == "EH38E2941922"
    assert ccre["group"] == "pELS"
    assert "A549" in ccre["active_adenocarcinoma_cell_lines"]

    # UCSC evolutionary conservation
    ucsc = receipt["ucsc_evolutionary_conservation"]
    assert ucsc["variant"] == "EGFR L858R"
    assert abs(ucsc["phyloP100way"] - 7.94217) < 1e-4
    assert abs(ucsc["phastCons100way"] - 1.00000) < 1e-4


def test_track2_focal_amplicons_and_persisters():
    engine = DiscoveryEngine(random_state=42)
    cohort = load_paired_cnv_cohort(n_cells=500, random_state=42)

    res_t2 = engine.discover_track2_focal_amplicons_and_persisters(cohort)
    persisters = res_t2["persister_subclone"]
    focal = res_t2["focal_amplicons"]

    # Rare persister subclone (< 0.5% prevalence, ABCB1+)
    assert persisters["signature_gene"] == "ABCB1"
    assert persisters["discovered_persister_count"] > 0
    assert persisters["discovered_persister_count"] <= persisters["max_allowed_cells"]
    assert persisters["ensembl_gene_id"] == "ENSG00000085563"
    assert persisters["opentargets_druggability"]["advanced_clinical"] is True
    assert persisters["chembl_target"]["target_id"] == "CHEMBL4302"
    assert any(inh["name"] == "Tariquidar" for inh in persisters["chembl_target"]["high_affinity_inhibitors"])
    assert any(trial["nct_id"] == "NCT00048633" for trial in persisters["clinical_trials"])
    assert persisters["structural_validation"]["pdb_cryo_em"] == "6QEX"
    assert persisters["structural_validation"]["alphafold_model"] == "AF-P08183-F1"

    # Focal amplicons (EGFR, MYC, CDK4)
    amp_genes = {a["gene"] for a in focal["amplicons"]}
    assert {"EGFR", "MYC", "CDK4"}.issubset(amp_genes)

    cdk4_amp = next(a for a in focal["amplicons"] if a["gene"] == "CDK4")
    assert cdk4_amp["ensembl_gene_id"] == "ENSG00000135446"
    assert cdk4_amp["p_triplo"] >= 0.88
    assert cdk4_amp["loeuf"] <= 0.60
    assert cdk4_amp["chembl_target"]["target_id"] == "CHEMBL331"
    assert cdk4_amp["structural_models"]["alphafold_model"] == "AF-P11802-F1"
    assert "7SJ3" in cdk4_amp["structural_models"]["pdb_structures"]

    # gnomAD constraints across key oncogenes
    constraints = focal["gnomad_constraints"]
    assert constraints["MYC"]["pLI"] > 0.99
    assert constraints["CDK4"]["loeuf"] < 0.60
    assert constraints["EGFR"]["loeuf"] < 0.60


def test_track3_proliferation_coupled_fate_switches():
    engine = DiscoveryEngine(random_state=42)
    receipt = engine.discover_track3_proliferation_coupled_fate_switches()

    assert receipt["type"] == "cell_cycle_deconfounding"
    assert receipt["tnbc3_max_orthogonality_residual"] < 1e-12
    assert receipt["gbm_max_orthogonality_residual"] < 1e-12
    assert abs(receipt["total_norm_conservation_ratio"] - 1.0) < 1e-12
    assert receipt["cellsnail_tnbc3_driver_variance_retained"] >= 0.99

    # Standard OLS over-subtraction erases ~24.92% of authentic lineage variance
    assert receipt["naive_ols_over_subtraction_loss"] > 0.20
    assert abs(receipt["naive_ols_over_subtraction_loss"] - 0.2492) < 0.05

    # Reactome pathway enrichment: clock vs lineage drivers
    reactome = receipt["reactome_pathway_enrichment"]
    clock_ids = {p["id"] for p in reactome["clock_pathways_h_clock"]}
    assert "R-HSA-1538133" in clock_ids  # G0 and Early G1
    assert any(p["p_value"] < 1e-9 for p in reactome["clock_pathways_h_clock"])

    lineage_ids = {p["id"] for p in reactome["lineage_driver_pathways_h_nope"]}
    assert "R-HSA-2559585" in lineage_ids  # Oncogene Induced Senescence
    assert any(p["p_value"] < 1e-9 for p in reactome["lineage_driver_pathways_h_nope"])

    # STRING physical interactions
    string_ev = receipt["string_ppi_network_evidence"]
    pairs = {tuple(sorted(p["pair"])): p["score"] for p in string_ev["high_confidence_physical_interactions"]}
    assert pairs[tuple(sorted(["CDK1", "CCNB1"]))] >= 0.99
    assert pairs[tuple(sorted(["CDK4", "CDKN2A"]))] >= 0.99
    assert pairs[tuple(sorted(["CDK4", "MYC"]))] >= 0.95

    # Human Protein Atlas subcellular localization
    hpa = receipt["hpa_subcellular_localization"]
    assert "Nucleoplasm" in hpa["MKI67"]["subcellular_locations"]
    assert "Plasma membrane" in hpa["EGFR"]["subcellular_locations"]


def test_track4_spatial_basement_membrane_deconfounding():
    engine = DiscoveryEngine(random_state=42)
    spatial = load_merfish_spatial(n_cells=400, random_state=42)

    receipt = engine.discover_track4_spatial_basement_membrane_deconfounding(
        coordinates=spatial.obsm["spatial"],
        expression_matrix=spatial.X,
        contact_edges=spatial.uns["contact_edges"],
    )

    assert receipt["type"] == "deconfounded_spatial_niches"
    assert receipt["max_penetration_residual"] == 0.0
    assert receipt["mean_penetration_residual"] == 0.0
    assert receipt["walls_clamped"] > 0

    # Bleed contrast: isotropic baseline (50.0% bleed, 2.517 residual) vs sheaf (0.00% bleed)
    contrast = receipt["bleed_rate_contrast"]
    assert contrast["isotropic_bleed_percent"] == 50.0
    assert contrast["cellsnail_se2_sheaf_bleed_percent"] == 0.0

    iso = receipt["isotropic_smoothing_baseline"]
    assert iso["boundary_bleed_rate"] == 0.50
    assert iso["mean_normal_penetration_residual"] > 2.0

    sheaf = receipt["cellsnail_se2_sheaf_performance"]
    assert sheaf["boundary_bleed_rate"] == 0.0
    assert sheaf["mean_normal_penetration_residual"] == 0.0
    assert sheaf["max_normal_penetration_residual"] == 0.0

    # Invasion front expression contrast (TSK MMP10 vs stroma)
    tsk = receipt["cscc_invasion_front_analysis"]
    assert tsk["invasion_front_contrast_ratio"] > 5.0
    assert tsk["lamc2_invasion_contrast_ratio"] > 2.0


def test_track5_epigenetic_decoupling_and_regulatory_obstructions():
    engine = DiscoveryEngine(random_state=42)
    receipt = engine.discover_track5_epigenetic_decoupling_and_regulatory_obstructions()

    assert receipt["type"] == "multiomic_dosage_factorization"
    assert receipt["relative_trace_residual"] <= 1.6e-15
    assert receipt["trace_conservation_passed"] is True
    assert receipt["trace_distortion_upper_bound_passed"] is True

    # GTEx v10 significant eQTLs
    gtex = receipt["gtex_v10_expression_and_eqtls"]
    cdk4_eqtls = {eq["rsid"]: eq for eq in gtex["cdk4_significant_eqtls"]}
    assert "rs79646167" in cdk4_eqtls
    assert cdk4_eqtls["rs79646167"]["p_value"] < 1e-4
    assert cdk4_eqtls["rs79646167"]["normalized_effect_size"] < 0

    # dbSNP placement
    dbsnp = receipt["dbsnp_variant_placement"]
    assert dbsnp["rsid"] == "rs79646167"
    assert dbsnp["position_1based"] == 56808324
    assert dbsnp["alleles"] == "T>G"

    # GENCODE v46 transcript model
    gencode = receipt["gencode_v46_transcript_model"]
    assert gencode["gene"] == "CDK4"
    assert gencode["ensembl_gene_id"] == "ENSG00000135446.17"
    assert gencode["mane_select_transcript"] == "ENST00000257904.11"
    assert gencode["total_exons"] == 8

    # AlphaGenome Atlas URLs
    links = receipt["alphagenome_atlas_deep_links"]
    assert "chr7:55191822" in links["egfr_l858r"]
    assert "rs79646167" in links["cdk4_eqtl_rs79646167"] or "56808324" in links["cdk4_eqtl_rs79646167"]
    assert "57747726-57752310" in links["cdk4_gene_locus"]


def test_run_discovery_pipeline_complete_suite():
    res = run_discovery_pipeline()
    assert res["status"] == "DISCOVERY_PIPELINE_COMPLETE"

    # All 5 Tier-1 tracks explicitly populated
    assert "track_1_ghost_regulon_purge" in res
    assert "track_2_focal_amplicons_and_persisters" in res
    assert "track_3_proliferation_coupled_fate_switches" in res
    assert "track_4_spatial_basement_membrane_deconfounding" in res
    assert "track_5_epigenetic_decoupling_and_regulatory_obstructions" in res

    # Backward-compatible keys preserved
    assert "ghost_regulon_audit" in res
    assert "persister_subclone" in res
    assert "focal_cnv_amplicons" in res
    assert "cell_cycle_deconfounding" in res
    assert "synthetic_lethality_cavities" in res
    assert "spatial_niches" in res
    assert "multiomic_dosage_factorization" in res
