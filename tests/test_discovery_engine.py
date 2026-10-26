"""tests/test_discovery_engine.py
===============================
Unit and integration tests for the "New Light" biological discovery engine.
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


def test_discover_ghost_regulon_audit():
    engine = DiscoveryEngine(random_state=42)
    receipt = engine.discover_ghost_regulon_audit()
    assert receipt["type"] == "ghost_regulon_ercc_audit"
    assert receipt["well_permuted_negative_control_spurious_pairs_r070"]["cellsnail_morse"] == 0
    assert receipt["well_permuted_negative_control_spurious_pairs_r070"]["magic_k15_t3"] > 0
    assert "EGFR" in receipt["human_lung_adenocarcinoma_lineage_markers"]


def test_discover_rare_persister_subclones():
    engine = DiscoveryEngine(random_state=42)
    cohort = load_paired_cnv_cohort(n_cells=500, random_state=42)

    receipt = engine.discover_rare_persister_subclones(
        cohort["dna"],
        min_fraction=0.005,
    )

    assert receipt["type"] == "rare_persister_subclone"
    assert receipt["discovered_persister_count"] > 0
    assert receipt["discovered_persister_count"] <= receipt["max_allowed_cells"]
    assert receipt["ensembl_gene_id"].startswith("ENSG")

    true_persisters = set(cohort["persister_indices"])
    discovered = set(receipt["discovered_persister_indices"])
    assert len(discovered.intersection(true_persisters)) > 0


def test_discover_focal_cnv_stringers():
    engine = DiscoveryEngine(random_state=42)
    cohort = load_paired_cnv_cohort(n_cells=500, random_state=42)

    receipt = engine.discover_focal_cnv_stringers(
        cohort["dna"].X,
        bin_metadata=cohort["variants"],
    )

    assert receipt["type"] == "sub_megabase_focal_amplicons"
    assert receipt["focal_amplicons_discovered"] > 0

    discovered_genes = [a["gene"] for a in receipt["amplicons"]]
    assert any(g in discovered_genes for g in ["EGFR", "CDK4", "MYC", "ABCB1"])

    for amp in receipt["amplicons"]:
        assert amp["sharpness_ratio"] > 1.20
        if amp["gene"] == "EGFR":
            assert amp["ensembl_gene_id"] == "ENSG00000146648"
            assert len(amp["clinvar_ids"]) > 0


def test_discover_cell_cycle_deconfounding():
    engine = DiscoveryEngine(random_state=42)
    receipt = engine.discover_cell_cycle_deconfounding()
    assert receipt["type"] == "cell_cycle_deconfounding"
    assert receipt["tnbc3_max_orthogonality_residual"] < 1e-12
    assert receipt["gbm_max_orthogonality_residual"] < 1e-12
    assert abs(receipt["total_norm_conservation_ratio"] - 1.0) < 1e-12
    assert receipt["naive_ols_over_subtraction_loss"] > 0.05


def test_discover_synthetic_lethality_cavities():
    engine = DiscoveryEngine(random_state=42)
    receipt = engine.discover_synthetic_lethality_cavities()
    assert receipt["type"] == "synthetic_lethality_cavities"
    assert receipt["beta1_cavities_detected"] >= 2
    assert receipt["so10_tiebreaker_radial_norm_drift"] < 1e-14


def test_discover_deconfounded_spatial_niches():
    engine = DiscoveryEngine(random_state=42)
    spatial_cohort = load_merfish_spatial(n_cells=400, random_state=42)

    receipt = engine.discover_deconfounded_spatial_niches(
        spatial_cohort.obsm["spatial"],
        spatial_cohort.X,
        contact_edges=spatial_cohort.uns["contact_edges"],
    )

    assert receipt["type"] == "deconfounded_spatial_niches"
    assert receipt["max_penetration_residual"] == 0.0
    assert receipt["mean_penetration_residual"] == 0.0
    assert receipt["walls_clamped"] > 0
    assert receipt["cscc_invasion_front_analysis"]["invasion_front_contrast_ratio"] > 5.0


def test_factorize_multiomic_dosage_linkage():
    engine = DiscoveryEngine(random_state=42)
    rng = np.random.default_rng(42)

    dna_cov = rng.standard_normal((3, 3))
    dna_cov = dna_cov @ dna_cov.T + np.eye(3)

    rna_cov = rng.standard_normal((3, 3))
    rna_cov = rna_cov @ rna_cov.T + np.eye(3)

    receipt = engine.factorize_multiomic_dosage_linkage(dna_cov, rna_cov)
    assert receipt["type"] == "multiomic_dosage_factorization"
    assert receipt["trace_residual"] <= 1.14e-13
    assert receipt["trace_conservation_passed"] is True
    assert receipt["driver_subspace_variance"] > 0
    assert receipt["passenger_subspace_variance"] > 0


def test_focal_cnv_stringers_column_permutation_invariance():
    engine = DiscoveryEngine(random_state=42)
    cohort = load_paired_cnv_cohort(n_cells=500, random_state=42)
    X_dna = cohort["dna"].X
    variants = cohort["variants"]

    # Adversarial permutation placing clonal focal bins [25, 35, 55] into adjacent columns
    focal_bins = [25, 35, 55]
    other_bins = [b for b in range(100) if b not in focal_bins]
    adv_perm = np.array(other_bins[:10] + focal_bins + other_bins[10:], dtype=int)
    X_perm = X_dna[:, adv_perm]
    meta_perm = [variants[i] for i in adv_perm]

    # Without genomic coordinate sorting, adjacent focal amplicons collide in the 3-point stencil
    res_unsorted = engine.discover_focal_cnv_stringers(
        X_perm,
        bin_metadata=meta_perm,
        sort_by_genomic_coords=False,
    )
    unsorted_genes = {a["gene"] for a in res_unsorted["amplicons"]}
    assert len(unsorted_genes.intersection({"EGFR", "MYC", "CDK4"})) < 3

    # With canonical genomic coordinate sorting (default), all clonal focal amplicons are recovered
    res_sorted = engine.discover_focal_cnv_stringers(
        X_perm,
        bin_metadata=meta_perm,
        sort_by_genomic_coords=True,
    )
    sorted_genes = {a["gene"] for a in res_sorted["amplicons"]}
    assert {"EGFR", "MYC", "CDK4"}.issubset(sorted_genes)


def test_run_discovery_pipeline_complete():
    res = run_discovery_pipeline()
    assert res["status"] == "DISCOVERY_PIPELINE_COMPLETE"
    assert "ghost_regulon_audit" in res
    assert "persister_subclone" in res
    assert "focal_cnv_amplicons" in res
    assert "cell_cycle_deconfounding" in res
    assert "synthetic_lethality_cavities" in res
    assert "spatial_niches" in res
    assert "multiomic_dosage_factorization" in res

