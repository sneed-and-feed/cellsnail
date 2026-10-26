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

    # True persisters in cohort
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

    # Verify discovered amplicons include key oncogenes
    discovered_genes = [a["gene"] for a in receipt["amplicons"]]
    assert any(g in discovered_genes for g in ["EGFR", "CDK4", "MYC", "ABCB1"])

    # Verify sharpness
    for amp in receipt["amplicons"]:
        assert amp["sharpness_ratio"] > 1.5


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


def test_run_discovery_pipeline_complete():
    res = run_discovery_pipeline()
    assert res["status"] == "DISCOVERY_PIPELINE_COMPLETE"
    assert "persister_subclone" in res
    assert "focal_cnv_amplicons" in res
    assert "spatial_niches" in res
    assert "multiomic_dosage_factorization" in res
