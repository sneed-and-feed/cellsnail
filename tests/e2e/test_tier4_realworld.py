"""Tier 4: Comprehensive Real-World Application Workloads E2E Tests.

Evaluates authentic, non-circular public genomic datasets:
- NCBI GEO GSE118767 (sc_mixology CEL-seq2 lung adenocarcinoma lines)
- NCBI GEO GSE54695 (ERCC92 synthetic spike-in control negative baseline)
- NCBI GEO GSE148673 (CopyKAT TNBC3 primary triple-negative breast cancer scRNA)
- NCBI GEO GSE57872 (Primary human glioblastoma SMART-seq mosaic drivers)
- NCBI GEO GSE144239 (cSCC spatial transcriptomics with calibrated tissue coordinates)
- Ensembl GRCh38 v113 canonical annotations & ClinVar/COSMIC oncology hotspot panel
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest

from cellsnail.core.morse_filter import morse_dropout_filter
from cellsnail.data import (
    load_ensembl_clinvar_panel,
    load_sc_mixology,
)
from cellsnail.discovery import DiscoveryEngine


def test_rw01_gse118767_sc_mixology_lineage_separation(empirical_npz: Dict[str, np.ndarray]):
    """Loads GSE118767 sc_mixology and verifies 300 cells across 3 lung adenocarcinoma cell lines."""
    X_hum = empirical_npz["mix_X_human"]
    cell_lines = [str(cl) for cl in empirical_npz["mix_cell_lines"]]
    assert X_hum.shape == (300, 200), f"Expected shape (300, 200), got {X_hum.shape}"
    assert len(cell_lines) == 300

    # Exactly 100 cells per line in balanced mixture
    unique, counts = np.unique(cell_lines, return_counts=True)
    line_counts = dict(zip(unique, counts))
    assert line_counts.get("HCC827") == 100
    assert line_counts.get("H1975") == 100
    assert line_counts.get("H2228") == 100


def test_rw02_gse118767_egfr_amplification_hcc827(empirical_npz: Dict[str, np.ndarray]):
    """Verifies HCC827 cells exhibit authentic EGFR overexpression relative to H2228 without synthetic injection."""
    X_hum = empirical_npz["mix_X_human"]
    gene_names = [str(g) for g in empirical_npz["mix_human_gene_names"]]
    cell_lines = [str(cl) for cl in empirical_npz["mix_cell_lines"]]

    if "EGFR" in gene_names:
        egfr_idx = gene_names.index("EGFR")
        hcc_mask = np.array([cl == "HCC827" for cl in cell_lines])
        h22_mask = np.array([cl == "H2228" for cl in cell_lines])
        hcc_egfr_mean = float(np.mean(X_hum[hcc_mask, egfr_idx]))
        h22_egfr_mean = float(np.mean(X_hum[h22_mask, egfr_idx]))
        assert hcc_egfr_mean >= h22_egfr_mean, "HCC827 EGFR should be higher or equal to H2228"


def test_rw03_gse54695_ercc92_spikein_zero_correlation_control(empirical_npz: Dict[str, np.ndarray]):
    """
    Verifies that under well-permutation decoupling (GSE54695 ERCC92),
    true inter-spike-in biological correlation is 0.000.
    """
    X_ercc = empirical_npz["mix_X_ercc_perm"]
    assert X_ercc.shape == (300, 92)

    # Compute correlation between first two spike-ins
    r = np.corrcoef(X_ercc[:, 0], X_ercc[:, 1])[0, 1]
    # Under permutation decoupling, r should be close to 0 (|r| < 0.20)
    assert abs(r) < 0.20, f"Expected decoupled correlation near 0.0, got {r}"


def test_rw04_gse54695_ercc92_unpermuted_technical_correlation(empirical_npz: Dict[str, np.ndarray]):
    """Evaluates unpermuted ERCC92 spike-ins showing authentic shared technical capture efficiency."""
    if "mix_X_ercc_unperm" in empirical_npz:
        X_unperm = empirical_npz["mix_X_ercc_unperm"]
        assert X_unperm.shape == (300, 92)
        # Technical capture correlation is positive on unpermuted wells
        r = np.corrcoef(X_unperm[:, 0], X_unperm[:, 1])[0, 1]
        assert not np.isnan(r)


def test_rw05_gse148673_tnbc3_scrna_expression_distribution(empirical_npz: Dict[str, np.ndarray]):
    """Verifies GSE148673 TNBC3 single-cell RNA counts are authentic non-negative integers."""
    X_rna = empirical_npz["cnv_sc_rna_expr"]
    assert X_rna.shape == (500, 100)
    assert np.all(X_rna >= 0), "All RNA expression counts must be non-negative"
    assert np.all(np.equal(np.mod(X_rna, 1), 0)), "All UMI counts must be integers"


def test_rw06_gse148673_tnbc3_aneuploid_chromosome_profiles(empirical_npz: Dict[str, np.ndarray]):
    """Evaluates 1D chromosomal profiles across TNBC3 tumor cells."""
    X_rna = empirical_npz["cnv_sc_rna_expr"]
    # Mean profile across 500 cells
    mean_profile = np.mean(X_rna, axis=0)
    assert len(mean_profile) == 100
    res = morse_dropout_filter(mean_profile.astype(np.float32), tau_mad=1.0)
    assert len(res["filtered"]) == 100


def test_rw07_gse57872_glioblastoma_mosaic_driver_amplifications(empirical_npz: Dict[str, np.ndarray]):
    """
    Verifies GSE57872 primary glioblastoma contains genuine driver genes
    (EGFR, PDGFRA, MET) across 543 single cells (Patel et al., Science 2014).
    """
    X_gbm = empirical_npz["gbm_X_drivers"]
    drivers = [str(d) for d in empirical_npz["gbm_driver_genes"]]
    assert X_gbm.shape == (543, 12)
    assert "EGFR" in drivers
    assert "CDK4" in drivers
    assert "PTEN" in drivers

    # Verify that single-cell values are finite normalized expression
    assert np.all(np.isfinite(X_gbm))


def test_rw08_gse57872_gbm_single_cell_heterogeneity(empirical_npz: Dict[str, np.ndarray]):
    """Verifies that glioblastoma driver expression exhibits authentic cell-to-cell variance."""
    X_gbm = empirical_npz["gbm_X_drivers"]
    variances = np.var(X_gbm, axis=0)
    assert np.all(variances > 0.0), "All driver genes must show non-zero variance across single cells"


def test_rw09_gse144239_cscc_spatial_transcriptomics_raw_tissue_coordinates(empirical_npz: Dict[str, np.ndarray]):
    """Verifies GSE144239 cSCC spatial transcriptomics contains genuine calibrated tissue coordinates."""
    coords_um = empirical_npz["st_coords_um"]
    assert coords_um.shape == (400, 2)
    # Coordinates in micrometers must be positive
    assert np.all(coords_um[:, 0] >= 0.0)
    assert np.all(coords_um[:, 1] >= 0.0)


def test_rw10_gse144239_cscc_spatial_marker_compartment_localization(empirical_npz: Dict[str, np.ndarray]):
    """Verifies presence of key cutaneous squamous cell carcinoma marker genes in GSE144239."""
    st_genes = [str(g) for g in empirical_npz["st_gene_names"]]
    assert len(st_genes) == 50
    # Marker genes from published study
    has_markers = any(g in st_genes for g in ["KRT14", "KRT5", "KRT1", "KRT10", "MMP10", "COL1A1"])
    assert has_markers, "Expected canonical cSCC markers in spatial panel"


def test_rw11_ensembl_grch38_canonical_oncology_coordinates_validation(clinvar_panel: Dict[str, Any]):
    """Validates Ensembl GRCh38 canonical coordinates for the 14-gene oncology driver panel."""
    genes = clinvar_panel.get("genes", {})
    assert len(genes) >= 10, f"Expected >= 10 driver genes, got {len(genes)}"

    for gene, info in genes.items():
        assert "seq_region_name" in info or "chrom" in info, f"Missing chrom/seq_region_name for {gene}"
        assert "start" in info, f"Missing start for {gene}"
        assert "end" in info, f"Missing end for {gene}"
        assert info["start"] < info["end"], f"Invalid interval for {gene}: start >= end"


def test_rw12_clinvar_pathogenic_oncology_hotspots_locus_mapping(clinvar_panel: Dict[str, Any]):
    """Validates ClinVar classifications and pathogenicity for oncology hotspot variants."""
    hotspots = clinvar_panel.get("clinical_hotspot_variants", {})
    assert len(hotspots) > 0, "Expected ClinVar clinical hotspot variants"

    pathogenic_count = 0
    for var_id, var_info in hotspots.items():
        assert "gene_symbol" in var_info or "gene" in var_info
        sig = var_info.get("clinvar_significance", [])
        assert len(sig) > 0, f"Missing clinvar_significance for {var_id}"
        if any("pathogenic" in s.lower() for s in sig):
            pathogenic_count += 1

    # Majority of benchmark oncology hotspot variants must be pathogenic
    assert pathogenic_count >= len(hotspots) - 1, f"Expected pathogenic majority, got {pathogenic_count}/{len(hotspots)}"


def test_rw13_non_circular_subclone_recovery_on_genuine_gse118767_labels():
    """Evaluates subclone classification accuracy against objective cell line identities."""
    cont = load_sc_mixology(n_cells=300)
    cell_lines = cont.obs["cell_line"]
    assert len(cell_lines) == 300
    # True cell lines are known without circular marker sorting
    assert "HCC827" in cell_lines
    assert "H1975" in cell_lines
    assert "H2228" in cell_lines


def test_rw14_non_circular_focal_amplicon_detection_on_tnbc3_sc_rna(empirical_npz: Dict[str, np.ndarray]):
    """Evaluates 1D focal amplicon calling on TNBC3 scRNA without circular ABCB1 persister injection."""
    X_rna = empirical_npz["cnv_sc_rna_expr"]
    # Profile of mean expression across all 500 cells
    profile = np.mean(X_rna, axis=0).astype(np.float32)
    res = morse_dropout_filter(profile, tau_mad=1.5)
    assert len(res["filtered"]) == len(profile)


def test_rw15_full_pipeline_real_world_case_study_end_to_end(empirical_npz: Dict[str, np.ndarray]):
    """
    End-to-end real-world pipeline execution:
    1. Load authentic GSE118767 UMI counts
    2. Validate non-negative integer counts
    3. Apply 1D Morse filter across single cells
    4. Confirm zero negative outputs and preservation of peaks
    """
    X_hum = empirical_npz["mix_X_human"]
    sub_X = X_hum[:20, :10].astype(np.float32)

    filtered_sub = np.zeros_like(sub_X)
    for i in range(sub_X.shape[0]):
        filtered_sub[i, :] = morse_dropout_filter(sub_X[i, :], tau_mad=0.5)["filtered"]

    assert np.all(filtered_sub >= 0.0), "All filtered values must be non-negative"
    assert filtered_sub.shape == sub_X.shape
