"""Shared fixtures and testing utilities for cellsnail E2E test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUST_DIR = PROJECT_ROOT / "rust" / "cellsnail_rs"
FORMALIZATION_DIR = PROJECT_ROOT / "formalization"
DOCS_DIR = PROJECT_ROOT / "docs"
SRC_DIR = PROJECT_ROOT / "cellsnail"
TESTS_DIR = PROJECT_ROOT / "tests"

# Forbidden domain-mismatched terms per ORIGINAL_REQUEST.md and PROJECT.md
OIL_WELL_TERMS = [
    "caliper",
    "calipers",
    "washout",
    "washouts",
    "stringer",
    "stringers",
    "thin stringer",
    "thin stringers",
    "thin-bed",
    "thin-beds",
    "thin_bed",
    "thin_beds",
    "Stringer Hunt",
    "Pan-Autosome Stringer Atlas",
    "Stealth domain-native alias",
    "Stealth Single-Cell Domain Alias",
]

PLASMA_PHYSICS_TERMS = [
    "stellarator",
    "stellarators",
    "magnetohydrodynamic",
    "magnetohydrodynamics",
    "mhd",
    "mhds",
    "beltrami",
    "beltramis",
    "mrxmhd",
    "mrxmhds",
    "cross-helicity",
    "alfven",
    "Alfvén",
]

RECYCLED_ECOSYSTEM_TERMS = [
    "darksnail",
    "darksnails",
    "triad-core",
    "triad-cores",
    "triad_core",
    "triad_cores",
    "TriadCore",
]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def rust_dir() -> Path:
    return RUST_DIR


@pytest.fixture(scope="session")
def formalization_dir() -> Path:
    return FORMALIZATION_DIR


@pytest.fixture(scope="session")
def docs_dir() -> Path:
    return DOCS_DIR


@pytest.fixture(scope="session")
def forbidden_oil_terms() -> List[str]:
    return OIL_WELL_TERMS


@pytest.fixture(scope="session")
def forbidden_plasma_terms() -> List[str]:
    return PLASMA_PHYSICS_TERMS


@pytest.fixture(scope="session")
def empirical_npz() -> Dict[str, np.ndarray]:
    npz_path = SRC_DIR / "data" / "empirical" / "wetlab_cohorts.npz"
    assert npz_path.exists(), f"Missing wetlab cohorts archive: {npz_path}"
    with np.load(npz_path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def clinvar_panel() -> Dict[str, Any]:
    json_path = SRC_DIR / "data" / "empirical" / "ensembl_clinvar_drivers.json"
    assert json_path.exists(), f"Missing ClinVar drivers panel: {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def sample_sorted_genomic_intervals() -> Dict[str, np.ndarray]:
    """Generates standard, strictly sorted 1D genomic intervals across chr1 and chr2."""
    chrom_ids = np.array([1, 1, 1, 1, 2, 2, 2], dtype=np.uint32)
    starts = np.array([1000, 2000, 3000, 4000, 1000, 2500, 5000], dtype=np.uint64)
    ends = np.array([1999, 2999, 3999, 4999, 2499, 4999, 7499], dtype=np.uint64)
    return {"chrom_ids": chrom_ids, "starts": starts, "ends": ends}


@pytest.fixture
def sample_signals() -> Dict[str, np.ndarray]:
    """Synthesizes canonical test signals with known mathematical properties."""
    # 1. Isolated focal amplicon peak: background 1.0, single-bin spike 8.0 at index 5
    focal_peak = np.full(11, 1.0, dtype=np.float32)
    focal_peak[5] = 8.0

    # 2. Monotone increasing ramp
    monotone_ramp = np.linspace(1.0, 5.0, 11, dtype=np.float32)

    # 3. Monotone flat step transition
    step_transition = np.concatenate([np.full(6, 1.0, dtype=np.float32), np.full(5, 4.0, dtype=np.float32)])

    # 4. Broad plateau with boundary pinning: length 20, center plateau of width 8
    broad_plateau = np.full(20, 2.0, dtype=np.float32)
    broad_plateau[6:14] = 6.0

    # 5. Flat profile with zero variance
    flat_zero_var = np.full(15, 2.0, dtype=np.float32)

    return {
        "focal_peak": focal_peak,
        "monotone_ramp": monotone_ramp,
        "step_transition": step_transition,
        "broad_plateau": broad_plateau,
        "flat_zero_var": flat_zero_var,
    }


def validate_genomic_coordinates_py(
    chrom_ids: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> int:
    """
    Reference specification oracle for 1D genomic coordinate validation (PROJECT.md contract).
    Returns 0 if strictly sorted by chrom_id, non-overlapping, and start < end.
    Returns negative or 1-based index on violation.
    """
    n = len(chrom_ids)
    if n == 0 or len(starts) != n or len(ends) != n:
        return -1

    for i in range(n):
        if starts[i] >= ends[i]:
            return i + 1  # invalid interval span

        if i + 1 < n:
            if chrom_ids[i + 1] < chrom_ids[i]:
                return i + 1  # chromosome order decreasing
            if chrom_ids[i + 1] == chrom_ids[i] and starts[i + 1] < ends[i]:
                return i + 1  # overlapping or unsorted start within chromosome

    return 0
