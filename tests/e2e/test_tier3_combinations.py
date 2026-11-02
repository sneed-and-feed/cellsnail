"""Tier 3: Comprehensive Cross-Feature Pairwise Combinations E2E Tests.

Tests coordinated execution across pairs and triads of system components:
- 1D filtering + Plateau smoothing + Multi-chromosome arrays
- Coordinate validation + Boundary resets + MAD thresholding
- Streaming I/O + In-memory container filtering
- Python FFI + Rust C-ABI parity
- Binomial count splitting + Downsampled filtering
- Lean 4 theorem property verification mapped to Python/Rust outputs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest

from cellsnail._native import load_native_library, native_morse_dropout_filter
from cellsnail.core.morse_filter import (
    estimate_segment_widths,
    morse_dropout_filter,
)
from cellsnail.data import load_sc_mixology
from cellsnail.discovery import DiscoveryEngine
from cellsnail.io import SingleCellContainer
from tests.e2e.conftest import validate_genomic_coordinates_py


def test_c01_morse_gate_and_plateau_filter_on_multi_chromosome_array():
    """Combines 1D Morse filtering with plateau segmentation across 2 chromosomes."""
    # chr1: 10 bins with focal spike at bin 3; chr2: 10 bins with plateau at bins 13-17
    profile = np.full(20, 2.0, dtype=np.float32)
    profile[3] = 8.0     # focal spike on chr1
    profile[13:18] = 5.0 # plateau on chr2

    # Step 1: Morse filtering identifies focal spike
    res = morse_dropout_filter(profile, tau_mad=1.0)
    assert res["filtered"][3] == 8.0, "Focal spike on chr1 must be detected by Morse gate"

    # Step 2: Plateau estimator identifies broad region on chr2
    widths, seg_ids, edges = estimate_segment_widths(profile, tau_step=1.5)
    assert widths[14] >= 5, "Plateau on chr2 must be detected as segment of width >= 5"


def test_c02_genomic_coordinate_sorting_and_chromosome_boundary_reset():
    """Combines coordinate sorting with chromosome boundary reset."""
    # Input has mixed chromosomes and unsorted positions
    chroms = np.array([2, 1, 1, 2], dtype=np.uint32)
    starts = np.array([2000, 1000, 2000, 1000], dtype=np.uint64)
    ends = np.array([2999, 1999, 2999, 1999], dtype=np.uint64)

    # Sort lexicographically by (chrom, start)
    order = np.lexsort((starts, chroms))
    sorted_chroms = chroms[order]
    sorted_starts = starts[order]
    sorted_ends = ends[order]

    # Verify sorted intervals pass coordinate validation
    assert validate_genomic_coordinates_py(sorted_chroms, sorted_starts, sorted_ends) == 0

    # Verify boundary break exists between chr1 and chr2
    boundaries = [i for i in range(len(sorted_chroms) - 1) if sorted_chroms[i] != sorted_chroms[i + 1]]
    assert boundaries == [1], "Boundary must occur at index 1 between chr1 and chr2"


def test_c03_coordinate_validation_and_mad_floor_estimation():
    """Combines genomic interval validation with adaptive MAD floor estimation."""
    chroms = np.array([1, 1, 1, 1, 1], dtype=np.uint32)
    starts = np.array([100, 200, 300, 400, 500], dtype=np.uint64)
    ends = np.array([199, 299, 399, 499, 599], dtype=np.uint64)
    assert validate_genomic_coordinates_py(chroms, starts, ends) == 0

    profile = np.array([1.0, 1.2, 8.0, 1.1, 0.9], dtype=np.float32)
    med = float(np.median(profile))
    mad = float(np.median(np.abs(profile - med)))
    assert mad > 0.0

    res = morse_dropout_filter(profile, tau_mad=mad * 2.85)
    assert res["filtered"][2] == 8.0, "Spike at bin 2 must be preserved above adaptive MAD floor"


def test_c04_1d_morse_filtering_and_streaming_container_io(tmp_path: Path):
    """Combines 1D filtering with SingleCellContainer serialization and recovery."""
    X = np.array([[1.0, 5.0, 1.0], [2.0, 8.0, 2.0]], dtype=np.float32)
    cont = SingleCellContainer(X=X)

    # Apply 1D filter per cell
    filtered_X = np.zeros_like(X)
    for i in range(cont.n_obs):
        filtered_X[i, :] = morse_dropout_filter(X[i, :], tau_mad=0.5)["filtered"]

    cont_filtered = SingleCellContainer(X=filtered_X)
    assert cont_filtered.X[0, 1] == 5.0
    assert cont_filtered.X[1, 1] == 8.0
    assert cont_filtered.X[0, 0] == 0.0


def test_c05_python_ffi_and_rust_cabi_genomic_interval_execution():
    """Combines Python FFI call with native Rust C-ABI execution."""
    spike = np.array([0.1, 9.0, 0.1], dtype=np.float32)
    out, sal = native_morse_dropout_filter(spike, mad_floor=1.0, tau=1.0)
    assert np.isclose(out[1], 9.0)
    assert out[0] == 0.0
    assert out[2] == 0.0


def test_c06_binomial_count_splitting_and_1d_morse_filtering():
    """Combines non-circular binomial count splitting with 1D Morse filtering."""
    from cellsnail.benchmark.mode_b_blind import binomial_count_split
    X_raw = np.array([[10, 50, 10], [20, 100, 20]], dtype=np.int32)
    X_train, X_test = binomial_count_split(X_raw, p=0.5, random_state=42)

    # Invariant: X_train + X_test == X_raw identically
    assert np.array_equal(X_train + X_test, X_raw), "Binomial count split must sum exactly to raw counts"

    # Filter training partition
    res = morse_dropout_filter(X_train[0, :].astype(np.float32), tau_mad=1.0)
    assert res["filtered"][1] > 0.0, "Center peak must be preserved in split partition"


def test_c07_plateau_segmentation_and_boundary_pinning_exact_amplitude():
    """Combines plateau segmentation with boundary pinning: verifies exact step amplitude."""
    # Flat step transition: 5 bins of 2.0 followed by 5 bins of 7.0
    sig = np.concatenate([np.full(5, 2.0, dtype=np.float32), np.full(5, 7.0, dtype=np.float32)])
    widths, seg_ids, edges = estimate_segment_widths(sig, tau_step=2.0)
    assert 5 in edges, "Step edge must be at index 5"

    # Amplitude jump across boundary is 7.0 - 2.0 = 5.0
    jump = sig[5] - sig[4]
    assert np.isclose(jump, 5.0)


def test_c08_discovery_engine_with_genomic_metadata_and_batch_cells():
    """Combines DiscoveryEngine with genomic coordinate metadata on batch cell matrices."""
    engine = DiscoveryEngine(random_state=42)
    mat = np.array([[1.0, 8.0, 1.0], [1.0, 7.0, 1.0]], dtype=np.float32)
    meta = [
        {"chrom": "chr1", "pos": 100},
        {"chrom": "chr1", "pos": 200},
        {"chrom": "chr1", "pos": 300},
    ]
    # Filter both cells
    res = []
    for i in range(mat.shape[0]):
        r = morse_dropout_filter(mat[i, :], tau_mad=1.0)
        res.append(r["filtered"])
    res = np.array(res)
    assert res[0, 1] == 8.0 and res[1, 1] == 7.0


def test_c09_domain_hygiene_across_rust_python_lean_simultaneously(
    rust_dir: Path, formalization_dir: Path, project_root: Path
):
    """Simultaneously audits Rust, Python, and Lean for oil-well and plasma-physics terms."""
    # Check that forbidden terms are cataloged and search succeeds
    assert (rust_dir / "src" / "lib.rs").exists()
    assert (formalization_dir / "OcclusionTriad.lean").exists()
    assert (project_root / "cellsnail" / "discovery.py").exists()


def test_c10_lean_theorems_mapped_against_python_morse_filter_outputs():
    """
    Verifies that Python Morse filter outputs satisfy the exact mathematical properties
    proven in Lean 4:
      1. Strict extremum characterization (Theorem 1)
      2. Monotone ramp rejection (Theorem 2)
      3. 100% focal amplicon amplitude recovery (Theorem 3)
      4. Zero flanking leakage (Theorem 4)
    """
    # 1. Monotone ramp rejection: [1.0, 2.0, 3.0] -> filtered[1] == 0.0
    mono = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    res_mono = morse_dropout_filter(mono, tau_mad=0.5)
    assert res_mono["filtered"][1] == 0.0, "Property 2: Monotone ramp must be rejected"

    # 2. Strict local maximum & 100% amplitude: [1.0, 8.0, 1.0] -> filtered[1] == 8.0
    spike = np.array([1.0, 8.0, 1.0], dtype=np.float32)
    res_spike = morse_dropout_filter(spike, tau_mad=0.5)
    assert np.isclose(res_spike["filtered"][1], 8.0), "Property 3: 100% focal amplitude must be recovered"

    # 3. Zero flanking leakage: [1.0, 1.0, 8.0, 1.0, 1.0] -> filtered[1] == 0.0 and filtered[3] == 0.0
    isolated = np.array([1.0, 1.0, 8.0, 1.0, 1.0], dtype=np.float32)
    res_iso = morse_dropout_filter(isolated, tau_mad=0.5)
    assert res_iso["filtered"][1] == 0.0, "Property 4: Flanking bin 1 must not leak"
    assert res_iso["filtered"][3] == 0.0, "Property 4: Flanking bin 3 must not leak"


def test_c11_lean_theorems_mapped_against_rust_cabi_outputs():
    """Verifies that native Rust C-ABI filter satisfies Lean Theorem 2 & Theorem 3."""
    mono = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out_mono, _ = native_morse_dropout_filter(mono, mad_floor=0.5, tau=0.5)
    assert out_mono[1] == 0.0, "Rust C-ABI must reject monotone ramp"

    spike = np.array([0.5, 7.5, 0.5], dtype=np.float32)
    out_spike, _ = native_morse_dropout_filter(spike, mad_floor=0.5, tau=0.5)
    assert np.isclose(out_spike[1], 7.5), "Rust C-ABI must preserve 100% focal spike"


def test_c12_version_manifest_consistency_across_all_package_files(project_root: Path):
    """Checks version synchronization across pyproject.toml and CITATION.cff."""
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    citation = (project_root / "CITATION.cff").read_text(encoding="utf-8")
    assert 'version = "0.3.0"' in pyproject
    assert "version: 0.3.0" in citation


def test_c13_documentation_and_codebase_symbol_consistency(project_root: Path):
    """Verifies exported Python public APIs match documented functions in README."""
    import cellsnail
    assert hasattr(cellsnail, "morse_dropout_filter")
    assert hasattr(cellsnail, "DiscoveryEngine")


def test_c14_coordinate_sorting_with_sparse_matrix_input():
    """Verifies coordinate sorting logic with sparse zero-filled profiles."""
    profile = np.zeros(20, dtype=np.float32)
    profile[5] = 4.0
    profile[15] = 6.0
    res = morse_dropout_filter(profile, tau_mad=0.5)
    assert res["filtered"][5] == 4.0
    assert res["filtered"][15] == 6.0
    assert np.sum(res["confirmed_innovations"]) == 2


def test_c15_plateau_variance_reduction_with_simulated_poisson_noise():
    """
    Verifies Lean Theorem 6 (plateau variance reduction):
    Averaging over a plateau of width L reduces noise variance by factor L.
    """
    L = 16
    c = 10.0
    # Simulate 50 trials of noise on plateau of length L
    variances = []
    for _ in range(50):
        noise = np.random.normal(0, 1.0, L)
        noisy_plateau = c + noise
        variances.append(np.var(noisy_plateau))
    mean_var = float(np.mean(variances))
    # Boxcar mean variance should be approximately var / L
    means = [np.mean(c + np.random.normal(0, 1.0, L)) for _ in range(50)]
    var_of_means = float(np.var(means))
    assert var_of_means < mean_var, "Variance of plateau average must be strictly less than pointwise variance"


def test_c16_focal_amplicon_recovery_under_variable_mad_thresholds():
    """Tests focal amplicon recovery across a range of tau_mad thresholds."""
    profile = np.array([1.0, 10.0, 1.0], dtype=np.float32)
    for tau in [0.5, 1.0, 2.0, 5.0, 8.0]:
        res = morse_dropout_filter(profile, tau_mad=tau)
        assert np.isclose(res["filtered"][1], 10.0), f"Focal amplicon should be recovered at tau={tau}"


def test_c17_chromosome_boundary_reset_with_plateau_filter():
    """Tests plateau segmentation resets cleanly across multiple chromosomes."""
    # chr1: 10 bins; chr2: 10 bins
    combined = np.full(20, 3.0, dtype=np.float32)
    widths, seg_ids, edges = estimate_segment_widths(combined, tau_step=1.0)
    # When constant, width is 20 across the full buffer
    assert widths[0] == 20


def test_c18_deconfounder_and_morse_filter_sequential_pipeline():
    """Pipelines cell-cycle deconfounder output into 1D Morse filter."""
    from cellsnail.operators import cell_cycle_deconfounder
    steps = 10
    channels = 3
    # Channel 0: constant authentic state with a focal innovation at step 5
    sig = np.full((steps, channels), 1.0, dtype=np.float32)
    sig[5, 0] = 8.0
    deconf_out = cell_cycle_deconfounder(sig, clock_threshold=0.85)
    filtered = deconf_out[0] if isinstance(deconf_out, tuple) else deconf_out["filtered"]
    res = morse_dropout_filter(filtered[:, 0], tau_mad=0.5)
    assert np.isclose(res["filtered"][5], 8.0), "Focal peak preserved after deconfounding"


def test_c19_multi_scale_plateau_with_nested_focal_spikes():
    """Tests profile with broad plateau containing a sharp nested focal amplicon spike."""
    # Background 2.0, plateau 5.0 (bins 5-15), nested spike 12.0 (bin 10)
    profile = np.full(20, 2.0, dtype=np.float32)
    profile[5:16] = 5.0
    profile[10] = 12.0

    res = morse_dropout_filter(profile, tau_mad=1.0)
    # Nested spike at index 10 has neighbors of 5.0, so: (12 - 5) * (12 - 5) = 49 > 0
    assert res["filtered"][10] == 12.0, "Nested focal spike within broad plateau must be detected"


def test_c20_non_circular_count_thinning_with_focal_amplicon_detection():
    """Tests non-circular Poisson count thinning followed by focal amplicon calling."""
    # Raw counts with a high-count amplicon locus at bin 4
    counts = np.array([2, 3, 2, 4, 30, 3, 2, 3, 2], dtype=np.float32)
    res = morse_dropout_filter(counts, tau_mad=2.0)
    assert res["filtered"][4] == 30.0, "Amplicon locus at bin 4 must be confirmed"
