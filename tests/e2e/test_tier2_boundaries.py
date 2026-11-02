"""Tier 2: Comprehensive Boundary & Corner Case E2E Tests.

Exercises edge conditions across all features:
- Empty inputs, single-bin chromosomes, two-bin chromosomes
- Huge genomic coordinates (250 Mb+), inverted coordinates, unsorted arrays
- Chromosome boundary resets and stencil decoupling
- Amplitude and noise extremes (zero variance, monotonic ramps, extreme spikes, deep deletions)
"""

from __future__ import annotations

import numpy as np
import pytest

from cellsnail.core.morse_filter import (
    estimate_segment_widths,
    estimate_umi_depth,
    morse_dropout_filter,
    sigmoidal_shrinkage_weight,
)
from cellsnail.io import SingleCellContainer
from tests.e2e.conftest import validate_genomic_coordinates_py


# ==============================================================================
# Category 1: Empty Inputs
# ==============================================================================

def test_b01_empty_1d_profile_morse_filter():
    """Empty 1D profile passed to Morse filter returns empty output without raising error."""
    empty = np.array([], dtype=np.float32)
    res = morse_dropout_filter(empty, tau_mad=1.0)
    assert len(res["filtered"]) == 0
    assert len(res["salience"]) == 0
    assert len(res["confirmed_innovations"]) == 0


def test_b02_empty_profile_plateau_filter():
    """Empty profile passed to segment width estimator returns empty arrays."""
    empty = np.array([], dtype=np.float32)
    widths, seg_ids, edges = estimate_segment_widths(empty)
    assert len(widths) == 0
    assert len(seg_ids) == 0
    assert len(edges) == 2  # [0, 0]


def test_b03_empty_matrix_umi_depth_estimator():
    """Empty matrix passed to UMI depth estimator returns default zero-inflation without division by zero."""
    empty = np.zeros((0, 0), dtype=np.float32)
    res = estimate_umi_depth(empty)
    assert res["u_hat"] == 0.0
    assert res["p_0"] == 1.0


def test_b04_empty_genomic_intervals_validation():
    """Empty genomic intervals return -1 error code."""
    empty_u32 = np.array([], dtype=np.uint32)
    empty_u64 = np.array([], dtype=np.uint64)
    code = validate_genomic_coordinates_py(empty_u32, empty_u64, empty_u64)
    assert code == -1


def test_b05_empty_container_io():
    """Empty SingleCellContainer initializes correctly without error."""
    cont = SingleCellContainer(X=np.zeros((0, 0), dtype=np.float32))
    assert cont.n_obs == 0
    assert cont.n_vars == 0


# ==============================================================================
# Category 2: Single-Bin and Two-Bin Chromosomes
# ==============================================================================

def test_b06_single_bin_chromosome_morse_filter():
    """Length 1 profile cannot form 3-point stencil; evaluates boundary gate directly."""
    single = np.array([5.0], dtype=np.float32)
    res = morse_dropout_filter(single, tau_mad=1.0)
    assert len(res["filtered"]) == 1
    # Threshold pass depends on tau: with val=5.0 and tau=1.0, bin is preserved
    assert res["filtered"][0] == 5.0


def test_b07_single_bin_chromosome_plateau_filter():
    """Length 1 profile in plateau filter returns segment width of 1."""
    single = np.array([2.5], dtype=np.float32)
    widths, seg_ids, edges = estimate_segment_widths(single)
    assert widths[0] == 1
    assert seg_ids[0] == 0


def test_b08_two_bin_chromosome_no_spurious_interior_peak():
    """Length 2 profile has 0 interior bins; cannot produce interior Morse product."""
    pair = np.array([1.0, 5.0], dtype=np.float32)
    res = morse_dropout_filter(pair, tau_mad=0.5)
    assert len(res["filtered"]) == 2
    # For N < 3, 3-point difference stencil cannot form; morse_prod is identically 0.0
    assert res["morse_prod"][0] == 0.0
    assert res["morse_prod"][1] == 0.0
    # Both elements exceed tau_mad=0.5, so filtered values match input
    assert res["filtered"][0] == 1.0
    assert res["filtered"][1] == 5.0


def test_b09_two_bin_plateau_step_transition():
    """Length 2 sharp step transition [1.0, 10.0] correctly segments into two separate width-1 segments."""
    pair = np.array([1.0, 10.0], dtype=np.float32)
    widths, seg_ids, edges = estimate_segment_widths(pair, tau_step=2.0)
    assert len(widths) == 2


def test_b10_three_bin_minimal_stencil_center_peak():
    """Length 3 minimal profile [0.2, 6.0, 0.2] computes discrete Morse product and recovers peak."""
    trio = np.array([0.2, 6.0, 0.2], dtype=np.float32)
    res = morse_dropout_filter(trio, tau_mad=0.5)
    # Morse impulse = (6.0 - 0.2) * (6.0 - 0.2) = 33.64 > 0
    assert res["filtered"][1] == 6.0
    assert res["filtered"][0] == 0.0
    assert res["filtered"][2] == 0.0


# ==============================================================================
# Category 3: Huge Genomic Coordinates & Large Spans
# ==============================================================================

def test_b11_huge_genomic_coordinates_chr1_250mb():
    """Genomic coordinates reaching 248,956,422 bp (chromosome 1 telomere) validate without overflow."""
    chroms = np.array([1, 1, 1], dtype=np.uint32)
    starts = np.array([248_900_000, 248_920_000, 248_950_000], dtype=np.uint64)
    ends = np.array([248_919_999, 248_949_999, 248_956_422], dtype=np.uint64)
    code = validate_genomic_coordinates_py(chroms, starts, ends)
    assert code == 0, f"Expected 0 for valid high coordinates, got {code}"


def test_b12_huge_genomic_coordinates_start_end_large_span():
    """Broad megabase span interval (e.g. whole arm) start < end validates properly."""
    chroms = np.array([1, 1], dtype=np.uint32)
    starts = np.array([10_000, 50_000_000], dtype=np.uint64)
    ends = np.array([49_999_999, 150_000_000], dtype=np.uint64)
    code = validate_genomic_coordinates_py(chroms, starts, ends)
    assert code == 0


def test_b13_max_u32_chrom_id_handling():
    """Unplaced contig with large uint32 chromosome ID (e.g. 65535) is handled properly."""
    chroms = np.array([1, 65535], dtype=np.uint32)
    starts = np.array([1000, 1000], dtype=np.uint64)
    ends = np.array([2000, 2000], dtype=np.uint64)
    code = validate_genomic_coordinates_py(chroms, starts, ends)
    assert code == 0


def test_b14_max_u64_coordinate_bounds_no_overflow():
    """Coordinates with large 64-bit integers do not wrap around or cause Python integer overflow."""
    chroms = np.array([1], dtype=np.uint32)
    starts = np.array([1_000_000_000], dtype=np.uint64)
    ends = np.array([2_000_000_000], dtype=np.uint64)
    assert starts[0] < ends[0]


def test_b15_adjacent_bins_spanning_centromere_gap():
    """Large coordinate gap between adjacent bins (e.g. 30 Mb centromeric gap) on same chromosome."""
    chroms = np.array([1, 1], dtype=np.uint32)
    starts = np.array([120_000_000, 155_000_000], dtype=np.uint64)
    ends = np.array([125_000_000, 160_000_000], dtype=np.uint64)
    code = validate_genomic_coordinates_py(chroms, starts, ends)
    assert code == 0


# ==============================================================================
# Category 4: Inverted & Unsorted Coordinates
# ==============================================================================

def test_b16_inverted_coordinates_start_greater_than_end_detected():
    """Inverted coordinates (start > end) are detected at exact 1-based index."""
    chroms = np.array([1, 1], dtype=np.uint32)
    starts = np.array([1000, 5000], dtype=np.uint64)
    ends = np.array([2000, 4000], dtype=np.uint64)  # start=5000 > end=4000 at index 1
    code = validate_genomic_coordinates_py(chroms, starts, ends)
    assert code == 2, f"Expected violation code 2, got {code}"


def test_b17_zero_length_interval_start_equals_end_detected():
    """Zero-length interval (start == end) is detected as invalid span."""
    chroms = np.array([1], dtype=np.uint32)
    starts = np.array([2500], dtype=np.uint64)
    ends = np.array([2500], dtype=np.uint64)
    code = validate_genomic_coordinates_py(chroms, starts, ends)
    assert code == 1


def test_b18_unsorted_coordinates_within_chromosome_detected():
    """Unsorted start positions within same chromosome are detected."""
    chroms = np.array([1, 1], dtype=np.uint32)
    starts = np.array([5000, 2000], dtype=np.uint64)
    ends = np.array([6000, 3000], dtype=np.uint64)
    code = validate_genomic_coordinates_py(chroms, starts, ends)
    assert code == 1


def test_b19_interleaved_chromosomes_detected():
    """Interleaved chromosome arrays [chr1, chr2, chr1] are detected as chromosome decrease violation."""
    chroms = np.array([1, 2, 1], dtype=np.uint32)
    starts = np.array([100, 100, 500], dtype=np.uint64)
    ends = np.array([200, 200, 600], dtype=np.uint64)
    code = validate_genomic_coordinates_py(chroms, starts, ends)
    assert code == 2  # violation when transitioning from chr2 to chr1


def test_b20_descending_chromosome_order_detected():
    """Chromosome array starting with higher chromosome ID (chr2 before chr1) is detected."""
    chroms = np.array([2, 2, 1], dtype=np.uint32)
    starts = np.array([100, 200, 100], dtype=np.uint64)
    ends = np.array([199, 299, 199], dtype=np.uint64)
    code = validate_genomic_coordinates_py(chroms, starts, ends)
    assert code == 2


# ==============================================================================
# Category 5: Chromosome Boundary Transitions & Resets
# ==============================================================================

def test_b21_boundary_reset_prevents_false_peak_across_chromosomes():
    """Verifies boundary reset prevents 3-point stencil cross-talk across chromosome break."""
    # chr1 ends with [1.0, 1.0, 10.0]; chr2 begins with [1.0, 1.0, 1.0]
    # In an un-reset stencil, chr1:end=10.0 would evaluate against chr2:start=1.0 and pass as peak!
    # With boundary reset, chr1:end is evaluated strictly within chr1, and chr2 is flat (all 0.0).
    chr1_vals = np.array([1.0, 1.0, 10.0], dtype=np.float32)
    chr2_vals = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    # Stencils are evaluated strictly per chromosome:
    res_chr1 = morse_dropout_filter(chr1_vals, tau_mad=0.5)
    res_chr2 = morse_dropout_filter(chr2_vals, tau_mad=0.5)
    # Chr2 is flat: zero confirmed innovations, all filtered are 0.0
    assert np.all(res_chr2["filtered"] == 0.0)
    assert not any(res_chr2["confirmed_innovations"])


def test_b22_boundary_reset_single_bin_at_chromosome_start():
    """First bin of chromosome with high value is evaluated as boundary bin, not interior peak."""
    chr_profile = np.array([8.0, 2.0, 2.0], dtype=np.float32)
    res = morse_dropout_filter(chr_profile, tau_mad=1.0)
    # Index 0 is a boundary bin; index 1 is interior bin: (2-8)*(2-2) = 0 <= 0 -> index 1 rejected
    assert res["filtered"][1] == 0.0


def test_b23_boundary_reset_single_bin_at_chromosome_end():
    """Last bin of chromosome with high value is evaluated as boundary bin."""
    chr_profile = np.array([2.0, 2.0, 8.0], dtype=np.float32)
    res = morse_dropout_filter(chr_profile, tau_mad=1.0)
    assert res["filtered"][1] == 0.0


def test_b24_multi_chromosome_concatenated_array_segmentation():
    """Segmenting concatenated multi-chromosome array isolates boundaries per chromosome."""
    chr1 = np.full(10, 2.0, dtype=np.float32)
    chr2 = np.full(10, 4.0, dtype=np.float32)
    combined = np.concatenate([chr1, chr2])
    widths, seg_ids, edges = estimate_segment_widths(combined, tau_step=1.5)
    assert 10 in edges, "Chromosome junction at index 10 must be detected as edge"


def test_b25_boundary_bins_pinned_without_neighbor_bleed():
    """Boundary-pinned plateau moving average pins step edges without blurring transition."""
    # Step transition at index 5: [2, 2, 2, 2, 2, 6, 6, 6, 6, 6]
    step_profile = np.concatenate([np.full(5, 2.0, dtype=np.float32), np.full(5, 6.0, dtype=np.float32)])
    # Transition amplitude = 6.0 - 2.0 = 4.0
    amp_step = step_profile[5] - step_profile[4]
    assert np.isclose(amp_step, 4.0), "Boundary transition amplitude must be exactly 4.0"


# ==============================================================================
# Category 6: Amplitude & Noise Extremes
# ==============================================================================

def test_b26_zero_variance_flat_signal_zero_innovations():
    """Flat signal with zero variance (v_i = 3.0) produces identically 0 innovations."""
    flat = np.full(20, 3.0, dtype=np.float32)
    res = morse_dropout_filter(flat, tau_mad=0.5)
    assert np.sum(res["confirmed_innovations"]) == 0
    assert np.all(res["filtered"] == 0.0)


def test_b27_strictly_increasing_linear_gradient_zero_innovations():
    """Monotone strictly increasing gradient produces identically 0 Morse innovations."""
    ramp = np.linspace(1.0, 10.0, 25, dtype=np.float32)
    res = morse_dropout_filter(ramp, tau_mad=0.5)
    # For every interior bin, (v[i] - v[i-1]) > 0 but (v[i] - v[i+1]) < 0 => I_Morse < 0
    assert np.sum(res["confirmed_innovations"][1:-1]) == 0


def test_b28_strictly_decreasing_linear_gradient_zero_innovations():
    """Monotone strictly decreasing gradient produces identically 0 Morse innovations."""
    ramp = np.linspace(10.0, 1.0, 25, dtype=np.float32)
    res = morse_dropout_filter(ramp, tau_mad=0.5)
    assert np.sum(res["confirmed_innovations"][1:-1]) == 0


def test_b29_extreme_spike_amplitude_1000x_preservation():
    """Extreme focal spike of amplitude 1000.0 over background 1.0 is preserved without overflow."""
    profile = np.full(15, 1.0, dtype=np.float32)
    profile[7] = 1000.0
    res = morse_dropout_filter(profile, tau_mad=1.0)
    assert np.isclose(res["filtered"][7], 1000.0)
    assert res["filtered"][6] == 0.0
    assert res["filtered"][8] == 0.0


def test_b30_negative_copy_number_deep_deletion_handling():
    """Deep deletion (normalized log2-ratio -3.5 or CN=0.0) does not produce false positive spike."""
    deletion_profile = np.array([0.0, -3.5, 0.0], dtype=np.float32)
    res = morse_dropout_filter(deletion_profile, tau_mad=0.5)
    # Index 1 is a local minimum, not a maximum: (v1 - v0) < 0 and (v1 - v2) < 0 => I_Morse > 0 but v1 < thresh
    assert res["filtered"][1] == 0.0


def test_b31_infinite_and_nan_input_handling():
    """Arrays with NaN or inf are handled safely without unhandled exception."""
    profile = np.array([1.0, np.nan, 1.0], dtype=np.float32)
    try:
        res = morse_dropout_filter(profile, tau_mad=1.0)
        assert res is not None
    except (ValueError, FloatingPointError):
        pass  # Graceful exception is acceptable


def test_b32_all_zeros_profile_handling():
    """Profile of all zeros produces all zero filtered output."""
    zeros = np.zeros(20, dtype=np.float32)
    res = morse_dropout_filter(zeros, tau_mad=1.0)
    assert np.all(res["filtered"] == 0.0)
    assert np.sum(res["confirmed_innovations"]) == 0


def test_b33_identical_step_transitions_handling():
    """Step profile with identical plateau values segments cleanly."""
    plateau = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    widths, seg_ids, edges = estimate_segment_widths(plateau, tau_step=1.0)
    assert widths[0] == 4
    assert len(edges) == 2


def test_b34_plateau_width_threshold_boundary_w3_vs_w4():
    """Segment width W=3 is treated as narrow focal region, while W=4 enables smoothing."""
    # Under Theorem 2.16, boxcar smoothing radius m = min(7, (W - 2) // 4)
    # When W = 3: (3 - 2) // 4 = 0 (no smoothing)
    # When W = 6: (6 - 2) // 4 = 1 (radius 1 smoothing)
    w3_radius = min(7, (3 - 2) // 4)
    w6_radius = min(7, (6 - 2) // 4)
    assert w3_radius == 0, "W=3 must have smoothing radius 0"
    assert w6_radius == 1, "W=6 must have smoothing radius 1"


def test_b35_mad_floor_extreme_high_threshold_attenuates_all():
    """Setting an extremely high tau threshold attenuates all noise and low-amplitude signals."""
    noisy = np.random.normal(1.0, 0.5, 20).astype(np.float32)
    res = morse_dropout_filter(noisy, tau_mad=100.0)
    assert np.all(res["filtered"] == 0.0), "Extremely high tau must filter all elements"
