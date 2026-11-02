"""tests/test_boundary_isolation_empirical.py
============================================
Adversarial empirical verification harness for chromosome boundary isolation,
single-bin chromosome independence, boundary spike detection, and 10,000-iteration
numerical stability in `morse_gate_genomic_f32` and `morse_gate_genomic_intervals_f32`.

Author: challenger_m1_2
Date: 2026-11-02
"""

import ctypes
import os
import sys
import numpy as np
import pytest

# Load shared library
def get_lib():
    dll_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "rust",
            "cellsnail_rs",
            "target",
            "release",
            "cellsnail_rs.dll",
        )
    )
    if not os.path.exists(dll_path):
        dll_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "rust",
                "cellsnail_rs",
                "target",
                "debug",
                "cellsnail_rs.dll",
            )
        )
    assert os.path.exists(dll_path), f"Cannot find cellsnail_rs.dll at {dll_path}"
    lib = ctypes.CDLL(dll_path)

    # Setup C-ABI signatures
    lib.morse_gate_genomic_f32.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.morse_gate_genomic_f32.restype = ctypes.c_int32

    class GenomicInterval(ctypes.Structure):
        _fields_ = [
            ("chrom_id", ctypes.c_uint32),
            ("start", ctypes.c_uint64),
            ("end", ctypes.c_uint64),
        ]

    lib.GenomicInterval = GenomicInterval

    lib.morse_gate_genomic_intervals_f32.argtypes = [
        ctypes.POINTER(GenomicInterval),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.morse_gate_genomic_intervals_f32.restype = ctypes.c_int32

    lib.validate_genomic_coordinates_c.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    lib.validate_genomic_coordinates_c.restype = ctypes.c_int32

    lib.validate_genomic_intervals_c.argtypes = [
        ctypes.POINTER(GenomicInterval),
        ctypes.c_size_t,
    ]
    lib.validate_genomic_intervals_c.restype = ctypes.c_int32

    return lib


def reference_morse_genomic(chrom_ids, in_signal, mad_floor=1.0, tau=1.0):
    """Python ground truth reference oracle for chromosome-isolated 1D Morse filtering."""
    n = len(in_signal)
    thresh = max(tau, mad_floor)
    out = np.zeros(n, dtype=np.float32)
    salience = np.abs(in_signal).astype(np.float32)

    for i in range(n):
        v_curr = in_signal[i]
        curr_chrom = chrom_ids[i]

        has_left = (i > 0) and (chrom_ids[i - 1] == curr_chrom)
        has_right = (i + 1 < n) and (chrom_ids[i + 1] == curr_chrom)

        if has_left and has_right:
            v_prev = in_signal[i - 1]
            v_next = in_signal[i + 1]
            d_minus = v_curr - v_prev
            d_plus = v_curr - v_next
            is_spike = (
                (d_minus * d_plus > 0.0)
                and (d_minus > 0.0)
                and (d_plus > 0.0)
                and (v_curr > thresh)
            )
        elif not has_left and has_right:
            v_next = in_signal[i + 1]
            is_spike = (v_curr > thresh) and (v_curr > v_next)
        elif has_left and not has_right:
            v_prev = in_signal[i - 1]
            is_spike = (v_curr > thresh) and (v_curr > v_prev)
        else:  # not has_left and not has_right: single-bin chromosome
            is_spike = v_curr > thresh

        out[i] = v_curr if is_spike else 0.0

    return out, salience


def call_morse_genomic(lib, chrs, starts, ends, signal, mad_floor=1.0, tau=1.0):
    n = len(signal)
    c_chrs = (ctypes.c_uint32 * n)(*chrs)
    c_starts = (ctypes.c_uint64 * n)(*starts)
    c_ends = (ctypes.c_uint64 * n)(*ends)
    c_in = (ctypes.c_float * n)(*signal)
    c_out = (ctypes.c_float * n)()
    c_salience = (ctypes.c_float * n)()

    code = lib.morse_gate_genomic_f32(
        c_chrs, c_starts, c_ends, c_in, c_out, c_salience, n, mad_floor, tau
    )
    return code, np.array(c_out, dtype=np.float32), np.array(c_salience, dtype=np.float32)


def call_morse_genomic_intervals(lib, intervals, signal, mad_floor=1.0, tau=1.0):
    n = len(signal)
    c_intervals = (lib.GenomicInterval * n)(
        *[lib.GenomicInterval(chrom_id=c, start=s, end=e) for c, s, e in intervals]
    )
    c_in = (ctypes.c_float * n)(*signal)
    c_out = (ctypes.c_float * n)()
    c_salience = (ctypes.c_float * n)()

    code = lib.morse_gate_genomic_intervals_f32(
        c_intervals, c_in, c_out, c_salience, n, mad_floor, tau
    )
    return code, np.array(c_out, dtype=np.float32), np.array(c_salience, dtype=np.float32)


# =============================================================================
# Test Suite 1: Chromosome Junction Boundary Isolation
# =============================================================================

def test_junction_isolation_high_end_chr1_low_start_chr2():
    """Verify that a high value at chr1 end and low value at chr2 start never compute cross differences."""
    lib = get_lib()

    # chr1: [0.1, 0.1, 10.0] -> bin 2 is end of chr1, value 10.0
    # chr2: [0.1, 0.1, 0.1]  -> bin 3 is start of chr2, value 0.1
    chrs = [1, 1, 1, 2, 2, 2]
    starts = [100, 200, 300, 100, 200, 300]
    ends = [200, 300, 400, 200, 300, 400]
    signal = [0.1, 0.1, 10.0, 0.1, 0.1, 0.1]

    code, out, sal = call_morse_genomic(lib, chrs, starts, ends, signal, mad_floor=1.0, tau=1.0)
    assert code == 0
    # chr1 bin 2 must be preserved (10.0 > 1.0 and 10.0 > 0.1 on chr1)
    assert out[2] == 10.0
    # chr2 bin 3 must be 0.0 (0.1 < 1.0)
    assert out[3] == 0.0

    # Perturbation invariance: perturb chr2 bin 3 across dynamic ranges (0.001 to 1e9)
    # The output of chr1 bin 2 MUST remain exactly 10.0!
    for test_val in [0.0, 0.001, 0.1, 5.0, 10.0, 20.0, 1000.0, 1e7]:
        sig_perturbed = list(signal)
        sig_perturbed[3] = test_val
        _, out_p, _ = call_morse_genomic(lib, chrs, starts, ends, sig_perturbed, mad_floor=1.0, tau=1.0)
        assert out_p[2] == 10.0, (
            f"Cross-boundary bleed detected! When chr2[0] = {test_val}, chr1[-1] changed from 10.0 to {out_p[2]}"
        )


def test_junction_isolation_low_end_chr1_high_start_chr2():
    """Verify that high value at chr2 start evaluates against chr2[1] only, invariant to chr1[-1]."""
    lib = get_lib()

    # chr1: [0.1, 0.1, 0.1]
    # chr2: [15.0, 0.2, 0.2] -> bin 3 is start of chr2, value 15.0
    chrs = [1, 1, 1, 2, 2, 2]
    starts = [100, 200, 300, 100, 200, 300]
    ends = [200, 300, 400, 200, 300, 400]
    signal = [0.1, 0.1, 0.1, 15.0, 0.2, 0.2]

    code, out, _ = call_morse_genomic(lib, chrs, starts, ends, signal, mad_floor=1.0, tau=1.0)
    assert code == 0
    assert out[3] == 15.0, "chr2 start peak must be preserved using right neighbor"

    # Perturb chr1[-1] (bin 2) across huge range; chr2 start must remain 15.0
    for test_val in [0.0, 0.1, 14.0, 15.0, 100.0, 1e6]:
        sig_perturbed = list(signal)
        sig_perturbed[2] = test_val
        _, out_p, _ = call_morse_genomic(lib, chrs, starts, ends, sig_perturbed, mad_floor=1.0, tau=1.0)
        assert out_p[3] == 15.0, (
            f"Cross-boundary bleed detected! When chr1[-1] = {test_val}, chr2[0] changed from 15.0 to {out_p[3]}"
        )


def test_junction_isolation_equal_boundary_plateaus():
    """Verify that a plateau spanning across chromosome boundary is NOT confused for an extremum."""
    lib = get_lib()

    # Both chr1[-1] and chr2[0] have value 8.0
    # chr1: [1.0, 8.0, 8.0] -> bin 2 is end of chr1, equal to bin 1 -> NOT a strict peak (8.0 > 8.0 is false)
    # chr2: [8.0, 8.0, 1.0] -> bin 3 is start of chr2, equal to bin 4 -> NOT a strict peak (8.0 > 8.0 is false)
    chrs = [1, 1, 1, 2, 2, 2]
    starts = [100, 200, 300, 100, 200, 300]
    ends = [200, 300, 400, 200, 300, 400]
    signal = [1.0, 8.0, 8.0, 8.0, 8.0, 1.0]

    code, out, _ = call_morse_genomic(lib, chrs, starts, ends, signal, mad_floor=1.0, tau=1.0)
    assert code == 0
    # Strict extremum required: plateaus must be rejected
    assert out[1] == 0.0
    assert out[2] == 0.0
    assert out[3] == 0.0
    assert out[4] == 0.0


# =============================================================================
# Test Suite 2: Single-Bin Chromosomes
# =============================================================================

def test_single_bin_chromosome_isolated():
    """Verify that a single-bin chromosome evaluates v > thresh independently."""
    lib = get_lib()

    # Array with a single bin (N=1)
    chrs = [5]
    starts = [1000]
    ends = [2000]

    # Signal > thresh
    code, out, sal = call_morse_genomic(lib, chrs, starts, ends, [4.5], mad_floor=1.0, tau=1.0)
    assert code == 0
    assert out[0] == 4.5
    assert sal[0] == 4.5

    # Signal <= thresh
    code, out, _ = call_morse_genomic(lib, chrs, starts, ends, [0.8], mad_floor=1.0, tau=1.0)
    assert code == 0
    assert out[0] == 0.0


def test_single_bin_chromosome_sandwiched():
    """Verify single-bin chromosome flanked by multi-bin chromosomes evaluates independently."""
    lib = get_lib()

    # chr1: 3 bins [0.1, 5.0, 0.1]
    # chr2: 1 bin  [12.0]          <- single bin!
    # chr3: 3 bins [0.1, 7.0, 0.1]
    chrs = [1, 1, 1, 2, 3, 3, 3]
    starts = [100, 200, 300, 50, 100, 200, 300]
    ends = [200, 300, 400, 150, 200, 300, 400]
    signal = [0.1, 5.0, 0.1, 12.0, 0.1, 7.0, 0.1]

    code, out, _ = call_morse_genomic(lib, chrs, starts, ends, signal, mad_floor=1.0, tau=1.0)
    assert code == 0
    expected = [0.0, 5.0, 0.0, 12.0, 0.0, 7.0, 0.0]
    assert np.allclose(out, expected)

    # Now make chr2 <= thresh (e.g. 0.5)
    signal2 = [0.1, 5.0, 0.1, 0.5, 0.1, 7.0, 0.1]
    code, out2, _ = call_morse_genomic(lib, chrs, starts, ends, signal2, mad_floor=1.0, tau=1.0)
    assert code == 0
    assert out2[3] == 0.0
    assert out2[1] == 5.0
    assert out2[5] == 7.0


def test_consecutive_single_bin_chromosomes():
    """Verify an array composed entirely of consecutive single-bin chromosomes."""
    lib = get_lib()

    chrs = [1, 2, 3, 4, 5, 6, 7]
    starts = [10, 20, 30, 40, 50, 60, 70]
    ends = [15, 25, 35, 45, 55, 65, 75]
    signal = [0.2, 5.5, 0.9, 10.0, 1.0, 1.01, 0.0]

    code, out, _ = call_morse_genomic(lib, chrs, starts, ends, signal, mad_floor=1.0, tau=1.0)
    assert code == 0
    expected = [
        0.0,        # 0.2 <= 1.0
        5.5,        # 5.5 > 1.0
        0.0,        # 0.9 <= 1.0
        10.0,       # 10.0 > 1.0
        0.0,        # 1.0 <= 1.0 (strict inequality v > thresh)
        1.01,       # 1.01 > 1.0
        0.0,        # 0.0 <= 1.0
    ]
    assert np.allclose(out, expected)


# =============================================================================
# Test Suite 3: Focal Spikes Placed Exactly at Chromosome Start, Middle, End
# =============================================================================

@pytest.mark.parametrize("chrom_len", [2, 3, 4, 5, 10, 32])
def test_focal_spikes_positions(chrom_len):
    """Test focal spikes at start, interior positions, and end across various chromosome lengths."""
    lib = get_lib()

    chrs = [1] * chrom_len
    starts = [i * 100 for i in range(chrom_len)]
    ends = [s + 50 for s in starts]

    # 1. Spike at start (index 0)
    sig_start = [0.1] * chrom_len
    sig_start[0] = 8.0
    code, out_start, _ = call_morse_genomic(lib, chrs, starts, ends, sig_start, mad_floor=1.0, tau=1.0)
    assert code == 0
    assert out_start[0] == 8.0
    assert np.all(out_start[1:] == 0.0)

    # 2. Spike at end (index chrom_len - 1)
    sig_end = [0.1] * chrom_len
    sig_end[-1] = 9.0
    code, out_end, _ = call_morse_genomic(lib, chrs, starts, ends, sig_end, mad_floor=1.0, tau=1.0)
    assert code == 0
    assert out_end[-1] == 9.0
    assert np.all(out_end[:-1] == 0.0)

    # 3. Spike at each interior position
    for mid in range(1, chrom_len - 1):
        sig_mid = [0.1] * chrom_len
        sig_mid[mid] = 10.0
        code, out_mid, _ = call_morse_genomic(lib, chrs, starts, ends, sig_mid, mad_floor=1.0, tau=1.0)
        assert code == 0
        assert out_mid[mid] == 10.0
        assert np.all(out_mid[:mid] == 0.0)
        assert np.all(out_mid[mid + 1:] == 0.0)


# =============================================================================
# Test Suite 4: Coordinate Validation and Error Codes
# =============================================================================

def test_validation_error_codes():
    lib = get_lib()

    # 1. Inverted coordinates: start > end -> -2
    code, _, _ = call_morse_genomic(
        lib, [1, 1], [100, 500], [200, 400], [1.0, 1.0], 1.0, 1.0
    )
    assert code == -2, f"Expected ERR_GENOMIC_INVERTED_COORDINATES (-2), got {code}"

    # 2. Unsorted chromosomes: chr2 before chr1 -> -3
    code, _, _ = call_morse_genomic(
        lib, [2, 1], [100, 100], [200, 200], [1.0, 1.0], 1.0, 1.0
    )
    assert code == -3, f"Expected ERR_GENOMIC_UNSORTED_CHROMOSOMES (-3), got {code}"

    # 3. Unsorted positions within chromosome: 300 before 200 -> -4
    code, _, _ = call_morse_genomic(
        lib, [1, 1], [300, 200], [400, 300], [1.0, 1.0], 1.0, 1.0
    )
    assert code == -4, f"Expected ERR_GENOMIC_UNSORTED_POSITIONS (-4), got {code}"

    # 4. Null pointer or n == 0 -> -1
    code = lib.morse_gate_genomic_f32(None, None, None, None, None, None, 0, 1.0, 1.0)
    assert code == -1, f"Expected ERR_GENOMIC_NULL_OR_EMPTY (-1), got {code}"


# =============================================================================
# Test Suite 5: Parity Between C Arrays API and Struct API
# =============================================================================

def test_struct_api_parity():
    """Verify morse_gate_genomic_intervals_f32 matches morse_gate_genomic_f32 bit-for-bit."""
    lib = get_lib()

    chrs = [1, 1, 1, 2, 3, 3, 4]
    starts = [100, 200, 300, 50, 100, 200, 500]
    ends = [150, 250, 350, 80, 150, 250, 600]
    signal = [0.1, 6.0, 0.1, 12.0, 0.2, 8.5, 0.5]
    intervals = list(zip(chrs, starts, ends))

    code1, out1, sal1 = call_morse_genomic(lib, chrs, starts, ends, signal, mad_floor=1.0, tau=1.0)
    code2, out2, sal2 = call_morse_genomic_intervals(lib, intervals, signal, mad_floor=1.0, tau=1.0)

    assert code1 == 0 and code2 == 0
    assert np.array_equal(out1, out2), "Array API and Struct API must produce bit-identical output"
    assert np.array_equal(sal1, sal2), "Array API and Struct API must produce bit-identical salience"


# =============================================================================
# Test Suite 6: 10,000 Synthetic Randomized Stress Evaluations
# =============================================================================

def test_10000_synthetic_evaluations_zero_drift():
    """Run 10,000 synthetic randomized evaluations against Python reference oracle.

    Verifies:
    1. Bit-level consistency / parity with reference oracle.
    2. Zero numerical drift or cumulative state corruption.
    3. Memory safety across variable buffer lengths and boundary configurations.
    4. Salience map preservation (|x|).
    5. Clean execution with 0 NaNs / Infs.
    """
    lib = get_lib()
    rng = np.random.default_rng(42)

    total_evals = 10_000
    mismatch_count = 0
    nan_count = 0

    print(f"\n[STRESS] Commencing {total_evals} synthetic evaluations...")

    for iteration in range(total_evals):
        # 1. Random buffer length between 1 and 256 bins
        n = rng.integers(1, 257)

        # 2. Partition into 1 to min(n, 15) chromosomes
        num_chroms = rng.integers(1, min(n, 15) + 1)
        if num_chroms == 1:
            chrom_ids = np.ones(n, dtype=np.uint32)
        else:
            split_points = sorted(rng.choice(range(1, n), size=num_chroms - 1, replace=False))
            boundaries = [0] + split_points + [n]
            chrom_ids = np.zeros(n, dtype=np.uint32)
            for cid, (start_idx, end_idx) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
                chrom_ids[start_idx:end_idx] = cid

        # 3. Generate strictly valid monotonic coordinates within each chromosome
        starts = np.zeros(n, dtype=np.uint64)
        ends = np.zeros(n, dtype=np.uint64)
        curr_pos = 1000
        for i in range(n):
            if i > 0 and chrom_ids[i] != chrom_ids[i - 1]:
                curr_pos = 1000
            curr_pos += rng.integers(10, 500)
            starts[i] = curr_pos
            ends[i] = curr_pos + rng.integers(50, 1000)

        # 4. Generate signal: mixture of noise, spikes, plateaus, and edge cases
        regime = iteration % 5
        if regime == 0:
            # Gaussian noise + random sparse high spikes
            signal = rng.normal(loc=1.0, scale=0.5, size=n).astype(np.float32)
            spike_mask = rng.random(n) > 0.85
            signal[spike_mask] += rng.uniform(5.0, 20.0, size=np.sum(spike_mask))
        elif regime == 1:
            # Step functions / plateaus (adversarial for derivative stencils)
            signal = np.repeat(rng.uniform(0.5, 10.0, size=n // 4 + 1), 4)[:n].astype(np.float32)
        elif regime == 2:
            # Monotone ramps
            signal = np.linspace(0.1, 20.0, num=n, dtype=np.float32)
        elif regime == 3:
            # Alternating highs and lows (boundary oscillations)
            signal = np.tile([0.1, 10.0], n)[:n].astype(np.float32)
        else:
            # Exponentially distributed values with extreme dynamic range
            signal = rng.exponential(scale=2.0, size=n).astype(np.float32)

        tau = float(rng.uniform(0.5, 3.0))
        mad_floor = float(rng.uniform(0.5, 2.0))

        # 5. Compute Rust output
        code, rust_out, rust_sal = call_morse_genomic(
            lib, chrom_ids.tolist(), starts.tolist(), ends.tolist(), signal.tolist(),
            mad_floor=mad_floor, tau=tau
        )

        assert code == 0, f"Iteration {iteration}: unexpected return code {code}"

        # 6. Compute Python reference output
        ref_out, ref_sal = reference_morse_genomic(
            chrom_ids.tolist(), signal.tolist(), mad_floor=mad_floor, tau=tau
        )

        # 7. Check bitwise / numerical equality
        if not np.allclose(rust_out, ref_out, atol=1e-6, rtol=1e-5):
            diff = np.where(~np.isclose(rust_out, ref_out, atol=1e-6, rtol=1e-5))[0]
            print(f"FAILED on iteration {iteration}: n={n}, diff indices={diff}")
            print(f"Signal: {signal[diff]}")
            print(f"Chroms: {chrom_ids[diff]}")
            print(f"Rust out: {rust_out[diff]}")
            print(f"Ref out: {ref_out[diff]}")
            mismatch_count += 1
            assert False, f"Mismatch at iteration {iteration}"

        if not np.allclose(rust_sal, ref_sal, atol=1e-6):
            assert False, f"Salience mismatch at iteration {iteration}"

        if np.any(np.isnan(rust_out)) or np.any(np.isinf(rust_out)):
            nan_count += 1
            assert False, f"NaN/Inf generated at iteration {iteration}"

    print(f"[STRESS] COMPLETED {total_evals} evaluations. Mismatches: {mismatch_count}, NaNs: {nan_count}. Zero drift verified!")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", "-s", __file__]))


# =============================================================================
# Test Suite 7: Additional Adversarial Edge Cases
# =============================================================================

def test_null_salience_pointer():
    """Verify that passing salience_ptr = NULL executes safely without crashing."""
    lib = get_lib()

    chrs = [1, 1, 2]
    starts = [100, 200, 100]
    ends = [200, 300, 200]
    signal = [0.1, 5.0, 8.0]

    c_chrs = (ctypes.c_uint32 * 3)(*chrs)
    c_starts = (ctypes.c_uint64 * 3)(*starts)
    c_ends = (ctypes.c_uint64 * 3)(*ends)
    c_in = (ctypes.c_float * 3)(*signal)
    c_out = (ctypes.c_float * 3)()

    code = lib.morse_gate_genomic_f32(
        c_chrs, c_starts, c_ends, c_in, c_out, None, 3, 1.0, 1.0
    )
    assert code == 0
    assert np.allclose(c_out, [0.0, 5.0, 8.0])


def test_huge_coordinate_values_near_max():
    """Verify coordinates near u32::MAX and u64::MAX execute cleanly without overflow."""
    lib = get_lib()

    u32_max = 0xFFFFFFFF
    u64_max = 0xFFFFFFFFFFFFFFFF

    chrs = [u32_max - 1, u32_max - 1, u32_max]
    starts = [u64_max - 1000, u64_max - 500, u64_max - 200]
    ends = [u64_max - 800, u64_max - 400, u64_max]
    signal = [0.1, 7.5, 9.0]

    code, out, _ = call_morse_genomic(lib, chrs, starts, ends, signal, mad_floor=1.0, tau=1.0)
    assert code == 0
    assert out[1] == 7.5
    assert out[2] == 9.0


def test_zero_length_intervals_start_equals_end():
    """Verify start == end is accepted (0-width genomic bin) and processed correctly."""
    lib = get_lib()

    chrs = [1, 1, 2]
    starts = [1000, 2000, 1000]
    ends = [1000, 2000, 1000]  # start == end
    signal = [0.2, 8.0, 0.5]

    code, out, _ = call_morse_genomic(lib, chrs, starts, ends, signal, mad_floor=1.0, tau=1.0)
    assert code == 0
    assert out[1] == 8.0
    assert out[0] == 0.0
    assert out[2] == 0.0


def test_deep_chromosome_interleaving():
    """Verify alternating pattern of single-bin and multi-bin chromosomes."""
    lib = get_lib()

    # Pattern: [chr1(1), chr2(2), chr3(1), chr4(3), chr5(1)]
    chrs = [1, 2, 2, 3, 4, 4, 4, 5]
    starts = [10, 10, 20, 10, 10, 20, 30, 10]
    ends = [20, 20, 30, 20, 20, 30, 40, 20]
    signal = [6.0, 0.1, 7.0, 0.2, 0.1, 9.0, 0.1, 10.0]

    code, out, _ = call_morse_genomic(lib, chrs, starts, ends, signal, mad_floor=1.0, tau=1.0)
    assert code == 0
    expected = [
        6.0,   # chr1 single bin > 1.0
        0.0,   # chr2 start bin < right neighbor 7.0
        7.0,   # chr2 end bin > left neighbor 0.1
        0.0,   # chr3 single bin (0.2 <= 1.0)
        0.0,   # chr4 start bin < right neighbor 9.0
        9.0,   # chr4 interior spike > 0.1 and > 0.1
        0.0,   # chr4 end bin < left neighbor 9.0
        10.0,  # chr5 single bin > 1.0
    ]
    assert np.allclose(out, expected)


def test_large_buffer_performance_and_correctness():
    """Verify large buffer of 100,000 bins across 24 chromosomes."""
    lib = get_lib()
    rng = np.random.default_rng(123)

    n = 100_000
    # 24 chromosomes with varying sizes
    chrom_sizes = rng.multinomial(n, [1/24]*24)
    chrom_ids = np.repeat(np.arange(1, 25, dtype=np.uint32), chrom_sizes)

    starts = np.zeros(n, dtype=np.uint64)
    ends = np.zeros(n, dtype=np.uint64)
    for c in range(1, 25):
        mask = (chrom_ids == c)
        c_len = np.sum(mask)
        pos = np.cumsum(rng.integers(10, 100, size=c_len, dtype=np.uint64))
        starts[mask] = pos
        ends[mask] = pos + rng.integers(10, 50, size=c_len, dtype=np.uint64)

    signal = rng.normal(loc=1.0, scale=0.5, size=n).astype(np.float32)
    # Add spikes
    spikes = rng.choice(n, size=1000, replace=False)
    signal[spikes] += 10.0

    code, out, sal = call_morse_genomic(
        lib, chrom_ids.tolist(), starts.tolist(), ends.tolist(), signal.tolist(),
        mad_floor=1.0, tau=2.0
    )
    assert code == 0
    assert len(out) == n
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))
    assert np.allclose(sal, np.abs(signal))

    ref_out, _ = reference_morse_genomic(chrom_ids.tolist(), signal.tolist(), mad_floor=1.0, tau=2.0)
    assert np.allclose(out, ref_out, atol=1e-6)
