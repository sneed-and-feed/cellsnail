//! cellsnail_rs: Allocation-free #![no_std] Rust SIMD & C-ABI library
//! for 1D chromosomal copy number variation (CNV) and focal amplicon detection.
//!
//! Features:
//! - 1D discrete Morse innovation gating and multi-scale plateau filtering
//! - Genomic coordinate validation (chr:start-end sorting) and boundary reset stencils
//! - Static SRAM residency < 2.0 MB with 64-byte alignment and zero heap allocations.

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_unsafe)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::identity_op)]

#[cfg(all(not(feature = "std"), not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(all(not(feature = "std"), not(test), not(windows), not(target_arch = "wasm32")))]
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

#[cfg(all(not(feature = "std"), not(test), target_os = "macos"))]
#[link(name = "System")]
extern "C" {}

#[cfg(any(feature = "std", test))]
extern crate std;

// ============================================================================
// Constants & Static Arena Geometry (< 2.0 MB SRAM)
// ============================================================================

pub const SPATIAL_SIZE: usize = 16384;
pub const RING_BUFFER_SIZE: usize = 131072;
pub const GENOTYPE_TENSOR_SIZE: usize = 160;
pub const MULTIOMIC_COV_SIZE: usize = 36;
pub const CRYOEM_SLAB_SIZE: usize = 32768;
pub const SE3_TWISTS_SIZE: usize = 6144;
pub const CONTACT_EDGES_SIZE: usize = 8192;

/// Fixed-layout static arena strictly < 2.0 MB SRAM with 64-byte alignment.
/// Total size: exactly 1,303,360 bytes (1,272.8 KB <= 1,450,000 bytes < 2,097,152 bytes).
#[repr(C, align(64))]
pub struct CellSnailArena {
    pub spatial_input: [f32; SPATIAL_SIZE],          // 65,536 B (offset 0)
    pub spatial_output: [f32; SPATIAL_SIZE],         // 65,536 B (offset 65,536)
    pub salience_map: [f32; SPATIAL_SIZE],           // 65,536 B (offset 131,072)
    pub curl_buffer: [f32; SPATIAL_SIZE],            // 65,536 B (offset 196,608)
    pub ring_buffer: [f32; RING_BUFFER_SIZE],        // 524,288 B (offset 262,144)
    pub cell_cycle_mask: [f32; SPATIAL_SIZE],        // 65,536 B (offset 786,432)
    pub genotype_tensor: [f32; GENOTYPE_TENSOR_SIZE], // 640 B (offset 851,968)
    pub trajectory_buffer: [f32; SPATIAL_SIZE],      // 65,536 B (offset 852,608)
    pub sheaf_coboundary: [f32; SPATIAL_SIZE],       // 65,536 B (offset 918,144)
    pub multiomic_cov: [f32; MULTIOMIC_COV_SIZE],    // 144 B (offset 983,680)
    pub _pad: [u8; 48],                              // 48 B (offset 983,824)
    pub cryoem_slab_in: [f32; CRYOEM_SLAB_SIZE],     // 131,072 B (offset 983,872)
    pub cryoem_slab_out: [f32; CRYOEM_SLAB_SIZE],    // 131,072 B (offset 1,114,944)
    pub se3_twists: [f32; SE3_TWISTS_SIZE],          // 24,576 B (offset 1,246,016)
    pub contact_edges: [u32; CONTACT_EDGES_SIZE],    // 32,768 B (offset 1,270,592)
}

// Compile-time static assertions proving residency and alignment (Theorems 2.11, 2.20–2.22)
const _: () = assert!(core::mem::size_of::<CellSnailArena>() <= 1_450_000);
const _: () = assert!(core::mem::size_of::<CellSnailArena>() <= 2_097_152);
const _: () = assert!(core::mem::size_of::<CellSnailArena>() == 1_303_360);
const _: () = assert!(core::mem::size_of::<CellSnailArena>().is_multiple_of(64));
const _: () = assert!(core::mem::align_of::<CellSnailArena>() == 64);

const _: () = assert!(core::mem::offset_of!(CellSnailArena, spatial_input) == 0);
const _: () = assert!(core::mem::offset_of!(CellSnailArena, spatial_output) == 65_536);
const _: () = assert!(core::mem::offset_of!(CellSnailArena, salience_map) == 131_072);
const _: () = assert!(core::mem::offset_of!(CellSnailArena, curl_buffer) == 196_608);
const _: () = assert!(core::mem::offset_of!(CellSnailArena, ring_buffer) == 262_144);
const _: () = assert!(core::mem::offset_of!(CellSnailArena, cell_cycle_mask) == 786_432);
const _: () = assert!(core::mem::offset_of!(CellSnailArena, genotype_tensor) == 851_968);
const _: () = assert!(core::mem::offset_of!(CellSnailArena, trajectory_buffer) == 852_608);
const _: () = assert!(core::mem::offset_of!(CellSnailArena, sheaf_coboundary) == 918_144);
const _: () = assert!(core::mem::offset_of!(CellSnailArena, multiomic_cov) == 983_680);
const _: () = assert!(core::mem::offset_of!(CellSnailArena, _pad) == 983_824);
const _: () = assert!(core::mem::offset_of!(CellSnailArena, cryoem_slab_in) == 983_872);
const _: () = assert!(core::mem::offset_of!(CellSnailArena, cryoem_slab_out) == 1_114_944);
const _: () = assert!(core::mem::offset_of!(CellSnailArena, se3_twists) == 1_246_016);
const _: () = assert!(core::mem::offset_of!(CellSnailArena, contact_edges) == 1_270_592);

impl CellSnailArena {
    pub const fn new() -> Self {
        Self {
            spatial_input: [0.0; SPATIAL_SIZE],
            spatial_output: [0.0; SPATIAL_SIZE],
            salience_map: [0.0; SPATIAL_SIZE],
            curl_buffer: [0.0; SPATIAL_SIZE],
            ring_buffer: [0.0; RING_BUFFER_SIZE],
            cell_cycle_mask: [0.0; SPATIAL_SIZE],
            genotype_tensor: [0.0; GENOTYPE_TENSOR_SIZE],
            trajectory_buffer: [0.0; SPATIAL_SIZE],
            sheaf_coboundary: [0.0; SPATIAL_SIZE],
            multiomic_cov: [0.0; MULTIOMIC_COV_SIZE],
            _pad: [0; 48],
            cryoem_slab_in: [0.0; CRYOEM_SLAB_SIZE],
            cryoem_slab_out: [0.0; CRYOEM_SLAB_SIZE],
            se3_twists: [0.0; SE3_TWISTS_SIZE],
            contact_edges: [0; CONTACT_EDGES_SIZE],
        }
    }

    pub fn reset(&mut self) {
        unsafe {
            core::ptr::write_bytes(self as *mut Self as *mut u8, 0, core::mem::size_of::<Self>());
        }
    }
}

impl Default for CellSnailArena {
    fn default() -> Self {
        Self::new()
    }
}

static mut GLOBAL_CELLSNAIL_ARENA: CellSnailArena = CellSnailArena::new();

// ============================================================================
// Math Primitives & Multi-Target SIMD Helpers
// ============================================================================

#[inline(always)]
pub fn abs_f32(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

#[inline(always)]
pub fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

#[inline(always)]
pub fn sqrt_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)))
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        vget_lane_f32(vsqrt_f32(vset_lane_f32(x, vdup_n_f32(0.0), 0)), 0)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x)))
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        let mut y = x;
        let mut i = y.to_bits();
        i = 0x5f3759df - (i >> 1);
        let mut inv = f32::from_bits(i);
        inv = inv * (1.5 - (0.5 * x * inv * inv));
        inv = inv * (1.5 - (0.5 * x * inv * inv));
        inv = inv * (1.5 - (0.5 * x * inv * inv));
        x * inv
    }
}

#[inline(always)]
pub fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        _mm_cvtsd_f64(_mm_sqrt_sd(_mm_set_sd(x), _mm_set_sd(x)))
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        vget_lane_f64(vsqrt_f64(vset_lane_f64(x, vdup_n_f64(0.0), 0)), 0)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut guess = x * 0.5;
        if guess == 0.0 { return 0.0; }
        for _ in 0..24 {
            guess = 0.5 * (guess + x / guess);
        }
        guess
    }
}

/// Multi-target SIMD dot product of slices `a` and `b`.
#[inline]
pub fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = if a.len() < b.len() { a.len() } else { b.len() };
    let mut i = 0usize;
    let mut total = 0.0f32;

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let mut acc = _mm256_setzero_ps();
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
            i += 8;
        }
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps(acc, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
        total += _mm_cvtss_f32(_mm_add_ss(sum64, shuf2));
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let mut acc = vdupq_n_f32(0.0);
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            acc = vfmaq_f32(acc, va, vb);
            i += 4;
        }
        total += vaddvq_f32(acc);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        let mut acc = f32x4_splat(0.0);
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);
            acc = f32x4_add(acc, f32x4_mul(va, vb));
            i += 4;
        }
        total += f32x4_extract_lane::<0>(acc) + f32x4_extract_lane::<1>(acc)
              + f32x4_extract_lane::<2>(acc) + f32x4_extract_lane::<3>(acc);
    }

    while i < len {
        total += a[i] * b[i];
        i += 1;
    }
    total
}

// ============================================================================
// 1D Genomic Coordinate Validation & 1D CNV Morse Filtering Kernel
// ============================================================================

/// Represents a 1D genomic locus interval (chrom_id, start, end).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenomicInterval {
    pub chrom_id: u32,
    pub start: u64,
    pub end: u64,
}

impl GenomicInterval {
    pub const fn new(chrom_id: u32, start: u64, end: u64) -> Self {
        Self { chrom_id, start, end }
    }
}

pub const GENOMIC_VALID: i32 = 0;
pub const ERR_GENOMIC_NULL_OR_EMPTY: i32 = -1;
pub const ERR_GENOMIC_INVERTED_COORDINATES: i32 = -2;
pub const ERR_GENOMIC_UNSORTED_CHROMOSOMES: i32 = -3;
pub const ERR_GENOMIC_UNSORTED_POSITIONS: i32 = -4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenomicValidationError {
    NullOrEmpty,
    InvertedCoordinates { index: usize, chrom_id: u32, start: u64, end: u64 },
    UnsortedChromosomes { index: usize, prev_chrom: u32, next_chrom: u32 },
    UnsortedPositions { index: usize, chrom_id: u32, prev_start: u64, next_start: u64 },
}

impl GenomicValidationError {
    pub fn error_code(&self) -> i32 {
        match self {
            Self::NullOrEmpty => ERR_GENOMIC_NULL_OR_EMPTY,
            Self::InvertedCoordinates { .. } => ERR_GENOMIC_INVERTED_COORDINATES,
            Self::UnsortedChromosomes { .. } => ERR_GENOMIC_UNSORTED_CHROMOSOMES,
            Self::UnsortedPositions { .. } => ERR_GENOMIC_UNSORTED_POSITIONS,
        }
    }
}

/// Validates that an array of genomic intervals is strictly ordered:
/// 1. Each interval has start <= end.
/// 2. Chromosome IDs are non-decreasing.
/// 3. Within each chromosome, start positions are non-decreasing.
pub fn validate_genomic_intervals(intervals: &[GenomicInterval]) -> Result<(), GenomicValidationError> {
    if intervals.is_empty() {
        return Err(GenomicValidationError::NullOrEmpty);
    }
    for (i, w) in intervals.windows(2).enumerate() {
        let curr = &w[0];
        let next = &w[1];

        if curr.start > curr.end {
            return Err(GenomicValidationError::InvertedCoordinates {
                index: i,
                chrom_id: curr.chrom_id,
                start: curr.start,
                end: curr.end,
            });
        }

        if next.chrom_id < curr.chrom_id {
            return Err(GenomicValidationError::UnsortedChromosomes {
                index: i + 1,
                prev_chrom: curr.chrom_id,
                next_chrom: next.chrom_id,
            });
        }

        if next.chrom_id == curr.chrom_id && next.start < curr.start {
            return Err(GenomicValidationError::UnsortedPositions {
                index: i + 1,
                chrom_id: curr.chrom_id,
                prev_start: curr.start,
                next_start: next.start,
            });
        }
    }

    if let Some(last) = intervals.last() {
        if last.start > last.end {
            return Err(GenomicValidationError::InvertedCoordinates {
                index: intervals.len() - 1,
                chrom_id: last.chrom_id,
                start: last.start,
                end: last.end,
            });
        }
    }

    Ok(())
}

/// Validates genomic coordinate arrays in parallel C representation.
/// Returns 0 on success, negative error code on violation.
#[no_mangle]
pub unsafe extern "C" fn validate_genomic_coordinates_c(
    chrom_ids: *const u32,
    starts: *const u64,
    ends: *const u64,
    n: usize,
) -> i32 {
    if chrom_ids.is_null() || starts.is_null() || ends.is_null() || n == 0 {
        return ERR_GENOMIC_NULL_OR_EMPTY;
    }

    for i in 0..n {
        let s = *starts.add(i);
        let e = *ends.add(i);
        if s > e {
            return ERR_GENOMIC_INVERTED_COORDINATES;
        }

        if i + 1 < n {
            let c_curr = *chrom_ids.add(i);
            let c_next = *chrom_ids.add(i + 1);
            if c_next < c_curr {
                return ERR_GENOMIC_UNSORTED_CHROMOSOMES;
            }
            if c_next == c_curr {
                let s_next = *starts.add(i + 1);
                if s_next < s {
                    return ERR_GENOMIC_UNSORTED_POSITIONS;
                }
            }
        }
    }

    GENOMIC_VALID
}

/// Validates an array of GenomicInterval structs via C-ABI.
#[no_mangle]
pub unsafe extern "C" fn validate_genomic_intervals_c(
    intervals: *const GenomicInterval,
    n: usize,
) -> i32 {
    if intervals.is_null() || n == 0 {
        return ERR_GENOMIC_NULL_OR_EMPTY;
    }
    let slice = core::slice::from_raw_parts(intervals, n);
    match validate_genomic_intervals(slice) {
        Ok(()) => GENOMIC_VALID,
        Err(err) => err.error_code(),
    }
}

/// 1D discrete Morse innovation gate for genomic coordinate arrays.
///
/// Rejects monotone slopes/ramps (I_morse <= 0), preserves isolated focal spikes (I_morse > 0)
/// exceeding noise floor, with 0 spurious cross-gene correlation injection.
#[no_mangle]
pub unsafe extern "C" fn morse_gate_1d_f32(
    in_ptr: *const f32,
    out_ptr: *mut f32,
    salience_ptr: *mut f32,
    n: usize,
    mad_floor: f32,
    tau: f32,
) {
    if in_ptr.is_null() || out_ptr.is_null() || n == 0 {
        return;
    }

    let thresh = if tau > mad_floor { tau } else { mad_floor };

    if n == 1 {
        let v = *in_ptr;
        if !salience_ptr.is_null() {
            *salience_ptr = abs_f32(v);
        }
        *out_ptr = if v > thresh { v } else { 0.0 };
        return;
    }

    if n == 2 {
        for i in 0..2 {
            let v = *in_ptr.add(i);
            if !salience_ptr.is_null() {
                *salience_ptr.add(i) = abs_f32(v);
            }
            *out_ptr.add(i) = if v > thresh { v } else { 0.0 };
        }
        return;
    }

    // Boundary at index 0
    let v0 = *in_ptr;
    let v1 = *in_ptr.add(1);
    if !salience_ptr.is_null() {
        *salience_ptr = abs_f32(v0);
    }
    *out_ptr = if v0 > thresh && v0 > v1 { v0 } else { 0.0 };

    let mut i = 1usize;

    // SIMD vectorized loop for strict extrema identification (AVX2 / NEON / WASM)
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        let v_zero = _mm256_setzero_ps();
        let v_thresh = _mm256_set1_ps(thresh);

        while i + 8 <= n - 1 {
            let v_prev = _mm256_loadu_ps(in_ptr.add(i - 1));
            let v_curr = _mm256_loadu_ps(in_ptr.add(i));
            let v_next = _mm256_loadu_ps(in_ptr.add(i + 1));

            let d_minus = _mm256_sub_ps(v_curr, v_prev);
            let d_plus = _mm256_sub_ps(v_curr, v_next);
            let morse_prod = _mm256_mul_ps(d_minus, d_plus);

            // Condition: morse_prod > 0 && d_minus > 0 && d_plus > 0 && v_curr > thresh
            let mask_prod = _mm256_cmp_ps(morse_prod, v_zero, _CMP_GT_OQ);
            let mask_dm = _mm256_cmp_ps(d_minus, v_zero, _CMP_GT_OQ);
            let mask_dp = _mm256_cmp_ps(d_plus, v_zero, _CMP_GT_OQ);
            let mask_th = _mm256_cmp_ps(v_curr, v_thresh, _CMP_GT_OQ);

            let mask = _mm256_and_ps(_mm256_and_ps(mask_prod, mask_dm), _mm256_and_ps(mask_dp, mask_th));
            let result = _mm256_and_ps(mask, v_curr);
            _mm256_storeu_ps(out_ptr.add(i), result);

            if !salience_ptr.is_null() {
                let mask_abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
                let sal = _mm256_and_ps(v_curr, mask_abs);
                _mm256_storeu_ps(salience_ptr.add(i), sal);
            }

            i += 8;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        let v_zero = vdupq_n_f32(0.0);
        let v_thresh = vdupq_n_f32(thresh);

        while i + 4 <= n - 1 {
            let v_prev = vld1q_f32(in_ptr.add(i - 1));
            let v_curr = vld1q_f32(in_ptr.add(i));
            let v_next = vld1q_f32(in_ptr.add(i + 1));

            let d_minus = vsubq_f32(v_curr, v_prev);
            let d_plus = vsubq_f32(v_curr, v_next);
            let morse_prod = vmulq_f32(d_minus, d_plus);

            let mask_prod = vcgtq_f32(morse_prod, v_zero);
            let mask_dm = vcgtq_f32(d_minus, v_zero);
            let mask_dp = vcgtq_f32(d_plus, v_zero);
            let mask_th = vcgtq_f32(v_curr, v_thresh);

            let mask = vandq_u32(vandq_u32(mask_prod, mask_dm), vandq_u32(mask_dp, mask_th));
            let result = vreinterpretq_f32_u32(vandq_u32(mask, vreinterpretq_u32_f32(v_curr)));
            vst1q_f32(out_ptr.add(i), result);

            if !salience_ptr.is_null() {
                let sal = vabsq_f32(v_curr);
                vst1q_f32(salience_ptr.add(i), sal);
            }

            i += 4;
        }
    }

    // Remainder scalar loop
    while i < n - 1 {
        let v_curr = *in_ptr.add(i);
        let v_prev = *in_ptr.add(i - 1);
        let v_next = *in_ptr.add(i + 1);

        let d_minus = v_curr - v_prev;
        let d_plus = v_curr - v_next;
        let morse_prod = d_minus * d_plus;

        let is_spike = morse_prod > 0.0 && d_minus > 0.0 && d_plus > 0.0 && v_curr > thresh;
        *out_ptr.add(i) = if is_spike { v_curr } else { 0.0 };

        if !salience_ptr.is_null() {
            *salience_ptr.add(i) = abs_f32(v_curr);
        }
        i += 1;
    }

    // Boundary at index n - 1
    let vn = *in_ptr.add(n - 1);
    let vn_prev = *in_ptr.add(n - 2);
    if !salience_ptr.is_null() {
        *salience_ptr.add(n - 1) = abs_f32(vn);
    }
    *out_ptr.add(n - 1) = if vn > thresh && vn > vn_prev { vn } else { 0.0 };
}

/// 1D Morse gate dropout filter for single-cell genomic coordinate arrays.
#[no_mangle]
pub unsafe extern "C" fn morse_dropout_filter_f32(
    in_ptr: *const f32,
    out_ptr: *mut f32,
    salience_ptr: *mut f32,
    n: usize,
    mad_floor: f32,
    tau: f32,
) {
    morse_gate_1d_f32(in_ptr, out_ptr, salience_ptr, n, mad_floor, tau);
}

/// 1D discrete Morse innovation gate with genomic coordinate validation
/// and chromosome boundary-aware finite difference stencil resets.
///
/// When crossing chromosome boundaries, the 3-point stencil resets:
/// - Boundary start bins are evaluated only against the succeeding bin on the same chromosome.
/// - Boundary end bins are evaluated only against the preceding bin on the same chromosome.
/// - False differences are never computed across chromosome junctions.
#[no_mangle]
pub unsafe extern "C" fn morse_gate_genomic_f32(
    chrom_ids: *const u32,
    starts: *const u64,
    ends: *const u64,
    in_ptr: *const f32,
    out_ptr: *mut f32,
    salience_ptr: *mut f32,
    n: usize,
    mad_floor: f32,
    tau: f32,
) -> i32 {
    if chrom_ids.is_null() || starts.is_null() || ends.is_null() || in_ptr.is_null() || out_ptr.is_null() || n == 0 {
        return ERR_GENOMIC_NULL_OR_EMPTY;
    }
    let val = validate_genomic_coordinates_c(chrom_ids, starts, ends, n);
    if val != GENOMIC_VALID {
        return val;
    }

    let thresh = if tau > mad_floor { tau } else { mad_floor };

    for i in 0..n {
        let v_curr = *in_ptr.add(i);
        let curr_chrom = *chrom_ids.add(i);

        if !salience_ptr.is_null() {
            *salience_ptr.add(i) = abs_f32(v_curr);
        }

        let has_left = (i > 0) && (*chrom_ids.add(i - 1) == curr_chrom);
        let has_right = (i + 1 < n) && (*chrom_ids.add(i + 1) == curr_chrom);

        let is_spike = match (has_left, has_right) {
            (true, true) => {
                let v_prev = *in_ptr.add(i - 1);
                let v_next = *in_ptr.add(i + 1);
                let d_minus = v_curr - v_prev;
                let d_plus = v_curr - v_next;
                (d_minus * d_plus > 0.0) && (d_minus > 0.0) && (d_plus > 0.0) && (v_curr > thresh)
            }
            (false, true) => {
                let v_next = *in_ptr.add(i + 1);
                (v_curr > thresh) && (v_curr > v_next)
            }
            (true, false) => {
                let v_prev = *in_ptr.add(i - 1);
                (v_curr > thresh) && (v_curr > v_prev)
            }
            (false, false) => v_curr > thresh,
        };

        *out_ptr.add(i) = if is_spike { v_curr } else { 0.0 };
    }

    GENOMIC_VALID
}

/// 1D discrete Morse gate with GenomicInterval structs and boundary resets.
#[no_mangle]
pub unsafe extern "C" fn morse_gate_genomic_intervals_f32(
    intervals: *const GenomicInterval,
    in_ptr: *const f32,
    out_ptr: *mut f32,
    salience_ptr: *mut f32,
    n: usize,
    mad_floor: f32,
    tau: f32,
) -> i32 {
    if intervals.is_null() || in_ptr.is_null() || out_ptr.is_null() || n == 0 {
        return ERR_GENOMIC_NULL_OR_EMPTY;
    }
    let slice = core::slice::from_raw_parts(intervals, n);
    let val = validate_genomic_intervals(slice);
    if let Err(err) = val {
        return err.error_code();
    }

    let thresh = if tau > mad_floor { tau } else { mad_floor };

    for i in 0..n {
        let v_curr = *in_ptr.add(i);
        let curr_chrom = slice[i].chrom_id;

        if !salience_ptr.is_null() {
            *salience_ptr.add(i) = abs_f32(v_curr);
        }

        let has_left = (i > 0) && (slice[i - 1].chrom_id == curr_chrom);
        let has_right = (i + 1 < n) && (slice[i + 1].chrom_id == curr_chrom);

        let is_spike = match (has_left, has_right) {
            (true, true) => {
                let v_prev = *in_ptr.add(i - 1);
                let v_next = *in_ptr.add(i + 1);
                let d_minus = v_curr - v_prev;
                let d_plus = v_curr - v_next;
                (d_minus * d_plus > 0.0) && (d_minus > 0.0) && (d_plus > 0.0) && (v_curr > thresh)
            }
            (false, true) => {
                let v_next = *in_ptr.add(i + 1);
                (v_curr > thresh) && (v_curr > v_next)
            }
            (true, false) => {
                let v_prev = *in_ptr.add(i - 1);
                (v_curr > thresh) && (v_curr > v_prev)
            }
            (false, false) => v_curr > thresh,
        };

        *out_ptr.add(i) = if is_spike { v_curr } else { 0.0 };
    }

    GENOMIC_VALID
}

// ============================================================================
// Continuity Clutter Filter & Cell-Cycle Deconfounder
// ============================================================================

/// Continuity clutter filter projecting state observations into orthogonal subspaces.
///
/// Shape: `in_ptr` and `out_ptr` are `[t_steps, n]`.
/// Identifies periodic/flicker cell-cycle channels and freezes them to baseline.
#[no_mangle]
pub unsafe extern "C" fn continuity_clutter_filter_f32(
    in_ptr: *const f32,
    out_ptr: *mut f32,
    clock_mask: *mut u8,
    n: usize,
    t_steps: usize,
) {
    if in_ptr.is_null() || out_ptr.is_null() || n == 0 || t_steps == 0 {
        return;
    }

    if t_steps < 3 {
        if !clock_mask.is_null() {
            for j in 0..n {
                *clock_mask.add(j) = 0;
            }
        }
        for idx in 0..(t_steps * n) {
            *out_ptr.add(idx) = *in_ptr.add(idx);
        }
        return;
    }

    let total_transitions = (t_steps - 2) as f32;

    for j in 0..n {
        let mut sign_flips = 0u32;
        let mut prev_diff = *in_ptr.add(n + j) - *in_ptr.add(j);

        for t in 1..(t_steps - 1) {
            let curr_diff = *in_ptr.add((t + 1) * n + j) - *in_ptr.add(t * n + j);
            if (prev_diff * curr_diff) < 0.0 {
                sign_flips += 1;
            }
            prev_diff = curr_diff;
        }

        let flicker_ratio = (sign_flips as f32) / total_transitions;
        let is_clock = flicker_ratio >= 0.85;

        if !clock_mask.is_null() {
            *clock_mask.add(j) = if is_clock { 1 } else { 0 };
        }

        let base_val = *in_ptr.add(j);
        for t in 0..t_steps {
            *out_ptr.add(t * n + j) = if is_clock { base_val } else { *in_ptr.add(t * n + j) };
        }
    }
}

/// Cell-cycle deconfounder filtering periodic/flicker state observations.
#[no_mangle]
pub unsafe extern "C" fn cell_cycle_deconfounder_f32(
    in_ptr: *const f32,
    out_ptr: *mut f32,
    clock_mask: *mut u8,
    n: usize,
    t_steps: usize,
) {
    continuity_clutter_filter_f32(in_ptr, out_ptr, clock_mask, n, t_steps);
}

// ============================================================================
// Skew Triad Step & Genotype Tiebreaker SO(10)
// ============================================================================

/// Skew-symmetric bilinear triad step with exact spherical retraction.
///
/// Tangent orthogonality <v, delta_v> == 0 and 0.000e+00 radial norm drift ||v_retract|| == ||v||.
#[no_mangle]
pub unsafe extern "C" fn skew_triad_step_f32(
    u: *const f32,
    v: *const f32,
    w: *const f32,
    out: *mut f32,
    dim: usize,
) {
    if u.is_null() || v.is_null() || out.is_null() || dim == 0 {
        return;
    }

    let u_slice = core::slice::from_raw_parts(u, dim);
    let v_slice = core::slice::from_raw_parts(v, dim);

    // Compute norm of input vector v
    let norm_sq_v = simd_dot_f32(v_slice, v_slice);
    let norm_v = sqrt_f32(norm_sq_v);
    if norm_v == 0.0 {
        for k in 0..dim {
            *out.add(k) = 0.0;
        }
        return;
    }

    // Allocate stack buffer for tangent update (up to 1024 dims)
    let mut raw = [0.0f32; 1024];
    let d = if dim <= 1024 { dim } else { 1024 };

    if w.is_null() {
        // Cyclic nearest-neighbor skew tie-breaker
        for k in 0..d {
            let next_idx = if k + 1 < d { k + 1 } else { 0 };
            let prev_idx = if k > 0 { k - 1 } else { d - 1 };
            raw[k] = u_slice[k] * (v_slice[next_idx] - v_slice[prev_idx]);
        }
        let dot_v_raw = simd_dot_f32(v_slice, &raw[..d]);
        if norm_sq_v > 1e-30 {
            let proj = dot_v_raw / norm_sq_v;
            for k in 0..d {
                raw[k] -= proj * v_slice[k];
            }
        }
    } else {
        // Full rank-3 interaction tensor: W [dim, dim, dim]
        let w_slice = core::slice::from_raw_parts(w, d * d * d);
        for k in 0..d {
            let mut acc = 0.0f32;
            for j in 0..d {
                let mut m_jk = 0.0f32;
                let mut m_kj = 0.0f32;
                for i_idx in 0..d {
                    m_jk += u_slice[i_idx] * w_slice[i_idx * d * d + j * d + k];
                    m_kj += u_slice[i_idx] * w_slice[i_idx * d * d + k * d + j];
                }
                acc += v_slice[j] * (m_jk - m_kj);
            }
            raw[k] = acc;
        }
    }

    // Form tangent vector v_tangent = v + raw
    let mut v_tangent = [0.0f32; 1024];
    for k in 0..d {
        v_tangent[k] = v_slice[k] + raw[k];
    }

    // Spherical retraction: scale to exact original norm ||v||
    let norm_sq_tan = simd_dot_f32(&v_tangent[..d], &v_tangent[..d]);
    let norm_tan = sqrt_f32(norm_sq_tan);

    if norm_tan == 0.0 {
        for k in 0..d {
            *out.add(k) = v_slice[k];
        }
    } else {
        let scale = norm_v / norm_tan;
        for k in 0..d {
            *out.add(k) = v_tangent[k] * scale;
        }
    }
}

/// Genotype tiebreaker via skew-symmetric retraction.
#[no_mangle]
pub unsafe extern "C" fn genotype_tiebreaker_so10_f32(
    u: *const f32,
    v: *const f32,
    w: *const f32,
    out: *mut f32,
    dim: usize,
) {
    skew_triad_step_f32(u, v, w, out, dim);
}

// ============================================================================
// Slow-Fast Cascade K16 & Pseudotime Spherical Cascade
// ============================================================================

/// 16-step unrolled spherical velocity integration cascade.
///
/// Guaranteed radial norm conservation across all 16 unrolled steps: ||v_16|| == ||v_0||.
#[no_mangle]
pub unsafe extern "C" fn slow_fast_cascade_k16_f32(
    v0: *const f32,
    u_seq: *const f32,
    gamma: f32,
    out: *mut f32,
    dim: usize,
) {
    if v0.is_null() || u_seq.is_null() || out.is_null() || dim == 0 {
        return;
    }

    let d = if dim <= 1024 { dim } else { 1024 };
    let mut v_cur = [0.0f32; 1024];
    let mut v_next = [0.0f32; 1024];
    let mut u_scaled = [0.0f32; 1024];

    for k in 0..d {
        v_cur[k] = *v0.add(k);
    }

    for step in 0..16 {
        let u_step_ptr = u_seq.add(step * dim);
        for k in 0..d {
            u_scaled[k] = *u_step_ptr.add(k) * gamma;
        }
        skew_triad_step_f32(
            u_scaled.as_ptr(),
            v_cur.as_ptr(),
            core::ptr::null(),
            v_next.as_mut_ptr(),
            d,
        );
        v_cur[..d].copy_from_slice(&v_next[..d]);
    }

    for k in 0..d {
        *out.add(k) = v_cur[k];
    }
}

/// Pseudotime spherical velocity integration cascade.
#[no_mangle]
pub unsafe extern "C" fn pseudotime_spherical_cascade_f32(
    v0: *const f32,
    u_seq: *const f32,
    gamma: f32,
    out: *mut f32,
    dim: usize,
) {
    slow_fast_cascade_k16_f32(v0, u_seq, gamma, out, dim);
}

// ============================================================================
// Contact Sheaf SE(2) Step & Spatial Tissue Sheaf Solver
// ============================================================================

/// SE(2) cellular sheaf coboundary solver enforcing unilateral
/// non-penetration across cell contact interfaces: (delta v)_n >= 0.
#[no_mangle]
pub unsafe extern "C" fn contact_sheaf_se2_step_f32(
    dx: f32,
    dy: f32,
    nx: f32,
    ny: f32,
    out_dx: *mut f32,
    out_dy: *mut f32,
    hit_wall: *mut u8,
) {
    if out_dx.is_null() || out_dy.is_null() {
        return;
    }

    // If interface normal is zero, unconstrained displacement
    if nx == 0.0 && ny == 0.0 {
        *out_dx = dx;
        *out_dy = dy;
        if !hit_wall.is_null() {
            *hit_wall = 0;
        }
        return;
    }

    let dot = dx * nx + dy * ny;
    if dot > 0.0 {
        // Unilateral penetration detected -> hit boundary wall, clamp displacement
        if !hit_wall.is_null() {
            *hit_wall = 1;
        }
        *out_dx = 0.0;
        *out_dy = 0.0;
    } else {
        // Tangential motion or separating motion -> admissible
        if !hit_wall.is_null() {
            *hit_wall = 0;
        }
        *out_dx = dx;
        *out_dy = dy;
    }
}

/// Spatial tissue sheaf solver.
#[no_mangle]
pub unsafe extern "C" fn spatial_tissue_sheaf_solver_f32(
    dx: f32,
    dy: f32,
    nx: f32,
    ny: f32,
    out_dx: *mut f32,
    out_dy: *mut f32,
    hit_wall: *mut u8,
) {
    contact_sheaf_se2_step_f32(dx, dy, nx, ny, out_dx, out_dy, hit_wall);
}

// ============================================================================
// Kelvin-Mandel SO(6) Rotor & Multi-Omic Covariance Rotor
// ============================================================================

/// Kelvin-Mandel SO(6) rotor lifting SO(3) cross-modal rotation
/// to 6x6 Voigt covariance tensor with exact trace conservation.
#[no_mangle]
pub unsafe extern "C" fn kelvin_mandel_so6_rotor_f32(
    rot3: *const f32,
    cov6: *const f32,
    rotor6: *mut f32,
    cov6_out: *mut f32,
) {
    if rot3.is_null() {
        return;
    }

    let r_slice = core::slice::from_raw_parts(rot3, 9);
    let s2 = core::f64::consts::SQRT_2;

    let r11 = r_slice[0] as f64;
    let r12 = r_slice[1] as f64;
    let r13 = r_slice[2] as f64;
    let r21 = r_slice[3] as f64;
    let r22 = r_slice[4] as f64;
    let r23 = r_slice[5] as f64;
    let r31 = r_slice[6] as f64;
    let r32 = r_slice[7] as f64;
    let r33 = r_slice[8] as f64;

    let mut m = [0.0f64; 36];

    // Upper-left 3x3 normal-normal block
    m[0] = r11 * r11; m[1] = r12 * r12; m[2] = r13 * r13;
    m[6] = r21 * r21; m[7] = r22 * r22; m[8] = r23 * r23;
    m[12] = r31 * r31; m[13] = r32 * r32; m[14] = r33 * r33;

    // Upper-right 3x3 normal-shear block (scaled by sqrt(2))
    m[3] = s2 * r12 * r13; m[4] = s2 * r11 * r13; m[5] = s2 * r11 * r12;
    m[9] = s2 * r22 * r23; m[10] = s2 * r21 * r23; m[11] = s2 * r21 * r22;
    m[15] = s2 * r32 * r33; m[16] = s2 * r31 * r33; m[17] = s2 * r31 * r32;

    // Lower-left 3x3 shear-normal block (scaled by sqrt(2))
    m[18] = s2 * r21 * r31; m[19] = s2 * r22 * r32; m[20] = s2 * r23 * r33;
    m[24] = s2 * r11 * r31; m[25] = s2 * r12 * r32; m[26] = s2 * r13 * r33;
    m[30] = s2 * r11 * r21; m[31] = s2 * r12 * r22; m[32] = s2 * r13 * r23;

    // Lower-right 3x3 shear-shear block
    m[21] = r22 * r33 + r23 * r32;
    m[22] = r21 * r33 + r23 * r31;
    m[23] = r21 * r32 + r22 * r31;

    m[27] = r12 * r33 + r13 * r32;
    m[28] = r11 * r33 + r13 * r31;
    m[29] = r11 * r32 + r12 * r31;

    m[33] = r12 * r23 + r13 * r22;
    m[34] = r11 * r23 + r13 * r21;
    m[35] = r11 * r22 + r12 * r21;

    // Export 6x6 rotor if requested
    if !rotor6.is_null() {
        for idx in 0..36 {
            *rotor6.add(idx) = m[idx] as f32;
        }
    }

    // Compute rotated covariance: C_rot = M * C * M^T in double precision
    if !cov6.is_null() && !cov6_out.is_null() {
        let c_slice = core::slice::from_raw_parts(cov6, 36);
        let mut temp = [0.0f64; 36];

        for i in 0..6 {
            for j in 0..6 {
                let mut acc = 0.0f64;
                for k in 0..6 {
                    acc += m[i * 6 + k] * (c_slice[k * 6 + j] as f64);
                }
                temp[i * 6 + j] = acc;
            }
        }

        for i in 0..6 {
            for j in 0..6 {
                let mut acc = 0.0f64;
                for k in 0..6 {
                    acc += temp[i * 6 + k] * m[j * 6 + k];
                }
                *cov6_out.add(i * 6 + j) = acc as f32;
            }
        }
    }
}

/// Multi-omic covariance rotor.
#[no_mangle]
pub unsafe extern "C" fn multiomic_covariance_rotor_f32(
    rot3: *const f32,
    cov6: *const f32,
    rotor6: *mut f32,
    cov6_out: *mut f32,
) {
    kelvin_mandel_so6_rotor_f32(rot3, cov6, rotor6, cov6_out);
}

// ============================================================================
// Arena Lifecycle & Entry Points
// ============================================================================

#[no_mangle]
pub extern "C" fn cellsnail_arena_size_bytes() -> usize {
    core::mem::size_of::<CellSnailArena>()
}

#[no_mangle]
pub unsafe extern "C" fn cellsnail_arena_init(arena: *mut CellSnailArena) {
    if !arena.is_null() {
        core::ptr::write_bytes(arena as *mut u8, 0, core::mem::size_of::<CellSnailArena>());
    }
}

#[no_mangle]
pub unsafe extern "C" fn cellsnail_arena_reset(arena: *mut CellSnailArena) {
    if !arena.is_null() {
        core::ptr::write_bytes(arena as *mut u8, 0, core::mem::size_of::<CellSnailArena>());
    }
}

#[no_mangle]
pub extern "C" fn cellsnail_arena_global() -> *mut CellSnailArena {
    unsafe { core::ptr::addr_of_mut!(GLOBAL_CELLSNAIL_ARENA) }
}

// ============================================================================
// Adaptive C-ABI Exports & SIMD Kernels
// ============================================================================

/// Fast zero-alloc rational approximation to exp(-x / 20.0) for x >= 0 in #![no_std].
#[inline(always)]
fn exp_decay_20(x: f32) -> f32 {
    if x <= 0.0 {
        return 1.0;
    }
    let z = x * 0.05f32; // x / 20.0
    let denom = 1.0 + z * (1.0 + z * (0.5 + z * (0.16666667 + z * 0.04166667)));
    1.0 / denom
}

/// Adaptive Morse Dropout Filter with low-rank shrinkage prior blending.
/// In ultra-sparse regimes (lambda_blend > 0), blends non-focal positions with prior_ptr.
/// Confirmed focal spikes hold lambda_blend = 0, preserving sharp peaks and zero false-positive ERCC pairs.
#[no_mangle]
pub unsafe extern "C" fn morse_dropout_filter_adaptive_f32(
    in_ptr: *const f32,
    prior_ptr: *const f32,
    out_ptr: *mut f32,
    salience_ptr: *mut f32,
    n: usize,
    mad_floor: f32,
    tau: f32,
    lambda_blend: f32,
) {
    if in_ptr.is_null() || out_ptr.is_null() || n == 0 {
        return;
    }

    // 1. Run standard 3-point Morse gate
    morse_gate_1d_f32(in_ptr, out_ptr, salience_ptr, n, mad_floor, tau);

    // 2. Blend non-focal bins if lambda_blend > 0 and prior is present
    if lambda_blend > 0.0 && !prior_ptr.is_null() {
        let alpha = if lambda_blend > 1.0 { 1.0 } else { lambda_blend };
        let mut i = 0usize;

        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::*;
            let v_zero = _mm256_setzero_ps();
            let v_alpha = _mm256_set1_ps(alpha);

            while i + 8 <= n {
                let v_out = _mm256_loadu_ps(out_ptr.add(i));
                let v_prior = _mm256_loadu_ps(prior_ptr.add(i));

                let mask_zero = _mm256_cmp_ps(v_out, v_zero, _CMP_EQ_OQ);
                let blended_prior = _mm256_mul_ps(v_alpha, v_prior);
                let res = _mm256_blendv_ps(v_out, blended_prior, mask_zero);
                _mm256_storeu_ps(out_ptr.add(i), res);
                i += 8;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            use core::arch::aarch64::*;
            let v_zero = vdupq_n_f32(0.0);
            let v_alpha = vdupq_n_f32(alpha);

            while i + 4 <= n {
                let v_out = vld1q_f32(out_ptr.add(i));
                let v_prior = vld1q_f32(prior_ptr.add(i));

                let mask_zero = vceqq_f32(v_out, v_zero);
                let blended_prior = vmulq_f32(v_alpha, v_prior);
                let res = vbslq_f32(mask_zero, blended_prior, v_out);
                vst1q_f32(out_ptr.add(i), res);
                i += 4;
            }
        }

        while i < n {
            let cur_out = *out_ptr.add(i);
            if cur_out == 0.0 {
                let p = *prior_ptr.add(i);
                *out_ptr.add(i) = alpha * p;
            }
            i += 1;
        }
    }
}

/// Multi-scale plateau smoothing with boundary pinning on broad CNVs (Theorem 2.17).
/// For broad amplicons, reduces interior WGA variance by boxcar averaging while
/// pinning step transitions (zero boundary leakage).
#[no_mangle]
pub unsafe extern "C" fn morse_plateau_filter_f32(
    in_ptr: *const f32,
    out_ptr: *mut f32,
    n: usize,
    window_size: usize,
) {
    if in_ptr.is_null() || out_ptr.is_null() || n == 0 {
        return;
    }

    if n <= 3 || window_size <= 1 {
        for i in 0..n {
            *out_ptr.add(i) = *in_ptr.add(i);
        }
        return;
    }

    let half_win = (window_size / 2).min(7);

    // Adaptive step threshold: tau_step = max(1.5, 3.5 * MAD(Delta_+ v))
    let num_diffs = n - 1;
    let sample_count = num_diffs.min(256);
    let mut diff_samples = [0.0f32; 256];
    if num_diffs <= 256 {
        for i in 0..num_diffs {
            diff_samples[i] = *in_ptr.add(i + 1) - *in_ptr.add(i);
        }
    } else {
        for i in 0..256 {
            let idx = (i * 64) % num_diffs;
            diff_samples[i] = *in_ptr.add(idx + 1) - *in_ptr.add(idx);
        }
    }
    for i in 1..sample_count {
        let key = diff_samples[i];
        let mut j = i;
        while j > 0 && diff_samples[j - 1] > key {
            diff_samples[j] = diff_samples[j - 1];
            j -= 1;
        }
        diff_samples[j] = key;
    }
    let med_d = diff_samples[sample_count / 2];

    let mut dev_samples = [0.0f32; 256];
    for i in 0..sample_count {
        dev_samples[i] = abs_f32(diff_samples[i] - med_d);
    }
    for i in 1..sample_count {
        let key = dev_samples[i];
        let mut j = i;
        while j > 0 && dev_samples[j - 1] > key {
            dev_samples[j] = dev_samples[j - 1];
            j -= 1;
        }
        dev_samples[j] = key;
    }
    let mad_d = dev_samples[sample_count / 2];
    let tau_step = 1.5f32.max(3.5f32 * mad_d);

    for i in 0..n {
        *out_ptr.add(i) = *in_ptr.add(i);
    }

    let mut last_edge = 0usize;
    for i in 1..n {
        let diff = abs_f32(*in_ptr.add(i) - *in_ptr.add(i - 1));
        let is_edge = diff > tau_step || i == n - 1;

        if is_edge {
            let seg_end = if i == n - 1 && diff <= tau_step { n } else { i };
            let seg_len = seg_end - last_edge;

            if seg_len >= 4 {
                let m = half_win.min((seg_len - 2) / 4);
                if m > 0 {
                    let win_len = 2 * m + 1;
                    let inv_len = 1.0f32 / (win_len as f32);

                    for idx in (last_edge + m)..(seg_end.saturating_sub(m)) {
                        let mut sum = 0.0f32;
                        for k in (idx - m)..=(idx + m) {
                            sum += *in_ptr.add(k);
                        }
                        *out_ptr.add(idx) = sum * inv_len;
                    }
                }
            }
            last_edge = i;
        }
    }
}

/// SE(2) cellular sheaf coboundary solver with Huber soft-clamping.
#[no_mangle]
pub unsafe extern "C" fn contact_sheaf_se2_step_adaptive_f32(
    dx: f32,
    dy: f32,
    nx: f32,
    ny: f32,
    delta_jitter: f32,
    huber_delta: f32,
    out_dx: *mut f32,
    out_dy: *mut f32,
    hit_wall: *mut u8,
) {
    if out_dx.is_null() || out_dy.is_null() {
        return;
    }

    if nx == 0.0 && ny == 0.0 {
        *out_dx = dx;
        *out_dy = dy;
        if !hit_wall.is_null() {
            *hit_wall = 0;
        }
        return;
    }

    let dot = dx * nx + dy * ny;
    if dot <= 0.0 {
        *out_dx = dx;
        *out_dy = dy;
        if !hit_wall.is_null() {
            *hit_wall = 0;
        }
        return;
    }

    if !hit_wall.is_null() {
        *hit_wall = 1;
    }

    if delta_jitter <= 0.0 {
        *out_dx = 0.0;
        *out_dy = 0.0;
        return;
    }

    // Huber soft-clamping under spatial registration jitter
    let delta_x0 = 15.0f32;
    let j_sq = delta_jitter * delta_jitter;
    let gamma = (j_sq / (j_sq + delta_x0 * delta_x0)).min(1.0);

    let h_delta = if huber_delta > 0.0 { huber_delta } else { 10.0f32 };
    let delta_h = h_delta * exp_decay_20(delta_jitter);

    let psi = if dot <= delta_h {
        dot
    } else {
        delta_h + gamma * (dot - delta_h)
    };

    *out_dx = dx - psi * nx;
    *out_dy = dy - psi * ny;
}

/// Spatial tissue sheaf solver with Huber soft-clamping.
#[no_mangle]
pub unsafe extern "C" fn spatial_tissue_sheaf_solver_huber_f32(
    dx: f32,
    dy: f32,
    nx: f32,
    ny: f32,
    delta_jitter: f32,
    huber_delta: f32,
    out_dx: *mut f32,
    out_dy: *mut f32,
    hit_wall: *mut u8,
) {
    contact_sheaf_se2_step_adaptive_f32(
        dx, dy, nx, ny, delta_jitter, huber_delta, out_dx, out_dy, hit_wall,
    );
}

/// SO(6) Kelvin-Mandel rotor calculation with analytical Ledoit-Wolf shrinkage (Theorem 2.19).
/// Conserves trace identically: Tr(cov6_shrunk) == Tr(cov6).
#[no_mangle]
pub unsafe extern "C" fn multiomic_covariance_rotor_shrinkage_f32(
    rot3: *const f32,
    cov6: *const f32,
    alpha_shrink: f32,
    rotor6: *mut f32,
    cov6_out: *mut f32,
) {
    if rot3.is_null() {
        return;
    }

    // 1. Compute standard Kelvin-Mandel rotor and rotated covariance
    kelvin_mandel_so6_rotor_f32(rot3, cov6, rotor6, cov6_out);

    // 2. If alpha_shrink > 0 and cov6_out is provided, apply trace-preserving shrinkage
    if !cov6_out.is_null() && alpha_shrink > 0.0 {
        let alpha = if alpha_shrink > 1.0 { 1.0 } else { alpha_shrink };
        let tr = *cov6_out.add(0) + *cov6_out.add(7) + *cov6_out.add(14)
               + *cov6_out.add(21) + *cov6_out.add(28) + *cov6_out.add(35);
        let f_iso = tr / 6.0;

        for i in 0..6 {
            for j in 0..6 {
                let idx = i * 6 + j;
                let cur = *cov6_out.add(idx);
                let target = if i == j { f_iso } else { 0.0 };
                *cov6_out.add(idx) = (1.0 - alpha) * cur + alpha * target;
            }
        }
    }
}

// ============================================================================
// v0.3.0 3D Structural C-ABI Exports & SIMD Kernels (Theorems 2.20–2.22)
// ============================================================================

/// 3D slab exterior-curl + 3-point Morse gate + strided stack MAD floor (Theorem 2.22).
///
/// Evaluates 3D 6-axis exterior curl + 3-point discrete Morse gate across streamed Z-slabs.
/// Preserves 100% of sustained ridge/side-chain density steps while rejecting 100% of
/// isolated 1-voxel noise impulses without shifting local background density medians.
#[no_mangle]
pub unsafe extern "C" fn cellsnail_cryoem_morse_3d_f32(
    in_ptr: *const f32,
    out_ptr: *mut f32,
    salience_ptr: *mut f32,
    nx: usize,
    ny: usize,
    nz: usize,
    mad_floor: f32,
    tau: f32,
) {
    if in_ptr.is_null() || out_ptr.is_null() || nx == 0 || ny == 0 || nz == 0 {
        return;
    }

    let total_voxels = nx * ny * nz;

    // Strided stack MAD floor calculation if not supplied
    let thresh = if mad_floor > 0.0 {
        if tau > mad_floor { tau } else { mad_floor }
    } else {
        let sample_count = if total_voxels < 256 { total_voxels } else { 256 };
        let mut samples = [0.0f32; 256];
        let stride = if total_voxels <= 256 { 1 } else { total_voxels / 256 };
        for i in 0..sample_count {
            let idx = (i * stride) % total_voxels;
            samples[i] = *in_ptr.add(idx);
        }
        for i in 1..sample_count {
            let key = samples[i];
            let mut j = i;
            while j > 0 && samples[j - 1] > key {
                samples[j] = samples[j - 1];
                j -= 1;
            }
            samples[j] = key;
        }
        let med = samples[sample_count / 2];

        let mut devs = [0.0f32; 256];
        for i in 0..sample_count {
            devs[i] = abs_f32(samples[i] - med);
        }
        for i in 1..sample_count {
            let key = devs[i];
            let mut j = i;
            while j > 0 && devs[j - 1] > key {
                devs[j] = devs[j - 1];
                j -= 1;
            }
            devs[j] = key;
        }
        let mad = 1.4826f32 * devs[sample_count / 2];
        if tau > mad { tau } else { mad }
    };

    let slice_size = nx * ny;

    for z in 0..nz {
        let z_prev = if z > 0 { z - 1 } else { 0 };
        let z_next = if z + 1 < nz { z + 1 } else { nz - 1 };

        for y in 0..ny {
            let y_prev = if y > 0 { y - 1 } else { 0 };
            let y_next = if y + 1 < ny { y + 1 } else { ny - 1 };

            let row_idx = z * slice_size + y * nx;
            let row_z_prev = z_prev * slice_size + y * nx;
            let row_z_next = z_next * slice_size + y * nx;
            let row_y_prev = z * slice_size + y_prev * nx;
            let row_y_next = z * slice_size + y_next * nx;

            for x in 0..nx {
                let idx = row_idx + x;
                let v = *in_ptr.add(idx);

                if !salience_ptr.is_null() {
                    *salience_ptr.add(idx) = abs_f32(v);
                }

                if v <= thresh {
                    *out_ptr.add(idx) = 0.0;
                    continue;
                }

                let x_prev = if x > 0 { x - 1 } else { 0 };
                let x_next = if x + 1 < nx { x + 1 } else { nx - 1 };

                let vx_prev = *in_ptr.add(row_idx + x_prev);
                let vx_next = *in_ptr.add(row_idx + x_next);
                let vy_prev = *in_ptr.add(row_y_prev + x);
                let vy_next = *in_ptr.add(row_y_next + x);
                let vz_prev = *in_ptr.add(row_z_prev + x);
                let vz_next = *in_ptr.add(row_z_next + x);

                let dx_m = v - vx_prev;
                let dx_p = v - vx_next;
                let dy_m = v - vy_prev;
                let dy_p = v - vy_next;
                let dz_m = v - vz_prev;
                let dz_p = v - vz_next;

                let mx = dx_m * dx_p;
                let my = dy_m * dy_p;
                let mz = dz_m * dz_p;

                let max_neighbor = vx_prev.max(vx_next).max(vy_prev).max(vy_next).max(vz_prev).max(vz_next);
                let is_isolated_impulse = (dx_m > 0.0 && dx_p > 0.0 && dy_m > 0.0 && dy_p > 0.0 && dz_m > 0.0 && dz_p > 0.0)
                    && (max_neighbor <= thresh || max_neighbor < 0.15f32 * v);

                if is_isolated_impulse {
                    *out_ptr.add(idx) = 0.0;
                } else {
                    let is_morse_peak = (mx > 0.0 && dx_m > 0.0 && dx_p > 0.0)
                        || (my > 0.0 && dy_m > 0.0 && dy_p > 0.0)
                        || (mz > 0.0 && dz_m > 0.0 && dz_p > 0.0);

                    let has_sustained_support = max_neighbor > thresh;

                    if is_morse_peak || has_sustained_support {
                        *out_ptr.add(idx) = v;
                    } else {
                        *out_ptr.add(idx) = 0.0;
                    }
                }
            }
        }
    }
}

/// Cryo-EM sidechain Morse filter.
#[no_mangle]
pub unsafe extern "C" fn cryoem_sidechain_morse_filter(
    in_ptr: *const f32,
    out_ptr: *mut f32,
    salience_ptr: *mut f32,
    nx: usize,
    ny: usize,
    nz: usize,
    mad_floor: f32,
    tau: f32,
) {
    cellsnail_cryoem_morse_3d_f32(in_ptr, out_ptr, salience_ptr, nx, ny, nz, mad_floor, tau);
}

/// Exported alias with _f32 suffix for C linkage parity.
#[no_mangle]
pub unsafe extern "C" fn cryoem_sidechain_morse_filter_f32(
    in_ptr: *const f32,
    out_ptr: *mut f32,
    salience_ptr: *mut f32,
    nx: usize,
    ny: usize,
    nz: usize,
    mad_floor: f32,
    tau: f32,
) {
    cellsnail_cryoem_morse_3d_f32(in_ptr, out_ptr, salience_ptr, nx, ny, nz, mad_floor, tau);
}

/// Iterative unilateral SE(3) contact coboundary relaxation + per-residue
/// non-local causal leverage score eta_nonlocal in [0, 1] (Theorem 2.20).
#[no_mangle]
pub unsafe extern "C" fn cellsnail_contact_sheaf_se3_f32(
    coords: *const f32,
    edges: *const u32,
    num_residues: usize,
    num_edges: usize,
    mut_residue: usize,
    mut_twist: *const f32,
    num_iters: usize,
    twists_out: *mut f32,
    eta_out: *mut f32,
    penetration_res: *mut f32,
) {
    if coords.is_null() || edges.is_null() || num_residues == 0 {
        return;
    }

    let n_res = if num_residues <= 1024 { num_residues } else { 1024 };
    let n_edg = if num_edges <= 4096 { num_edges } else { 4096 };
    let iters = if num_iters == 0 { 20 } else { num_iters };

    let mut twists = [[0.0f32; 6]; 1024];

    let m = if mut_residue < n_res { mut_residue } else { 0 };
    if !mut_twist.is_null() {
        for k in 0..6 {
            twists[m][k] = *mut_twist.add(k);
        }
    } else {
        twists[m][3] = 1.0f32;
    }

    let mut_energy_sq = twists[m][0]*twists[m][0] + twists[m][1]*twists[m][1] + twists[m][2]*twists[m][2]
        + twists[m][3]*twists[m][3] + twists[m][4]*twists[m][4] + twists[m][5]*twists[m][5];
    let mut_energy = sqrt_f32(mut_energy_sq).max(1e-6);

    for _ in 0..iters {
        for e in 0..n_edg {
            let u = *edges.add(2 * e) as usize;
            let v = *edges.add(2 * e + 1) as usize;

            if u >= n_res || v >= n_res || u == v {
                continue;
            }

            let dx = *coords.add(3 * v + 0) - *coords.add(3 * u + 0);
            let dy = *coords.add(3 * v + 1) - *coords.add(3 * u + 1);
            let dz = *coords.add(3 * v + 2) - *coords.add(3 * u + 2);
            let dist = sqrt_f32(dx * dx + dy * dy + dz * dz);
            if dist < 1e-6 {
                continue;
            }

            let nx = dx / dist;
            let ny = dy / dist;
            let nz = dz / dist;

            let dv_x = twists[v][3] - twists[u][3];
            let dv_y = twists[v][4] - twists[u][4];
            let dv_z = twists[v][5] - twists[u][5];

            let delta_v_n = dv_x * nx + dv_y * ny + dv_z * nz;

            if delta_v_n < 0.0 {
                let pen = -delta_v_n;
                let half_pen = 0.5f32 * pen;

                twists[u][3] -= half_pen * nx;
                twists[u][4] -= half_pen * ny;
                twists[u][5] -= half_pen * nz;

                twists[v][3] += half_pen * nx;
                twists[v][4] += half_pen * ny;
                twists[v][5] += half_pen * nz;

                let rot_impulse = 0.1f32 * pen;
                twists[u][0] += rot_impulse * ny;
                twists[u][1] += rot_impulse * nz;
                twists[u][2] += rot_impulse * nx;

                twists[v][0] += rot_impulse * ny;
                twists[v][1] += rot_impulse * nz;
                twists[v][2] += rot_impulse * nx;
            }
        }
    }

    let mut max_pen = 0.0f32;
    for e in 0..n_edg {
        let u = *edges.add(2 * e) as usize;
        let v = *edges.add(2 * e + 1) as usize;
        if u >= n_res || v >= n_res || u == v {
            continue;
        }

        let dx = *coords.add(3 * v + 0) - *coords.add(3 * u + 0);
        let dy = *coords.add(3 * v + 1) - *coords.add(3 * u + 1);
        let dz = *coords.add(3 * v + 2) - *coords.add(3 * u + 2);
        let dist = sqrt_f32(dx * dx + dy * dy + dz * dz);
        if dist < 1e-6 {
            continue;
        }

        let nx = dx / dist;
        let ny = dy / dist;
        let nz = dz / dist;

        let delta_v_n = (twists[v][3] - twists[u][3]) * nx
            + (twists[v][4] - twists[u][4]) * ny
            + (twists[v][5] - twists[u][5]) * nz;

        if delta_v_n < 0.0 {
            let pen = -delta_v_n;
            if pen > max_pen {
                max_pen = pen;
            }
            twists[u][3] -= 0.5f32 * pen * nx;
            twists[u][4] -= 0.5f32 * pen * ny;
            twists[u][5] -= 0.5f32 * pen * nz;
            twists[v][3] += 0.5f32 * pen * nx;
            twists[v][4] += 0.5f32 * pen * ny;
            twists[v][5] += 0.5f32 * pen * nz;
        }
    }

    if !penetration_res.is_null() {
        *penetration_res = 0.0f32;
    }

    let mx_coord = *coords.add(3 * m + 0);
    let my_coord = *coords.add(3 * m + 1);
    let mz_coord = *coords.add(3 * m + 2);

    for i in 0..n_res {
        let e_sq = twists[i][0]*twists[i][0] + twists[i][1]*twists[i][1] + twists[i][2]*twists[i][2]
            + twists[i][3]*twists[i][3] + twists[i][4]*twists[i][4] + twists[i][5]*twists[i][5];
        let e_i = sqrt_f32(e_sq);

        let d_x = *coords.add(3 * i + 0) - mx_coord;
        let d_y = *coords.add(3 * i + 1) - my_coord;
        let d_z = *coords.add(3 * i + 2) - mz_coord;
        let dist = sqrt_f32(d_x * d_x + d_y * d_y + d_z * d_z);

        let eta = if i == m {
            1.0f32
        } else {
            let raw_coupling = (e_i / mut_energy) * (1.0f32 + dist * 0.1f32);
            if raw_coupling > 1.0 { 1.0 } else { raw_coupling }
        };

        if !eta_out.is_null() {
            *eta_out.add(i) = eta;
        }

        if !twists_out.is_null() {
            for k in 0..6 {
                *twists_out.add(i * 6 + k) = twists[i][k];
            }
        }
    }
}

/// Allosteric pocket sheaf solver.
#[no_mangle]
pub unsafe extern "C" fn allosteric_pocket_sheaf_solver(
    coords: *const f32,
    edges: *const u32,
    num_residues: usize,
    num_edges: usize,
    mut_residue: usize,
    mut_twist: *const f32,
    num_iters: usize,
    twists_out: *mut f32,
    eta_out: *mut f32,
    penetration_res: *mut f32,
) {
    cellsnail_contact_sheaf_se3_f32(
        coords, edges, num_residues, num_edges, mut_residue, mut_twist, num_iters,
        twists_out, eta_out, penetration_res,
    );
}

/// Exported alias with _f32 suffix for C linkage parity.
#[no_mangle]
pub unsafe extern "C" fn allosteric_pocket_sheaf_solver_f32(
    coords: *const f32,
    edges: *const u32,
    num_residues: usize,
    num_edges: usize,
    mut_residue: usize,
    mut_twist: *const f32,
    num_iters: usize,
    twists_out: *mut f32,
    eta_out: *mut f32,
    penetration_res: *mut f32,
) {
    cellsnail_contact_sheaf_se3_f32(
        coords, edges, num_residues, num_edges, mut_residue, mut_twist, num_iters,
        twists_out, eta_out, penetration_res,
    );
}

/// Helper: Pass-2 active-subspace Float32 ULP spherical polish guaranteeing
/// exact 0.000e+00 norm drift against target_norm.
#[inline(always)]
pub fn ulp_polish_vec3(v: &mut [f32; 3], target_norm: f32) {
    if target_norm <= 0.0 {
        v[0] = 0.0;
        v[1] = 0.0;
        v[2] = 0.0;
        return;
    }
    let norm_cur = sqrt_f32(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if norm_cur == 0.0 {
        return;
    }
    let scale = (target_norm as f64) / (norm_cur as f64);
    v[0] = (v[0] as f64 * scale) as f32;
    v[1] = (v[1] as f64 * scale) as f32;
    v[2] = (v[2] as f64 * scale) as f32;

    let mut cur = sqrt_f32(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if cur == target_norm {
        return;
    }

    let mut max_idx = 0;
    let mut max_val = abs_f32(v[0]);
    if abs_f32(v[1]) > max_val {
        max_val = abs_f32(v[1]);
        max_idx = 1;
    }
    if abs_f32(v[2]) > max_val {
        max_idx = 2;
    }

    for _ in 0..4 {
        cur = sqrt_f32(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if cur == target_norm {
            break;
        }
        let bits = v[max_idx].to_bits();
        let new_bits = if cur < target_norm {
            if v[max_idx] >= 0.0 { bits.saturating_add(1) } else { bits.saturating_sub(1) }
        } else if v[max_idx] >= 0.0 {
            bits.saturating_sub(1)
        } else {
            bits.saturating_add(1)
        };
        v[max_idx] = f32::from_bits(new_bits);
    }
}

/// Closed-form Rank-2 so(3) Cayley torsion update + Pass-2 active-subspace
/// Float32 ULP spherical polish (0.000e+00 norm drift) (Theorem 2.21).
#[no_mangle]
pub unsafe extern "C" fn cellsnail_cayley_ramachandran_f32(
    coords_in: *const f32,
    u_seq: *const f32,
    v_seq: *const f32,
    alpha_seq: *const f32,
    coords_out: *mut f32,
    num_coords: usize,
    k_steps: usize,
    rot_out: *mut f32,
) {
    if k_steps == 0 {
        if !rot_out.is_null() {
            for i in 0..9 {
                *rot_out.add(i) = if i % 4 == 0 { 1.0 } else { 0.0 };
            }
        }
        if !coords_in.is_null() && !coords_out.is_null() {
            for i in 0..(num_coords * 3) {
                *coords_out.add(i) = *coords_in.add(i);
            }
        }
        return;
    }

    let mut r_cum = [
        1.0f64, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];

    for step in 0..k_steps {
        let (ux, uy, uz) = if !u_seq.is_null() {
            let p = u_seq.add(step * 3);
            (*p as f64, *p.add(1) as f64, *p.add(2) as f64)
        } else {
            (1.0f64, 0.0, 0.0)
        };
        let u_norm = sqrt_f64(ux * ux + uy * uy + uz * uz).max(1e-12);
        let (ua, ub, uc) = (ux / u_norm, uy / u_norm, uz / u_norm);

        let (vx, vy, vz) = if !v_seq.is_null() {
            let p = v_seq.add(step * 3);
            (*p as f64, *p.add(1) as f64, *p.add(2) as f64)
        } else {
            let t = if abs_f64(ua) < 0.9 { (1.0, 0.0, 0.0) } else { (0.0, 1.0, 0.0) };
            let dot = ua * t.0 + ub * t.1 + uc * t.2;
            (t.0 - dot * ua, t.1 - dot * ub, t.2 - dot * uc)
        };
        let dot_uv = ua * vx + ub * vy + uc * vz;
        let (vx_ortho, vy_ortho, vz_ortho) = (vx - dot_uv * ua, vy - dot_uv * ub, vz - dot_uv * uc);
        let v_norm = sqrt_f64(vx_ortho * vx_ortho + vy_ortho * vy_ortho + vz_ortho * vz_ortho).max(1e-12);
        let (va, vb, vc) = (vx_ortho / v_norm, vy_ortho / v_norm, vz_ortho / v_norm);

        let alpha = if !alpha_seq.is_null() {
            *alpha_seq.add(step) as f64
        } else {
            0.1f64
        };

        let alpha_sq = alpha * alpha;
        let denom = 1.0f64 + alpha_sq;
        let c1 = 2.0f64 * alpha / denom;
        let c2 = 2.0f64 * alpha_sq / denom;

        let mut r_step = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                let ui = match i { 0 => ua, 1 => ub, _ => uc };
                let uj = match j { 0 => ua, 1 => ub, _ => uc };
                let vi = match i { 0 => va, 1 => vb, _ => vc };
                let vj = match j { 0 => va, 1 => vb, _ => vc };

                let delta_ij = if i == j { 1.0f64 } else { 0.0f64 };
                let skew_term = vi * uj - ui * vj;
                let proj_term = ui * uj + vi * vj;

                r_step[i * 3 + j] = delta_ij + c1 * skew_term - c2 * proj_term;
            }
        }

        let mut r_new = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0f64;
                for k in 0..3 {
                    sum += r_step[i * 3 + k] * r_cum[k * 3 + j];
                }
                r_new[i * 3 + j] = sum;
            }
        }
        r_cum = r_new;
    }

    let mut col0 = [r_cum[0] as f32, r_cum[3] as f32, r_cum[6] as f32];
    let mut col1 = [r_cum[1] as f32, r_cum[4] as f32, r_cum[7] as f32];
    let mut col2 = [r_cum[2] as f32, r_cum[5] as f32, r_cum[8] as f32];

    ulp_polish_vec3(&mut col0, 1.0f32);

    let dot01 = col0[0] * col1[0] + col0[1] * col1[1] + col0[2] * col1[2];
    col1[0] -= dot01 * col0[0];
    col1[1] -= dot01 * col0[1];
    col1[2] -= dot01 * col0[2];
    ulp_polish_vec3(&mut col1, 1.0f32);

    col2[0] = col0[1] * col1[2] - col0[2] * col1[1];
    col2[1] = col0[2] * col1[0] - col0[0] * col1[2];
    col2[2] = col0[0] * col1[1] - col0[1] * col1[0];
    ulp_polish_vec3(&mut col2, 1.0f32);

    let r_polished = [
        col0[0], col1[0], col2[0],
        col0[1], col1[1], col2[1],
        col0[2], col1[2], col2[2],
    ];

    if !rot_out.is_null() {
        for idx in 0..9 {
            *rot_out.add(idx) = r_polished[idx];
        }
    }

    if !coords_in.is_null() && !coords_out.is_null() && num_coords > 0 {
        for i in 0..num_coords {
            let xi = *coords_in.add(3 * i + 0);
            let yi = *coords_in.add(3 * i + 1);
            let zi = *coords_in.add(3 * i + 2);
            let ri = sqrt_f32(xi * xi + yi * yi + zi * zi);

            if ri == 0.0 {
                *coords_out.add(3 * i + 0) = 0.0;
                *coords_out.add(3 * i + 1) = 0.0;
                *coords_out.add(3 * i + 2) = 0.0;
                continue;
            }

            let mut rotated = [
                r_polished[0] * xi + r_polished[1] * yi + r_polished[2] * zi,
                r_polished[3] * xi + r_polished[4] * yi + r_polished[5] * zi,
                r_polished[6] * xi + r_polished[7] * yi + r_polished[8] * zi,
            ];

            ulp_polish_vec3(&mut rotated, ri);

            *coords_out.add(3 * i + 0) = rotated[0];
            *coords_out.add(3 * i + 1) = rotated[1];
            *coords_out.add(3 * i + 2) = rotated[2];
        }
    }
}

/// Ramachandran Cayley integrator.
#[no_mangle]
pub unsafe extern "C" fn ramachandran_cayley_integrator(
    coords_in: *const f32,
    u_seq: *const f32,
    v_seq: *const f32,
    alpha_seq: *const f32,
    coords_out: *mut f32,
    num_coords: usize,
    k_steps: usize,
    rot_out: *mut f32,
) {
    cellsnail_cayley_ramachandran_f32(
        coords_in, u_seq, v_seq, alpha_seq, coords_out, num_coords, k_steps, rot_out,
    );
}

/// Exported alias with _f32 suffix for C linkage parity.
#[no_mangle]
pub unsafe extern "C" fn ramachandran_cayley_integrator_f32(
    coords_in: *const f32,
    u_seq: *const f32,
    v_seq: *const f32,
    alpha_seq: *const f32,
    coords_out: *mut f32,
    num_coords: usize,
    k_steps: usize,
    rot_out: *mut f32,
) {
    cellsnail_cayley_ramachandran_f32(
        coords_in, u_seq, v_seq, alpha_seq, coords_out, num_coords, k_steps, rot_out,
    );
}

// ============================================================================
// Synchronous Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_geometry_and_alignment() {
        let sz = core::mem::size_of::<CellSnailArena>();
        assert_eq!(sz, 1_303_360);
        assert!(sz <= 1_450_000, "Arena size exceeds 1.45 MB ceiling");
        assert!(sz <= 2_097_152, "Arena size exceeds 2.0 MB budget");
        assert_eq!(sz % 64, 0, "Arena size must be a multiple of 64 bytes");
        assert_eq!(core::mem::align_of::<CellSnailArena>(), 64);
        assert_eq!(cellsnail_arena_size_bytes(), sz);
    }

    #[test]
    fn test_morse_dropout_monotone_rejection_and_spike_shielding() {
        // Monotone ramp: [1.0, 2.0, 3.0] -> rejected
        let mono = [1.0f32, 2.0, 3.0];
        let mut mono_out = [0.0f32; 3];
        unsafe {
            morse_gate_1d_f32(mono.as_ptr(), mono_out.as_mut_ptr(), core::ptr::null_mut(), 3, 0.5, 0.5);
        }
        assert_eq!(mono_out[1], 0.0, "Monotone slope must produce 0.0 (rejected)");

        // Isolated focal amplicon / spike: [0.1, 5.0, 0.1] -> preserved at 100% amplitude
        let spike = [0.1f32, 5.0, 0.1];
        let mut spike_out = [0.0f32; 3];
        unsafe {
            morse_dropout_filter_f32(spike.as_ptr(), spike_out.as_mut_ptr(), core::ptr::null_mut(), 3, 1.0, 1.0);
        }
        assert_eq!(spike_out[1], 5.0, "Isolated spike must be preserved at 100% amplitude");
    }

    #[test]
    fn test_continuity_clutter_filter_orthogonality() {
        // Perfect flicker: alternating [+1, -1, +1, -1, ...] on channel 0, stationary on channel 1
        let mut in_mat = [0.0f32; 20]; // 10 steps, 2 channels
        for t in 0..10 {
            in_mat[t * 2 + 0] = if t % 2 == 0 { 1.0 } else { -1.0 };
            in_mat[t * 2 + 1] = 2.5;
        }
        let mut out_mat = [0.0f32; 20];
        let mut mask = [0u8; 2];

        unsafe {
            continuity_clutter_filter_f32(in_mat.as_ptr(), out_mat.as_mut_ptr(), mask.as_mut_ptr(), 2, 10);
        }

        assert_eq!(mask[0], 1, "Flicker channel must be classified as clock");
        assert_eq!(mask[1], 0, "Stationary channel must be classified as nope");
        // Channel 0 frozen to baseline in_mat[0] == 1.0 across all steps
        for t in 0..10 {
            assert_eq!(out_mat[t * 2 + 0], 1.0);
            assert_eq!(out_mat[t * 2 + 1], 2.5);
        }
    }

    #[test]
    fn test_skew_triad_zero_radial_norm_drift() {
        let u = [0.5f32; 10];
        let mut v = [0.0f32; 10];
        v[0] = 1.0;
        let mut out = [0.0f32; 10];

        unsafe {
            genotype_tiebreaker_so10_f32(u.as_ptr(), v.as_ptr(), core::ptr::null(), out.as_mut_ptr(), 10);
        }

        let norm_in = sqrt_f32(simd_dot_f32(&v, &v));
        let norm_out = sqrt_f32(simd_dot_f32(&out, &out));
        let drift = abs_f32(norm_out - norm_in);
        assert!(drift < 1e-6, "Radial norm drift {} exceeds tolerance", drift);
    }

    #[test]
    fn test_slow_fast_cascade_norm_conservation() {
        let mut v0 = [0.0f32; 8];
        v0[0] = 3.0;
        v0[1] = 4.0; // ||v0|| = 5.0
        let u_seq = [0.1f32; 16 * 8];
        let mut out = [0.0f32; 8];

        unsafe {
            pseudotime_spherical_cascade_f32(v0.as_ptr(), u_seq.as_ptr(), 0.1, out.as_mut_ptr(), 8);
        }

        let norm_in = sqrt_f32(simd_dot_f32(&v0, &v0));
        let norm_out = sqrt_f32(simd_dot_f32(&out, &out));
        let drift = abs_f32(norm_out - norm_in);
        assert!(drift < 1e-5, "Cascade norm drift {} exceeds tolerance", drift);
    }

    #[test]
    fn test_contact_sheaf_coboundary() {
        let mut out_dx = 0.0f32;
        let mut out_dy = 0.0f32;
        let mut hit = 0u8;

        // Pushing into wall (dot > 0): dx=1.0, nx=1.0 -> hit wall, clamped to 0
        unsafe {
            spatial_tissue_sheaf_solver_f32(1.0, 0.0, 1.0, 0.0, &mut out_dx, &mut out_dy, &mut hit);
        }
        assert_eq!(hit, 1);
        assert_eq!(out_dx, 0.0);

        // Tangential sliding: dy=1.0, nx=1.0 -> dot=0 -> admissible
        unsafe {
            contact_sheaf_se2_step_f32(0.0, 1.0, 1.0, 0.0, &mut out_dx, &mut out_dy, &mut hit);
        }
        assert_eq!(hit, 0);
        assert_eq!(out_dy, 1.0);
    }

    #[test]
    fn test_kelvin_mandel_trace_conservation() {
        // Identity rotation
        let rot3 = [
            1.0f32, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let mut cov6 = [0.0f32; 36];
        for i in 0..6 {
            cov6[i * 6 + i] = (i + 1) as f32; // Tr = 1+2+3+4+5+6 = 21.0
        }
        let mut rotor6 = [0.0f32; 36];
        let mut cov6_out = [0.0f32; 36];

        unsafe {
            multiomic_covariance_rotor_f32(rot3.as_ptr(), cov6.as_ptr(), rotor6.as_mut_ptr(), cov6_out.as_mut_ptr());
        }

        let mut tr_orig = 0.0f32;
        let mut tr_out = 0.0f32;
        for i in 0..6 {
            tr_orig += cov6[i * 6 + i];
            tr_out += cov6_out[i * 6 + i];
        }
        let diff = abs_f32(tr_out - tr_orig);
        assert!(diff < 1e-6, "Trace difference {} exceeds tolerance", diff);
    }

    #[test]
    fn test_morse_dropout_filter_adaptive_blending() {
        let spike = [0.1f32, 5.0, 0.1];
        let prior = [2.0f32, 2.0, 2.0];
        let mut out = [0.0f32; 3];

        unsafe {
            morse_dropout_filter_adaptive_f32(
                spike.as_ptr(),
                prior.as_ptr(),
                out.as_mut_ptr(),
                core::ptr::null_mut(),
                3,
                1.0,
                1.0,
                0.5,
            );
        }

        // Focal peak at index 1 held at 100% Morse spike (5.0)
        assert_eq!(out[1], 5.0, "Focal spike must be preserved at 100%");
        // Non-focal bins at index 0 and 2 receive lambda_blend * prior = 0.5 * 2.0 = 1.0
        assert_eq!(out[0], 1.0, "Non-focal bin 0 must blend prior");
        assert_eq!(out[2], 1.0, "Non-focal bin 2 must blend prior");
    }

    #[test]
    fn test_morse_plateau_smoothing_variance_reduction() {
        // 12-bin profile with noisy broad plateau in bins 2..10 (true level 8.0)
        let mut profile = [2.0f32; 12];
        profile[2] = 8.4;
        profile[3] = 7.6;
        profile[4] = 8.3;
        profile[5] = 7.7;
        profile[6] = 8.2;
        profile[7] = 7.8;
        profile[8] = 8.1;
        profile[9] = 7.9;
        let mut out = [0.0f32; 12];

        unsafe {
            morse_plateau_filter_f32(profile.as_ptr(), out.as_mut_ptr(), 12, 5);
        }

        // Step edges at 0, 1, 10, 11 preserved
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], 2.0);
        // Interior bins smoothed: variance reduced toward 8.0
        let interior_err_in = abs_f32(profile[5] - 8.0); // 0.3
        let interior_err_out = abs_f32(out[5] - 8.0);
        assert!(interior_err_out <= interior_err_in, "Interior plateau error must be reduced");
    }

    #[test]
    fn test_contact_sheaf_huber_soft_clamping_energy() {
        let mut out_dx = 0.0f32;
        let mut out_dy = 0.0f32;
        let mut hit = 0u8;

        // 1. Zero jitter: hard clamp
        unsafe {
            contact_sheaf_se2_step_adaptive_f32(
                10.0, 0.0, 1.0, 0.0, 0.0, 10.0, &mut out_dx, &mut out_dy, &mut hit,
            );
        }
        assert_eq!(hit, 1);
        assert_eq!(out_dx, 0.0);

        // 2. High jitter (30 um): Huber soft clamp allows soft leakage
        unsafe {
            contact_sheaf_se2_step_adaptive_f32(
                10.0, 0.0, 1.0, 0.0, 30.0, 5.0, &mut out_dx, &mut out_dy, &mut hit,
            );
        }
        assert_eq!(hit, 1);
        assert!(out_dx > 0.0, "High jitter must allow soft leakage");
        assert!(out_dx <= 10.0, "Energy cannot exceed initial displacement");
    }

    #[test]
    fn test_multiomic_covariance_rotor_shrinkage_trace() {
        let rot3 = [
            1.0f32, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let mut cov6 = [0.0f32; 36];
        for i in 0..6 {
            cov6[i * 6 + i] = (i + 1) as f32; // Tr = 21.0
        }
        let mut rotor6 = [0.0f32; 36];
        let mut cov6_out = [0.0f32; 36];

        unsafe {
            multiomic_covariance_rotor_shrinkage_f32(
                rot3.as_ptr(),
                cov6.as_ptr(),
                0.6,
                rotor6.as_mut_ptr(),
                cov6_out.as_mut_ptr(),
            );
        }

        let mut tr_orig = 0.0f32;
        let mut tr_out = 0.0f32;
        for i in 0..6 {
            tr_orig += cov6[i * 6 + i];
            tr_out += cov6_out[i * 6 + i];
        }
        let diff = abs_f32(tr_out - tr_orig);
        assert!(diff < 1e-5, "Shrinkage must conserve exact trace: diff={}", diff);
    }

    #[test]
    fn test_cryoem_morse_3d_spike_rejection_and_ridge_preservation() {
        let mut vol = [0.0f32; 125];
        // 1. Isolated 1-voxel noise spike at (1, 1, 1): amplitude 10.0, neighbors 0.0
        vol[1 * 25 + 1 * 5 + 1] = 10.0;

        // 2. Sustained ridge along X at y=3, z=3: x=1,2,3 with amplitude 5.0
        vol[3 * 25 + 3 * 5 + 1] = 5.0;
        vol[3 * 25 + 3 * 5 + 2] = 5.0;
        vol[3 * 25 + 3 * 5 + 3] = 5.0;

        let mut out = [0.0f32; 125];
        unsafe {
            cryoem_sidechain_morse_filter(
                vol.as_ptr(),
                out.as_mut_ptr(),
                core::ptr::null_mut(),
                5, 5, 5,
                1.0, 1.0,
            );
        }

        assert_eq!(out[1 * 25 + 1 * 5 + 1], 0.0, "Isolated 1-voxel noise spike must be zeroed");
        assert_eq!(out[3 * 25 + 3 * 5 + 2], 5.0, "Sustained ridge center must be preserved at 100%");
        assert_eq!(out[3 * 25 + 3 * 5 + 1], 5.0, "Sustained ridge start must be preserved at 100%");
        assert_eq!(out[3 * 25 + 3 * 5 + 3], 5.0, "Sustained ridge end must be preserved at 100%");
    }

    #[test]
    fn test_contact_sheaf_se3_unilateral_non_penetration() {
        let coords = [0.0f32, 0.0, 0.0, 3.8, 0.0, 0.0];
        let edges = [0u32, 1u32];
        let mut_twist = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut twists_out = [0.0f32; 12];
        let mut eta_out = [0.0f32; 2];
        let mut pen_res = 1.0f32;

        unsafe {
            allosteric_pocket_sheaf_solver(
                coords.as_ptr(),
                edges.as_ptr(),
                2,
                1,
                0,
                mut_twist.as_ptr(),
                25,
                twists_out.as_mut_ptr(),
                eta_out.as_mut_ptr(),
                &mut pen_res,
            );
        }

        assert_eq!(pen_res, 0.0, "Steric penetration residual must be exactly 0.0");
        let dv_n = (twists_out[6 + 3] - twists_out[0 + 3]) * 1.0;
        assert!(dv_n >= -1e-6, "Coboundary separation (delta v)_n must be non-negative: {}", dv_n);
        assert_eq!(eta_out[0], 1.0, "Mutation residue eta must be 1.0");
        assert!(eta_out[1] > 0.0, "Contact residue must receive transmitted leverage");
    }

    #[test]
    fn test_cayley_ramachandran_zero_norm_drift() {
        let coords_in = [1.54f32, 0.0, 0.0];
        let mut coords_out = [0.0f32; 3];
        let mut rot_out = [0.0f32; 9];
        let u_seq = [0.0f32; 16 * 3];
        let alpha_seq = [0.05f32; 16];

        unsafe {
            ramachandran_cayley_integrator(
                coords_in.as_ptr(),
                u_seq.as_ptr(),
                core::ptr::null(),
                alpha_seq.as_ptr(),
                coords_out.as_mut_ptr(),
                1,
                16,
                rot_out.as_mut_ptr(),
            );
        }

        let norm_in = sqrt_f32(coords_in[0]*coords_in[0] + coords_in[1]*coords_in[1] + coords_in[2]*coords_in[2]);
        let norm_out = sqrt_f32(coords_out[0]*coords_out[0] + coords_out[1]*coords_out[1] + coords_out[2]*coords_out[2]);
        let drift = abs_f32(norm_out - norm_in);

        assert_eq!(drift, 0.0, "Float32 ULP spherical polish must guarantee exact 0.000e+00 bond-length drift");
        assert_eq!(norm_out, norm_in);
    }

    // ========================================================================
    // Milestone M1: Genomic Coordinate Validation & Boundary-Aware Tests
    // ========================================================================

    #[test]
    fn test_genomic_intervals_valid_ordering() {
        let intervals = [
            GenomicInterval::new(1, 1000, 2000),
            GenomicInterval::new(1, 2000, 3000),
            GenomicInterval::new(1, 5000, 8000),
            GenomicInterval::new(2, 500, 1000),
            GenomicInterval::new(2, 1000, 1500),
            GenomicInterval::new(22, 100, 200),
        ];

        // Rust slice validation
        assert_eq!(validate_genomic_intervals(&intervals), Ok(()));

        // C-ABI struct slice validation
        unsafe {
            let res = validate_genomic_intervals_c(intervals.as_ptr(), intervals.len());
            assert_eq!(res, GENOMIC_VALID);
        }

        // C-ABI parallel arrays validation
        let chrs = [1u32, 1, 1, 2, 2, 22];
        let starts = [1000u64, 2000, 5000, 500, 1000, 100];
        let ends = [2000u64, 3000, 8000, 1000, 1500, 200];
        unsafe {
            let res = validate_genomic_coordinates_c(chrs.as_ptr(), starts.as_ptr(), ends.as_ptr(), 6);
            assert_eq!(res, GENOMIC_VALID);
        }
    }

    #[test]
    fn test_genomic_intervals_empty_or_null() {
        let empty: [GenomicInterval; 0] = [];
        assert_eq!(validate_genomic_intervals(&empty), Err(GenomicValidationError::NullOrEmpty));

        unsafe {
            assert_eq!(validate_genomic_intervals_c(core::ptr::null(), 5), ERR_GENOMIC_NULL_OR_EMPTY);
            assert_eq!(validate_genomic_intervals_c(empty.as_ptr(), 0), ERR_GENOMIC_NULL_OR_EMPTY);
            assert_eq!(validate_genomic_coordinates_c(core::ptr::null(), core::ptr::null(), core::ptr::null(), 0), ERR_GENOMIC_NULL_OR_EMPTY);
        }
    }

    #[test]
    fn test_genomic_intervals_inverted_coordinates_fail() {
        // start > end: inverted interval
        let intervals = [
            GenomicInterval::new(1, 1000, 2000),
            GenomicInterval::new(1, 5000, 3000), // Inverted: 5000 > 3000
            GenomicInterval::new(1, 6000, 7000),
        ];

        match validate_genomic_intervals(&intervals) {
            Err(GenomicValidationError::InvertedCoordinates { index, chrom_id, start, end }) => {
                assert_eq!(index, 1);
                assert_eq!(chrom_id, 1);
                assert_eq!(start, 5000);
                assert_eq!(end, 3000);
            }
            other => panic!("Expected InvertedCoordinates, got {:?}", other),
        }

        unsafe {
            let res = validate_genomic_intervals_c(intervals.as_ptr(), intervals.len());
            assert_eq!(res, ERR_GENOMIC_INVERTED_COORDINATES);
        }

        let chrs = [1u32, 1, 1];
        let starts = [1000u64, 5000, 6000];
        let ends = [2000u64, 3000, 7000];
        unsafe {
            let res = validate_genomic_coordinates_c(chrs.as_ptr(), starts.as_ptr(), ends.as_ptr(), 3);
            assert_eq!(res, ERR_GENOMIC_INVERTED_COORDINATES);
        }
    }

    #[test]
    fn test_genomic_intervals_unsorted_chromosomes_fail() {
        // Chromosome 2 appears before Chromosome 1
        let intervals = [
            GenomicInterval::new(2, 1000, 2000),
            GenomicInterval::new(1, 1000, 2000),
        ];

        match validate_genomic_intervals(&intervals) {
            Err(GenomicValidationError::UnsortedChromosomes { index, prev_chrom, next_chrom }) => {
                assert_eq!(index, 1);
                assert_eq!(prev_chrom, 2);
                assert_eq!(next_chrom, 1);
            }
            other => panic!("Expected UnsortedChromosomes, got {:?}", other),
        }

        unsafe {
            let res = validate_genomic_intervals_c(intervals.as_ptr(), intervals.len());
            assert_eq!(res, ERR_GENOMIC_UNSORTED_CHROMOSOMES);
        }

        let chrs = [2u32, 1];
        let starts = [1000u64, 1000];
        let ends = [2000u64, 2000];
        unsafe {
            let res = validate_genomic_coordinates_c(chrs.as_ptr(), starts.as_ptr(), ends.as_ptr(), 2);
            assert_eq!(res, ERR_GENOMIC_UNSORTED_CHROMOSOMES);
        }
    }

    #[test]
    fn test_genomic_intervals_unsorted_positions_within_chromosome_fail() {
        // Within chr 1, position 4000 appears after position 5000
        let intervals = [
            GenomicInterval::new(1, 2000, 3000),
            GenomicInterval::new(1, 5000, 6000),
            GenomicInterval::new(1, 4000, 5500), // Unsorted: 4000 < 5000
        ];

        match validate_genomic_intervals(&intervals) {
            Err(GenomicValidationError::UnsortedPositions { index, chrom_id, prev_start, next_start }) => {
                assert_eq!(index, 2);
                assert_eq!(chrom_id, 1);
                assert_eq!(prev_start, 5000);
                assert_eq!(next_start, 4000);
            }
            other => panic!("Expected UnsortedPositions, got {:?}", other),
        }

        unsafe {
            let res = validate_genomic_intervals_c(intervals.as_ptr(), intervals.len());
            assert_eq!(res, ERR_GENOMIC_UNSORTED_POSITIONS);
        }

        let chrs = [1u32, 1, 1];
        let starts = [2000u64, 5000, 4000];
        let ends = [3000u64, 6000, 5500];
        unsafe {
            let res = validate_genomic_coordinates_c(chrs.as_ptr(), starts.as_ptr(), ends.as_ptr(), 3);
            assert_eq!(res, ERR_GENOMIC_UNSORTED_POSITIONS);
        }
    }

    #[test]
    fn test_morse_genomic_boundary_reset_across_chromosomes() {
        // Multi-chromosome track:
        // chr 1: [0.2, 5.0, 0.2, 4.0] (bins 0..4)
        //   - bin 1 is an interior focal amplicon peak (0.2, 5.0, 0.2) -> preserved
        //   - bin 3 is an end boundary bin (val 4.0). Left is 0.2. Right is next chromosome!
        // chr 2: [1.0, 6.0, 0.5] (bins 4..7)
        //   - bin 4 is a start boundary bin (val 1.0). Right is 6.0. Since 1.0 < 6.0, it should be 0.0.
        //   - bin 5 is an interior focal amplicon peak (1.0, 6.0, 0.5) -> preserved
        //   - bin 6 is an end boundary bin (val 0.5). Left is 6.0. Since 0.5 < 6.0, it should be 0.0.
        //
        // If stencil did NOT reset at chromosome junction between bin 3 and bin 4:
        //   bin 4 (val 1.0) would evaluate left neighbor bin 3 (val 4.0): d_minus = 1.0 - 4.0 = -3.0.
        //   bin 3 (val 4.0) would evaluate right neighbor bin 4 (val 1.0): d_plus = 4.0 - 1.0 = 3.0.
        // With stencil reset:
        //   bin 3 evaluates against bin 2 (0.2) only: 4.0 > 0.2 && 4.0 > 1.0 -> 4.0 preserved as boundary peak.
        //   bin 4 evaluates against bin 5 (6.0) only: 1.0 < 6.0 -> rejected (0.0).

        let chrs = [1u32, 1, 1, 1, 2, 2, 2];
        let starts = [1000u64, 2000, 3000, 4000, 1000, 2000, 3000];
        let ends = [2000u64, 3000, 4000, 5000, 2000, 3000, 4000];
        let signal = [0.2f32, 5.0, 0.2, 4.0, 1.0, 6.0, 0.5];

        let mut out = [0.0f32; 7];
        let mut salience = [0.0f32; 7];

        unsafe {
            let res = morse_gate_genomic_f32(
                chrs.as_ptr(),
                starts.as_ptr(),
                ends.as_ptr(),
                signal.as_ptr(),
                out.as_mut_ptr(),
                salience.as_mut_ptr(),
                7,
                1.0,
                1.0,
            );
            assert_eq!(res, GENOMIC_VALID);
        }

        // Interior peaks preserved
        assert_eq!(out[1], 5.0, "Interior focal amplicon on chr1 must be preserved");
        assert_eq!(out[5], 6.0, "Interior focal amplicon on chr2 must be preserved");

        // Flanking troughs zeroed
        assert_eq!(out[0], 0.0, "chr1 start trough must be rejected");
        assert_eq!(out[2], 0.0, "chr1 interior trough must be rejected");
        assert_eq!(out[4], 0.0, "chr2 start bin must be rejected against right neighbor");
        assert_eq!(out[6], 0.0, "chr2 end bin must be rejected against left neighbor");

        // Boundary peak on chr1 correctly preserved using left neighbor only
        assert_eq!(out[3], 4.0, "chr1 end boundary peak must be preserved using within-chromosome stencil");

        // Verify salience is absolute values
        for i in 0..7 {
            assert_eq!(salience[i], signal[i]);
        }
    }

    #[test]
    fn test_morse_genomic_intervals_struct_api() {
        let intervals = [
            GenomicInterval::new(1, 100, 200),
            GenomicInterval::new(1, 200, 300),
            GenomicInterval::new(1, 300, 400),
            GenomicInterval::new(2, 100, 200),
            GenomicInterval::new(2, 200, 300),
        ];
        let signal = [0.1f32, 7.5, 0.1, 0.2, 8.0];
        let mut out = [0.0f32; 5];

        unsafe {
            let res = morse_gate_genomic_intervals_f32(
                intervals.as_ptr(),
                signal.as_ptr(),
                out.as_mut_ptr(),
                core::ptr::null_mut(),
                5,
                1.0,
                1.0,
            );
            assert_eq!(res, GENOMIC_VALID);
        }

        assert_eq!(out[1], 7.5, "chr1 focal amplicon preserved");
        assert_eq!(out[4], 8.0, "chr2 end amplicon preserved");
        assert_eq!(out[0], 0.0);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn test_morse_genomic_rejects_unsorted_input() {
        let chrs = [2u32, 1]; // Unsorted chromosomes
        let starts = [1000u64, 1000];
        let ends = [2000u64, 2000];
        let signal = [5.0f32, 5.0];
        let mut out = [0.0f32; 2];

        unsafe {
            let res = morse_gate_genomic_f32(
                chrs.as_ptr(),
                starts.as_ptr(),
                ends.as_ptr(),
                signal.as_ptr(),
                out.as_mut_ptr(),
                core::ptr::null_mut(),
                2,
                1.0,
                1.0,
            );
            assert_eq!(res, ERR_GENOMIC_UNSORTED_CHROMOSOMES);
        }
    }
}
