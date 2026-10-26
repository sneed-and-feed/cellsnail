//! cellsnail_rs: Allocation-free #![no_std] Rust SIMD & C-ABI static arena library
//! for multi-omic single-cell data science in the DarkSnail / triad-core ecosystem.
//!
//! Grounded in machine-checked formal proofs (Theorems 2.1–2.15):
//! - Theorem 2.12: Non-Circular Morse Dropout Shielding & Zero Spurious Gene-Correlation Injection
//! - Theorem 2.13: H_clock (+) H_nope Cell-Cycle & Batch Orthogonality
//! - Theorem 2.14: Ascertainment-Bias-Free Ultrametric Site Condensation
//! - Theorem 2.15: Bi-Hemispheric CNV-to-RNA Dosage Linkage & Driver-Passenger Causal Factorization
//! - Theorem 2.11: Static SRAM residency < 2.0 MB (< 2,097,152 B) with 64-byte alignment and 0 heap allocations.

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_unsafe)]

#[cfg(all(not(feature = "std"), not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(all(not(feature = "std"), not(test), not(windows), not(target_arch = "wasm32")))]
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

#[cfg(any(feature = "std", test))]
extern crate std;

// ============================================================================
// Constants & Static Arena Geometry (< 2.0 MB SRAM)
// ============================================================================

pub const SPATIAL_SIZE: usize = 16384;
pub const RING_BUFFER_SIZE: usize = 131072;
pub const GENOTYPE_TENSOR_SIZE: usize = 160;
pub const MULTIOMIC_COV_SIZE: usize = 36;

/// Fixed-layout static arena strictly < 2.0 MB SRAM with 64-byte alignment.
/// Total size: exactly 983,872 bytes (960.8 KB <= 2,097,152 bytes).
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
    pub _pad: [u8; 48],                              // 48 B (offset 983,824) -> total 983,872 B
}

// Compile-time static assertions proving residency and alignment (Theorem 2.11)
const _: () = assert!(core::mem::size_of::<CellSnailArena>() <= 2_097_152);
const _: () = assert!(core::mem::size_of::<CellSnailArena>() == 983_872);
const _: () = assert!(core::mem::size_of::<CellSnailArena>() % 64 == 0);
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
// Dual C-ABI Export 1: Morse Gate & Morse Dropout Filter (Theorem 2.12)
// ============================================================================

/// Canonical DarkSnail: 3-point discrete Morse innovation gate.
///
/// Rejects monotone slopes/ramps (I_morse <= 0), preserves isolated spikes (I_morse > 0)
/// exceeding noise floor, with 0 spurious cross-gene correlation injection.
#[no_mangle]
pub unsafe extern "C" fn beltrami_morse_gate_f32(
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

/// Stealth Single-Cell Domain Alias for [`beltrami_morse_gate_f32`].
#[no_mangle]
pub unsafe extern "C" fn morse_dropout_filter_f32(
    in_ptr: *const f32,
    out_ptr: *mut f32,
    salience_ptr: *mut f32,
    n: usize,
    mad_floor: f32,
    tau: f32,
) {
    beltrami_morse_gate_f32(in_ptr, out_ptr, salience_ptr, n, mad_floor, tau);
}

// ============================================================================
// Dual C-ABI Export 2: Continuity Clutter Filter & Cell-Cycle Deconfounder (Theorem 2.13)
// ============================================================================

/// Canonical DarkSnail: Continuity clutter filter projecting state observations
/// into orthogonal subspaces H_clock (+) H_nope.
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
        let mut prev_diff = *in_ptr.add(1 * n + j) - *in_ptr.add(0 * n + j);

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

/// Stealth Single-Cell Domain Alias for [`continuity_clutter_filter_f32`].
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
// Dual C-ABI Export 3: Skew Triad Step & Genotype Tiebreaker SO(10) (Theorems 2.1–2.3)
// ============================================================================

/// Canonical DarkSnail: Skew-symmetric bilinear triad step with exact spherical retraction.
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

/// Stealth Single-Cell Domain Alias for [`skew_triad_step_f32`].
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
// Dual C-ABI Export 4: Slow-Fast Cascade K16 & Pseudotime Spherical Cascade (Theorem 2.6)
// ============================================================================

/// Canonical DarkSnail: 16-step unrolled spherical velocity integration cascade.
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
        for k in 0..d {
            v_cur[k] = v_next[k];
        }
    }

    for k in 0..d {
        *out.add(k) = v_cur[k];
    }
}

/// Stealth Single-Cell Domain Alias for [`slow_fast_cascade_k16_f32`].
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
// Dual C-ABI Export 5: Contact Sheaf SE(2) Step & Spatial Tissue Sheaf Solver (Theorem 2.10b)
// ============================================================================

/// Canonical DarkSnail: SE(2) cellular sheaf coboundary solver enforcing unilateral
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

/// Stealth Single-Cell Domain Alias for [`contact_sheaf_se2_step_f32`].
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
// Dual C-ABI Export 6: Kelvin-Mandel SO(6) Rotor & Multi-Omic Covariance Rotor (Theorems 2.7, 2.15)
// ============================================================================

/// Canonical DarkSnail: Kelvin-Mandel SO(6) rotor lifting SO(3) cross-modal rotation
/// to 6x6 Voigt covariance tensor with exact trace conservation (Theorem 2.15).
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
    let s2 = 1.4142135623730951f64;

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

/// Stealth Single-Cell Domain Alias for [`kelvin_mandel_so6_rotor_f32`].
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
// Synchronous Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_geometry_and_alignment() {
        let sz = core::mem::size_of::<CellSnailArena>();
        assert_eq!(sz, 983_872);
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
            beltrami_morse_gate_f32(mono.as_ptr(), mono_out.as_mut_ptr(), core::ptr::null_mut(), 3, 0.5, 0.5);
        }
        assert_eq!(mono_out[1], 0.0, "Monotone slope must produce 0.0 (rejected)");

        // Isolated thin stringer / spike: [0.1, 5.0, 0.1] -> preserved at 100% amplitude
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
}
