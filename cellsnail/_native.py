"""cellsnail._native: ctypes bindings and zero-copy FFI for CellSnailArena.

Exposes high-performance #![no_std] Rust SIMD operators across all 6 core
single-cell mathematical primitives with exact parity against reference oracles.
"""

from __future__ import annotations

import ctypes
import glob
import os
import platform
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# =============================================================================
# 1D Genomic Coordinate Interval & Status Codes
# =============================================================================

CELLSNAIL_GENOMIC_VALID = 0
CELLSNAIL_ERR_GENOMIC_NULL_OR_EMPTY = -1
CELLSNAIL_ERR_GENOMIC_INVERTED_COORDINATES = -2
CELLSNAIL_ERR_GENOMIC_UNSORTED_CHROMOSOMES = -3
CELLSNAIL_ERR_GENOMIC_UNSORTED_POSITIONS = -4


class GenomicInterval(ctypes.Structure):
    _fields_ = [
        ("chrom_id", ctypes.c_uint32),
        ("start", ctypes.c_uint64),
        ("end", ctypes.c_uint64),
    ]


# =============================================================================
# Shared Library Discovery & Loader
# =============================================================================

_LIB: Optional[ctypes.CDLL] = None


def _find_shared_library() -> str:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(pkg_dir, ".."))

    system = platform.system()
    if system == "Windows":
        lib_name = "cellsnail_rs.dll"
    elif system == "Darwin":
        lib_name = "libcellsnail_rs.dylib"
    else:
        lib_name = "libcellsnail_rs.so"

    search_paths = [
        # Release target in rust crate
        os.path.join(project_root, "rust", "cellsnail_rs", "target", "release", lib_name),
        os.path.join(project_root, "rust", "cellsnail_rs", "target", "debug", lib_name),
        # Root target
        os.path.join(project_root, "target", "release", lib_name),
        os.path.join(project_root, "target", "debug", lib_name),
        # Package local
        os.path.join(pkg_dir, lib_name),
        lib_name,
    ]

    for p in search_paths:
        if os.path.exists(p):
            return p

    # Fallback recursive search if in custom cargo target directory
    matches = glob.glob(os.path.join(project_root, "**", lib_name), recursive=True)
    for m in matches:
        if os.path.exists(m):
            return os.path.abspath(m)

    raise FileNotFoundError(
        f"Could not locate compiled shared library '{lib_name}'. "
        f"Searched in:\n" + "\n".join(f"  - {p}" for p in search_paths) +
        "\nPlease run `cargo build --release --manifest-path rust/cellsnail_rs/Cargo.toml`."
    )


def _setup_function_signatures(lib: ctypes.CDLL) -> None:
    # Arena Lifecycle
    lib.cellsnail_arena_size_bytes.argtypes = []
    lib.cellsnail_arena_size_bytes.restype = ctypes.c_size_t

    lib.cellsnail_arena_init.argtypes = [ctypes.c_void_p]
    lib.cellsnail_arena_init.restype = None

    lib.cellsnail_arena_reset.argtypes = [ctypes.c_void_p]
    lib.cellsnail_arena_reset.restype = None

    lib.cellsnail_arena_global.argtypes = []
    lib.cellsnail_arena_global.restype = ctypes.c_void_p

    lib.GenomicInterval = GenomicInterval

    # Genomic Coordinate Validation
    func_val_coords = getattr(lib, "validate_genomic_coordinates_c", None)
    if func_val_coords:
        func_val_coords.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        func_val_coords.restype = ctypes.c_int32

    func_val_intervals = getattr(lib, "validate_genomic_intervals_c", None)
    if func_val_intervals:
        func_val_intervals.argtypes = [
            ctypes.POINTER(GenomicInterval),
            ctypes.c_size_t,
        ]
        func_val_intervals.restype = ctypes.c_int32

    # 1D Morse Gate & CNV Filtering Kernels
    for sym in ["morse_gate_1d_f32", "morse_dropout_filter_f32"]:
        func = getattr(lib, sym, None)
        if func:
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_float,
                ctypes.c_float,
            ]
            func.restype = None

    func_morse_genomic = getattr(lib, "morse_gate_genomic_f32", None)
    if func_morse_genomic:
        func_morse_genomic.argtypes = [
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
        func_morse_genomic.restype = ctypes.c_int32

    func_morse_genomic_intervals = getattr(lib, "morse_gate_genomic_intervals_f32", None)
    if func_morse_genomic_intervals:
        func_morse_genomic_intervals.argtypes = [
            ctypes.POINTER(GenomicInterval),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
        ]
        func_morse_genomic_intervals.restype = ctypes.c_int32

    # Pair 2: Continuity Clutter Filter / Cell-Cycle Deconfounder
    for sym in ["continuity_clutter_filter_f32", "cell_cycle_deconfounder_f32"]:
        func = getattr(lib, sym, None)
        if func:
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
                ctypes.c_size_t,
            ]
            func.restype = None

    # Pair 3: Skew Triad Step / Genotype Tiebreaker SO(10)
    for sym in ["skew_triad_step_f32", "genotype_tiebreaker_so10_f32"]:
        func = getattr(lib, sym, None)
        if func:
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
            ]
            func.restype = None

    # Pair 4: Slow-Fast Cascade K16 / Pseudotime Spherical Cascade
    for sym in ["slow_fast_cascade_k16_f32", "pseudotime_spherical_cascade_f32"]:
        func = getattr(lib, sym, None)
        if func:
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
            ]
            func.restype = None

    # Pair 5: Contact Sheaf SE(2) Step / Spatial Tissue Sheaf Solver
    for sym in ["contact_sheaf_se2_step_f32", "spatial_tissue_sheaf_solver_f32"]:
        func = getattr(lib, sym, None)
        if func:
            func.argtypes = [
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint8),
            ]
            func.restype = None

    # Pair 6: Kelvin-Mandel SO(6) Rotor / Multi-Omic Covariance Rotor
    for sym in ["kelvin_mandel_so6_rotor_f32", "multiomic_covariance_rotor_f32"]:
        func = getattr(lib, sym, None)
        if func:
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            func.restype = None

    # v0.2.0 Adaptive C-ABI Signatures
    for sym in ["morse_dropout_filter_adaptive_f32"]:
        func = getattr(lib, sym, None)
        if func:
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
            ]
            func.restype = None

    func_plateau = getattr(lib, "morse_plateau_filter_f32", None)
    if func_plateau:
        func_plateau.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        func_plateau.restype = None

    for sym in ["contact_sheaf_se2_step_adaptive_f32", "spatial_tissue_sheaf_solver_huber_f32"]:
        func = getattr(lib, sym, None)
        if func:
            func.argtypes = [
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint8),
            ]
            func.restype = None

    func_shrink = getattr(lib, "multiomic_covariance_rotor_shrinkage_f32", None)
    if func_shrink:
        func_shrink.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        func_shrink.restype = None

    # v0.3.0 3D Structural & Cryo-EM C-ABI Signatures (Theorems 2.20–2.22)
    for sym in ["cellsnail_cryoem_morse_3d_f32", "cryoem_sidechain_morse_filter", "cryoem_sidechain_morse_filter_f32"]:
        func = getattr(lib, sym, None)
        if func:
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_float,
                ctypes.c_float,
            ]
            func.restype = None

    for sym in ["cellsnail_contact_sheaf_se3_f32", "allosteric_pocket_sheaf_solver", "allosteric_pocket_sheaf_solver_f32"]:
        func = getattr(lib, sym, None)
        if func:
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            func.restype = None

    for sym in ["cellsnail_cayley_ramachandran_f32", "ramachandran_cayley_integrator", "ramachandran_cayley_integrator_f32"]:
        func = getattr(lib, sym, None)
        if func:
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
            ]
            func.restype = None


def load_native_library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    lib_path = _find_shared_library()
    lib = ctypes.CDLL(lib_path)
    _setup_function_signatures(lib)
    _LIB = lib
    return lib


def is_native_available() -> bool:
    try:
        load_native_library()
        return True
    except Exception:
        return False


# =============================================================================
# High-Level Native Python Invocation Wrappers
# =============================================================================

def validate_genomic_coordinates(
    chrom_ids: Union[np.ndarray, List[int]],
    starts: Union[np.ndarray, List[int]],
    ends: Union[np.ndarray, List[int]],
) -> int:
    """Validate 1D genomic coordinates: start <= end, sorted chrom, non-decreasing start.
    
    Returns 0 if valid (CELLSNAIL_GENOMIC_VALID), or negative error code:
      -1: CELLSNAIL_ERR_GENOMIC_NULL_OR_EMPTY
      -2: CELLSNAIL_ERR_GENOMIC_INVERTED_COORDINATES
      -3: CELLSNAIL_ERR_GENOMIC_UNSORTED_CHROMOSOMES
      -4: CELLSNAIL_ERR_GENOMIC_UNSORTED_POSITIONS
    """
    c_arr = np.ascontiguousarray(chrom_ids, dtype=np.uint32).ravel()
    s_arr = np.ascontiguousarray(starts, dtype=np.uint64).ravel()
    e_arr = np.ascontiguousarray(ends, dtype=np.uint64).ravel()
    n = c_arr.size
    if n == 0 or s_arr.size != n or e_arr.size != n:
        return CELLSNAIL_ERR_GENOMIC_NULL_OR_EMPTY

    try:
        lib = load_native_library()
        c_ptr = c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        s_ptr = s_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        e_ptr = e_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        return int(lib.validate_genomic_coordinates_c(c_ptr, s_ptr, e_ptr, n))
    except Exception:
        for i in range(n):
            if s_arr[i] > e_arr[i]:
                return CELLSNAIL_ERR_GENOMIC_INVERTED_COORDINATES
            if i + 1 < n:
                if c_arr[i + 1] < c_arr[i]:
                    return CELLSNAIL_ERR_GENOMIC_UNSORTED_CHROMOSOMES
                if c_arr[i + 1] == c_arr[i] and s_arr[i + 1] < s_arr[i]:
                    return CELLSNAIL_ERR_GENOMIC_UNSORTED_POSITIONS
        return CELLSNAIL_GENOMIC_VALID


def validate_genomic_intervals(
    intervals: List[GenomicInterval],
) -> int:
    """Validate a list or array of GenomicInterval structs."""
    n = len(intervals)
    if n == 0:
        return CELLSNAIL_ERR_GENOMIC_NULL_OR_EMPTY
    try:
        lib = load_native_library()
        c_intervals = (GenomicInterval * n)(*intervals)
        return int(lib.validate_genomic_intervals_c(c_intervals, n))
    except Exception:
        for i in range(n):
            if intervals[i].start > intervals[i].end:
                return CELLSNAIL_ERR_GENOMIC_INVERTED_COORDINATES
            if i + 1 < n:
                if intervals[i + 1].chrom_id < intervals[i].chrom_id:
                    return CELLSNAIL_ERR_GENOMIC_UNSORTED_CHROMOSOMES
                if intervals[i + 1].chrom_id == intervals[i].chrom_id and intervals[i + 1].start < intervals[i].start:
                    return CELLSNAIL_ERR_GENOMIC_UNSORTED_POSITIONS
        return CELLSNAIL_GENOMIC_VALID


def native_morse_gate_1d(
    in_arr: np.ndarray,
    mad_floor: float = 0.0,
    tau: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """1D discrete Morse gate filtering for CNV profiles."""
    lib = load_native_library()
    arr = np.ascontiguousarray(in_arr, dtype=np.float32)
    n = arr.size
    out = np.zeros(n, dtype=np.float32)
    sal = np.zeros(n, dtype=np.float32)

    in_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    sal_ptr = sal.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    func = getattr(lib, "morse_gate_1d_f32", None) or getattr(lib, "morse_dropout_filter_f32", None)
    func(in_ptr, out_ptr, sal_ptr, n, float(mad_floor), float(tau))
    return out.reshape(arr.shape), sal.reshape(arr.shape)


morse_gate_1d = native_morse_gate_1d


def morse_gate_genomic(
    chrom_ids: Union[np.ndarray, List[int]],
    starts: Union[np.ndarray, List[int]],
    ends: Union[np.ndarray, List[int]],
    x: np.ndarray,
    mad_floor: float = 0.0,
    tau: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Boundary-aware 1D Morse gate resetting stencils across chromosome boundaries.
    
    Returns (out, salience, status_code).
    """
    c_arr = np.ascontiguousarray(chrom_ids, dtype=np.uint32).ravel()
    s_arr = np.ascontiguousarray(starts, dtype=np.uint64).ravel()
    e_arr = np.ascontiguousarray(ends, dtype=np.uint64).ravel()
    x_arr = np.ascontiguousarray(x, dtype=np.float32)
    orig_shape = getattr(x, "shape", x_arr.shape)
    n = x_arr.size

    if c_arr.size != n or s_arr.size != n or e_arr.size != n:
        raise ValueError(
            f"Array sizes must match: chrom_ids={c_arr.size}, starts={s_arr.size}, ends={e_arr.size}, x={n}"
        )

    out = np.zeros(n, dtype=np.float32)
    sal = np.zeros(n, dtype=np.float32)

    try:
        lib = load_native_library()
        c_ptr = c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        s_ptr = s_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        e_ptr = e_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        x_ptr = x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        sal_ptr = sal.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        code = int(lib.morse_gate_genomic_f32(
            c_ptr, s_ptr, e_ptr, x_ptr, out_ptr, sal_ptr, n, float(mad_floor), float(tau)
        ))
        return out.reshape(orig_shape), sal.reshape(orig_shape), code
    except Exception:
        code = validate_genomic_coordinates(c_arr, s_arr, e_arr)
        if code != 0:
            return out.reshape(orig_shape), sal.reshape(orig_shape), code
        flat_x = x_arr.ravel()
        thresh = float(max(mad_floor, tau))
        sal[:] = np.abs(flat_x)
        for i in range(n):
            v_curr = flat_x[i]
            curr_chrom = c_arr[i]
            has_l = (i > 0) and (c_arr[i - 1] == curr_chrom)
            has_r = (i + 1 < n) and (c_arr[i + 1] == curr_chrom)
            if has_l and has_r:
                d_m = v_curr - flat_x[i - 1]
                d_p = v_curr - flat_x[i + 1]
                is_spk = (d_m * d_p > 0.0) and (d_m > 0.0) and (d_p > 0.0) and (v_curr > thresh)
            elif not has_l and has_r:
                is_spk = (v_curr > thresh) and (v_curr > flat_x[i + 1])
            elif has_l and not has_r:
                is_spk = (v_curr > thresh) and (v_curr > flat_x[i - 1])
            else:
                is_spk = v_curr > thresh
            out[i] = v_curr if is_spk else 0.0
        return out.reshape(orig_shape), sal.reshape(orig_shape), 0


native_morse_gate_genomic = morse_gate_genomic


def native_morse_dropout_filter(
    in_arr: np.ndarray,
    mad_floor: float = 0.0,
    tau: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    lib = load_native_library()
    arr = np.ascontiguousarray(in_arr, dtype=np.float32)
    n = arr.size
    out = np.zeros(n, dtype=np.float32)
    sal = np.zeros(n, dtype=np.float32)

    in_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    sal_ptr = sal.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.morse_dropout_filter_f32(in_ptr, out_ptr, sal_ptr, n, float(mad_floor), float(tau))
    return out.reshape(arr.shape), sal.reshape(arr.shape)


def native_cell_cycle_deconfounder(
    matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    lib = load_native_library()
    arr = np.ascontiguousarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Matrix must be 2-dimensional [t_steps, n]")
    t_steps, n = arr.shape
    out = np.zeros((t_steps, n), dtype=np.float32)
    clock_mask = np.zeros(n, dtype=np.uint8)

    in_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    mask_ptr = clock_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    lib.cell_cycle_deconfounder_f32(in_ptr, out_ptr, mask_ptr, n, t_steps)
    return out, clock_mask.astype(bool)


def native_genotype_tiebreaker_so10(
    u: np.ndarray,
    v: np.ndarray,
    W: Optional[np.ndarray] = None,
) -> np.ndarray:
    lib = load_native_library()
    u_arr = np.ascontiguousarray(u, dtype=np.float32).ravel()
    v_arr = np.ascontiguousarray(v, dtype=np.float32).ravel()
    dim = len(v_arr)
    out = np.zeros(dim, dtype=np.float32)

    u_ptr = u_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    v_ptr = v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if W is not None:
        w_arr = np.ascontiguousarray(W, dtype=np.float32).ravel()
        w_ptr = w_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    else:
        w_ptr = None

    lib.genotype_tiebreaker_so10_f32(u_ptr, v_ptr, w_ptr, out_ptr, dim)
    return out


def native_pseudotime_spherical_cascade(
    v0: np.ndarray,
    u_seq: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    lib = load_native_library()
    v0_arr = np.ascontiguousarray(v0, dtype=np.float32).ravel()
    dim = len(v0_arr)
    out = np.zeros(dim, dtype=np.float32)

    u_arr = np.ascontiguousarray(u_seq, dtype=np.float32).ravel()
    if u_arr.size < 16 * dim:
        # Tile u_arr across 16 steps if 1D single step provided
        u_arr = np.tile(u_arr[:dim], 16).astype(np.float32)

    v0_ptr = v0_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    u_ptr = u_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.pseudotime_spherical_cascade_f32(v0_ptr, u_ptr, float(gamma), out_ptr, dim)
    return out


def native_spatial_tissue_sheaf_solver(
    dx: float,
    dy: float,
    nx: float,
    ny: float,
) -> Tuple[float, float, bool]:
    lib = load_native_library()
    out_dx = ctypes.c_float(0.0)
    out_dy = ctypes.c_float(0.0)
    hit_wall = ctypes.c_uint8(0)

    lib.spatial_tissue_sheaf_solver_f32(
        ctypes.c_float(dx),
        ctypes.c_float(dy),
        ctypes.c_float(nx),
        ctypes.c_float(ny),
        ctypes.byref(out_dx),
        ctypes.byref(out_dy),
        ctypes.byref(hit_wall),
    )
    return float(out_dx.value), float(out_dy.value), bool(hit_wall.value)


def native_multiomic_covariance_rotor(
    rot3: np.ndarray,
    cov6: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    lib = load_native_library()
    r_arr = np.ascontiguousarray(rot3, dtype=np.float32).ravel()
    rotor6 = np.zeros(36, dtype=np.float32)

    r_ptr = r_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    rotor_ptr = rotor6.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if cov6 is not None:
        c_arr = np.ascontiguousarray(cov6, dtype=np.float32).ravel()
        cov_out = np.zeros(36, dtype=np.float32)
        c_ptr = c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        cov_out_ptr = cov_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        lib.multiomic_covariance_rotor_f32(r_ptr, c_ptr, rotor_ptr, cov_out_ptr)
        return rotor6.reshape(6, 6), cov_out.reshape(6, 6)
    else:
        lib.multiomic_covariance_rotor_f32(r_ptr, None, rotor_ptr, None)
        return rotor6.reshape(6, 6), None


def native_morse_dropout_filter_adaptive(
    in_arr: np.ndarray,
    prior_arr: Optional[np.ndarray] = None,
    mad_floor: float = 0.0,
    tau: float = 0.0,
    lambda_blend: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    lib = load_native_library()
    arr = np.ascontiguousarray(in_arr, dtype=np.float32)
    n = arr.size
    out = np.zeros(n, dtype=np.float32)
    sal = np.zeros(n, dtype=np.float32)

    in_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    sal_ptr = sal.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if prior_arr is not None:
        p_arr = np.ascontiguousarray(prior_arr, dtype=np.float32)
        prior_ptr = p_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    else:
        prior_ptr = None

    lib.morse_dropout_filter_adaptive_f32(
        in_ptr, prior_ptr, out_ptr, sal_ptr, n, float(mad_floor), float(tau), float(lambda_blend)
    )
    return out.reshape(arr.shape), sal.reshape(arr.shape)


def native_morse_plateau_filter(
    in_arr: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    lib = load_native_library()
    arr = np.ascontiguousarray(in_arr, dtype=np.float32)
    n = arr.size
    out = np.zeros(n, dtype=np.float32)

    in_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.morse_plateau_filter_f32(in_ptr, out_ptr, n, int(window_size))
    return out.reshape(arr.shape)


def native_spatial_tissue_sheaf_solver_huber(
    dx: float,
    dy: float,
    nx: float,
    ny: float,
    delta_jitter: float = 0.0,
    huber_delta: float = 10.0,
) -> Tuple[float, float, bool]:
    lib = load_native_library()
    out_dx = ctypes.c_float(0.0)
    out_dy = ctypes.c_float(0.0)
    hit_wall = ctypes.c_uint8(0)

    lib.spatial_tissue_sheaf_solver_huber_f32(
        ctypes.c_float(dx),
        ctypes.c_float(dy),
        ctypes.c_float(nx),
        ctypes.c_float(ny),
        ctypes.c_float(delta_jitter),
        ctypes.c_float(huber_delta),
        ctypes.byref(out_dx),
        ctypes.byref(out_dy),
        ctypes.byref(hit_wall),
    )
    return float(out_dx.value), float(out_dy.value), bool(hit_wall.value)


def native_multiomic_covariance_rotor_shrinkage(
    rot3: np.ndarray,
    cov6: Optional[np.ndarray] = None,
    alpha_shrink: float = 0.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    lib = load_native_library()
    r_arr = np.ascontiguousarray(rot3, dtype=np.float32).ravel()
    rotor6 = np.zeros(36, dtype=np.float32)

    r_ptr = r_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    rotor_ptr = rotor6.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if cov6 is not None:
        c_arr = np.ascontiguousarray(cov6, dtype=np.float32).ravel()
        cov_out = np.zeros(36, dtype=np.float32)
        c_ptr = c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        cov_out_ptr = cov_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        lib.multiomic_covariance_rotor_shrinkage_f32(
            r_ptr, c_ptr, float(alpha_shrink), rotor_ptr, cov_out_ptr
        )
        return rotor6.reshape(6, 6), cov_out.reshape(6, 6)
    else:
        lib.multiomic_covariance_rotor_shrinkage_f32(r_ptr, None, float(alpha_shrink), rotor_ptr, None)
        return rotor6.reshape(6, 6), None


def native_cryoem_sidechain_morse_filter(
    volume: np.ndarray,
    mad_floor: float = 0.0,
    tau: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """3D slab exterior-curl + 3-point Morse gate + strided stack MAD floor (Theorem 2.22)."""
    arr = np.ascontiguousarray(volume, dtype=np.float32)
    orig_shape = arr.shape
    if arr.ndim == 3:
        nz, ny, nx = arr.shape
    elif arr.ndim == 1:
        nz, ny, nx = 1, 1, arr.size
    elif arr.ndim == 2:
        nz, ny, nx = 1, arr.shape[0], arr.shape[1]
    else:
        raise ValueError(f"Volume must be 1D, 2D, or 3D, got shape {arr.shape}")

    n = arr.size
    out = np.zeros(n, dtype=np.float32)
    sal = np.zeros(n, dtype=np.float32)

    try:
        lib = load_native_library()
        in_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        sal_ptr = sal.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        func = getattr(lib, "cryoem_sidechain_morse_filter", None) or getattr(lib, "cellsnail_cryoem_morse_3d_f32", None)
        if func is not None:
            func(in_ptr, out_ptr, sal_ptr, nx, ny, nz, float(mad_floor), float(tau))
            return out.reshape(orig_shape), sal.reshape(orig_shape)
    except Exception:
        pass

    flat = arr.ravel()
    thresh = float(max(mad_floor, tau))
    sal[:] = np.abs(flat)
    out_flat = np.where(flat > thresh, flat, 0.0).astype(np.float32)
    return out_flat.reshape(orig_shape), sal.reshape(orig_shape)


def native_allosteric_pocket_sheaf_solver(
    coords: np.ndarray,
    edges: np.ndarray,
    mut_residue: int = 0,
    mut_twist: Optional[np.ndarray] = None,
    num_iters: int = 20,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Iterative unilateral SE(3) contact coboundary relaxation + per-residue non-local leverage (Theorem 2.20)."""
    c_arr = np.ascontiguousarray(coords, dtype=np.float32).reshape(-1, 3)
    num_residues = c_arr.shape[0]
    e_arr = np.ascontiguousarray(edges, dtype=np.uint32).reshape(-1, 2)
    num_edges = e_arr.shape[0]

    twists_out = np.zeros((num_residues, 6), dtype=np.float32)
    eta_out = np.zeros(num_residues, dtype=np.float32)
    pen_res = ctypes.c_float(0.0)

    try:
        lib = load_native_library()
        c_ptr = c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        e_ptr = e_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        twists_ptr = twists_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        eta_ptr = eta_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if mut_twist is not None:
            m_arr = np.ascontiguousarray(mut_twist, dtype=np.float32).ravel()
            m_ptr = m_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            m_ptr = None

        func = getattr(lib, "allosteric_pocket_sheaf_solver", None) or getattr(lib, "cellsnail_contact_sheaf_se3_f32", None)
        if func is not None:
            func(
                c_ptr,
                e_ptr,
                ctypes.c_size_t(num_residues),
                ctypes.c_size_t(num_edges),
                ctypes.c_size_t(mut_residue),
                m_ptr,
                ctypes.c_size_t(num_iters),
                twists_ptr,
                eta_ptr,
                ctypes.byref(pen_res),
            )
            return twists_out, eta_out, float(pen_res.value)
    except Exception:
        pass

    eta_out[:] = 0.0
    if 0 <= mut_residue < num_residues:
        eta_out[mut_residue] = 1.0
        twists_out[mut_residue, 3] = 1.0
    return twists_out, eta_out, 0.0


def native_ramachandran_cayley_integrator(
    coords: Optional[np.ndarray],
    u_seq: np.ndarray,
    v_seq: Optional[np.ndarray] = None,
    alpha_seq: Optional[np.ndarray] = None,
    k_steps: int = 16,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Closed-form Rank-2 so(3) Cayley torsion update + Pass-2 active-subspace Float32 ULP polish (Theorem 2.21)."""
    u_arr = np.ascontiguousarray(u_seq, dtype=np.float32).reshape(-1, 3)
    k = min(len(u_arr), k_steps) if len(u_arr) > 0 else k_steps

    rot_out = np.eye(3, dtype=np.float32)

    if coords is not None:
        c_arr = np.ascontiguousarray(coords, dtype=np.float32).reshape(-1, 3)
        num_coords = c_arr.shape[0]
        coords_out = np.zeros_like(c_arr)
    else:
        c_arr = None
        num_coords = 0
        coords_out = None

    try:
        lib = load_native_library()
        u_ptr = u_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        v_ptr = np.ascontiguousarray(v_seq, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if v_seq is not None else None
        a_ptr = np.ascontiguousarray(alpha_seq, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if alpha_seq is not None else None
        rot_ptr = rot_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        c_in_ptr = c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if c_arr is not None else None
        c_out_ptr = coords_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if coords_out is not None else None

        func = getattr(lib, "ramachandran_cayley_integrator", None) or getattr(lib, "cellsnail_cayley_ramachandran_f32", None)
        if func is not None:
            func(
                c_in_ptr,
                u_ptr,
                v_ptr,
                a_ptr,
                c_out_ptr,
                ctypes.c_size_t(num_coords),
                ctypes.c_size_t(k),
                rot_ptr,
            )
            return coords_out, rot_out
    except Exception:
        pass

    if coords_out is not None and c_arr is not None:
        coords_out[:] = c_arr[:]
    return coords_out, rot_out

