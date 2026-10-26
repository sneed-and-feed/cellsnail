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

    # Pair 1: Morse Gate / Dropout Filter
    for sym in ["beltrami_morse_gate_f32", "morse_dropout_filter_f32"]:
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
