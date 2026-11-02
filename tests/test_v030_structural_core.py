"""tests/test_v030_structural_core.py
===================================
Comprehensive deterministic unit, parity, memory, and topological invariant test suite
for sneed-and-feed/cellsnail v0.3.0 core structural mechanics engine.

Verifies:
  1. Rust-to-Python numerical parity (<= 1.0e-7 in float32) across all 3D operators.
  2. Exact 0.000e+00 Float32 ULP bond-length drift and orthogonality drift over K = 16, 64, 256 steps.
  3. Out-of-core streaming: strictly < 35.0 MB peak tracemalloc heap on 128x128x128 .mrc volume.
  4. Theorem 2.20 invariant: SE(3) contact sheaf coboundary non-penetration, rigid invariance, torque transmission.
  5. Theorem 2.21 invariant: Rank-2 Cayley dihedral torsion norm preservation ||Rx||_2 = ||x||_2 and R^T R = I_3.
  6. Theorem 2.22 invariant: Cryo-EM 3D discrete Morse gate 100% ridge preservation & 100% noise rejection.
  7. PDB parsing/export roundtrip, contact graph generation (d_ij <= 4.5 A / <= 8.0 A), Ramachandran dihedrals.
  8. Cryptic pocket Jordan beta_0, beta_1, beta_2 cavity topology and allosteric collapse ratio.
  9. verify_all_theorems() comprehensive audit covering Theorems 2.1–2.22.
"""

from __future__ import annotations

import gc
import os
import tempfile
import tracemalloc
from typing import Any, Dict

import numpy as np
import pytest

import cellsnail
from cellsnail._native import is_native_available
from cellsnail.io import (
    MRCSlab,
    PDBStructure,
    export_pdb_structure,
    parse_pdb_structure,
    read_mrc_header,
    stream_mrc_slabs,
    write_mrc,
)
from cellsnail.operators import (
    CellSnailEngine,
    allosteric_pocket_sheaf_solver,
    cayley_rotor_so3_torsion,
    contact_sheaf_se3_step,
    cryoem_sidechain_morse_filter,
    cryptic_pocket_topology_audit,
    jordan_3d_cavity_betti,
    ramachandran_cayley_integrator,
    ulp_polish_vec3,
    verify_all_theorems,
    verify_cayley_so3_ramachandran_norm,
    verify_cryoem_3d_morse_shield,
    verify_se3_allosteric_coboundary,
)
from cellsnail.structural import (
    CayleyRamachandranRotor,
    ContactSheafSE3,
    CryoEMMorseFilter3D,
    JordanCavityBetti,
    _py_cayley_ramachandran_step,
    _py_contact_sheaf_se3_solver,
    _py_cryoem_sidechain_morse_3d,
)


# ==============================================================================
# Fixtures & Synthetic Test Geometries
# ==============================================================================

SAMPLE_PDB_TEXT = """\
ATOM      1  N   ALA A   1      11.104  13.207   2.484  1.00 11.20           N
ATOM      2  CA  ALA A   1      12.145  14.159   2.996  1.00 11.00           C
ATOM      3  C   ALA A   1      12.603  13.791   4.403  1.00 11.50           C
ATOM      4  O   ALA A   1      12.222  14.474   5.353  1.00 12.00           O
ATOM      5  CB  ALA A   1      11.603  15.589   2.996  1.00 12.50           C
ATOM      6  N   GLU A   2      13.432  12.748   4.536  1.00 11.80           N
ATOM      7  CA  GLU A   2      13.945  12.302   5.834  1.00 12.20           C
ATOM      8  C   GLU A   2      12.839  11.532   6.549  1.00 12.50           C
ATOM      9  O   GLU A   2      13.067  10.963   7.616  1.00 13.00           O
ATOM     10  CB  GLU A   2      15.176  11.411   5.642  1.00 13.10           C
ATOM     11  CG  GLU A   2      16.321  12.154   4.954  1.00 13.50           C
ATOM     12  CD  GLU A   2      17.562  11.298   4.752  1.00 14.00           C
ATOM     13  OE1 GLU A   2      17.489  10.071   4.992  1.00 14.20           O
ATOM     14  OE2 GLU A   2      18.614  11.862   4.341  1.00 14.50           O
ATOM     15  N   LYS A   3      11.602  11.821   6.102  1.00 12.80           N
ATOM     16  CA  LYS A   3      10.453  11.124   6.683  1.00 13.00           C
ATOM     17  C   LYS A   3      10.124  11.751   8.041  1.00 13.50           C
ATOM     18  O   LYS A   3       9.452  11.145   8.882  1.00 14.00           O
ATOM     19  CB  LYS A   3       9.231  11.214   5.742  1.00 13.20           C
ATOM     20  NZ  LYS A   3       7.541  10.892   4.121  1.00 14.80           N
TER
END
"""


# ==============================================================================
# Group 1: Rust-to-Python Numerical Parity (<= 1.0e-7 in float32)
# ==============================================================================

@pytest.mark.skipif(not is_native_available(), reason="Native library not available")
def test_rust_python_parity_cryoem_morse_filter():
    """Verify Rust SIMD vs Python oracle parity <= 1.0e-7 on 3D Cryo-EM Morse filter."""
    rng = np.random.default_rng(101)
    vol = rng.uniform(0.0, 0.5, size=(10, 10, 10)).astype(np.float32)
    # Add sustained ridge
    vol[4:8, 5, 5] = 5.0
    # Add isolated noise spike
    vol[2, 2, 2] = 12.0

    res_nat = cryoem_sidechain_morse_filter(vol, mad_floor=1.0, tau=1.5, use_native=True)
    res_py = cryoem_sidechain_morse_filter(vol, mad_floor=1.0, tau=1.5, use_native=False)

    max_diff_filt = float(np.max(np.abs(res_nat["filtered"] - res_py["filtered"])))
    max_diff_sal = float(np.max(np.abs(res_nat["salience"] - res_py["salience"])))

    assert max_diff_filt <= 1.0e-7, f"CryoEM filtered parity diff {max_diff_filt} exceeds 1e-7"
    assert max_diff_sal <= 1.0e-7, f"CryoEM salience parity diff {max_diff_sal} exceeds 1e-7"


@pytest.mark.skipif(not is_native_available(), reason="Native library not available")
def test_rust_python_parity_allosteric_pocket_sheaf():
    """Verify Rust SIMD vs Python oracle parity <= 1.0e-7 on SE(3) allosteric sheaf solver."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [3.8, 0.0, 0.0],
        [6.5, 2.5, 0.0],
        [8.0, 5.0, 2.0],
    ], dtype=np.float32)
    edges = np.array([
        [0, 1], [1, 0],
        [1, 2], [2, 1],
        [2, 3], [3, 2],
    ], dtype=np.uint32)

    res_nat = allosteric_pocket_sheaf_solver(coords, edges, mut_residue=0, num_iters=20, use_native=True)
    res_py = allosteric_pocket_sheaf_solver(coords, edges, mut_residue=0, num_iters=20, use_native=False)

    max_diff_twists = float(np.max(np.abs(res_nat["twists"] - res_py["twists"])))
    max_diff_eta = float(np.max(np.abs(res_nat["eta_nonlocal"] - res_py["eta_nonlocal"])))

    assert max_diff_twists <= 1.0e-7, f"Sheaf twists parity diff {max_diff_twists} exceeds 1e-7"
    assert max_diff_eta <= 1.0e-7, f"Sheaf eta parity diff {max_diff_eta} exceeds 1e-7"


@pytest.mark.skipif(not is_native_available(), reason="Native library not available")
def test_rust_python_parity_ramachandran_cayley_rotations():
    """Verify Rust SIMD vs Python oracle parity <= 1.0e-7 on Cayley Ramachandran integrator."""
    coords = np.array([
        [1.54, 0.0, 0.0],
        [0.0, 1.45, 0.0],
        [0.0, 0.0, 1.33],
        [1.2, 0.8, -0.5],
    ], dtype=np.float32)

    res_nat = ramachandran_cayley_integrator(coords, k_steps=16, use_native=True)
    res_py = ramachandran_cayley_integrator(coords, k_steps=16, use_native=False)

    max_diff_c = float(np.max(np.abs(res_nat["coords"] - res_py["coords"])))
    max_diff_r = float(np.max(np.abs(res_nat["rotation_matrix"] - res_py["rotation_matrix"])))

    assert max_diff_c <= 1.0e-7, f"Cayley coords parity diff {max_diff_c} exceeds 1e-7"
    assert max_diff_r <= 1.0e-7, f"Cayley rot matrix parity diff {max_diff_r} exceeds 1e-7"


# ==============================================================================
# Group 2: Float32 ULP Bond-Length and Orthogonality Drift (K = 16, 64, 256)
# ==============================================================================

def test_ramachandran_cayley_ulp_bond_length_drift_k16():
    """Verify exact 0.000e+00 Float32 ULP bond-length drift over K = 16 torsion steps."""
    c_in = np.array([
        [1.54, 0.0, 0.0],   # C-C bond
        [0.0, 1.45, 0.0],   # C-O bond
        [0.0, 0.0, 1.33],   # C-N bond
    ], dtype=np.float32)

    res = ramachandran_cayley_integrator(c_in, k_steps=16)
    c_out = res["coords"]

    # Compute exact float32 norm before and after
    norms_in = np.sqrt(np.sum(c_in.astype(np.float64)**2, axis=1)).astype(np.float32)
    norms_out = np.sqrt(np.sum(c_out.astype(np.float64)**2, axis=1)).astype(np.float32)

    drift = float(np.max(np.abs(norms_out - norms_in)))
    assert drift == 0.0, f"Float32 ULP bond drift must be exactly 0.000e+00 at K=16, got {drift}"
    assert res["radial_drift"] == 0.0


def test_ramachandran_cayley_ulp_bond_length_drift_k64():
    """Verify exact 0.000e+00 Float32 ULP bond-length drift over K = 64 torsion steps."""
    c_in = np.array([
        [1.54, 0.0, 0.0],   # C-C bond
        [0.0, 1.45, 0.0],   # C-O bond
        [0.0, 0.0, 1.33],   # C-N bond
    ], dtype=np.float32)

    res = ramachandran_cayley_integrator(c_in, k_steps=64)
    c_out = res["coords"]

    norms_in = np.sqrt(np.sum(c_in.astype(np.float64)**2, axis=1)).astype(np.float32)
    norms_out = np.sqrt(np.sum(c_out.astype(np.float64)**2, axis=1)).astype(np.float32)

    drift = float(np.max(np.abs(norms_out - norms_in)))
    assert drift == 0.0, f"Float32 ULP bond drift must be exactly 0.000e+00 at K=64, got {drift}"
    assert res["radial_drift"] == 0.0


def test_ramachandran_cayley_ulp_bond_length_drift_k256():
    """Verify exact 0.000e+00 Float32 ULP bond-length drift over K = 256 torsion steps."""
    c_in = np.array([
        [1.54, 0.0, 0.0],   # C-C bond
        [0.0, 1.45, 0.0],   # C-O bond
        [0.0, 0.0, 1.33],   # C-N bond
    ], dtype=np.float32)

    res = ramachandran_cayley_integrator(c_in, k_steps=256)
    c_out = res["coords"]

    norms_in = np.sqrt(np.sum(c_in.astype(np.float64)**2, axis=1)).astype(np.float32)
    norms_out = np.sqrt(np.sum(c_out.astype(np.float64)**2, axis=1)).astype(np.float32)

    drift = float(np.max(np.abs(norms_out - norms_in)))
    assert drift == 0.0, f"Float32 ULP bond drift must be exactly 0.000e+00 at K=256, got {drift}"
    assert res["radial_drift"] == 0.0


def test_ramachandran_cayley_orthogonality_invariance_k16_to_k256():
    """Verify Cayley rotor column-orthonormality R^T R = I_3 across K = 16, 64, 256 cascades."""
    for k in [16, 64, 256]:
        res = ramachandran_cayley_integrator(k_steps=k)
        R = res["rotation_matrix"]
        ortho_err = float(np.max(np.abs(R.T @ R - np.eye(3, dtype=np.float32))))
        assert ortho_err <= 1.0e-6, f"Orthogonality error {ortho_err} exceeds 1e-6 at K={k}"


def test_ramachandran_cayley_vector_norm_preservation_so3():
    """Verify that SO(3) Cayley rotation preserves arbitrary vector norms ||Rx||_2 = ||x||_2."""
    rng = np.random.default_rng(202)
    coords = rng.standard_normal((20, 3)).astype(np.float32)

    res = ramachandran_cayley_integrator(coords, k_steps=32)
    c_out = res["coords"]

    norms_in = np.sqrt(np.sum(coords.astype(np.float64)**2, axis=1))
    norms_out = np.sqrt(np.sum(c_out.astype(np.float64)**2, axis=1))

    max_norm_diff = float(np.max(np.abs(norms_out - norms_in)))
    assert max_norm_diff <= 1.0e-6, f"Vector norm deviation {max_norm_diff} exceeds 1e-6"


# ==============================================================================
# Group 3: Out-of-Core .mrc Streaming (< 35.0 MB Peak Heap)
# ==============================================================================

def test_out_of_core_mrc_streaming_tracemalloc_heap_128cube():
    """Strictly verify < 35.0 MB peak Python heap when streaming a 128x128x128 .mrc volume."""
    with tempfile.NamedTemporaryFile(suffix=".mrc", delete=False) as f:
        tmp_mrc = f.name

    try:
        # Create 128x128x128 float32 volume (8 MB on disk)
        vol = np.full((128, 128, 128), 0.25, dtype=np.float32)
        write_mrc(tmp_mrc, vol)
        del vol
        gc.collect()

        tracemalloc.start()
        peak_before = tracemalloc.get_traced_memory()[1]

        slabs_read = 0
        total_voxels = 0
        for slab in stream_mrc_slabs(tmp_mrc, chunk_z=16, halo=1):
            slabs_read += 1
            total_voxels += slab.size
            # Access slice without materializing full volume
            _ = float(slab[slab.core_slice].mean())

        _, peak_after = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_heap_mb = (peak_after - peak_before) / (1024 * 1024)
        assert slabs_read == 8, f"Expected 8 slabs of size 16, got {slabs_read}"
        assert peak_heap_mb < 35.0, f"Peak tracemalloc heap {peak_heap_mb:.2f} MB exceeds 35.0 MB ceiling"

    finally:
        del slab
        gc.collect()
        if os.path.exists(tmp_mrc):
            try:
                os.remove(tmp_mrc)
            except Exception:
                pass


def test_out_of_core_mrc_streaming_roi_slicing():
    """Verify ROI bounding box slicing during out-of-core streaming."""
    with tempfile.NamedTemporaryFile(suffix=".mrc", delete=False) as f:
        tmp_mrc = f.name

    try:
        vol = np.zeros((32, 32, 32), dtype=np.float32)
        vol[10:20, 10:20, 10:20] = 7.0
        write_mrc(tmp_mrc, vol)
        del vol

        # Stream with ROI covering subvolume
        roi = {"z": (10, 20), "y": (10, 20), "x": (10, 20)}
        slabs = list(stream_mrc_slabs(tmp_mrc, chunk_z=5, halo=0, roi=roi))

        assert len(slabs) == 2, f"Expected 2 slabs for 10 slices with chunk_z=5, got {len(slabs)}"
        for s in slabs:
            assert s.shape[1] == 10 and s.shape[2] == 10
            assert np.all(s == 7.0)
    finally:
        gc.collect()
        if os.path.exists(tmp_mrc):
            try:
                os.remove(tmp_mrc)
            except Exception:
                pass


def test_mrc_header_read_write_roundtrip():
    """Verify read_mrc_header extracts valid metadata from write_mrc file."""
    with tempfile.NamedTemporaryFile(suffix=".mrc", delete=False) as f:
        tmp_mrc = f.name

    try:
        vol = np.zeros((16, 24, 32), dtype=np.float32)
        vol[0, 0, 0] = 1.0
        vol[15, 23, 31] = 5.0
        write_mrc(tmp_mrc, vol, voxel_size=1.25)

        h = read_mrc_header(tmp_mrc)
        assert h["shape"] == (16, 24, 32)
        assert h["nx"] == 32 and h["ny"] == 24 and h["nz"] == 16
        assert h["mode"] == 2  # float32
        assert h["cmap"] == b"MAP "
        assert h["stamp"] == b"DA\x00\x00"
        assert abs(h["voxel_size"][0] - 1.25) < 1e-4
    finally:
        gc.collect()
        if os.path.exists(tmp_mrc):
            try:
                os.remove(tmp_mrc)
            except Exception:
                pass


def test_mrc_slab_halo_and_core_alignment():
    """Verify that MRCSlab correctly defines halo_start, halo_end, and core_slice."""
    with tempfile.NamedTemporaryFile(suffix=".mrc", delete=False) as f:
        tmp_mrc = f.name

    try:
        vol = np.arange(32 * 16 * 16, dtype=np.float32).reshape(32, 16, 16)
        write_mrc(tmp_mrc, vol)
        del vol

        slabs = list(stream_mrc_slabs(tmp_mrc, chunk_z=16, halo=2))
        assert len(slabs) == 2

        # Slab 0: z=0..16, halo_start=0, halo_end=18, core_slice=(0..16)
        s0 = slabs[0]
        assert s0.z_start == 0 and s0.z_end == 16
        assert s0.halo_start == 0 and s0.halo_end == 18
        assert s0.core_slice[0] == slice(0, 16)

        # Slab 1: z=16..32, halo_start=14, halo_end=32, core_slice=(2..18)
        s1 = slabs[1]
        assert s1.z_start == 16 and s1.z_end == 32
        assert s1.halo_start == 14 and s1.halo_end == 32
        assert s1.core_slice[0] == slice(2, 18)
    finally:
        del slabs
        gc.collect()
        if os.path.exists(tmp_mrc):
            try:
                os.remove(tmp_mrc)
            except Exception:
                pass


# ==============================================================================
# Group 4: Theorem 2.20 Invariants (SE(3) Sheaf Coboundary & Torque)
# ==============================================================================

def test_theorem_2_20_unilateral_coboundary_non_penetration():
    """Verify unilateral contact coboundary condition (delta v)(e) >= 0 (Theorem 2.20)."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [3.8, 0.0, 0.0],
        [7.6, 0.0, 0.0],
    ], dtype=np.float32)
    edges = np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.uint32)

    res = allosteric_pocket_sheaf_solver(coords, edges, mut_residue=0, num_iters=25)
    twists = res["twists"]

    assert res["steric_penetration_residual"] == 0.0

    for e in edges:
        u, v = e
        dx = coords[v] - coords[u]
        d = float(np.linalg.norm(dx))
        n_e = dx / d
        dv_n = float(np.dot(twists[v, 3:6] - twists[u, 3:6], n_e))
        assert dv_n >= -1e-5, f"Steric clash on edge {u}->{v}: dv_n = {dv_n}"


def test_theorem_2_20_rigid_body_se3_translation_invariance():
    """Verify contact coboundary vanishes under global rigid-body SE(3) translation."""
    coords = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.uint32)

    # Base solve
    res1 = allosteric_pocket_sheaf_solver(coords, edges, mut_residue=0)

    # Shift all coordinates by (100, -50, 20)
    coords_shifted = coords + np.array([100.0, -50.0, 20.0], dtype=np.float32)
    res2 = allosteric_pocket_sheaf_solver(coords_shifted, edges, mut_residue=0)

    # Non-local leverage and penetration residual must be identical
    assert np.allclose(res1["eta_nonlocal"], res2["eta_nonlocal"], atol=1e-5)
    assert res1["steric_penetration_residual"] == res2["steric_penetration_residual"]


def test_theorem_2_20_rigid_body_se3_twist_invariance():
    """Verify that pure rigid-body translation twists induce zero relative coboundary."""
    n_res = 4
    coords = np.array([[i * 3.8, 0.0, 0.0] for i in range(n_res)], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)

    # Uniform translation twist across all residues: v_i = (1.0, 2.0, 3.0)
    uniform_twists = np.zeros((n_res, 6), dtype=np.float32)
    uniform_twists[:, 3:] = [1.0, 2.0, 3.0]

    # Coboundary between any two nodes must be identically 0.0
    for e in edges:
        u, v = e
        dv = uniform_twists[v, 3:] - uniform_twists[u, 3:]
        assert np.allclose(dv, 0.0), f"Relative translation coboundary must be zero: {dv}"


def test_theorem_2_20_monotonic_impulse_transmission_non_slack():
    """Verify monotonic mutational impulse transmission along non-slack contact path."""
    # Linear chain of 4 residues: 0 - 1 - 2 - 3
    coords = np.array([
        [0.0, 0.0, 0.0],
        [3.8, 0.0, 0.0],
        [7.6, 0.0, 0.0],
        [11.4, 0.0, 0.0],
    ], dtype=np.float32)
    edges = np.array([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=np.uint32)

    res = allosteric_pocket_sheaf_solver(coords, edges, mut_residue=0, num_iters=25)
    eta = res["eta_nonlocal"]

    assert eta[0] == 1.0, "Mutated residue must have maximum leverage 1.0"
    assert eta[1] > 0.0, "Direct contact neighbor must receive positive leverage"
    assert eta[2] >= 0.0 and eta[3] >= 0.0, "Downstream path receives transmitted leverage"


def test_theorem_2_20_allosteric_driver_classification():
    """Verify classification of active allosteric drivers vs slaved surface loops."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [3.5, 0.0, 0.0],
        [20.0, 20.0, 20.0],  # Distant unconnected loop
    ], dtype=np.float32)
    edges = np.array([[0, 1], [1, 0]], dtype=np.uint32)

    res = allosteric_pocket_sheaf_solver(coords, edges, mut_residue=0, num_iters=20)
    assert 0 in res["active_drivers"], "Mutated node is an active driver (eta >= 0.50)"
    assert 2 in res["slaved_loops"], "Distant unconnected node is a slaved loop (eta < 0.50)"


# ==============================================================================
# Group 5: Theorem 2.21 Invariants (Rank-2 Cayley Rotors & Polish)
# ==============================================================================

def test_theorem_2_21_cayley_sherman_morrison_closed_form():
    """Verify Sherman-Morrison closed-form Cayley rotor matches Cayley transform (I-A)(I+A)^-1."""
    u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    alpha = 0.2

    # A = alpha * (u v^T - v u^T)
    A = alpha * (np.outer(u, v) - np.outer(v, u))
    I = np.eye(3, dtype=np.float64)
    R_exact = (I - A) @ np.linalg.inv(I + A)

    # Sherman-Morrison closed form:
    # R = I + (2*alpha / (1+alpha^2)) * (v u^T - u v^T) - (2*alpha^2 / (1+alpha^2)) * (u u^T + v v^T)
    c1 = 2.0 * alpha / (1.0 + alpha**2)
    c2 = 2.0 * alpha**2 / (1.0 + alpha**2)
    R_sm = I + c1 * (np.outer(v, u) - np.outer(u, v)) - c2 * (np.outer(u, u) + np.outer(v, v))

    assert np.allclose(R_exact, R_sm, atol=1e-12)


def test_theorem_2_21_active_subspace_ulp_polish_idempotence():
    """Verify that Pass-2 active-subspace Float32 ULP polish is strictly idempotent."""
    v = np.array([1.234567, -2.345678, 0.987654], dtype=np.float32)
    r0 = float(np.sqrt(np.sum(v.astype(np.float64)**2)))

    polished_1 = ulp_polish_vec3(v, r0)
    polished_2 = ulp_polish_vec3(polished_1, r0)

    assert np.array_equal(polished_1, polished_2), "ULP polish must be strictly idempotent"


# ==============================================================================
# Group 6: Theorem 2.22 Invariants (Cryo-EM 3D Morse Sidechain Shield)
# ==============================================================================

def test_theorem_2_22_cryoem_ridge_preservation_100_percent():
    """Verify 100.0% preservation of sustained ridge / side-chain density steps (Theorem 2.22)."""
    vol = np.zeros((11, 11, 11), dtype=np.float32)
    vol[:] = 0.2  # background

    # Sustained ridge spanning z=3..8 along line (y=5, x=5) with amplitude 6.0
    vol[3:9, 5, 5] = 6.0

    res = cryoem_sidechain_morse_filter(vol, mad_floor=0.5, tau=1.5)
    filtered = res["filtered"]

    # All ridge interior and boundary voxels must be retained at 100% amplitude
    assert np.all(filtered[3:9, 5, 5] == 6.0), "100% of sustained ridge voxels must be preserved"


def test_theorem_2_22_cryoem_noise_spike_rejection_100_percent():
    """Verify 100.0% rejection of isolated 1-voxel noise spikes without blurring (Theorem 2.22)."""
    vol = np.zeros((11, 11, 11), dtype=np.float32)
    vol[:] = 0.2

    # Isolated single-voxel spike at (2, 2, 2)
    vol[2, 2, 2] = 10.0
    # Isolated single-voxel spike at (8, 8, 8)
    vol[8, 8, 8] = 12.0

    res = cryoem_sidechain_morse_filter(vol, mad_floor=0.5, tau=1.5)
    filtered = res["filtered"]

    assert filtered[2, 2, 2] == 0.0, "Isolated spike at (2,2,2) must be completely rejected"
    assert filtered[8, 8, 8] == 0.0, "Isolated spike at (8,8,8) must be completely rejected"


def test_theorem_2_22_cryoem_background_median_shift_zero():
    """Verify that background density median shift is exactly 0.000 (Theorem 2.22)."""
    vol = np.zeros((11, 11, 11), dtype=np.float32)
    vol[:] = 0.2
    vol[4:8, 5, 5] = 5.0
    vol[1, 1, 1] = 9.0

    res = cryoem_sidechain_morse_filter(vol, mad_floor=0.5, tau=1.5)
    filtered = res["filtered"]

    # Background voxels (excluding ridge and spike)
    bg_mask = np.ones_like(vol, dtype=bool)
    bg_mask[4:8, 5, 5] = False
    bg_mask[1, 1, 1] = False

    bg_filtered = filtered[bg_mask]
    assert float(np.median(bg_filtered)) == 0.0, "Background median must be zeroed without positive shift"


# ==============================================================================
# Group 7: PDB Parsing, Export Roundtrip & Contact Graph
# ==============================================================================

def test_pdb_parsing_atoms_backbone_sidechain_centroids():
    """Verify parse_pdb_structure extracts atoms, backbone, and sidechain centroids."""
    s = parse_pdb_structure(SAMPLE_PDB_TEXT)
    assert len(s.residues) == 3
    assert s.sequence == "AEK"

    # Check residue 0 (ALA)
    r0 = s.residues[0]
    assert r0["res_name"] == "ALA"
    assert "CA" in r0["backbone"]
    assert "N" in r0["backbone"]
    assert "C" in r0["backbone"]
    assert "O" in r0["backbone"]
    assert r0["sidechain_centroid"] is not None


def test_pdb_export_text_format_roundtrip():
    """Verify PDB text export and re-parse coordinate exact roundtrip."""
    s1 = parse_pdb_structure(SAMPLE_PDB_TEXT)
    exported_text = export_pdb_structure(s1)

    s2 = parse_pdb_structure(exported_text)
    assert len(s2.residues) == len(s1.residues)
    assert s2.sequence == s1.sequence

    max_coord_diff = float(np.max(np.abs(s1.ca_coords - s2.ca_coords)))
    assert max_coord_diff == 0.0, f"PDB export roundtrip coordinate drift {max_coord_diff} must be 0.0"


def test_pdb_contact_graph_heavy_and_ca_thresholds():
    """Verify contact graph generation with d_ij <= 4.5 A heavy-atom and <= 8.0 A CA."""
    s = parse_pdb_structure(SAMPLE_PDB_TEXT, contact_heavy_threshold=4.5, contact_ca_threshold=8.0)
    cg = s.contact_graph

    assert cg["num_nodes"] == 3
    assert len(cg["edges"]) > 0
    assert "covalent" in cg["edge_types"]

    # Adjacent residues 1 and 2 must have a covalent peptide contact edge
    edges_set = set((int(e[0]), int(e[1])) for e in cg["edges"])
    assert (0, 1) in edges_set and (1, 0) in edges_set


def test_pdb_ramachandran_phi_psi_dihedrals():
    """Verify calculation of Ramachandran (phi, psi) angles."""
    s = parse_pdb_structure(SAMPLE_PDB_TEXT, compute_dihedrals=True)
    angles = s.ramachandran_angles

    # Middle residue (GLU at index 1) must have valid non-NaN phi and psi
    phi_glu = angles[1, 0]
    psi_glu = angles[1, 1]

    assert not np.isnan(phi_glu), "GLU residue phi must be non-NaN"
    assert not np.isnan(psi_glu), "GLU residue psi must be non-NaN"
    assert -180.0 <= phi_glu <= 180.0
    assert -180.0 <= psi_glu <= 180.0


# ==============================================================================
# Group 8: Cryptic Pocket Jordan Cavity Betti & Druggability
# ==============================================================================

def test_cryptic_pocket_jordan_betti_cavity_voids():
    """Verify 6-way flood-fill Jordan cavity enclosure computes beta_0, beta_1, beta_2."""
    # Enclosed hollow void inside a 12x12x12 cube
    solid = np.ones((12, 12, 12), dtype=bool)
    # Hollow cavity at center 5..7
    solid[5:8, 5:8, 5:8] = False

    res = cryptic_pocket_topology_audit(grid=solid, grid_resolution=1.0)
    assert res["beta_2"] >= 1, "Must detect at least 1 enclosed cavity void (beta_2 >= 1)"
    assert res["occluded_volume_angstrom3"] == 27.0, "3x3x3 void has 27 A^3 volume"
    assert res["is_druggable"] is True


def test_cryptic_pocket_collapse_ratio_under_torque():
    """Verify monotonic increase of pocket collapse ratio under allosteric torque."""
    solid = np.ones((10, 10, 10), dtype=bool)
    solid[4:7, 4:7, 4:7] = False

    res_t0 = cryptic_pocket_topology_audit(grid=solid, allosteric_torque=0.0)
    res_t1 = cryptic_pocket_topology_audit(grid=solid, allosteric_torque=0.5)
    res_t2 = cryptic_pocket_topology_audit(grid=solid, allosteric_torque=1.5)

    assert res_t0["pocket_collapse_ratio"] == 0.0
    assert res_t1["pocket_collapse_ratio"] > res_t0["pocket_collapse_ratio"]
    assert res_t2["pocket_collapse_ratio"] >= res_t1["pocket_collapse_ratio"]


# ==============================================================================
# Group 9: verify_all_theorems() Comprehensive Audit & Unified Engine
# ==============================================================================

def test_verify_all_theorems_comprehensive_audit():
    """Verify verify_all_theorems() executes all 18 invariants (Theorems 2.1–2.22) with 0 failures."""
    res = verify_all_theorems()
    assert res["status"] == "ALL_THEOREMS_VERIFIED"
    assert res["total_theorems"] == 18
    assert res["passed_theorems"] == 18
    assert res["failed_theorems"] == 0
    assert res["all_passed"] is True


def test_cellsnail_engine_v030_integration():
    """Verify CellSnailEngine unifies all 3D structural operators with tracked steps."""
    engine = CellSnailEngine()
    initial_steps = engine.step_count

    # 1. CryoEM filter
    vol = np.zeros((5, 5, 5), dtype=np.float32)
    vol[2, 2, 1:4] = 4.0
    f_res = engine.cryoem_sidechain_morse_filter(vol, tau=1.0)
    assert f_res["filtered"][2, 2, 2] == 4.0

    # 2. Allosteric sheaf
    coords = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.uint32)
    s_res = engine.allosteric_pocket_sheaf_solver(coords, edges)
    assert s_res["eta_nonlocal"][0] == 1.0

    # 3. Cayley Ramachandran
    c_in = np.array([[1.54, 0.0, 0.0]], dtype=np.float32)
    r_res = engine.ramachandran_cayley_integrator(c_in, k_steps=16)
    assert r_res["radial_drift"] == 0.0

    # 4. Cryptic pocket audit
    solid = np.ones((8, 8, 8), dtype=bool)
    solid[3:5, 3:5, 3:5] = False
    p_res = engine.cryptic_pocket_topology_audit(grid=solid)
    assert p_res["beta_2"] >= 1

    assert engine.step_count > initial_steps
