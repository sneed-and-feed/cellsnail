"""tests/test_native_bindings.py
==============================
Unit tests for Cellsnail Rust SIMD native C-ABI bindings and Python operator parity.
Verifies all 6 dual operator pairs, zero-allocation SRAM arena geometry, and mathematical invariants.
"""

import numpy as np
import pytest

from cellsnail._native import (
    is_native_available,
    load_native_library,
    native_cell_cycle_deconfounder,
    native_genotype_tiebreaker_so10,
    native_morse_dropout_filter,
    native_multiomic_covariance_rotor,
    native_pseudotime_spherical_cascade,
    native_spatial_tissue_sheaf_solver,
)
from cellsnail.operators import (
    CellSnailEngine,
    beltrami_morse_gate,
    cell_cycle_deconfounder,
    compute_spatial_mad_floor_256,
    contact_sheaf_se2_step,
    continuity_clutter_filter,
    genotype_tiebreaker_so10,
    kelvin_mandel_so6_rotor,
    morse_dropout_filter,
    multiomic_covariance_rotor,
    pseudotime_spherical_cascade,
    skew_triad_step,
    slow_fast_cascade_k16,
    spatial_tissue_sheaf_solver,
)


def test_native_library_available():
    assert is_native_available(), "cellsnail_rs shared library must be compiled and loadable"
    lib = load_native_library()
    assert lib is not None


def test_arena_size_bytes_and_sram_ceiling():
    lib = load_native_library()
    sz = lib.cellsnail_arena_size_bytes()
    assert sz == 983_872, f"Expected exactly 983,872 bytes, got {sz}"
    assert sz <= 2_097_152, f"Arena size {sz} exceeds 2.0 MB SRAM ceiling"
    assert sz % 64 == 0, f"Arena size {sz} must be 64-byte aligned"


def test_pair1_morse_dropout_filter():
    # 1. Monotone ramp: [1.0, 2.0, 3.0] -> rejected
    mono = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    res_canonical = beltrami_morse_gate(mono, tau_mad=0.5)
    res_alias = morse_dropout_filter(mono, tau_mad=0.5)

    assert res_canonical["filtered"][1] == 0.0, "Monotone slope must be rejected"
    assert res_alias["filtered"][1] == 0.0, "Alias must produce identical output"
    assert np.allclose(res_canonical["filtered"], res_alias["filtered"])

    # 2. Isolated thin spike: [0.1, 5.0, 0.1] with tau=1.0 -> preserved at 100% amplitude
    spike = np.array([0.1, 5.0, 0.1], dtype=np.float32)
    res_spike = morse_dropout_filter(spike, tau_mad=1.0)
    assert res_spike["filtered"][1] == 5.0, "Isolated spike must be preserved at 100% amplitude"
    assert res_spike["confirmed_innovations"][1] is True or res_spike["confirmed_innovations"][1] == 1

    # 3. Native C-ABI direct invocation
    out_native, sal_native = native_morse_dropout_filter(spike, mad_floor=1.0, tau=1.0)
    assert out_native[1] == 5.0
    assert out_native[0] == 0.0
    assert out_native[2] == 0.0


def test_pair2_cell_cycle_deconfounder():
    # 10 steps, 3 channels
    # Channel 0: alternating flicker [+1, -1, +1, -1, ...] -> periodic clock clutter
    # Channel 1: stationary constant -> authentic state
    # Channel 2: steady ramp -> authentic state
    t_steps = 10
    mat = np.zeros((t_steps, 3), dtype=np.float32)
    for t in range(t_steps):
        mat[t, 0] = 1.0 if t % 2 == 0 else -1.0
        mat[t, 1] = 2.5
        mat[t, 2] = float(t) * 0.5

    filt_canon, mask_canon = continuity_clutter_filter(mat, clock_threshold=0.85)
    filt_alias, mask_alias = cell_cycle_deconfounder(mat, clock_threshold=0.85)

    assert mask_canon[0] is True or mask_canon[0] == 1, "Flicker channel must be identified as clock"
    assert mask_canon[1] is False or mask_canon[1] == 0, "Stationary channel must be nope"
    assert mask_canon[2] is False or mask_canon[2] == 0, "Ramp channel must be nope"
    assert np.array_equal(mask_canon, mask_alias)

    # Channel 0 frozen to baseline mat[0, 0] == 1.0
    assert np.all(filt_canon[:, 0] == 1.0)
    # Channel 1 and 2 preserved unmodified
    assert np.allclose(filt_canon[:, 1], mat[:, 1])
    assert np.allclose(filt_canon[:, 2], mat[:, 2])

    # Native direct invocation
    filt_nat, mask_nat = native_cell_cycle_deconfounder(mat)
    assert mask_nat[0] is True or mask_nat[0] == 1
    assert np.all(filt_nat[:, 0] == 1.0)


def test_pair3_genotype_tiebreaker_so10_norm_drift():
    # 10-state genotype vector
    u = np.ones(10, dtype=np.float64) * 0.2
    v = np.zeros(10, dtype=np.float64)
    v[0] = 1.0  # ||v|| == 1.0

    v_retract_canon = skew_triad_step(u, v)
    v_retract_alias = genotype_tiebreaker_so10(u, v)

    norm_in = np.linalg.norm(v)
    norm_canon = np.linalg.norm(v_retract_canon)
    norm_alias = np.linalg.norm(v_retract_alias)

    drift_canon = abs(norm_canon - norm_in)
    drift_alias = abs(norm_alias - norm_in)

    assert drift_canon < 1e-12, f"Canonical radial norm drift {drift_canon} exceeds 1e-12"
    assert drift_alias < 1e-12, f"Alias radial norm drift {drift_alias} exceeds 1e-12"
    assert np.allclose(v_retract_canon, v_retract_alias)

    # Native direct invocation
    v_retract_nat = native_genotype_tiebreaker_so10(u.astype(np.float32), v.astype(np.float32))
    norm_nat = np.linalg.norm(v_retract_nat)
    assert abs(norm_nat - 1.0) < 1e-6


def test_pair4_pseudotime_spherical_cascade():
    dim = 8
    v0 = np.zeros(dim, dtype=np.float64)
    v0[0] = 3.0
    v0[1] = 4.0  # ||v0|| == 5.0
    u_seq = np.random.RandomState(42).randn(16, dim).astype(np.float64)

    v16_canon = slow_fast_cascade_k16(u_seq, v0, gamma=0.05, k_steps=16)
    v16_alias = pseudotime_spherical_cascade(u_seq, v0, gamma=0.05, k_steps=16)

    norm_0 = np.linalg.norm(v0)
    norm_16_canon = np.linalg.norm(v16_canon)
    norm_16_alias = np.linalg.norm(v16_alias)

    assert abs(norm_16_canon - norm_0) < 1e-10, "16-step cascade must conserve norm"
    assert abs(norm_16_alias - norm_0) < 1e-10, "Pseudotime cascade must conserve norm"
    assert np.allclose(v16_canon, v16_alias)

    # Native direct invocation
    v16_nat = native_pseudotime_spherical_cascade(v0.astype(np.float32), u_seq.astype(np.float32), 0.05)
    assert abs(np.linalg.norm(v16_nat) - 5.0) < 1e-5


def test_pair5_spatial_tissue_sheaf_solver():
    # 1. Pushing into wall (dot > 0): dx=1.0, nx=1.0 -> clamped to 0, hit_wall=True
    ox, oy, telem = spatial_tissue_sheaf_solver(1.0, 0.0, 1.0, 0.0)
    assert telem["hit_wall"] is True
    assert ox == 0.0
    assert oy == 0.0

    # 2. Tangential motion (dot == 0): dy=1.0, nx=1.0 -> preserved, hit_wall=False
    ox_tan, oy_tan, telem_tan = spatial_tissue_sheaf_solver(0.0, 1.0, 1.0, 0.0)
    assert telem_tan["hit_wall"] is False
    assert ox_tan == 0.0
    assert oy_tan == 1.0

    # 3. Canonical and alias parity
    ox_c, oy_c, telem_c = contact_sheaf_se2_step(1.0, 0.0, 1.0, 0.0)
    assert ox_c == ox and oy_c == oy and telem_c["hit_wall"] == telem["hit_wall"]

    # 4. Native direct invocation
    ox_n, oy_n, hit_n = native_spatial_tissue_sheaf_solver(1.0, 0.0, 1.0, 0.0)
    assert hit_n is True
    assert ox_n == 0.0
    assert oy_n == 0.0


def test_pair6_multiomic_covariance_rotor():
    # 3D rotation matrix around Z-axis (pi/4)
    theta = np.pi / 4.0
    c, s = np.cos(theta), np.sin(theta)
    rot3 = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # 6x6 covariance matrix with distinct diagonal elements
    cov6 = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float64)
    tr_orig = np.trace(cov6)  # 21.0

    m_canon, c_rot_canon = kelvin_mandel_so6_rotor(rot3, cov6)
    m_alias, c_rot_alias = multiomic_covariance_rotor(rot3, cov6)

    # Trace conservation: Tr(C_rot) == Tr(C_base)
    tr_canon = np.trace(c_rot_canon)
    tr_alias = np.trace(c_rot_alias)

    assert abs(tr_canon - tr_orig) < 1e-12, f"Trace difference {abs(tr_canon - tr_orig)} exceeds 1e-12"
    assert abs(tr_alias - tr_orig) < 1e-12, f"Trace difference {abs(tr_alias - tr_orig)} exceeds 1e-12"
    assert np.allclose(m_canon, m_alias)
    assert np.allclose(c_rot_canon, c_rot_alias)

    # SO(6) isometry check: M^T @ M == I_6
    assert np.allclose(m_canon.T @ m_canon, np.eye(6), atol=1e-12)

    # Native direct invocation
    m_nat, c_rot_nat = native_multiomic_covariance_rotor(rot3.astype(np.float32), cov6.astype(np.float32))
    assert abs(np.trace(c_rot_nat) - 21.0) < 1e-5


def test_cellsnail_engine_unification():
    engine = CellSnailEngine()
    # Step through all operators via unified engine
    res = engine.morse_dropout_filter(np.array([0.1, 5.0, 0.1], dtype=np.float32), tau=1.0)
    assert res["filtered"][1] == 5.0

    deconf, mask = engine.cell_cycle_deconfounder(np.ones((4, 2), dtype=np.float32))
    assert deconf.shape == (4, 2)

    v_ret = engine.genotype_tiebreaker_so10(np.ones(10), np.eye(10)[0])
    assert abs(np.linalg.norm(v_ret) - 1.0) < 1e-6

    v16 = engine.pseudotime_spherical_cascade(np.array([1.0, 0.0]), np.ones((16, 2)) * 0.1)
    assert abs(np.linalg.norm(v16) - 1.0) < 1e-6

    dx, dy, t = engine.spatial_tissue_sheaf_solver(0.0, 1.0, 1.0, 0.0)
    assert t["hit_wall"] is False and dy == 1.0

    m, c = engine.multiomic_covariance_rotor(np.eye(3), np.eye(6))
    assert np.allclose(m, np.eye(6))

    assert engine.step_count > 0


def test_v020_adaptive_native_operators():
    from cellsnail._native import (
        native_morse_dropout_filter_adaptive,
        native_morse_plateau_filter,
        native_spatial_tissue_sheaf_solver_huber,
        native_multiomic_covariance_rotor_shrinkage,
    )

    # 1. Morse adaptive blending
    spike = np.array([0.1, 5.0, 0.1], dtype=np.float32)
    prior = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    out, sal = native_morse_dropout_filter_adaptive(spike, prior, mad_floor=1.0, tau=1.0, lambda_blend=0.5)
    assert out[1] == 5.0, "Focal spike must be preserved"
    assert out[0] == 1.0 and out[2] == 1.0, "Non-focal bins must blend prior"

    # 2. Morse plateau smoothing
    prof = np.array([2.0, 2.0, 8.4, 7.6, 8.3, 7.7, 8.2, 7.8, 8.1, 7.9, 2.0, 2.0], dtype=np.float32)
    out_p = native_morse_plateau_filter(prof, window_size=5)
    assert out_p[0] == 2.0 and out_p[1] == 2.0, "Plateau boundary step edges must be pinned"
    assert abs(out_p[5] - 8.0) <= abs(prof[5] - 8.0), "Plateau interior variance must be reduced"

    # 3. Spatial sheaf Huber soft-clamping
    ox, oy, hit = native_spatial_tissue_sheaf_solver_huber(10.0, 0.0, 1.0, 0.0, delta_jitter=0.0)
    assert hit and ox == 0.0, "Zero jitter must enforce hard clamp"
    ox_leak, oy_leak, hit_leak = native_spatial_tissue_sheaf_solver_huber(10.0, 0.0, 1.0, 0.0, delta_jitter=30.0, huber_delta=5.0)
    assert hit_leak and 0.0 < ox_leak <= 10.0, "High jitter must allow Huber soft leakage"

    # 4. Multi-omic covariance rotor Ledoit-Wolf shrinkage
    rot3 = np.eye(3, dtype=np.float32)
    cov6 = np.diag(np.arange(1, 7, dtype=np.float32))
    m6, cov_out = native_multiomic_covariance_rotor_shrinkage(rot3, cov6, alpha_shrink=0.5)
    assert abs(np.trace(cov_out) - np.trace(cov6)) < 1e-4, "Shrinkage must conserve exact trace"

