"""tests/test_mode_a_oracle.py
===========================
Machine-checked oracle verification test suite for Theorems 2.1–2.15.
"""

import pytest
from cellsnail.benchmark.mode_a_oracle import (
    run_mode_a_oracle_audit,
    verify_bcm2712_arena_geometry,
    verify_cascade_norm_conservation,
    verify_cayley_transform_unitarity,
    verify_cellular_sheaf_coboundary,
    verify_discrete_energy_identity,
    verify_h_clock_h_nope_orthogonality,
    verify_ideal_mhd_transverse_invariance,
    verify_kelvin_mandel_trace,
    verify_morse_dropout_shielding,
    verify_multiomic_dosage_linkage,
    verify_rope_isometric_invariance,
    verify_skew_antisymmetry,
    verify_step0_homotopy,
    verify_triad_orthogonality,
    verify_ultrametric_site_condensation,
)


def test_theorem_2_1_triad_orthogonality():
    res = verify_triad_orthogonality()
    assert res["passed"] is True
    assert res["residual"] < 1e-14


def test_theorem_2_2_skew_antisymmetry():
    res = verify_skew_antisymmetry()
    assert res["passed"] is True
    assert res["residual"] < 1e-15


def test_theorem_2_3_step0_homotopy():
    res = verify_step0_homotopy()
    assert res["passed"] is True
    assert res["residual"] < 1e-15


def test_theorem_2_4_discrete_energy_identity():
    res = verify_discrete_energy_identity()
    assert res["passed"] is True
    assert res["residual"] < 1e-14


def test_theorem_2_5_rope_isometric_invariance():
    res = verify_rope_isometric_invariance()
    assert res["passed"] is True
    assert res["residual"] < 1e-14


def test_theorem_2_6_cascade_norm_conservation():
    res = verify_cascade_norm_conservation()
    assert res["passed"] is True
    assert res["residual"] < 1e-14


def test_theorem_2_7_kelvin_mandel_trace():
    res = verify_kelvin_mandel_trace()
    assert res["passed"] is True
    assert res["residual"] <= 1.14e-13


def test_theorem_2_8_cayley_transform_unitarity():
    res = verify_cayley_transform_unitarity()
    assert res["passed"] is True
    assert res["residual"] < 1e-14


def test_theorem_2_9_ideal_mhd_transverse_invariance():
    res = verify_ideal_mhd_transverse_invariance()
    assert res["passed"] is True
    assert res["residual"] < 1e-15


def test_theorem_2_10_cellular_sheaf_coboundary():
    res = verify_cellular_sheaf_coboundary()
    assert res["passed"] is True
    assert res["residual"] == 0.0


def test_theorem_2_11_bcm2712_arena_geometry():
    res = verify_bcm2712_arena_geometry()
    assert res["passed"] is True
    assert res["arena_size_bytes"] <= 2097152
    assert res["arena_size_bytes"] % 64 == 0


def test_theorem_2_12_morse_dropout_shielding():
    res = verify_morse_dropout_shielding()
    assert res["passed"] is True
    assert res["ramp_rejected"] is True
    assert res["spike_preserved"] is True
    assert res["spurious_covariance"] == 0.0


def test_theorem_2_13_h_clock_h_nope_orthogonality():
    res = verify_h_clock_h_nope_orthogonality()
    assert res["passed"] is True
    assert res["clock_channel_detected"] is True
    assert res["residual"] < 1e-15


def test_theorem_2_14_ultrametric_site_condensation():
    res = verify_ultrametric_site_condensation()
    assert res["passed"] is True
    assert res["three_point_preserved"] is True
    assert res["height_equality_preserved"] is True


def test_theorem_2_15_multiomic_dosage_linkage():
    res = verify_multiomic_dosage_linkage()
    assert res["passed"] is True
    assert res["trace_residual"] <= 1.14e-13
    assert res["quad_form_residual"] <= 1.14e-13


def test_full_mode_a_oracle_audit():
    audit = run_mode_a_oracle_audit()
    assert audit["status"] == "ALL_15_INVARIANTS_VERIFIED"
    assert audit["passed_invariants"] == 15
    assert audit["failed_invariants"] == 0
    assert audit["max_residual"] <= 1.14e-13
