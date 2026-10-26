"""tests/test_adversarial_stress.py
===================================
Adversarial Stress Test Suite for Single-Cell Mathematical Operators.

Comprehensive empirical verification of biological edge cases, degenerate geometries,
runaway scales, and algebraic invariants across both Python reference oracles and
native #![no_std] Rust SIMD C-ABI bindings:

1. Extreme Noise (SNR < 0 dB down to -20 dB) and Extreme Dropout (50%, 80%, 95%) on
   `morse_dropout_filter` / `beltrami_morse_gate`:
   - Verifies exact zero spurious correlation injection (|r| > 0.70 == 0) across independent genes.
   - Evaluates real wet-lab ERCC spike-in controls from `sc_mixology` across 50%, 80%, and 95% dropout.
   - Contrasts against MAGIC graph diffusion and ALRA SVD low-rank smearing.
   - Evaluates 2D spatial fields under extreme noise and complete dropout.

2. Degenerate Inputs on `genotype_tiebreaker_so10`:
   - Zero candidate vectors (v=0, u=0, both=0).
   - Collinear vectors (u = alpha * v for alpha in [-10^6, +10^6]).
   - Identical probabilities (k-way ties for k in {2, 3, 5, 10} and flat uniform distributions).
   - Invariant: 0.000e+00 radial norm drift (| ||v_retract|| - ||v|| | < 1e-14 in f64, < 1e-6 in f32).

3. Runaway Velocities on `pseudotime_spherical_cascade`:
   - Extreme velocity magnitudes: 10^3, 10^6, 10^9.
   - Variable initial state norms: 1.0, 10^3, 10^6, 10^-6.
   - Invariant: Exact norm conservation across all 16 unrolled cascade steps.

4. Spatial Sheaf Boundary Collisions:
   - Direct wall penetration at arbitrary approach angles (0 deg to 89 deg).
   - Acute corner traps (convergent walls with angle 15 deg to 75 deg).
   - Invariant: Non-penetration residual = 0.00e+00 (max(0, (delta v)_n) == 0.00e+00).

5. Multiomic Covariance Rotor on Ill-Conditioned & Singular Matrices:
   - Singular covariance matrices (rank 0, 1, 2, 3, 4, 5).
   - Ill-conditioned covariance tensors (condition numbers 10^4, 10^8, 10^12, 10^16).
   - Invariant: Exact trace conservation Tr(M C M^T) == Tr(C) and SO(6) isometry M^T M == I_6.
"""

from typing import Dict, List, Tuple
import numpy as np
import pytest

from cellsnail._native import (
    is_native_available,
    native_genotype_tiebreaker_so10,
    native_morse_dropout_filter,
    native_multiomic_covariance_rotor,
    native_pseudotime_spherical_cascade,
    native_spatial_tissue_sheaf_solver,
)
from cellsnail.benchmark.baselines import (
    alra_impute,
    magic_impute,
)
from cellsnail.benchmark.mode_b_blind import (
    apply_snr_noise,
    apply_technical_dropout,
    benchmark_ercc_spurious_correlation,
    binomial_count_split,
)
from cellsnail.data import load_sc_mixology
from cellsnail.operators import (
    beltrami_morse_gate,
    contact_sheaf_se2_step,
    genotype_tiebreaker_so10,
    kelvin_mandel_so6_rotor,
    morse_dropout_filter,
    multiomic_covariance_rotor,
    pseudotime_spherical_cascade,
    skew_triad_step,
    slow_fast_cascade_k16,
    spatial_tissue_sheaf_solver,
)


# ==============================================================================
# Challenge 1: Extreme Noise & Dropout on morse_dropout_filter / beltrami_morse_gate
# ==============================================================================

class TestExtremeNoiseAndDropoutSpuriousCorrelation:
    """Stress tests morse_dropout_filter under extreme noise (SNR < 0 dB) and dropout (50-95%)."""

    @pytest.mark.parametrize("snr_db", [-20.0, -10.0, -5.0, -1.0, 0.0])
    @pytest.mark.parametrize("dropout_rate", [0.50, 0.80, 0.95])
    def test_synthetic_independent_genes_zero_spurious_correlation(self, snr_db: float, dropout_rate: float):
        """Verify that filtering independent Poisson gene counts under extreme SNR < 0 dB and

        dropout up to 95% injects zero spurious correlations (|r| > 0.70).
        """
        rng = np.random.default_rng(1337)
        n_cells = 300
        n_genes = 25

        # Ground truth: 25 completely independent genes with varying basal expression
        lambdas = rng.uniform(2.0, 25.0, size=n_genes)
        raw_counts = rng.poisson(lam=lambdas, size=(n_cells, n_genes)).astype(np.float32)

        # Inject extreme noise and technical dropout
        noisy = apply_snr_noise(raw_counts, snr_db=snr_db, random_state=42)
        corrupted = apply_technical_dropout(noisy, dropout_fraction=dropout_rate, random_state=42)

        # Split into training and testing splits via zero-leakage binomial count splitting
        x_train, _ = binomial_count_split(corrupted, p=0.5, random_state=42)

        # 1. cellsnail morse_dropout_filter (Theorem 2.12: separable pointwise filtering)
        x_cellsnail = np.zeros_like(x_train)
        for j in range(n_genes):
            res = morse_dropout_filter(x_train[:, j], tau_mad=1.0)
            x_cellsnail[:, j] = res["filtered"]

        corr_cellsnail = np.corrcoef(x_cellsnail, rowvar=False)
        corr_cellsnail = np.nan_to_num(corr_cellsnail, nan=0.0)
        np.fill_diagonal(corr_cellsnail, 0.0)

        spurious_pairs = int(np.sum(np.abs(corr_cellsnail) > 0.70) // 2)
        assert spurious_pairs == 0, (
            f"cellsnail injected {spurious_pairs} spurious correlations under SNR={snr_db} dB, dropout={dropout_rate}"
        )

        # 2. Check native C-ABI kernel parity
        if is_native_available():
            x_native = np.zeros_like(x_train)
            for j in range(n_genes):
                out_j, _ = native_morse_dropout_filter(x_train[:, j], mad_floor=1.0, tau=1.0)
                x_native[:, j] = out_j
            corr_nat = np.corrcoef(x_native, rowvar=False)
            corr_nat = np.nan_to_num(corr_nat, nan=0.0)
            np.fill_diagonal(corr_nat, 0.0)
            spurious_nat = int(np.sum(np.abs(corr_nat) > 0.70) // 2)
            assert spurious_nat == 0, f"Native kernel injected {spurious_nat} spurious correlations"

    @pytest.mark.parametrize("dropout_rate", [0.50, 0.80, 0.95])
    def test_ercc_wetlab_extreme_dropout_sweep(self, dropout_rate: float):
        """Test on curated wet-lab ERCC spike-in benchmark cohort (n=300) across extreme dropout.

        Confirms Theorem 2.12: cellsnail guarantees 0 spurious correlations, whereas
        MAGIC and ALRA diffuse across cells and inject dozens to hundreds of spurious pairs.
        """
        res = benchmark_ercc_spurious_correlation(dropout_rate=dropout_rate, random_state=42)

        # cellsnail maintains exact zero spurious pairs
        assert res["spurious_pairs"]["cellsnail"] == 0, (
            f"cellsnail had {res['spurious_pairs']['cellsnail']} spurious pairs at dropout={dropout_rate}"
        )
        assert res["spurious_rate"]["cellsnail"] == 0.0

        # Baselines suffer from massive spurious correlation injection
        if dropout_rate >= 0.80:
            assert res["spurious_pairs"]["alra"] > 100, "ALRA should inject > 100 spurious pairs"
            assert res["spurious_pairs"]["magic"] > 0, "MAGIC should inject spurious pairs"

    def test_beltrami_morse_gate_2d_extreme_noise_and_dropout(self):
        """Test 2D spatial Beltrami-Morse gate under severe noise and 95% zero dropout."""
        rng = np.random.RandomState(42)
        grid_size = 64
        # Two independent spatial feature channels
        ch1 = rng.randn(grid_size, grid_size).astype(np.float32)
        ch2 = rng.randn(grid_size, grid_size).astype(np.float32)

        # 95% dropout
        ch1[rng.rand(grid_size, grid_size) < 0.95] = 0.0
        ch2[rng.rand(grid_size, grid_size) < 0.95] = 0.0

        out1 = beltrami_morse_gate(ch1)
        out2 = beltrami_morse_gate(ch2)

        # Verify no NaN or Inf in outputs
        assert np.all(np.isfinite(out1["filtered"]))
        assert np.all(np.isfinite(out2["filtered"]))

        # Cross-channel correlation between independent spatial fields remains strictly uncoupled
        flat1 = out1["filtered"].ravel()
        flat2 = out2["filtered"].ravel()
        if np.std(flat1) > 0 and np.std(flat2) > 0:
            cross_r = abs(float(np.corrcoef(flat1, flat2)[0, 1]))
            assert cross_r < 0.20, f"Spurious spatial cross-correlation {cross_r} was too high"


# ==============================================================================
# Challenge 2: Degenerate Inputs on genotype_tiebreaker_so10
# ==============================================================================

class TestDegenerateGenotypeTiebreakerSO10:
    """Stress tests genotype_tiebreaker_so10 under degenerate inputs:

    zero vectors, collinear vectors, identical probabilities.
    """

    def test_zero_candidate_vectors(self):
        """Zero candidate vectors: v = 0, u = 0, or both."""
        dim = 10
        zero_v = np.zeros(dim, dtype=np.float64)
        u_rand = np.random.RandomState(42).randn(dim).astype(np.float64)

        # Case A: v is zero vector
        ret_v0 = genotype_tiebreaker_so10(u_rand, zero_v)
        norm_v0 = np.linalg.norm(zero_v)
        norm_ret_v0 = np.linalg.norm(ret_v0)
        drift_a = abs(norm_ret_v0 - norm_v0)
        assert drift_a == 0.0, f"Norm drift for v=0 was {drift_a}"
        assert np.all(ret_v0 == 0.0)

        # Case B: u is zero vector, v is non-zero
        v_rand = np.random.RandomState(101).randn(dim).astype(np.float64)
        norm_v = np.linalg.norm(v_rand)
        ret_u0 = genotype_tiebreaker_so10(zero_v, v_rand)
        norm_ret_u0 = np.linalg.norm(ret_u0)
        drift_b = abs(norm_ret_u0 - norm_v)
        assert drift_b < 1e-15, f"Norm drift for u=0 was {drift_b}"

        # Case C: Both u and v are zero
        ret_both0 = genotype_tiebreaker_so10(zero_v, zero_v)
        assert np.all(ret_both0 == 0.0)
        assert abs(np.linalg.norm(ret_both0)) == 0.0

        # Case D: With full interaction tensor W
        W = np.random.RandomState(88).randn(dim, dim, dim)
        ret_w_v0 = genotype_tiebreaker_so10(u_rand, zero_v, W=W)
        assert np.all(ret_w_v0 == 0.0)

        # Case E: Native Rust bindings parity
        if is_native_available():
            ret_nat_v0 = native_genotype_tiebreaker_so10(u_rand.astype(np.float32), zero_v.astype(np.float32))
            assert np.all(ret_nat_v0 == 0.0)

    @pytest.mark.parametrize("alpha", [-1e6, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.0, 10.0, 1e6])
    def test_collinear_vectors(self, alpha: float):
        """Collinear vectors: u = alpha * v."""
        dim = 10
        rng = np.random.RandomState(999)
        v = rng.randn(dim).astype(np.float64)
        v /= np.linalg.norm(v)  # unit norm
        norm_in = np.linalg.norm(v)  # exactly 1.0

        u = alpha * v

        # Python float64 implementation
        ret = genotype_tiebreaker_so10(u, v)
        norm_out = np.linalg.norm(ret)
        drift = abs(norm_out - norm_in)

        # Exact radial norm conservation (0.000e+00 drift)
        assert drift < 1e-14, f"Radial norm drift {drift:.3e} exceeds 1e-14 for alpha={alpha}"

        # With 3D interaction tensor W
        W = rng.randn(dim, dim, dim).astype(np.float64)
        ret_w = genotype_tiebreaker_so10(u, v, W=W)
        drift_w = abs(np.linalg.norm(ret_w) - norm_in)
        assert drift_w < 1e-14, f"Radial norm drift with W {drift_w:.3e} exceeds 1e-14 for alpha={alpha}"

        # Native Rust float32 implementation
        if is_native_available():
            v_f32 = v.astype(np.float32)
            u_f32 = u.astype(np.float32)
            ret_nat = native_genotype_tiebreaker_so10(u_f32, v_f32)
            norm_nat = np.linalg.norm(ret_nat)
            drift_nat = abs(norm_nat - 1.0)
            assert drift_nat < 1e-6, f"Native norm drift {drift_nat:.3e} exceeds 1e-6 for alpha={alpha}"

    @pytest.mark.parametrize("tie_k", [2, 3, 5, 10])
    def test_identical_probabilities_and_k_way_ties(self, tie_k: int):
        """Identical probabilities across genotypes (k-way ties and uniform distributions)."""
        dim = 10
        # Construct flat distribution across tie_k states
        v = np.zeros(dim, dtype=np.float64)
        v[:tie_k] = 1.0 / np.sqrt(tie_k)
        norm_in = np.linalg.norm(v)  # 1.000000

        # u also uniform or random
        u_flat = np.ones(dim, dtype=np.float64) * 0.1
        ret_flat = genotype_tiebreaker_so10(u_flat, v)
        drift_flat = abs(np.linalg.norm(ret_flat) - norm_in)
        assert drift_flat < 1e-14, f"Drift {drift_flat:.3e} for {tie_k}-way tie with flat u"

        # u random perturbation breaking the tie
        u_rand = np.random.RandomState(42).randn(dim).astype(np.float64)
        ret_rand = genotype_tiebreaker_so10(u_rand, v)
        drift_rand = abs(np.linalg.norm(ret_rand) - norm_in)
        assert drift_rand < 1e-14, f"Drift {drift_rand:.3e} for {tie_k}-way tie with random u"

        # Check native float32
        if is_native_available():
            ret_nat = native_genotype_tiebreaker_so10(u_rand.astype(np.float32), v.astype(np.float32))
            drift_nat = abs(np.linalg.norm(ret_nat) - 1.0)
            assert drift_nat < 1e-6


# ==============================================================================
# Challenge 3: Runaway Velocities on pseudotime_spherical_cascade
# ==============================================================================

class TestRunawayVelocitiesPseudotimeCascade:
    """Stress tests pseudotime_spherical_cascade under runaway velocities:

    testing velocity magnitudes 10^3, 10^6, 10^9 and large integration steps.
    """

    @pytest.mark.parametrize("velocity_scale", [1e3, 1e6, 1e9])
    @pytest.mark.parametrize("v0_norm", [1.0, 1e3, 1e6, 1e-6])
    def test_runaway_velocity_norm_conservation(self, velocity_scale: float, v0_norm: float):
        """Verify exact radial norm conservation under runaway velocities up to 10^9."""
        dim = 8
        rng = np.random.RandomState(2026)

        # Base vector v0 with specified norm
        v0 = rng.randn(dim).astype(np.float64)
        v0 = (v0 / np.linalg.norm(v0)) * v0_norm
        initial_norm = np.linalg.norm(v0)

        # Runaway velocity sequence: 16 time-steps with massive magnitudes
        u_runaway = rng.randn(16, dim).astype(np.float64) * velocity_scale

        # Integrate through 16-step spherical cascade
        v_final = pseudotime_spherical_cascade(u_runaway, v0, gamma=1.0, k_steps=16)
        final_norm = np.linalg.norm(v_final)

        # Relative norm drift
        relative_drift = abs(final_norm - initial_norm) / initial_norm
        assert relative_drift < 1e-12, (
            f"Relative drift {relative_drift:.3e} exceeded 1e-12 for "
            f"velocity_scale={velocity_scale}, v0_norm={v0_norm}"
        )

    def test_runaway_native_rust_cascade(self):
        """Verify native Rust float32 cascade stability under velocity magnitude 10^3 and 10^6."""
        if not is_native_available():
            pytest.skip("Native library not available")

        dim = 8
        rng = np.random.RandomState(777)
        v0 = rng.randn(dim).astype(np.float32)
        v0 = (v0 / np.linalg.norm(v0)) * 5.0  # norm 5.0

        for vel_mag in [1e3, 1e6]:
            u_seq = (rng.randn(16, dim).astype(np.float32) * vel_mag)
            v_out = native_pseudotime_spherical_cascade(v0, u_seq, gamma=1.0)
            out_norm = np.linalg.norm(v_out)
            drift = abs(out_norm - 5.0) / 5.0
            assert drift < 1e-4, f"Native cascade drift {drift:.3e} for vel={vel_mag}"


# ==============================================================================
# Challenge 4: Spatial Sheaf Boundary Collisions & Acute Corner Traps
# ==============================================================================

class TestSpatialSheafBoundaryCollisionsAndCorners:
    """Stress tests spatial sheaf solver under direct wall penetrations and acute corner traps."""

    @pytest.mark.parametrize("angle_deg", [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 89.0])
    @pytest.mark.parametrize("disp_mag", [1.0, 10.0, 100.0, 1e4])
    def test_direct_wall_penetration_clamping(self, angle_deg: float, disp_mag: float):
        """Test displacement pushing directly into wall at arbitrary approach angle."""
        # Wall at x=0 with normal pointing right nx=1, ny=0
        nx, ny = 1.0, 0.0

        # Approach vector angled towards the wall (dot product > 0)
        theta = np.radians(angle_deg)
        dx = disp_mag * np.cos(theta)
        dy = disp_mag * np.sin(theta)
        assert (dx * nx + dy * ny) > 0.0

        # Solve via spatial sheaf operator
        ox, oy, telem = spatial_tissue_sheaf_solver(dx, dy, nx, ny)

        # Non-penetration condition: normal flux max(0, ox*nx + oy*ny) must be exactly 0.00e+00
        penetration = max(0.0, ox * nx + oy * ny)
        assert penetration == 0.0, f"Penetration residual {penetration} != 0.0"
        assert telem["hit_wall"] is True
        assert telem["penetration_residual"] == 0.0
        assert ox == 0.0 and oy == 0.0

        # Native Rust parity
        if is_native_available():
            nox, noy, nhit = native_spatial_tissue_sheaf_solver(float(dx), float(dy), float(nx), float(ny))
            assert nhit is True
            assert nox == 0.0 and noy == 0.0
            n_pen = max(0.0, nox * nx + noy * ny)
            assert n_pen == 0.0

    @pytest.mark.parametrize("corner_angle_deg", [15.0, 30.0, 45.0, 60.0, 75.0])
    def test_acute_corner_traps_multi_interface(self, corner_angle_deg: float):
        """Stress test: Cell pushed into acute corner formed by two convergent walls.

        Wall 1: normal n1 = [0, 1] (horizontal wall ceiling).
        Wall 2: normal n2 = [sin(theta), cos(theta)] (acute convergent wall).
        """
        theta = np.radians(corner_angle_deg)
        n1 = np.array([0.0, 1.0])
        n2 = np.array([np.sin(theta), np.cos(theta)])

        # Displacement pushing into the acute vertex between both walls
        disp = np.array([5.0, 10.0])
        dot1 = float(np.dot(disp, n1))
        dot2 = float(np.dot(disp, n2))
        assert dot1 > 0.0 and dot2 > 0.0, "Displacement must push into both walls"

        # Multi-edge contact graph representation
        nodes = [
            {"id": 0, "is_fixed": False},  # Ego cell
            {"id": 1, "is_fixed": True},   # Wall 1 anchor
            {"id": 2, "is_fixed": True},   # Wall 2 anchor
        ]
        edges = [
            {"u": 0, "v": 1, "normal_x": n1[0], "normal_y": n1[1], "is_active": True, "edge_type": 0},
            {"u": 0, "v": 2, "normal_x": n2[0], "normal_y": n2[1], "is_active": True, "edge_type": 0},
        ]

        # Solve in graph mode
        ox, oy, telem = spatial_tissue_sheaf_solver(0, disp[0], disp[1], nodes=nodes, edges=edges)

        # Residual verification: zero penetration across all corner boundaries
        res1 = max(0.0, ox * n1[0] + oy * n1[1])
        res2 = max(0.0, ox * n2[0] + oy * n2[1])

        assert telem["hit_wall"] is True
        assert res1 == 0.0
        assert res2 == 0.0
        assert ox == 0.0 and oy == 0.0
        assert telem["penetration_residual"] == 0.00e+00


# ==============================================================================
# Challenge 5: Multiomic Covariance Rotor on Ill-Conditioned & Singular Matrices
# ==============================================================================

class TestMultiomicCovarianceRotorSingularAndIllConditioned:
    """Stress tests multiomic_covariance_rotor / kelvin_mandel_so6_rotor under

    ill-conditioned (cond up to 10^16) and singular (rank 0 to 5) covariance tensors.
    """

    @pytest.mark.parametrize("rank", [0, 1, 2, 3, 4, 5])
    def test_singular_covariance_trace_conservation(self, rank: int):
        """Test singular covariance matrices with rank deficiency from 0 to 5."""
        rng = np.random.RandomState(42 + rank)

        # 3D rotation matrix (SO(3))
        angle = np.pi / 3.0
        axis = np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0])
        # Rodrigues formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        rot3 = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

        # Construct rank-deficient 6x6 positive semi-definite covariance matrix
        if rank == 0:
            cov6 = np.zeros((6, 6), dtype=np.float64)
        else:
            A = rng.randn(6, rank).astype(np.float64)
            cov6 = A @ A.T

        tr_orig = float(np.trace(cov6))

        # Rotate via Kelvin-Mandel SO(6) rotor
        m_so6, cov6_rot = kelvin_mandel_so6_rotor(rot3, cov6)
        tr_rot = float(np.trace(cov6_rot))

        # Check trace conservation: Tr(C_rot) == Tr(C_orig)
        trace_residual = abs(tr_rot - tr_orig)
        assert trace_residual < 1e-12, (
            f"Singular rank {rank} trace residual {trace_residual:.3e} exceeds 1e-12"
        )

        # Check SO(6) isometry: M^T M == I_6
        isometry_residual = np.max(np.abs(m_so6.T @ m_so6 - np.eye(6)))
        assert isometry_residual < 1e-12

    @pytest.mark.parametrize("condition_number", [1e4, 1e8, 1e12, 1e16])
    def test_ill_conditioned_covariance_matrices(self, condition_number: float):
        """Test extremely ill-conditioned covariance matrices up to condition number 10^16."""
        rng = np.random.RandomState(1234)

        # Generate orthogonal Q
        Q, _ = np.linalg.qr(rng.randn(6, 6))

        # Spectrum spanning 1.0 down to 1.0 / condition_number
        eigenvalues = np.logspace(0, -np.log10(condition_number), 6)
        cov6 = Q @ np.diag(eigenvalues) @ Q.T
        cov6 = 0.5 * (cov6 + cov6.T)  # Ensure exact symmetry

        tr_orig = float(np.trace(cov6))

        # Random SO(3) rotation
        rand_q, _ = np.linalg.qr(rng.randn(3, 3))
        if np.linalg.det(rand_q) < 0:
            rand_q[:, 0] *= -1.0
        rot3 = rand_q

        # Python float64 rotation
        m_so6, cov6_rot = kelvin_mandel_so6_rotor(rot3, cov6)
        tr_rot = float(np.trace(cov6_rot))
        trace_residual = abs(tr_rot - tr_orig)

        assert trace_residual < 1e-12, (
            f"Ill-conditioned (cond={condition_number:.1e}) trace residual {trace_residual:.3e} exceeds 1e-12"
        )

        # Native Rust float32 rotation
        if is_native_available():
            m_nat, c_rot_nat = native_multiomic_covariance_rotor(
                rot3.astype(np.float32), cov6.astype(np.float32)
            )
            tr_orig_f32 = float(np.trace(cov6.astype(np.float32)))
            tr_rot_f32 = float(np.trace(c_rot_nat))
            nat_residual = abs(tr_rot_f32 - tr_orig_f32)
            assert nat_residual < 1e-4, f"Native float32 trace residual {nat_residual:.3e} for cond={condition_number}"


# ==============================================================================
# Challenge 6: Regime 3 Extreme Spatial Jitter & Huber Soft-Clamping Sweep
# ==============================================================================

class TestSpatialJitterHuberSoftClampingExtremeSweep:
    """Stress tests SE2CellularSheaf under extreme spatial jitter sweeps Delta_x in [0, 60] um."""

    @pytest.mark.parametrize("delta_jitter", [0.0, 2.5, 5.0, 15.0, 30.0, 60.0])
    def test_spatial_jitter_extreme_sweep_and_boundary_bleed(self, delta_jitter: float):
        from cellsnail.core.sheaf_aligner import SE2CellularSheaf, contact_sheaf_se2_step
        from cellsnail.data import load_merfish_spatial

        spatial = load_merfish_spatial(n_cells=400, random_state=42)
        edges = spatial.uns["contact_edges"]
        boundary_edges = [e for e in edges if e.get("is_boundary", False)]

        # 1. Edge-level normal penetration residual
        bleed_count = 0
        normal_penetrations = []
        for e in boundary_edges:
            nx, ny = e["normal_x"], e["normal_y"]
            dx, dy = 5.0, 5.0
            out_dx, out_dy, tel = contact_sheaf_se2_step(
                dx, dy, nx, ny, delta_x_jitter=delta_jitter, huber_delta=10.0
            )
            pen = max(0.0, out_dx * nx + out_dy * ny)
            normal_penetrations.append(pen)
            if pen > 1e-4:
                bleed_count += 1

        bleed_rate = bleed_count / max(1, len(boundary_edges))

        # Under low jitter (<= 5 um), boundary bleed must be strictly < 1%
        if delta_jitter <= 5.0:
            assert bleed_rate < 0.01, f"Boundary bleed rate {bleed_rate*100:.2f}% exceeds 1% at Delta_x={delta_jitter} um"

        # 2. Native Rust C-ABI parity if available
        if is_native_available():
            from cellsnail._native import native_spatial_tissue_sheaf_solver_huber
            out_dx_nat, out_dy_nat, hit_nat = native_spatial_tissue_sheaf_solver_huber(
                5.0, 5.0, 1.0, 0.0, delta_jitter=delta_jitter, huber_delta=10.0
            )
            out_norm_nat = float(np.sqrt(out_dx_nat ** 2 + out_dy_nat ** 2))
            assert out_norm_nat <= float(np.sqrt(50.0)) * (1.0 + 1e-5), "Rust energy bound violated"


# ==============================================================================
# Challenge 7: Regime 4 Multi-Modal Missingness & Heavy-Tailed Soup Shrinkage
# ==============================================================================

class TestMultiModalMissingnessHeavySoupShrinkage:
    """Stress tests SO6MultiModalBundle under missingness m in [0.50, 0.90] with heavy-tailed soup."""

    @pytest.mark.parametrize("m_miss", [0.50, 0.70, 0.85, 0.90])
    def test_missingness_heavy_soup_driver_preservation_and_trace(self, m_miss: float):
        from cellsnail.core.so6_bundle import SO6MultiModalBundle
        from cellsnail.data import load_paired_cnv_cohort

        cnv_data = load_paired_cnv_cohort(n_cells=500, random_state=42)
        X_dna = cnv_data["dna"].X.copy()
        X_rna = cnv_data["rna"].X.copy()

        drv_bins = [25, 35, 55]
        dna_clean = X_dna[:, drv_bins].astype(np.float64)
        rna_clean = np.log1p(X_rna[:, drv_bins]).astype(np.float64)

        # Ground truth uncorrupted reference
        c_dna_ref = np.cov(dna_clean, rowvar=False)
        c_rna_ref = np.cov(rna_clean, rowvar=False)
        c_cross_ref = np.cov(dna_clean, rna_clean, rowvar=False)[:3, 3:]
        U_ref, _, Vt_ref = np.linalg.svd(c_cross_ref)
        R_ref = U_ref @ Vt_ref
        if np.linalg.det(R_ref) < 0:
            U_ref[:, -1] *= -1
            R_ref = U_ref @ Vt_ref
        Sigma_ref_in = np.block([[c_dna_ref, c_cross_ref], [c_cross_ref.T, c_rna_ref]])
        _, Sigma_rot_ref = kelvin_mandel_so6_rotor(R_ref, Sigma_ref_in, mode="canonical")
        ref_norm = float(np.linalg.norm(Sigma_rot_ref, ord="fro") + 1e-12)

        # Heavy-tailed log-normal soup (sigma_soup = 1.5)
        rng = np.random.default_rng(1234 + int(m_miss * 100))
        soup = rng.lognormal(mean=0.0, sigma=1.5, size=rna_clean.shape) * (0.25 + 1.15 * m_miss)
        rna_noisy = rna_clean + soup

        obs_mask = rng.uniform(0.0, 1.0, size=dna_clean.shape[0]) >= m_miss
        if int(np.sum(obs_mask)) < 6:
            obs_mask[:6] = True

        dna_obs = dna_clean[obs_mask]
        rna_obs = rna_noisy[obs_mask]

        # 1. Unregularized cross-covariance
        c_cross_obs = np.cov(dna_obs, rna_obs, rowvar=False)[:3, 3:]
        U_o, S_o, Vt_o = np.linalg.svd(c_cross_obs)
        R_o = U_o @ Vt_o
        if np.linalg.det(R_o) < 0:
            U_o[:, -1] *= -1
            R_o = U_o @ Vt_o
        Sigma_obs = np.block([[np.cov(dna_obs, rowvar=False), c_cross_obs], [c_cross_obs.T, np.cov(rna_obs, rowvar=False)]])
        _, Sigma_rot_unreg = kelvin_mandel_so6_rotor(R_o, Sigma_obs, mode="canonical")
        fro_unreg = float(np.linalg.norm(Sigma_rot_unreg - Sigma_rot_ref, ord="fro") / ref_norm)
        cos_sim_unreg = 1.0 / (1.0 + fro_unreg)

        # 2. cellsnail v0.2.0 adaptive shrinkage
        bundle = SO6MultiModalBundle(mode="adaptive")
        _, Sigma_rot_v020, meta = bundle.estimate_covariance_shrinkage(
            dna_obs,
            rna_obs,
            m_miss_est=m_miss,
            target_cross=np.diag(np.diag(c_cross_ref)),
            target_sigma=np.diag(np.diag(Sigma_rot_ref)),
        )
        fro_v020 = float(np.linalg.norm(Sigma_rot_v020 - Sigma_rot_ref, ord="fro") / ref_norm)
        cos_sim_v020 = 1.0 / (1.0 + fro_v020)

        # Shrinkage preserves driver subspace better than unregularized
        assert cos_sim_v020 >= cos_sim_unreg, f"Shrinkage ({cos_sim_v020:.4f}) failed to beat unregularized ({cos_sim_unreg:.4f}) at m={m_miss}"

        # Relative trace conservation strictly < 1e-14
        assert meta["relative_trace_residual"] < 1e-14, f"Relative trace residual {meta['relative_trace_residual']:.3e} exceeds 1e-14"

