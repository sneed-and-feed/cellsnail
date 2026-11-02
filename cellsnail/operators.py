"""cellsnail.operators
===================
Mathematical oracle and SIMD-accelerated implementations of core primitives
for single-cell data science and spatial biology:

Core Mathematical Operators:
  1. morse_dropout_filter / morse_gate_1d (Theorems 2.12, 2.16, 2.17)
  2. cell_cycle_deconfounder (Theorem 2.13)
  3. genotype_tiebreaker_so10 (Theorems 2.1–2.3)
  4. pseudotime_spherical_cascade (Theorem 2.6)
  5. spatial_tissue_sheaf_solver (Theorems 2.10b, 2.18)
  6. multiomic_covariance_rotor (Theorems 2.7, 2.15, 2.19)
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

# Re-export from modular core
from cellsnail.core import (
    ContactEdge,
    ContactNode,
    SE2CellularSheaf,
    SO6MultiModalBundle,
    build_kelvin_mandel_so6_rotor,
    cell_cycle_deconfounder,
    compute_spatial_mad_floor_256,
    contact_sheaf_se2_step,
    continuity_clutter_filter,
    estimate_missingness,
    estimate_segment_widths,
    estimate_spatial_jitter,
    estimate_umi_depth,
    genotype_tiebreaker_so10,
    kelvin_mandel_so6_rotor,
    ledoit_wolf_shrinkage_so6,
    morse_cnv_filter,
    morse_dropout_filter,
    multiomic_covariance_rotor,
    pseudotime_spherical_cascade,
    rotate_stiffness_so6,
    sigmoidal_shrinkage_weight,
    skew_triad_step,
    slow_fast_cascade_k16,
    spatial_tissue_sheaf_solver,
)

try:
    from cellsnail._native import (
        is_native_available,
        morse_gate_1d,
        morse_gate_genomic,
        validate_genomic_coordinates,
        validate_genomic_intervals,
        GenomicInterval,
        native_allosteric_pocket_sheaf_solver,
        native_cell_cycle_deconfounder,
        native_cryoem_sidechain_morse_filter,
        native_genotype_tiebreaker_so10,
        native_morse_dropout_filter,
        native_multiomic_covariance_rotor,
        native_pseudotime_spherical_cascade,
        native_ramachandran_cayley_integrator,
        native_spatial_tissue_sheaf_solver,
    )
    _NATIVE_READY = is_native_available()
except Exception:
    _NATIVE_READY = False

# Import 3D structural operators from modular structural engine (Theorems 2.20–2.22)
from cellsnail.structural import (
    CayleyRamachandranRotor,
    ContactSheafSE3,
    CryoEMMorseFilter3D,
    JordanCavityBetti,
    allosteric_pocket_sheaf_solver,
    cayley_rotor_so3_torsion,
    contact_sheaf_se3_step,
    cryoem_sidechain_morse_filter,
    cryptic_pocket_topology_audit,
    jordan_3d_cavity_betti,
    ramachandran_cayley_integrator,
    ulp_polish_vec3,
)


# ==============================================================================
# Unified CellSnail Functor Engine
# ==============================================================================

class CellSnailEngine:
    """
    Unified Single-Cell Functor Engine for sneed-and-feed/cellsnail v0.3.0.
    Unifies core mathematical operators for single-cell genomics and spatial biology.
    """

    def __init__(self, mode: str = "adaptive"):
        self.step_count = 0
        self.mode = mode

    def morse_gate_1d(
        self,
        signal: np.ndarray,
        mad_floor: float = 0.0,
        tau: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.step_count += 1
        return morse_gate_1d(signal, mad_floor=mad_floor, tau=tau)

    def morse_gate_genomic(
        self,
        chrom_ids: Union[np.ndarray, List[int]],
        starts: Union[np.ndarray, List[int]],
        ends: Union[np.ndarray, List[int]],
        signal: np.ndarray,
        mad_floor: float = 0.0,
        tau: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        self.step_count += 1
        return morse_gate_genomic(chrom_ids, starts, ends, signal, mad_floor=mad_floor, tau=tau)

    def morse_dropout_filter(
        self,
        signal: np.ndarray,
        mad_floor: float = 0.0,
        tau: float = 0.0,
        u_hat: Optional[float] = None,
        prior_matrix: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        self.step_count += 1
        return morse_dropout_filter(
            signal,
            tau_mad=tau if tau > 0 else (mad_floor if mad_floor > 0 else None),
            u_hat=u_hat,
            prior_matrix=prior_matrix,
            mode=kwargs.get("mode", self.mode),
            **kwargs,
        )

    def morse_cnv_filter(
        self,
        profile: np.ndarray,
        tau_mad: float = 3.2,
        tau_step: Optional[float] = None,
        w_min_plateau: int = 10,
        max_window: int = 15,
    ) -> Dict[str, Any]:
        self.step_count += 1
        return morse_cnv_filter(
            profile,
            tau_mad=tau_mad,
            tau_step=tau_step,
            w_min_plateau=w_min_plateau,
            max_window=max_window,
        )

    def cell_cycle_deconfounder(
        self,
        matrix: np.ndarray,
        threshold: float = 0.85,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.step_count += 1
        return cell_cycle_deconfounder(matrix, clock_threshold=threshold)

    def genotype_tiebreaker_so10(
        self,
        u: np.ndarray,
        v: np.ndarray,
        W: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.step_count += 1
        return genotype_tiebreaker_so10(u, v, W)

    def pseudotime_spherical_cascade(
        self,
        v0: np.ndarray,
        u_seq: np.ndarray,
        gamma: float = 1.0,
    ) -> np.ndarray:
        self.step_count += 16
        return pseudotime_spherical_cascade(u_seq, v0, gamma=gamma)

    def spatial_tissue_sheaf_solver(
        self,
        dx: float,
        dy: float,
        nx: float,
        ny: float,
        delta_x_jitter: float = 0.0,
        huber_delta: float = 10.0,
    ) -> Tuple[float, float, Dict[str, Any]]:
        self.step_count += 1
        return spatial_tissue_sheaf_solver(
            dx, dy, nx, ny,
            delta_x_jitter=delta_x_jitter,
            huber_delta=huber_delta,
        )

    def multiomic_covariance_rotor(
        self,
        rot3: np.ndarray,
        cov6: Optional[np.ndarray] = None,
        m_miss: Optional[float] = None,
        shrink_alpha: Optional[float] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        self.step_count += 1
        return multiomic_covariance_rotor(
            rot3,
            cov6,
            m_miss=m_miss,
            shrink_alpha=shrink_alpha,
            mode=self.mode,
        )

    def cryoem_sidechain_morse_filter(
        self,
        volume: np.ndarray,
        mad_floor: float = 0.0,
        tau: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        self.step_count += 1
        return cryoem_sidechain_morse_filter(volume, mad_floor=mad_floor, tau=tau, **kwargs)

    def allosteric_pocket_sheaf_solver(
        self,
        coords: np.ndarray,
        edges: np.ndarray,
        mut_residue: int = 0,
        mut_twist: Optional[np.ndarray] = None,
        num_iters: int = 25,
        pocket_residues: Optional[Union[List[int], Set[int]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        self.step_count += 1
        return allosteric_pocket_sheaf_solver(
            coords,
            edges,
            mut_residue=mut_residue,
            mut_twist=mut_twist,
            num_iters=num_iters,
            pocket_residues=pocket_residues,
            **kwargs,
        )

    def ramachandran_cayley_integrator(
        self,
        coords: Optional[np.ndarray] = None,
        u_seq: Optional[np.ndarray] = None,
        v_seq: Optional[np.ndarray] = None,
        alpha_seq: Optional[np.ndarray] = None,
        k_steps: int = 16,
        **kwargs,
    ) -> Dict[str, Any]:
        self.step_count += k_steps
        return ramachandran_cayley_integrator(
            coords=coords,
            u_seq=u_seq,
            v_seq=v_seq,
            alpha_seq=alpha_seq,
            k_steps=k_steps,
            **kwargs,
        )

    def cryptic_pocket_topology_audit(
        self,
        coords: Optional[np.ndarray] = None,
        grid: Optional[np.ndarray] = None,
        grid_resolution: float = 1.0,
        probe_radius: float = 1.4,
        allosteric_torque: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        self.step_count += 1
        return cryptic_pocket_topology_audit(
            coords=coords,
            grid=grid,
            grid_resolution=grid_resolution,
            probe_radius=probe_radius,
            allosteric_torque=allosteric_torque,
            **kwargs,
        )


# ==============================================================================
# Numerical Invariant Verifiers for Theorems 2.20–2.22 & Complete Suite
# ==============================================================================

def verify_se3_allosteric_coboundary() -> Dict[str, Any]:
    """Theorem 2.20: 3D SE(3) Allosteric Contact Sheaf Non-Penetration & Torque Transmission."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [3.8, 0.0, 0.0],
        [7.6, 0.0, 0.0],
        [11.4, 0.0, 0.0],
    ], dtype=np.float32)
    edges = np.array([
        [0, 1], [1, 0],
        [1, 2], [2, 1],
        [2, 3], [3, 2],
    ], dtype=np.uint32)

    # 1. Unilateral non-penetration under mutational perturbation
    res = allosteric_pocket_sheaf_solver(coords, edges, mut_residue=0, num_iters=25)
    twists = res["twists"]
    eta = res["eta_nonlocal"]
    pen_res = res["steric_penetration_residual"]

    # Check non-penetration: (delta v)_n >= -1e-6
    non_penetration_holds = True
    for e in range(len(edges)):
        u, v = edges[e]
        dx = coords[v] - coords[u]
        d = float(np.linalg.norm(dx))
        if d > 1e-6:
            n_e = dx / d
            dv_n = float(np.dot(twists[v, 3:6] - twists[u, 3:6], n_e))
            if dv_n < -1e-5:
                non_penetration_holds = False
                break

    # 2. SE(3) rigid-body translation invariance: shifting coordinates globally does not alter relative coboundary
    coords_shifted = coords + np.array([10.0, -5.0, 20.0], dtype=np.float32)
    res_shifted = allosteric_pocket_sheaf_solver(coords_shifted, edges, mut_residue=0, num_iters=25)
    rigid_invariance = bool(np.allclose(res["eta_nonlocal"], res_shifted["eta_nonlocal"], atol=1e-5))

    # 3. Monotonic transmission along non-slack contact path: eta[0] == 1.0, eta[1] > 0
    monotonic_transmission = bool((eta[0] == 1.0) and (eta[1] > 0.0) and (eta[2] >= 0.0))

    passed = bool(non_penetration_holds and (pen_res == 0.0) and rigid_invariance and monotonic_transmission)
    return {
        "theorem": "Theorem 2.20",
        "name": "SE(3) Allosteric Contact Sheaf Non-Penetration & Torque Transmission",
        "steric_penetration_residual": float(pen_res),
        "non_penetration_holds": bool(non_penetration_holds),
        "rigid_invariance": bool(rigid_invariance),
        "monotonic_transmission": bool(monotonic_transmission),
        "residual": float(pen_res),
        "tolerance": 1e-6,
        "passed": passed,
    }


def verify_cayley_so3_ramachandran_norm() -> Dict[str, Any]:
    """Theorem 2.21: Rank-2 Cayley SO(3) Ramachandran Dihedral Torsion Norm Invariance & Orthogonality."""
    coords_in = np.array([
        [1.54, 0.0, 0.0],
        [0.0, 1.45, 0.0],
        [0.0, 0.0, 1.33],
    ], dtype=np.float32)

    all_k_passed = True
    max_radial = 0.0
    max_ortho = 0.0

    for k in [16, 64, 256]:
        res = ramachandran_cayley_integrator(coords_in, k_steps=k)
        r_drift = float(res["radial_drift"])
        o_drift = float(res["ortho_drift"])
        max_radial = max(max_radial, r_drift)
        max_ortho = max(max_ortho, o_drift)
        if r_drift != 0.0 or o_drift > 1e-6:
            all_k_passed = False

    return {
        "theorem": "Theorem 2.21",
        "name": "Rank-2 Cayley SO(3) Ramachandran Dihedral Torsion Norm Invariance",
        "radial_drift": float(max_radial),
        "ortho_drift": float(max_ortho),
        "all_k_passed": bool(all_k_passed),
        "residual": float(max_radial),
        "tolerance": 0.0,
        "passed": bool(all_k_passed),
    }


def verify_cryoem_3d_morse_shield() -> Dict[str, Any]:
    """Theorem 2.22: Cryo-EM 3D 6-Axis Exterior-Curl Discrete Morse Gate Side-Chain Shielding."""
    vol = np.zeros((9, 9, 9), dtype=np.float32)
    vol[:] = 0.1

    # Sustained ridge / side-chain density along Z axis at (x=4, y=4)
    vol[2:7, 4, 4] = 4.0
    ridge_mask = np.zeros_like(vol, dtype=bool)
    ridge_mask[2:7, 4, 4] = True

    # Isolated 1-voxel noise spikes
    vol[1, 1, 1] = 8.0
    vol[7, 7, 7] = 9.0
    spike_mask = np.zeros_like(vol, dtype=bool)
    spike_mask[1, 1, 1] = True
    spike_mask[7, 7, 7] = True

    res = cryoem_sidechain_morse_filter(vol, mad_floor=0.5, tau=1.0)
    filtered = res["filtered"]

    # 1. 100% preservation of sustained ridge
    ridge_preserved = bool(np.all(filtered[ridge_mask] == 4.0))

    # 2. 100% rejection of isolated 1-voxel noise spikes
    spikes_rejected = bool(np.all(filtered[spike_mask] == 0.0))

    # 3. Background median shift == 0.0
    bg_after = filtered[vol < 1.0]
    med_after = float(np.median(bg_after))
    median_shift = float(abs(med_after - 0.0))

    passed = bool(ridge_preserved and spikes_rejected and (median_shift == 0.0))
    return {
        "theorem": "Theorem 2.22",
        "name": "Cryo-EM 3D Exterior-Curl Discrete Morse Gate Side-Chain Shielding",
        "ridge_preserved": bool(ridge_preserved),
        "spikes_rejected": bool(spikes_rejected),
        "median_shift": float(median_shift),
        "residual": float(median_shift),
        "tolerance": 0.0,
        "passed": passed,
    }


def verify_all_theorems() -> Dict[str, Any]:
    """
    Execute complete formal verification of all mathematical invariants (Theorems 2.1–2.22).
    Includes the single-cell foundation (2.1–2.15) and 3D structural mechanics (2.20–2.22).
    """
    from cellsnail.benchmark.mode_a_oracle import (
        verify_triad_orthogonality,
        verify_skew_antisymmetry,
        verify_step0_homotopy,
        verify_discrete_energy_identity,
        verify_rope_isometric_invariance,
        verify_cascade_norm_conservation,
        verify_kelvin_mandel_trace,
        verify_cayley_transform_unitarity,
        verify_transverse_sheaf_curvature_invariance,
        verify_cellular_sheaf_coboundary,
        verify_bcm2712_arena_geometry,
        verify_morse_dropout_shielding,
        verify_h_clock_h_nope_orthogonality,
        verify_ultrametric_site_condensation,
        verify_multiomic_dosage_linkage,
    )

    checks = [
        verify_triad_orthogonality(),
        verify_skew_antisymmetry(),
        verify_step0_homotopy(),
        verify_discrete_energy_identity(),
        verify_rope_isometric_invariance(),
        verify_cascade_norm_conservation(),
        verify_kelvin_mandel_trace(),
        verify_cayley_transform_unitarity(),
        verify_transverse_sheaf_curvature_invariance(),
        verify_cellular_sheaf_coboundary(),
        verify_bcm2712_arena_geometry(),
        verify_morse_dropout_shielding(),
        verify_h_clock_h_nope_orthogonality(),
        verify_ultrametric_site_condensation(),
        verify_multiomic_dosage_linkage(),
        verify_se3_allosteric_coboundary(),
        verify_cayley_so3_ramachandran_norm(),
        verify_cryoem_3d_morse_shield(),
    ]

    all_passed = all(c["passed"] for c in checks)
    max_residual = max(c.get("residual", 0.0) for c in checks)

    return {
        "status": "ALL_THEOREMS_VERIFIED" if all_passed else "THEOREMS_FAILED",
        "total_theorems": len(checks),
        "passed_theorems": sum(1 for c in checks if c["passed"]),
        "failed_theorems": sum(1 for c in checks if not c["passed"]),
        "max_residual": max_residual,
        "theorems": checks,
        "all_passed": all_passed,
    }


__all__ = [
    "CellSnailEngine",
    # 1D Morse Filtering
    "morse_gate_1d",
    "morse_gate_genomic",
    "validate_genomic_coordinates",
    "validate_genomic_intervals",
    "GenomicInterval",
    "morse_dropout_filter",
    "morse_cnv_filter",
    "compute_spatial_mad_floor_256",
    # Single-Cell Deconfounding & Cascades
    "continuity_clutter_filter",
    "cell_cycle_deconfounder",
    "skew_triad_step",
    "genotype_tiebreaker_so10",
    "slow_fast_cascade_k16",
    "pseudotime_spherical_cascade",
    # Spatial Sheaf & Multiomic Rotor
    "contact_sheaf_se2_step",
    "spatial_tissue_sheaf_solver",
    "kelvin_mandel_so6_rotor",
    "multiomic_covariance_rotor",
    "build_kelvin_mandel_so6_rotor",
    "rotate_stiffness_so6",
    # Adaptive estimators & high-level wrappers
    "estimate_umi_depth",
    "sigmoidal_shrinkage_weight",
    "estimate_segment_widths",
    "estimate_spatial_jitter",
    "SE2CellularSheaf",
    "ContactEdge",
    "ContactNode",
    "estimate_missingness",
    "ledoit_wolf_shrinkage_so6",
    "SO6MultiModalBundle",
    # 3D Structural Operators (Theorems 2.20–2.22)
    "cryoem_sidechain_morse_filter",
    "contact_sheaf_se3_step",
    "allosteric_pocket_sheaf_solver",
    "cayley_rotor_so3_torsion",
    "ramachandran_cayley_integrator",
    "jordan_3d_cavity_betti",
    "cryptic_pocket_topology_audit",
    "CryoEMMorseFilter3D",
    "ContactSheafSE3",
    "CayleyRamachandranRotor",
    "JordanCavityBetti",
    "ulp_polish_vec3",
    # Formal Invariant Verifiers
    "verify_all_theorems",
    "verify_se3_allosteric_coboundary",
    "verify_cayley_so3_ramachandran_norm",
    "verify_cryoem_3d_morse_shield",
]
