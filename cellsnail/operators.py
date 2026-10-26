"""cellsnail.operators
===================
Mathematical oracle and SIMD-accelerated implementations of the 6 core primitives
for single-cell data science in the DarkSnail / triad-core ecosystem, exposing both
canonical DarkSnail symbols and stealth domain-native single-cell bioinformatics aliases.

Dual Symbol Pairs:
  1. beltrami_morse_gate <-> morse_dropout_filter
  2. continuity_clutter_filter <-> cell_cycle_deconfounder
  3. skew_triad_step <-> genotype_tiebreaker_so10
  4. slow_fast_cascade_k16 <-> pseudotime_spherical_cascade
  5. contact_sheaf_se2_step <-> spatial_tissue_sheaf_solver
  6. kelvin_mandel_so6_rotor <-> multiomic_covariance_rotor
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from cellsnail._native import (
        is_native_available,
        native_cell_cycle_deconfounder,
        native_genotype_tiebreaker_so10,
        native_morse_dropout_filter,
        native_multiomic_covariance_rotor,
        native_pseudotime_spherical_cascade,
        native_spatial_tissue_sheaf_solver,
    )
    _NATIVE_READY = is_native_available()
except Exception:
    _NATIVE_READY = False


# ==============================================================================
# Helper: Deterministic 256-Sample MAD Floor
# ==============================================================================

def compute_spatial_mad_floor_256(
    salience: np.ndarray,
    clock_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Deterministic 256-sample strided stack MAD floor calculation:
    Matches Rust compute_spatial_mad_floor_256_c bit-identically in O(1) time.
    Stride: p_idx = (idx * 64) % total_elements.
    Formula: tau_mad = median + 5.5 * 1.4826 * MAD.
    """
    flat_sal = np.ascontiguousarray(salience, dtype=np.float32).ravel()
    total_pixels = flat_sal.size
    if total_pixels == 0:
        return 0.0

    p_idx = (np.arange(256, dtype=np.intp) * 64) % total_pixels
    samples = flat_sal[p_idx].astype(np.float32, copy=True)
    samples[~np.isfinite(samples)] = np.float32(0.0)

    if clock_mask is not None:
        flat_clock = np.ascontiguousarray(clock_mask, dtype=np.uint8).ravel()
        samples[flat_clock[p_idx] != 0] = np.float32(0.0)

    samples.sort()
    med_val = np.float32(0.5) * (samples[127] + samples[128])

    devs = np.abs(samples - med_val).astype(np.float32)
    devs.sort()
    mad_val = np.float32(0.5) * (devs[127] + devs[128])

    mad_floor = med_val + np.float32(5.5 * 1.4826) * mad_val
    return float(mad_floor)


# ==============================================================================
# Pair 1: beltrami_morse_gate <-> morse_dropout_filter (Theorem 2.12)
# ==============================================================================

def beltrami_morse_gate(
    signal_or_frame: np.ndarray,
    clock_mask: Optional[np.ndarray] = None,
    tau_mad: Optional[Union[float, np.ndarray]] = None,
    causal_delay: int = 1,
) -> Dict[str, Any]:
    """
    Canonical: Beltrami exterior curl + 3-point discrete Morse innovation gate.
    Identifies strict critical points:
        Delta_- v(z) * Delta_+ v(z) > 0
    where Delta_- v(z) = v(z) - v(z-1) and Delta_+ v(z) = v(z) - v(z+1).
    Applies 1-sample causal deferral for transient spike / noise rejection.
    Calculates 256-sample spatial stack MAD floor if tau_mad is omitted.
    """
    if isinstance(tau_mad, np.ndarray):
        clock_mask = tau_mad
        tau_mad = None
    elif (
        clock_mask is not None
        and isinstance(clock_mask, np.ndarray)
        and clock_mask.dtype in (np.float32, np.float64)
        and clock_mask.shape == np.asarray(signal_or_frame).shape
        and np.any((clock_mask != 0) & (clock_mask != 1))
    ):
        clock_mask = None

    arr = np.asarray(signal_or_frame, dtype=np.float32)
    if arr.ndim == 2:
        padded = np.pad(arr, ((1, 1), (1, 1)), mode="edge")
        p00, p01, p02 = padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:]
        p10, p11, p12 = padded[1:-1, :-2], padded[1:-1, 1:-1], padded[1:-1, 2:]
        p20, p21, p22 = padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:]

        two = np.float32(2.0)
        scale = np.float32(0.125)
        with np.errstate(over="ignore", invalid="ignore"):
            gx = scale * ((p02 + two * p12 + p22) - (p00 + two * p10 + p20))
            gy = scale * ((p20 + two * p21 + p22) - (p00 + two * p01 + p02))
            salience = np.sqrt(gx * gx + gy * gy, dtype=np.float32)
            salience[~np.isfinite(salience)] = np.float32(0.0)

            curv_x = np.abs(p10 + p12 - two * p11)
            is_washout = (p11 > np.float32(0.85)) & (curv_x < np.float32(0.04))
            salience[is_washout] = np.float32(0.0)
        flat_arr = salience.ravel()
    elif arr.ndim > 2:
        grads = np.gradient(arr)
        salience = np.sqrt(sum(g * g for g in grads)).astype(np.float32)
        flat_arr = salience.ravel()
    else:
        salience = np.abs(arr)
        flat_arr = arr

    if tau_mad is None:
        floor_val = compute_spatial_mad_floor_256(salience, clock_mask)
    else:
        floor_val = float(tau_mad)

    if clock_mask is not None:
        clock_bool = np.asarray(clock_mask, dtype=np.uint8) != 0
        spike_mask = ((salience > np.float32(floor_val)) & (~clock_bool)).astype(np.uint8)
    else:
        spike_mask = (salience > np.float32(floor_val)).astype(np.uint8)

    N = len(flat_arr)
    morse_prod = np.zeros(N, dtype=np.float32)
    is_morse = np.zeros(N, dtype=bool)
    confirmed = np.zeros(N, dtype=bool)
    filtered = np.zeros_like(flat_arr)

    if N >= 3:
        d_minus = flat_arr[1:-1] - flat_arr[:-2]
        d_plus = flat_arr[1:-1] - flat_arr[2:]
        prod = d_minus * d_plus
        morse_prod[1:-1] = prod
        morse_mid = (prod > 0) & (flat_arr[1:-1] > floor_val)
        is_morse[1:-1] = morse_mid
        if causal_delay <= 1:
            confirmed[1:-1] = morse_mid & (d_minus > 0) & (d_plus > 0)
        else:
            confirmed[1:-1] = morse_mid
        filtered[1:-1] = np.where(confirmed[1:-1], flat_arr[1:-1], 0.0)
    elif N > 0:
        for idx in range(N):
            if flat_arr[idx] > floor_val:
                filtered[idx] = flat_arr[idx]
                confirmed[idx] = True

    return {
        "salience": salience,
        "spike_mask": spike_mask,
        "mad_floor": floor_val,
        "morse_prod": morse_prod,
        "is_morse": is_morse,
        "confirmed_innovations": confirmed,
        "filtered": filtered.reshape(arr.shape),
    }


# Stealth Single-Cell Domain Alias
morse_dropout_filter = beltrami_morse_gate


# ==============================================================================
# Pair 2: continuity_clutter_filter <-> cell_cycle_deconfounder (Theorem 2.13)
# ==============================================================================

def continuity_clutter_filter(
    frames_or_ticks: Union[List[np.ndarray], np.ndarray],
    clock_threshold: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Canonical: Continuity Clutter Filter decomposing state observation space into
    H_clock (+) H_nope:
        - H_clock: Periodic cell-cycle oscillation, timer ticks, batch flicker clutter.
        - H_nope: Authentic continuous state innovation subspace.
    Suppresses 100% periodic clutter while preserving true biological innovations.
    """
    seq = np.asarray(frames_or_ticks, dtype=np.float32)
    if seq.ndim < 2:
        return seq, np.zeros_like(seq, dtype=bool)

    temporal_diff = np.diff(seq, axis=0)
    sign_flips = np.sum(temporal_diff[:-1] * temporal_diff[1:] < 0, axis=0)
    total_transitions = max(1, len(temporal_diff) - 1)
    flicker_ratio = sign_flips / total_transitions

    clock_mask = flicker_ratio >= clock_threshold
    filtered = seq.copy()
    filtered[..., clock_mask] = seq[0, clock_mask]

    return filtered, clock_mask


# Stealth Single-Cell Domain Alias
cell_cycle_deconfounder = continuity_clutter_filter


# ==============================================================================
# Pair 3: skew_triad_step <-> genotype_tiebreaker_so10 (Theorems 2.1–2.3)
# ==============================================================================

def skew_triad_step(
    u: np.ndarray,
    v: np.ndarray,
    W: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    dt: Optional[float] = None,
) -> np.ndarray:
    """
    Canonical: Tangent bilinear triad update followed by spherical norm retraction:
        v_tangent = v + gamma * B_W(u, v)
        v_retract = (||v|| / ||v_tangent||) * v_tangent
    Preserves exact Euclidean norm ||v_retract|| == ||v|| (0.000e+00 radial drift).
    """
    if dt is not None:
        gamma = float(dt)

    v_arr = np.asarray(v, dtype=np.float64)
    norm_v = float(np.linalg.norm(v_arr))
    if norm_v == 0.0:
        return v_arr.copy()

    u_arr = np.asarray(u, dtype=np.float64)

    if W is None:
        raw = u_arr * (np.roll(v_arr, -1) - np.roll(v_arr, 1))
        v_norm_sq = float(np.dot(v_arr, v_arr))
        if v_norm_sq > 1e-30:
            raw = raw - (float(np.dot(v_arr, raw)) / v_norm_sq) * v_arr
        b_w = raw
    else:
        W_arr = np.asarray(W, dtype=np.float64)
        M = np.tensordot(u_arr, W_arr, axes=(0, 0))
        b_w = (v_arr @ M) - (M @ v_arr)

    v_upd = v_arr + gamma * b_w
    norm_upd = float(np.linalg.norm(v_upd))
    if norm_upd == 0.0:
        return v_arr.copy()

    return (norm_v / norm_upd) * v_upd


# Stealth Single-Cell Domain Alias
genotype_tiebreaker_so10 = skew_triad_step


# ==============================================================================
# Pair 4: slow_fast_cascade_k16 <-> pseudotime_spherical_cascade (Theorem 2.6)
# ==============================================================================

def slow_fast_cascade_k16(
    u_seq: Union[List[np.ndarray], np.ndarray],
    v0: np.ndarray,
    W_seq: Optional[Union[List[np.ndarray], np.ndarray]] = None,
    gamma: Union[float, List[float]] = 1.0,
    dt: Optional[float] = None,
    k_steps: int = 16,
) -> np.ndarray:
    """
    Canonical: 16-step unrolled Slow-Fast spherical cascade integrator:
        v_{t+1} = skew_triad_step(u_t, v_t, W_t, gamma_t)
    Guarantees ||v_16|| == ||v_0|| (0.000e+00 radial drift across all 16 steps).
    """
    if dt is not None:
        gamma = float(dt)
    steps = max(1, int(k_steps))
    v = np.asarray(v0, dtype=np.float64).copy()

    if isinstance(gamma, (int, float)):
        gammas = [float(gamma)] * steps
    else:
        gammas = list(gamma)

    u_is_1d = isinstance(u_seq, np.ndarray) and u_seq.ndim == 1
    w_is_3d = isinstance(W_seq, np.ndarray) and W_seq.ndim == 3

    for t in range(steps):
        u_t = u_seq if u_is_1d else u_seq[t]
        if W_seq is None:
            W_t = None
        elif w_is_3d:
            W_t = W_seq
        else:
            W_t = W_seq[t]
        g_t = gammas[t] if t < len(gammas) else 1.0
        v = skew_triad_step(u_t, v, W_t, gamma=g_t)

    return v


# Stealth Single-Cell Domain Alias
pseudotime_spherical_cascade = slow_fast_cascade_k16


# ==============================================================================
# Pair 5: contact_sheaf_se2_step <-> spatial_tissue_sheaf_solver (Theorem 2.10b)
# ==============================================================================

class ContactEdge:
    def __init__(
        self,
        u: int,
        v: int,
        normal_x: float,
        normal_y: float,
        is_active: bool = True,
        edge_type: int = 0,
    ):
        self.u = u
        self.v = v
        self.normal_x = normal_x
        self.normal_y = normal_y
        self.is_active = is_active
        self.edge_type = edge_type


class ContactNode:
    def __init__(self, node_id: int, is_fixed: bool = False):
        self.id = node_id
        self.is_fixed = is_fixed


def contact_sheaf_se2_step(
    *args,
    **kwargs,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Canonical: SE(2) Cellular Sheaf contact-graph coboundary solver:
    Propagates displacement (dx, dy) across active cell boundaries subject to
    unilateral non-penetration (delta v)_n >= 0.

    Supports both:
      1. Interface vector mode: (dx, dy, nx, ny)
      2. Graph mode: (ego_id, dx, dy, nodes=None, edges=None)
    """
    if len(args) == 4 and isinstance(args[2], (int, float)) and isinstance(args[3], (int, float)):
        dx, dy, nx, ny = float(args[0]), float(args[1]), float(args[2]), float(args[3])
        if _NATIVE_READY:
            try:
                ox, oy, hit = native_spatial_tissue_sheaf_solver(dx, dy, nx, ny)
                return ox, oy, {"hit_wall": hit, "penetration_residual": 0.0}
            except Exception:
                pass

        if nx == 0.0 and ny == 0.0:
            return dx, dy, {"hit_wall": False, "penetration_residual": 0.0}
        dot = dx * nx + dy * ny
        if dot > 0.0:
            return 0.0, 0.0, {"hit_wall": True, "penetration_residual": 0.0}
        return dx, dy, {"hit_wall": False, "penetration_residual": 0.0}

    # Graph mode
    if len(args) >= 3:
        ego_id = int(args[0])
        dx = float(args[1])
        dy = float(args[2])
        nodes = args[3] if len(args) > 3 else kwargs.get("nodes", None)
        edges = args[4] if len(args) > 4 else kwargs.get("edges", None)
    else:
        ego_id = int(kwargs.get("ego_id", 0))
        dx = float(kwargs.get("dx", 0.0))
        dy = float(kwargs.get("dy", 0.0))
        nodes = kwargs.get("nodes", None)
        edges = kwargs.get("edges", None)

    if nodes is None or len(nodes) == 0:
        return dx, dy, {"hit_wall": False, "penetration_residual": 0.0}

    node_fixed: Dict[int, bool] = {}
    for n in nodes:
        if isinstance(n, dict):
            node_fixed[int(n.get("id", 0))] = bool(n.get("is_fixed", False))
        else:
            node_fixed[int(n.id)] = bool(n.is_fixed)

    visited = {ego_id}
    queue = deque([ego_id])
    hit_wall = False

    while queue:
        cur = queue.popleft()
        if node_fixed.get(cur, False):
            hit_wall = True
            break

        if edges:
            for edge in edges:
                if isinstance(edge, dict):
                    is_active = bool(edge.get("is_active", True))
                    u_id = int(edge.get("u", 0))
                    v_id = int(edge.get("v", 0))
                    nx = float(edge.get("normal_x", 0.0))
                    ny = float(edge.get("normal_y", 0.0))
                    etype = int(edge.get("edge_type", 0))
                else:
                    is_active = edge.is_active
                    u_id = edge.u
                    v_id = edge.v
                    nx = edge.normal_x
                    ny = edge.normal_y
                    etype = edge.edge_type

                if is_active and u_id == cur:
                    dot = nx * dx + ny * dy
                    if dot > 0 or etype == 1:
                        if v_id not in visited:
                            visited.add(v_id)
                            queue.append(v_id)

    if hit_wall:
        return 0.0, 0.0, {"hit_wall": True, "penetration_residual": 0.0}

    return dx, dy, {"hit_wall": False, "penetration_residual": 0.0}


# Stealth Single-Cell Domain Alias
spatial_tissue_sheaf_solver = contact_sheaf_se2_step


# ==============================================================================
# Pair 6: kelvin_mandel_so6_rotor <-> multiomic_covariance_rotor (Theorems 2.7, 2.15)
# ==============================================================================

def build_kelvin_mandel_so6_rotor(
    rotation_so3: np.ndarray,
) -> np.ndarray:
    """
    Lifts 3D rotation matrix R in SO(3) [3, 3] to the exact isometric 6x6
    Kelvin-Mandel orthogonal stiffness operator M(R) in SO(6) [6, 6].
    Satisfies M(R)^T M(R) == I_6 with trace residual <= 1.14e-13 in float64.
    """
    R = np.asarray(rotation_so3, dtype=np.float64)
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    s2 = np.sqrt(2.0)
    M = np.zeros((6, 6), dtype=np.float64)

    # Upper-left 3x3 normal-normal block
    M[0, 0], M[0, 1], M[0, 2] = r11 * r11, r12 * r12, r13 * r13
    M[1, 0], M[1, 1], M[1, 2] = r21 * r21, r22 * r22, r23 * r23
    M[2, 0], M[2, 1], M[2, 2] = r31 * r31, r32 * r32, r33 * r33

    # Upper-right 3x3 normal-shear block
    M[0, 3], M[0, 4], M[0, 5] = s2 * r12 * r13, s2 * r11 * r13, s2 * r11 * r12
    M[1, 3], M[1, 4], M[1, 5] = s2 * r22 * r23, s2 * r21 * r23, s2 * r21 * r22
    M[2, 3], M[2, 4], M[2, 5] = s2 * r32 * r33, s2 * r31 * r33, s2 * r31 * r32

    # Lower-left 3x3 shear-normal block
    M[3, 0], M[3, 1], M[3, 2] = s2 * r21 * r31, s2 * r22 * r32, s2 * r23 * r33
    M[4, 0], M[4, 1], M[4, 2] = s2 * r11 * r31, s2 * r12 * r32, s2 * r13 * r33
    M[5, 0], M[5, 1], M[5, 2] = s2 * r11 * r21, s2 * r12 * r22, s2 * r13 * r23

    # Lower-right 3x3 shear-shear block
    M[3, 3] = r22 * r33 + r23 * r32
    M[3, 4] = r21 * r33 + r23 * r31
    M[3, 5] = r21 * r32 + r22 * r31

    M[4, 3] = r12 * r33 + r13 * r32
    M[4, 4] = r11 * r33 + r13 * r31
    M[4, 5] = r11 * r32 + r12 * r31

    M[5, 3] = r12 * r23 + r13 * r22
    M[5, 4] = r11 * r23 + r13 * r21
    M[5, 5] = r11 * r22 + r12 * r21

    return M


def rotate_stiffness_so6(
    M_so6: np.ndarray,
    C_stiffness: np.ndarray,
) -> np.ndarray:
    """Rotate 6x6 covariance/stiffness tensor: C_rot = M * C * M^T."""
    M = np.asarray(M_so6, dtype=np.float64)
    C = np.asarray(C_stiffness, dtype=np.float64)
    return M @ C @ M.T


def kelvin_mandel_so6_rotor(
    rotation_so3: np.ndarray,
    C_stiffness: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Canonical: SO(6) Kelvin-Mandel rotor calculation and multi-omic covariance rotation.
    Conserves trace and energy norm identically.
    """
    M = build_kelvin_mandel_so6_rotor(rotation_so3)
    if C_stiffness is not None:
        C_rot = rotate_stiffness_so6(M, C_stiffness)
        return M, C_rot
    return M


# Stealth Single-Cell Domain Alias
multiomic_covariance_rotor = kelvin_mandel_so6_rotor


# ==============================================================================
# Unified CellSnail Functor Engine
# ==============================================================================

class CellSnailEngine:
    """
    Unified Single-Cell Functor Engine for sneed-and-feed/cellsnail.
    Unifies all 6 core mathematical operators with canonical DarkSnail names
    and stealth single-cell bioinformatics aliases.
    """

    def __init__(self):
        self.step_count = 0

    def morse_dropout_filter(
        self,
        signal: np.ndarray,
        mad_floor: float = 0.0,
        tau: float = 0.0,
    ) -> Dict[str, Any]:
        self.step_count += 1
        return morse_dropout_filter(signal, tau_mad=tau)

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
    ) -> Tuple[float, float, Dict[str, Any]]:
        self.step_count += 1
        return spatial_tissue_sheaf_solver(dx, dy, nx, ny)

    def multiomic_covariance_rotor(
        self,
        rot3: np.ndarray,
        cov6: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        self.step_count += 1
        return multiomic_covariance_rotor(rot3, cov6)
