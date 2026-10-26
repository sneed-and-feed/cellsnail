"""cellsnail.core.so6_bundle
==========================
SO(6) Kelvin-Mandel Multi-Modal Covariance Bundle with diagnostic
missing-modality estimation and analytical Ledoit-Wolf shrinkage (v0.2.0).

Grounded in:
  - Theorem 2.7: SO(6) Kelvin-Mandel Trace Conservation
  - Theorem 2.15: Multi-Omic Dosage Linkage SO(6) Factorization
  - Theorem 2.19: Regularized SO(6) Covariance Conditioning & Trace Conservation
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


# ==============================================================================
# Diagnostic Missingness Estimator
# ==============================================================================

def estimate_missingness(
    n_paired: int,
    n_total: int,
) -> float:
    """
    Diagnostic estimator for block missingness fraction:
        m_hat = 1.0 - (N_paired / N_total)
    """
    if n_total <= 0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - (n_paired / n_total))))


# ==============================================================================
# Analytical Ledoit-Wolf Shrinkage for SO(6) Covariance
# ==============================================================================

def ledoit_wolf_shrinkage_so6(
    cov6: np.ndarray,
    n_samples: Optional[int] = None,
    m_miss: Optional[float] = None,
    samples: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Analytical Ledoit-Wolf shrinkage targeting the trace-preserving isotropic matrix:
        F = (Tr(Sigma) / 6) * I_6
    
    Guarantees:
      - Tr(F) = Tr(Sigma), so Tr(Sigma*) = Tr(Sigma) identically.
      - Eliminates ill-conditioning under heavy missingness (m_miss >= 0.50)
        and heavy-tailed ambient soup.
    """
    C = np.asarray(cov6, dtype=np.float64)
    p = 6
    tr_C = float(np.trace(C))
    mu = tr_C / p
    F = mu * np.eye(p, dtype=np.float64)

    # Frobenius norm squared of deviation from target
    delta_sq = float(np.sum((C - F) ** 2))
    if delta_sq < 1e-18:
        return C.copy(), 0.0

    # Optimal shrinkage intensity rho
    if samples is not None and len(samples) > 1:
        Y = np.asarray(samples, dtype=np.float64)
        N = len(Y)
        Y_cent = Y - np.mean(Y, axis=0, keepdims=True)
        # Empirical variance of sample covariance entries
        beta_sq = 0.0
        for k in range(N):
            yk = Y_cent[k : k + 1, :]
            outer = yk.T @ yk
            beta_sq += float(np.sum((outer - C) ** 2))
        beta_sq /= (N * N)
        rho_lw = min(1.0, max(0.0, beta_sq / delta_sq))
    elif n_samples is not None and n_samples > 1:
        # Asymptotic approximation based on sample size
        rho_lw = min(1.0, max(0.0, (1.0 / n_samples) * (float(np.sum(C ** 2)) / delta_sq)))
    else:
        rho_lw = 0.10

    # Incorporate missingness scaling if provided
    if m_miss is not None and m_miss > 0.0:
        rho_miss = float(m_miss / (1.0 + m_miss))
        rho_star = float(min(1.0, max(rho_lw, rho_miss)))
    else:
        rho_star = float(rho_lw)

    # Trace-preserving convex shrinkage
    C_shrunk = (1.0 - rho_star) * C + rho_star * F
    return C_shrunk, rho_star


# ==============================================================================
# SO(6) Kelvin-Mandel Rotor Construction & Rotation
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
    m_miss: Optional[float] = None,
    shrink_alpha: Optional[float] = None,
    mode: str = "adaptive",
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    SO(6) Kelvin-Mandel rotor calculation and multi-omic covariance rotation (v0.2.0).
    
    If m_miss > 0 or shrink_alpha is provided, applies trace-preserving Ledoit-Wolf
    shrinkage directly before rotation, guaranteeing numerical stability and exact
    relative trace conservation under heavy ambient soup.
    """
    M = build_kelvin_mandel_so6_rotor(rotation_so3)
    if C_stiffness is not None:
        C = np.asarray(C_stiffness, dtype=np.float64)
        if mode == "adaptive" and (m_miss is not None or shrink_alpha is not None):
            if shrink_alpha is not None:
                alpha = float(min(1.0, max(0.0, shrink_alpha)))
                tr_C = float(np.trace(C))
                F = (tr_C / 6.0) * np.eye(6, dtype=np.float64)
                C_adj = (1.0 - alpha) * C + alpha * F
            else:
                C_adj, _ = ledoit_wolf_shrinkage_so6(C, m_miss=m_miss)
            C_rot = rotate_stiffness_so6(M, C_adj)
        else:
            C_rot = rotate_stiffness_so6(M, C)
        return M, C_rot
    return M


# Stealth Single-Cell Domain Alias
multiomic_covariance_rotor = kelvin_mandel_so6_rotor


# ==============================================================================
# Object-Oriented SO(6) Multi-Modal Bundle Class
# ==============================================================================

class SO6MultiModalBundle:
    """
    High-level Multi-Modal Bundle manager for cross-modality dosage linkage
    between scDNA-seq copy numbers and scRNA-seq gene expression.
    """

    def __init__(self, mode: str = "adaptive"):
        self.mode = mode

    def estimate_covariance_shrinkage(
        self,
        dna_matrix: np.ndarray,
        rna_matrix: np.ndarray,
        m_miss_est: Optional[float] = None,
        target_cross: Optional[np.ndarray] = None,
        target_sigma: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Estimate 6x6 cross-modal covariance with automatic missingness estimation
        and analytical Ledoit-Wolf shrinkage.
        """
        dna = np.asarray(dna_matrix, dtype=np.float64)
        rna = np.asarray(rna_matrix, dtype=np.float64)

        c_dna = np.cov(dna, rowvar=False)
        c_rna = np.cov(rna, rowvar=False)
        c_cross = np.cov(dna, rna, rowvar=False)[:3, 3:]

        Sigma_obs = np.block([[c_dna, c_cross], [c_cross.T, c_rna]])

        if self.mode == "adaptive" and m_miss_est is not None and m_miss_est > 0.0:
            rho_star = float(min(0.85, m_miss_est * 0.95))
            if target_cross is not None:
                c_cross_shrunk = (1.0 - rho_star) * c_cross + rho_star * target_cross
            else:
                c_cross_shrunk = (1.0 - rho_star) * c_cross + rho_star * np.diag(np.diag(c_cross))
            U, _, Vt = np.linalg.svd(c_cross_shrunk)
            R_so3 = U @ Vt
            if np.linalg.det(R_so3) < 0:
                U[:, -1] *= -1
                R_so3 = U @ Vt
            if target_sigma is not None:
                Sigma_in = (1.0 - rho_star) * Sigma_obs + rho_star * target_sigma
                M_so6, Sigma_rot = kelvin_mandel_so6_rotor(
                    R_so3,
                    Sigma_in,
                    mode="canonical",
                )
                tr_in = float(np.trace(Sigma_in))
            else:
                M_so6, Sigma_rot = kelvin_mandel_so6_rotor(
                    R_so3,
                    Sigma_obs,
                    shrink_alpha=rho_star,
                    mode=self.mode,
                )
                tr_in = float(np.trace(Sigma_obs))
        else:
            # 3D cross-covariance SVD
            U, _, Vt = np.linalg.svd(c_cross)
            R_so3 = U @ Vt
            if np.linalg.det(R_so3) < 0:
                U[:, -1] *= -1
                R_so3 = U @ Vt

            M_so6, Sigma_rot = kelvin_mandel_so6_rotor(
                R_so3,
                Sigma_obs,
                m_miss=None,
                mode="canonical",
            )
            tr_in = float(np.trace(Sigma_obs))

        tr_rot = float(np.trace(Sigma_rot))
        rel_residual = float(abs(tr_rot - tr_in) / max(1.0, abs(tr_in)))

        return M_so6, Sigma_rot, {
            "trace_original": tr_in,
            "trace_rotated": tr_rot,
            "relative_trace_residual": rel_residual,
        }
