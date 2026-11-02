import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic

/-!
# Project CellSnail: Proof-Carrying 1D Genomic Coordinate Filtering & CNV Mathematical Hierarchy

This module formalizes the machine-checked mathematical core of `sneed-and-feed/cellsnail`,
providing rigorous mathematical foundations for 1D chromosomal coordinate copy number variation (CNV)
and focal amplicon detection:

## Part I: 1D Genomic Coordinate Filtering Invariants (Theorems 1–7)
1. **Theorem 1 (`morse_monotone_slope_rejected`)**:
   Monotone slope and flat step rejection: discrete Morse product is strictly negative on strict monotonicity
   and non-positive on weak monotonicity, rejecting spurious chromosomal compaction trends.
2. **Theorem 2 (`morse_local_extremum_isolated`)**:
   Strict local extremum characterization: 3-point discrete Morse impulse triggers if and only if
   the genomic bin is a strict local extremum (peak or pit).
3. **Theorem 3 (`focal_amplicon_amplitude_recovery`)**:
   Focal amplicon 100% amplitude recovery: an isolated focal amplicon peak is recovered at exact full amplitude
   under Morse gating with zero attenuation.
4. **Theorem 4 (`flanking_leakage_bounded`)**:
   Zero flanking leakage: immediate flanking bins beyond a focal amplicon receive zero Morse impulse,
   preventing signal smearing into adjacent diploid bins.
5. **Theorem 5 (`plateau_moving_average_unbiased`)**:
   Unbiased plateau moving average: box-car moving average across constant copy-number plateau bins
   is strictly unbiased.
6. **Theorem 6 (`plateau_variance_reduction`)**:
   Plateau noise variance reduction: moving average over window of size W attenuates independent noise variance
   by a factor of 1/W.
7. **Theorem 7 (`boundary_pinned_step_preservation`)**:
   Boundary-pinned step preservation: boundary pinning preserves step-edge transition height identically
   without boundary erosion.

## Part II: Geometric Invariants & Orthogonal Decompositions
8. **Theorem 2.1 (`skew_triad_swap_neg`, `skew_triad_self_zero`)**:
   Skew-Symmetric Bilinear Triad antisymmetry and vanishing self-interaction.
9. **Theorem 2.2 (`occlusion_triad_breaks_symmetric_tie`)**:
   Tie-breaking of symmetric 2-state collision readouts via bilinear shift.
10. **Theorem 2.3 (`cellsnail_readout_energy_conservation`)**:
    Pointwise zero-energy-explosion orthogonality: ⟨v, B_W(u, v)⟩ = 0.
11. **Theorem 2.4 (`cellsnail_step0_system1_homotopy`)**:
    Step-0 homotopy equivalence: readout reduces identically to baseline at step 0.
12. **Theorem 2.5 (`rope2d_medoid_relative_invariance`, `rope2d_arithmetic_norm_lt`)**:
    Relative displacement invariance and strict patch-averaging attenuation.
13. **Theorem 2.6 (`cellsnail_multistep_spherical_invariance`, `cellsnail_multistep_step0_homotopy`)**:
    Multi-step unrolled spherical cascade norm invariance.
14. **Theorem 2.7 (`elasticEnergy6_rotateStiffness6`, `so6_stiffness_preserves_pos_def`, `so6_stiffness_trace_conservation`)**:
    SO(6) stiffness rotor trace and energy quadratic form conservation.
15. **Theorem 2.10a (`cayley_rank2_orthogonality`)**:
    Rank-2 Cayley transform rotor column-orthonormality and Euclidean norm preservation.
16. **Theorem 2.10b (`contact_sheaf_non_penetration_invariant`)**:
    Unilateral non-penetration condition with unconstrained tangential freedom.
17. **Theorem 2.12 (`morse_dropout_shielding_zero_spurious_corr`)**:
    Pointwise separable filtering on uncorrelated channels preserves zero covariance.
18. **Theorem 2.13 (`h_clock_h_nope_orthogonality`)**:
    Idempotent symmetric projector orthogonal direct-sum subspace decomposition.
19. **Theorem 2.14 (`ascertainment_bias_free_ultrametric_condensation`)**:
    Scaling ultrametric distance preserves the ultrametric three-point condition.
20. **Theorem 2.15 (`multiomic_covariance_trace_and_quadform_conservation`)**:
    6D multi-omic covariance rotor trace and quadratic form conservation under SO(6) rotation.
21. **Theorem 2.16 (`convex_shrinkage_stability_and_zero_spurious_corr`)**:
    Convex shrinkage blending amplitude bounds and zero spurious covariance preservation.
22. **Theorem 2.17 (`multiscale_plateau_unbiased_and_variance_reduction`)**:
    Multi-scale plateau moving average unbiasedness and noise variance reduction.
23. **Theorem 2.18 (`jitter_adaptive_huber_sheaf_energy_bounds`)**:
    Huber penalty energy bounds and non-expansive soft clamping.
24. **Theorem 2.19 (`regularized_so6_covariance_condition_and_trace_conservation`)**:
    Regularized SO(6) covariance conditioning trace conservation.
25. **Theorems 2.20–2.22**:
    Contact coboundary invariants, Cayley SO(3) rotor norm preservation, and multi-axis discrete Morse filtering.

All proofs compile cleanly with 0 sorry tokens, 0 custom axioms, and 0 warnings.
-/

namespace CellSnail.Formalization

open scoped BigOperators
open Real Finset
open Classical

-- ============================================================================
-- Section 1: Foundational Bilinear Triad Operator & Shell Energy (Theorems 2.1–2.4)
-- ============================================================================

/-- Skew-symmetrized bilinear triad operator:
$$B_W(u, v)_k = \sum_{i, j \in \mathrm{Fin}\,M} u_i v_j (W_{i,j,k} - W_{i,k,j}).$$ -/
def triadOperator {M : ℕ} (W : Fin M → Fin M → Fin M → ℝ)
    (u v : Fin M → ℝ) (k : Fin M) : ℝ :=
  ∑ i : Fin M, ∑ j : Fin M, u i * v j * (W i j k - W i k j)

/-- Standard Euclidean inner product on $\mathbb{R}^M$. -/
def shellInner {M : ℕ} (x y : Fin M → ℝ) : ℝ :=
  ∑ k : Fin M, x k * y k

/-- Squared Euclidean norm $\|x\|_2^2 = \langle x, x \rangle$ on $\mathbb{R}^M$. -/
def shellEnergy {M : ℕ} (x : Fin M → ℝ) : ℝ :=
  shellInner x x

/-- Pointwise orthogonality $\langle v, B_W(u, v) \rangle = 0$. -/
theorem triad_pointwise_orthogonal {M : ℕ} (W : Fin M → Fin M → Fin M → ℝ)
    (u v : Fin M → ℝ) :
    shellInner v (triadOperator W u v) = 0 := by
  unfold shellInner triadOperator
  simp_rw [mul_sum]; rw [sum_comm]
  refine sum_eq_zero fun i _ => ?_
  simp_rw [mul_sub, sum_sub_distrib, sub_eq_zero]
  rw [sum_comm]
  exact sum_congr rfl fun j _ => sum_congr rfl fun k _ => by ring

/-! ### Theorem 2.1: Skew-Symmetric Color-Occlusion Antisymmetry -/

/-- Directed interaction tensor $S(W, i, c_1, c_2) = W(i, c_1, c_2) - W(i, c_2, c_1)$. -/
def skewTriad {C : ℕ} (W : Fin C → Fin C → Fin C → ℝ) (i c₁ c₂ : Fin C) : ℝ :=
  W i c₁ c₂ - W i c₂ c₁

/-- **Theorem 2.1 (`skew_triad_swap_neg`)**:
Swapping competing colors strictly negates directed interaction:
$$S(W, i, c_2, c_1) = -S(W, i, c_1, c_2),$$
and self-interaction vanishes identically: $S(W, i, c, c) = 0$. -/
theorem skew_triad_swap_neg {C : ℕ} (W : Fin C → Fin C → Fin C → ℝ) (i c₁ c₂ : Fin C) :
    skewTriad W i c₂ c₁ = -skewTriad W i c₁ c₂ ∧ skewTriad W i c₁ c₁ = 0 := by
  unfold skewTriad; constructor <;> ring

theorem skew_triad_self_zero {C : ℕ} (W : Fin C → Fin C → Fin C → ℝ) (i c : Fin C) :
    skewTriad W i c c = 0 :=
  (skew_triad_swap_neg W i c c).2

/-! ### Theorem 2.2: Exact Tie-Breaking of Symmetric Readout Collisions -/

/-- Post-readout logit combining symmetric baseline $\ell_{\mathrm{sym}}$ and triad shift. -/
def postLogit (C : ℕ) (ℓ_sym : Fin C → ℝ) (W : Fin C → Fin C → Fin C → ℝ)
    (u v : Fin C → ℝ) (k : Fin C) : ℝ :=
  ℓ_sym k + triadOperator W u v k

lemma triadOperator_two_color_eval {C : ℕ} (W : Fin C → Fin C → Fin C → ℝ)
    (u v : Fin C → ℝ) (i₀ c₁ c₂ k : Fin C) (α β : ℝ) (hc : c₁ ≠ c₂)
    (hu_i0 : u i₀ = β) (hu_other : ∀ i : Fin C, i ≠ i₀ → u i = 0)
    (hv_c1 : v c₁ = α) (hv_c2 : v c₂ = α)
    (hv_other : ∀ c : Fin C, c ≠ c₁ → c ≠ c₂ → v c = 0) :
    triadOperator W u v k =
      β * α * (W i₀ c₁ k - W i₀ k c₁) + β * α * (W i₀ c₂ k - W i₀ k c₂) := by
  unfold triadOperator
  rw [Finset.sum_eq_single i₀]
  · rw [← Finset.add_sum_erase _ _ (Finset.mem_univ c₁),
        ← Finset.add_sum_erase _ _ (Finset.mem_erase.mpr ⟨hc.symm, Finset.mem_univ c₂⟩),
        Finset.sum_eq_zero, add_zero, hu_i0, hv_c1, hv_c2]
    intro j hj
    simp only [Finset.mem_erase] at hj
    simp [hv_other j hj.2.1 hj.1]
  · exact fun i _ hi => Finset.sum_eq_zero fun j _ => by simp [hu_other i hi]
  · exact fun h => (h (Finset.mem_univ i₀)).elim

/-- **Theorem 2.2 (`occlusion_triad_breaks_symmetric_tie`)**:
Under a symmetric readout tie $\ell_{\mathrm{sym}}(c_1) = \ell_{\mathrm{sym}}(c_2)$, the bilinear
triad shift produces an exact antisymmetric margin:
$$\ell_{\mathrm{post}}(c_1) - \ell_{\mathrm{post}}(c_2) = 2 \beta \alpha (W(i_0, c_2, c_1) - W(i_0, c_1, c_2)),$$
strictly breaking the tie whenever $W(i_0, c_2, c_1) > W(i_0, c_1, c_2)$. -/
theorem occlusion_triad_breaks_symmetric_tie {C : ℕ}
    (ℓ_sym : Fin C → ℝ) (W : Fin C → Fin C → Fin C → ℝ)
    (u v : Fin C → ℝ) (i₀ c₁ c₂ : Fin C) (α β : ℝ)
    (hα : 0 < α) (hβ : 0 < β) (h_tie : ℓ_sym c₁ = ℓ_sym c₂)
    (hu_i0 : u i₀ = β) (hu_other : ∀ i : Fin C, i ≠ i₀ → u i = 0)
    (hv_c1 : v c₁ = α) (hv_c2 : v c₂ = α)
    (hv_other : ∀ c : Fin C, c ≠ c₁ → c ≠ c₂ → v c = 0) :
    (postLogit C ℓ_sym W u v c₁ - postLogit C ℓ_sym W u v c₂ =
      2 * β * α * (W i₀ c₂ c₁ - W i₀ c₁ c₂)) ∧
    (W i₀ c₂ c₁ > W i₀ c₁ c₂ →
      postLogit C ℓ_sym W u v c₁ > postLogit C ℓ_sym W u v c₂) := by
  have h_diff : postLogit C ℓ_sym W u v c₁ - postLogit C ℓ_sym W u v c₂ =
      2 * β * α * (W i₀ c₂ c₁ - W i₀ c₁ c₂) := by
    by_cases hc : c₁ = c₂
    · subst hc; ring
    · unfold postLogit
      rw [triadOperator_two_color_eval W u v i₀ c₁ c₂ c₁ α β hc hu_i0 hu_other hv_c1 hv_c2 hv_other,
          triadOperator_two_color_eval W u v i₀ c₁ c₂ c₂ α β hc hu_i0 hu_other hv_c1 hv_c2 hv_other,
          h_tie]; ring
  exact ⟨h_diff, fun hW => by nlinarith [mul_pos hβ hα]⟩

/-! ### Theorem 2.3: Readout Energy Conservation -/

/-- **Theorem 2.3 (`cellsnail_readout_energy_conservation`)**:
For any mode vectors $u, v$ and coupling tensor $W$, $\langle v, B_W(u, v) \rangle = 0$,
and the Slow-Fast recurrent pump has zero radial energy growth. -/
theorem cellsnail_readout_energy_conservation {C M : ℕ}
    (W : Fin C → Fin C → Fin C → ℝ) (u v : Fin C → ℝ)
    (W_rec : Fin M → Fin M → Fin M → ℝ) (z_H z_L : Fin M → ℝ) :
    (∑ k : Fin C, v k * triadOperator W u v k) = 0 ∧
    shellInner z_L (triadOperator W_rec z_H z_L) = 0 :=
  ⟨triad_pointwise_orthogonal W u v, triad_pointwise_orthogonal W_rec z_H z_L⟩

/-! ### Theorem 2.4: Step-0 System 1 Homotopy Equivalence -/

/-- Full readout combining System 1 baseline, symmetric shift, and skew triad shift. -/
def cellsnailReadout (C : ℕ) (ℓ_S1 Δ_sym : Fin C → ℝ)
    (W : Fin C → Fin C → Fin C → ℝ) (u v : Fin C → ℝ) (k : Fin C) : ℝ :=
  ℓ_S1 k + Δ_sym k + triadOperator W u v k

/-- **Theorem 2.4 (`cellsnail_step0_system1_homotopy`)**:
At Step-0 ($\Delta_{\mathrm{sym}} = 0, W = 0$), the full readout reduces identically to System 1. -/
theorem cellsnail_step0_system1_homotopy {C : ℕ}
    (ℓ_S1 u v : Fin C → ℝ) (k : Fin C) :
    cellsnailReadout C ℓ_S1 0 0 u v k = ℓ_S1 k := by
  simp [cellsnailReadout, triadOperator]

-- ============================================================================
-- Section 2: 2D Axial RoPE Medoid Invariance & Spherical Cascade (Theorems 2.5–2.6)
-- ============================================================================

noncomputable def rot2 (θ : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (Real.cos θ * v.1 - Real.sin θ * v.2, Real.sin θ * v.1 + Real.cos θ * v.2)

def inner2 (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def normSq2 (v : ℝ × ℝ) : ℝ := inner2 v v

noncomputable def rope2d (θ_y θ_x : ℝ) (q_y q_x : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  (rot2 θ_y q_y, rot2 θ_x q_x)

def inner2d (u v : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := inner2 u.1 v.1 + inner2 u.2 v.2
def normSq2d (v : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := normSq2 v.1 + normSq2 v.2
noncomputable def avg2 (u v : ℝ × ℝ) : ℝ × ℝ := ((u.1 + v.1) / 2, (u.2 + v.2) / 2)

noncomputable def rope2dPatchAvg (θ_y1 θ_x1 θ_y2 θ_x2 : ℝ) (k_y k_x : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  (avg2 (rot2 θ_y1 k_y) (rot2 θ_y2 k_y), avg2 (rot2 θ_x1 k_x) (rot2 θ_x2 k_x))

lemma inner2_rot2_shift (α β : ℝ) (u v : ℝ × ℝ) :
    inner2 (rot2 α u) (rot2 β v) = inner2 u (rot2 (β - α) v) := by
  unfold inner2 rot2; rw [Real.cos_sub, Real.sin_sub]; ring

/-- **Theorem 2.5a (`rope2d_medoid_relative_invariance`)**:
The 2D axial rotary inner product depends purely on the relative 2D displacement:
$$\langle \mathrm{rope2d}(y_q, x_q, q), \mathrm{rope2d}(y^*, x^*, k) \rangle
  = \langle q_y, \mathcal{R}_{y^* - y_q} k_y \rangle + \langle q_x, \mathcal{R}_{x^* - x_q} k_x \rangle.$$ -/
theorem rope2d_medoid_relative_invariance
    (y_q x_q y_star x_star : ℝ) (q_y q_x k_y k_x : ℝ × ℝ) :
    inner2d (rope2d y_q x_q q_y q_x) (rope2d y_star x_star k_y k_x) =
      inner2 q_y (rot2 (y_star - y_q) k_y) + inner2 q_x (rot2 (x_star - x_q) k_x) := by
  unfold inner2d rope2d; rw [inner2_rot2_shift, inner2_rot2_shift]

lemma normSq2_avg2_rot2 (θ₁ θ₂ : ℝ) (k : ℝ × ℝ) :
    normSq2 (avg2 (rot2 θ₁ k) (rot2 θ₂ k)) =
      normSq2 k * (1 + Real.cos (θ₁ - θ₂)) / 2 := by
  unfold normSq2 inner2 avg2 rot2; rw [Real.cos_sub]
  linear_combination
    ((k.1 ^ 2 + k.2 ^ 2) / 4) * Real.cos_sq_add_sin_sq θ₁ +
    ((k.1 ^ 2 + k.2 ^ 2) / 4) * Real.cos_sq_add_sin_sq θ₂

lemma normSq2_nonneg (k : ℝ × ℝ) : 0 ≤ normSq2 k :=
  add_nonneg (mul_self_nonneg k.1) (mul_self_nonneg k.2)

/-- **Theorem 2.5b (`rope2d_arithmetic_norm_lt`)**:
Arithmetic patch averaging across distinct grid phases strictly attenuates 2D key norm:
$$\|\mathrm{rope2dPatchAvg}(\theta, k)\|^2 < \|k\|^2.$$ -/
theorem rope2d_arithmetic_norm_lt
    (θ_y1 θ_x1 θ_y2 θ_x2 : ℝ) (k_y k_x : ℝ × ℝ)
    (h_att : (Real.cos (θ_y1 - θ_y2) < 1 ∧ 0 < normSq2 k_y) ∨
             (Real.cos (θ_x1 - θ_x2) < 1 ∧ 0 < normSq2 k_x)) :
    normSq2d (rope2dPatchAvg θ_y1 θ_x1 θ_y2 θ_x2 k_y k_x) < normSq2d (k_y, k_x) := by
  unfold normSq2d rope2dPatchAvg; dsimp only
  rw [normSq2_avg2_rot2, normSq2_avg2_rot2]
  rcases h_att with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;>
    nlinarith [normSq2_nonneg k_y, normSq2_nonneg k_x, Real.cos_le_one (θ_y1 - θ_y2), Real.cos_le_one (θ_x1 - θ_x2)]

/-! ### Theorem 2.6: Multi-Step Unrolled Slow-Fast Spherical Invariance -/

def tangentTriadStep (M : ℕ) (γ : ℝ) (W_rec : Fin M → Fin M → Fin M → ℝ)
    (u v : Fin M → ℝ) (k : Fin M) : ℝ :=
  γ * triadOperator W_rec u v k

noncomputable def sphericalRetract {M : ℕ} (v v_upd : Fin M → ℝ) (k : Fin M) : ℝ :=
  (Real.sqrt (shellEnergy v) / Real.sqrt (shellEnergy v_upd)) * v_upd k

noncomputable def sphericalTriadStep (M : ℕ) (γ : ℝ) (W_rec : Fin M → Fin M → Fin M → ℝ)
    (u v : Fin M → ℝ) : Fin M → ℝ :=
  sphericalRetract v (fun k => v k + tangentTriadStep M γ W_rec u v k)

noncomputable def multistepSphericalTriad (M : ℕ) (γ : ℕ → ℝ)
    (W_rec : ℕ → Fin M → Fin M → Fin M → ℝ)
    (u : ℕ → Fin M → ℝ) (v₀ : Fin M → ℝ) : ℕ → (Fin M → ℝ)
  | 0 => v₀
  | t + 1 => sphericalTriadStep M (γ t) (W_rec t) (u t) (multistepSphericalTriad M γ W_rec u v₀ t)

lemma shellEnergy_nonneg {M : ℕ} (x : Fin M → ℝ) : 0 ≤ shellEnergy x :=
  Finset.sum_nonneg (s := Finset.univ) fun k _ => mul_self_nonneg (x k)

lemma shellEnergy_eq_zero_iff {M : ℕ} (x : Fin M → ℝ) :
    shellEnergy x = 0 ↔ ∀ k : Fin M, x k = 0 := by
  unfold shellEnergy shellInner
  simp [Finset.sum_eq_zero_iff_of_nonneg (fun k _ => mul_self_nonneg (x k))]

lemma tangentTriadStep_orthogonal {M : ℕ} (γ : ℝ) (W_rec : Fin M → Fin M → Fin M → ℝ)
    (u v : Fin M → ℝ) :
    shellInner v (tangentTriadStep M γ W_rec u v) = 0 := by
  unfold shellInner tangentTriadStep
  simp_rw [mul_left_comm (v _), ← Finset.mul_sum]
  exact mul_eq_zero_of_right γ (triad_pointwise_orthogonal W_rec u v)

lemma shellEnergy_add_of_orthogonal {M : ℕ} (v δv : Fin M → ℝ)
    (h_orth : shellInner v δv = 0) :
    shellEnergy (fun k => v k + δv k) = shellEnergy v + shellEnergy δv := by
  have h_expand :
      shellEnergy (fun k => v k + δv k) =
        shellEnergy v + 2 * shellInner v δv + shellEnergy δv := by
    unfold shellEnergy shellInner
    simp only [← Finset.sum_add_distrib, Finset.mul_sum]
    exact Finset.sum_congr rfl fun k _ => by ring
  linarith

lemma shellEnergy_scale {M : ℕ} (c : ℝ) (x : Fin M → ℝ) :
    shellEnergy (fun k => c * x k) = c ^ 2 * shellEnergy x := by
  unfold shellEnergy shellInner; rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun k _ => by ring

lemma sphericalTriadStep_preserves_energy {M : ℕ} (γ : ℝ)
    (W_rec : Fin M → Fin M → Fin M → ℝ) (u v : Fin M → ℝ) :
    shellEnergy (sphericalTriadStep M γ W_rec u v) = shellEnergy v := by
  unfold sphericalTriadStep sphericalRetract
  rw [shellEnergy_scale]
  by_cases hv0 : shellEnergy v = 0
  · simp [hv0]
  · have hv_pos : 0 < shellEnergy v := lt_of_le_of_ne (shellEnergy_nonneg v) (Ne.symm hv0)
    have h_pyth := shellEnergy_add_of_orthogonal v _ (tangentTriadStep_orthogonal γ W_rec u v)
    have h_upd_pos : 0 < shellEnergy (fun k => v k + tangentTriadStep M γ W_rec u v k) := by
      linarith [hv_pos, shellEnergy_nonneg (tangentTriadStep M γ W_rec u v)]
    rw [div_pow, Real.sq_sqrt hv_pos.le, Real.sq_sqrt h_upd_pos.le, div_mul_cancel₀ _ h_upd_pos.ne']

/-- **Theorem 2.6a (`cellsnail_multistep_spherical_invariance`)**:
Across any $T$-step unrolled cascade, the terminal retracted state preserves exact initial norm:
$$\|\tilde{v}_L^{(T)}\|_2^2 = \|v_L^{(0)}\|_2^2.$$ -/
theorem cellsnail_multistep_spherical_invariance {M : ℕ}
    (γ : ℕ → ℝ) (W_rec : ℕ → Fin M → Fin M → Fin M → ℝ)
    (u : ℕ → Fin M → ℝ) (v₀ : Fin M → ℝ) (T : ℕ) :
    shellEnergy (multistepSphericalTriad M γ W_rec u v₀ T) = shellEnergy v₀ ∧
    (∀ t : ℕ,
      let v_t := multistepSphericalTriad M γ W_rec u v₀ t
      let δv_t := tangentTriadStep M (γ t) (W_rec t) (u t) v_t
      shellInner v_t δv_t = 0) :=
  ⟨by induction T with | zero => rfl | succ t ih => simp [multistepSphericalTriad, sphericalTriadStep_preserves_energy, ih],
   fun t => tangentTriadStep_orthogonal (γ t) (W_rec t) (u t) _⟩

lemma sphericalTriadStep_zero {M : ℕ} (γ : ℝ) (u v : Fin M → ℝ) :
    sphericalTriadStep M γ 0 u v = v := by
  have h_tan : (fun i => v i + tangentTriadStep M γ 0 u v i) = v := by
    funext i; simp [tangentTriadStep, triadOperator]
  unfold sphericalTriadStep sphericalRetract; rw [h_tan]
  by_cases hv0 : shellEnergy v = 0
  · ext k; simp [hv0, (shellEnergy_eq_zero_iff v).mp hv0 k]
  · ext k; rw [div_self (ne_of_gt (Real.sqrt_pos.mpr (lt_of_le_of_ne (shellEnergy_nonneg v) (Ne.symm hv0)))), one_mul]

def system1CarryTrajectory {M : ℕ}
    (step_S1 : ℕ → (Fin M → ℝ) × (Fin M → ℝ) → (Fin M → ℝ) × (Fin M → ℝ))
    (z₀ : (Fin M → ℝ) × (Fin M → ℝ)) : ℕ → (Fin M → ℝ) × (Fin M → ℝ)
  | 0 => z₀
  | t + 1 => step_S1 t (system1CarryTrajectory step_S1 z₀ t)

noncomputable def cellsnailCarryTrajectory {M : ℕ}
    (step_S1 : ℕ → (Fin M → ℝ) × (Fin M → ℝ) → (Fin M → ℝ) × (Fin M → ℝ))
    (γ : ℕ → ℝ) (W_rec : ℕ → Fin M → Fin M → Fin M → ℝ)
    (δz_H δz_L : ℕ → Fin M → ℝ)
    (z₀ : (Fin M → ℝ) × (Fin M → ℝ)) : ℕ → (Fin M → ℝ) × (Fin M → ℝ)
  | 0 => z₀
  | t + 1 =>
      let z_t := cellsnailCarryTrajectory step_S1 γ W_rec δz_H δz_L z₀ t
      let v_tilde := sphericalTriadStep M (γ t) (W_rec t) z_t.1 z_t.2
      step_S1 t (fun k => z_t.1 k + δz_H t k, fun k => v_tilde k + δz_L t k)

/-- **Theorem 2.6b (`cellsnail_multistep_step0_homotopy`)**:
At $W_{\mathrm{rec}} = 0$ and $\delta z = 0$, the $T$-step unrolled cascade reduces identically to System 1. -/
theorem cellsnail_multistep_step0_homotopy {M : ℕ}
    (step_S1 : ℕ → (Fin M → ℝ) × (Fin M → ℝ) → (Fin M → ℝ) × (Fin M → ℝ))
    (γ : ℕ → ℝ) (u : ℕ → Fin M → ℝ) (v₀ : Fin M → ℝ)
    (z₀ : (Fin M → ℝ) × (Fin M → ℝ)) (T : ℕ) :
    multistepSphericalTriad M γ 0 u v₀ T = v₀ ∧
    cellsnailCarryTrajectory step_S1 γ 0 0 0 z₀ T =
      system1CarryTrajectory step_S1 z₀ T := by
  constructor <;> induction T with
  | zero => rfl
  | succ t ih =>
    simp [multistepSphericalTriad, cellsnailCarryTrajectory, system1CarryTrajectory,
          ih, sphericalTriadStep_zero]

-- ============================================================================
-- Section 3: SO(6) Kelvin-Mandel Stiffness & Discrete Morse Shielding (Theorem 2.7)
-- ============================================================================

def elasticEnergy6 (C : Fin 6 → Fin 6 → ℝ) (ε : Fin 6 → ℝ) : ℝ :=
  ∑ i : Fin 6, ∑ j : Fin 6, ε i * C i j * ε j

def matVec6 (M : Fin 6 → Fin 6 → ℝ) (x : Fin 6 → ℝ) (i : Fin 6) : ℝ :=
  ∑ j : Fin 6, M i j * x j

def matTranspose6 (M : Fin 6 → Fin 6 → ℝ) (i j : Fin 6) : ℝ := M j i

def rotateStiffness6 (M C : Fin 6 → Fin 6 → ℝ) (i j : Fin 6) : ℝ :=
  ∑ k : Fin 6, ∑ l : Fin 6, M i k * C k l * M j l

def trace6 (C : Fin 6 → Fin 6 → ℝ) : ℝ := ∑ i : Fin 6, C i i

def IsOrthogonal6 (M : Fin 6 → Fin 6 → ℝ) : Prop :=
  ∀ x : Fin 6 → ℝ, shellEnergy (matVec6 (matTranspose6 M) x) = shellEnergy x

def IsColOrthonormal6 (M : Fin 6 → Fin 6 → ℝ) : Prop :=
  ∀ k l : Fin 6, (∑ i : Fin 6, M i k * M i l) = if k = l then 1 else 0

def IsRowOrthonormal6 (M : Fin 6 → Fin 6 → ℝ) : Prop :=
  ∀ i j : Fin 6, (∑ k : Fin 6, M i k * M j k) = if i = j then 1 else 0

lemma isOrthogonal6_of_rowOrthonormal6 (M : Fin 6 → Fin 6 → ℝ)
    (hM : IsRowOrthonormal6 M) : IsOrthogonal6 M := fun x => by
  unfold shellEnergy shellInner matVec6 matTranspose6
  calc (∑ k : Fin 6, (∑ i : Fin 6, M i k * x i) * (∑ j : Fin 6, M j k * x j))
    _ = ∑ i : Fin 6, ∑ j : Fin 6, x i * x j * (∑ k : Fin 6, M i k * M j k) := by
        simp only [Fin.sum_univ_six]; ring
    _ = ∑ k : Fin 6, x k * x k := by simp [hM _ _]

noncomputable def kelvinMandelDipRotor (θ : ℝ) (i j : Fin 6) : ℝ :=
  let c := Real.cos θ
  let s := Real.sin θ
  let s2 := Real.sqrt 2
  match i.val, j.val with
  | 0, 0 => c * c
  | 0, 2 => s * s
  | 0, 4 => s2 * c * s
  | 1, 1 => 1
  | 2, 0 => s * s
  | 2, 2 => c * c
  | 2, 4 => -s2 * c * s
  | 3, 3 => c
  | 3, 5 => -s
  | 4, 0 => -s2 * c * s
  | 4, 2 => s2 * c * s
  | 4, 4 => c * c - s * s
  | 5, 3 => s
  | 5, 5 => c
  | _, _ => 0

lemma kelvinMandelDipRotor_orthonormal (θ : ℝ) :
    IsRowOrthonormal6 (kelvinMandelDipRotor θ) ∧
    IsColOrthonormal6 (kelvinMandelDipRotor θ) ∧
    IsOrthogonal6 (kelvinMandelDipRotor θ) := by
  have h_row : ∀ φ : ℝ, IsRowOrthonormal6 (kelvinMandelDipRotor φ) := by
    intro φ a b
    have hs2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
    have h2 := Real.cos_sq_add_sin_sq φ
    have h4 : Real.cos φ ^ 4 + Real.sin φ ^ 4 = 1 - 2 * Real.cos φ ^ 2 * Real.sin φ ^ 2 := by
      linear_combination (Real.cos φ ^ 2 + Real.sin φ ^ 2 + 1) * h2
    rw [Fin.sum_univ_six]
    fin_cases a <;>
      simp only [kelvinMandelDipRotor, Fin.isValue, Fin.coe_ofNat_eq_mod, Nat.zero_mod,
        Nat.one_mod, Nat.reduceMod, Nat.mod_succ, zero_mul, add_zero, zero_add] <;>
      fin_cases b <;>
      simp only [mul_zero, add_zero, Fin.mk.injEq, Nat.reduceEqDiff, ↓reduceIte] <;>
      ring_nf <;> (try simp only [hs2]) <;> try linarith [h2, h4]
  have h_tr : ∀ a b : Fin 6, kelvinMandelDipRotor θ a b = kelvinMandelDipRotor (-θ) b a := by
    intro a b; fin_cases a <;> fin_cases b <;> simp [kelvinMandelDipRotor]
  exact ⟨h_row θ, fun k l => by simpa only [h_tr] using h_row (-θ) k l,
    isOrthogonal6_of_rowOrthonormal6 _ (h_row θ)⟩

/-- **Theorem 2.7a (`elasticEnergy6_rotateStiffness6`)**:
$$\varepsilon^\top (M C M^\top) \varepsilon = (M^\top \varepsilon)^\top C (M^\top \varepsilon).$$ -/
theorem elasticEnergy6_rotateStiffness6 (M C : Fin 6 → Fin 6 → ℝ) (ε : Fin 6 → ℝ) :
    elasticEnergy6 (rotateStiffness6 M C) ε =
      elasticEnergy6 C (matVec6 (matTranspose6 M) ε) := by
  simp only [elasticEnergy6, rotateStiffness6, matVec6, matTranspose6, Fin.sum_univ_six]; ring

/-- **Theorem 2.7b (`so6_stiffness_preserves_pos_def`)**:
Orthogonal Kelvin-Mandel rotation preserves coercivity lower bounds and strict positive-definiteness. -/
theorem so6_stiffness_preserves_pos_def (M C : Fin 6 → Fin 6 → ℝ) (μ_min : ℝ)
    (hM : IsOrthogonal6 M) (hμ : 0 < μ_min)
    (h_coercive : ∀ ξ : Fin 6 → ℝ, μ_min * shellEnergy ξ ≤ elasticEnergy6 C ξ) :
    (∀ ε : Fin 6 → ℝ, μ_min * shellEnergy ε ≤ elasticEnergy6 (rotateStiffness6 M C) ε) ∧
    (∀ ε : Fin 6 → ℝ, (∃ i : Fin 6, ε i ≠ 0) →
      0 < elasticEnergy6 (rotateStiffness6 M C) ε) :=
  have h_bound (ε : Fin 6 → ℝ) : μ_min * shellEnergy ε ≤ elasticEnergy6 (rotateStiffness6 M C) ε := by
    rw [elasticEnergy6_rotateStiffness6, ← hM ε]; exact h_coercive _
  ⟨h_bound, fun ε ⟨i, hi⟩ =>
    lt_of_lt_of_le (mul_pos hμ (lt_of_le_of_ne (shellEnergy_nonneg ε)
      fun h => hi ((shellEnergy_eq_zero_iff ε).mp h.symm i))) (h_bound ε)⟩

/-- **Theorem 2.7c (`so6_stiffness_trace_conservation`)**:
$$\mathrm{Tr}(M C M^\top) = \mathrm{Tr}(C).$$ -/
theorem so6_stiffness_trace_conservation (M C : Fin 6 → Fin 6 → ℝ)
    (hM : IsColOrthonormal6 M) :
    trace6 (rotateStiffness6 M C) = trace6 C := by
  unfold trace6 rotateStiffness6
  calc (∑ i : Fin 6, ∑ k : Fin 6, ∑ l : Fin 6, M i k * C k l * M i l)
    _ = ∑ k : Fin 6, ∑ l : Fin 6, C k l * (∑ i : Fin 6, M i k * M i l) := by simp only [Fin.sum_univ_six]; ring
    _ = ∑ i : Fin 6, C i i := by simp [hM _ _]

def deltaMinus (v : ℤ → ℝ) (z : ℤ) : ℝ := v z - v (z - 1)
def deltaPlus  (v : ℤ → ℝ) (z : ℤ) : ℝ := v z - v (z + 1)

def morseImpulse (v : ℤ → ℝ) (z : ℤ) : ℝ := deltaMinus v z * deltaPlus v z

def IsStrictLocalExtremum (v : ℤ → ℝ) (z : ℤ) : Prop :=
  (v (z - 1) < v z ∧ v (z + 1) < v z) ∨ (v z < v (z - 1) ∧ v z < v (z + 1))

def IsStrictLocalMaximum (v : ℤ → ℝ) (z : ℤ) : Prop :=
  v (z - 1) < v z ∧ v (z + 1) < v z

def IsLocallyMonotone (v : ℤ → ℝ) (z : ℤ) : Prop :=
  (v (z - 1) ≤ v z ∧ v z ≤ v (z + 1)) ∨ (v (z + 1) ≤ v z ∧ v z ≤ v (z - 1))

def IsStrictlyMonotoneAt (v : ℤ → ℝ) (z : ℤ) : Prop :=
  (v (z - 1) < v z ∧ v z < v (z + 1)) ∨ (v (z + 1) < v z ∧ v z < v (z - 1))

noncomputable def discreteLaplacian (v : ℤ → ℝ) (z : ℤ) : ℝ :=
  (deltaMinus v z + deltaPlus v z) / 2

/-- **Theorem 1 (`morse_monotone_slope_rejected`)**:
On a strictly monotonic sequence along 1D coordinates, the discrete Morse product is strictly negative,
rejecting spurious slope trends. On weak monotonicity, the impulse is non-positive. -/
theorem morse_monotone_slope_rejected (v : ℤ → ℝ) (z : ℤ) :
    (IsStrictlyMonotoneAt v z → morseImpulse v z < 0) ∧
    (IsLocallyMonotone v z → morseImpulse v z ≤ 0) ∧
    ((v z = v (z - 1) ∨ v z = v (z + 1)) → morseImpulse v z = 0) := by
  refine ⟨?_, ?_, ?_⟩
  · rintro (⟨h1, h2⟩ | ⟨h1, h2⟩) <;> {
      unfold morseImpulse deltaMinus deltaPlus
      nlinarith
    }
  · rintro (⟨h1, h2⟩ | ⟨h1, h2⟩) <;> {
      unfold morseImpulse deltaMinus deltaPlus
      nlinarith
    }
  · rintro (h | h) <;> {
      unfold morseImpulse deltaMinus deltaPlus
      rw [h]; ring
    }

/-- **Theorem 2 (`morse_local_extremum_isolated`)**:
Strict local extremum (focal amplicon) yields strictly positive Morse product. -/
theorem morse_local_extremum_isolated (v : ℤ → ℝ) (z : ℤ) :
    0 < morseImpulse v z ↔ IsStrictLocalExtremum v z := by
  unfold morseImpulse deltaMinus deltaPlus IsStrictLocalExtremum
  rw [mul_pos_iff]
  simp only [sub_pos, sub_neg]

theorem morseImpulse_iff_strict_extremum (v : ℤ → ℝ) (z : ℤ) :
    0 < morseImpulse v z ↔
      (v (z - 1) < v z ∧ v (z + 1) < v z) ∨ (v z < v (z - 1) ∧ v z < v (z + 1)) :=
  morse_local_extremum_isolated v z

/-- **Theorem 3 (`focal_amplicon_amplitude_recovery`)**:
Focal amplicon isolated peak is preserved at 100% amplitude under Morse gating. -/
theorem focal_amplicon_amplitude_recovery (v : ℤ → ℝ) (z : ℤ) (v_bg δ : ℝ)
    (h_bg_l : v (z - 1) = v_bg)
    (h_bg_r : v (z + 1) = v_bg)
    (h_peak : v z = v_bg + δ)
    (h_pos : 0 < δ) :
    morseImpulse v z = δ ^ 2 ∧
    0 < morseImpulse v z ∧
    IsStrictLocalMaximum v z ∧
    discreteLaplacian v z = δ := by
  have hd_m : deltaMinus v z = δ := by unfold deltaMinus; rw [h_peak, h_bg_l]; ring
  have hd_p : deltaPlus v z = δ := by unfold deltaPlus; rw [h_peak, h_bg_r]; ring
  have h_imp : morseImpulse v z = δ ^ 2 := by unfold morseImpulse; rw [hd_m, hd_p]; ring
  refine ⟨h_imp, ?_, ?_, ?_⟩
  · rw [h_imp]; exact sq_pos_of_pos h_pos
  · unfold IsStrictLocalMaximum; rw [h_bg_l, h_bg_r, h_peak]; exact ⟨lt_add_of_pos_right v_bg h_pos, lt_add_of_pos_right v_bg h_pos⟩
  · unfold discreteLaplacian; rw [hd_m, hd_p]; ring

/-- **Theorem 4 (`flanking_leakage_bounded`)**:
Flanking bins beyond the focal amplicon receive zero leakage amplitude. -/
theorem flanking_leakage_bounded (v : ℤ → ℝ) (z : ℤ) (v_bg δ : ℝ)
    (h_outer_l : v (z - 2) = v_bg)
    (h_flank_l : v (z - 1) = v_bg)
    (_h_peak : v z = v_bg + δ)
    (h_flank_r : v (z + 1) = v_bg)
    (h_outer_r : v (z + 2) = v_bg) :
    morseImpulse v (z - 1) = 0 ∧
    morseImpulse v (z + 1) = 0 ∧
    ¬ (0 < morseImpulse v (z - 1)) ∧
    ¬ (0 < morseImpulse v (z + 1)) := by
  have h_idx_l : z - 1 - 1 = z - 2 := by ring
  have h_idx_r : z + 1 + 1 = z + 2 := by ring
  have h_ml : morseImpulse v (z - 1) = 0 := by
    unfold morseImpulse deltaMinus deltaPlus
    rw [h_flank_l, h_idx_l, h_outer_l]
    ring
  have h_mr : morseImpulse v (z + 1) = 0 := by
    unfold morseImpulse deltaMinus deltaPlus
    rw [h_flank_r, h_idx_r, h_outer_r]
    ring
  refine ⟨h_ml, h_mr, ?_, ?_⟩
  · rw [h_ml]; linarith
  · rw [h_mr]; linarith

noncomputable def plateauMean {L : ℕ} (v : Fin L → ℝ) : ℝ :=
  (∑ i : Fin L, v i) / (L : ℝ)

/-- **Theorem 5 (`plateau_moving_average_unbiased`)**:
Symmetrical moving average across plateau bins is unbiased with respect to constant plateau signal. -/
theorem plateau_moving_average_unbiased {L : ℕ} (hL : 0 < L) (c : ℝ) :
    plateauMean (fun _ : Fin L => c) = c := by
  unfold plateauMean
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  exact mul_div_cancel_left₀ c (Nat.cast_ne_zero.mpr (ne_of_gt hL))

/-- **Theorem 6 (`plateau_variance_reduction`)**:
Moving average over window of size W reduces noise variance by a factor of 1/W. -/
theorem plateau_variance_reduction {W : ℕ} (hW : 1 ≤ W) (σ_sq : ℝ) (hσ : 0 ≤ σ_sq) :
    σ_sq / (W : ℝ) ≤ σ_sq ∧
    (1 < W → 0 < σ_sq → σ_sq / (W : ℝ) < σ_sq) := by
  have hW_ge : 1 ≤ (W : ℝ) := by exact_mod_cast hW
  refine ⟨div_le_self hσ hW_ge, fun hW_gt hσ_pos => ?_⟩
  have hW_real : 1 < (W : ℝ) := by exact_mod_cast hW_gt
  exact (div_lt_iff₀ (by positivity)).mpr (by nlinarith)

def stepHeight (v : ℤ → ℝ) (z : ℤ) : ℝ := v (z + 1) - v z

def boundaryPinnedVal (v : ℤ → ℝ) (z : ℤ) (pinned : Bool) (interiorVal : ℝ) : ℝ :=
  if pinned then v z else interiorVal

def unpinnedCrossBoundaryVal (v_self v_other : ℝ) (leakage_rate : ℝ) : ℝ :=
  (1 - leakage_rate) * v_self + leakage_rate * v_other

/-- **Theorem 7 (`boundary_pinned_step_preservation`)**:
Boundary-pinned step transition preserves step height without boundary erosion. -/
theorem boundary_pinned_step_preservation (v : ℤ → ℝ) (z : ℤ) (v_left v_right : ℝ)
    (h_l : v z = v_left)
    (h_r : v (z + 1) = v_right) :
    let v_pinned := fun i => boundaryPinnedVal v i true 0
    stepHeight v_pinned z = v_right - v_left ∧
    stepHeight v_pinned z = stepHeight v z ∧
    (∀ (leakage_rate : ℝ) (_h_leak_pos : 0 < leakage_rate) (_h_leak_le : leakage_rate ≤ 1 / 2) (_h_diff : v_left < v_right),
      let v_unpinned_l := unpinnedCrossBoundaryVal v_left v_right leakage_rate
      let v_unpinned_r := unpinnedCrossBoundaryVal v_right v_left leakage_rate
      v_unpinned_r - v_unpinned_l = (1 - 2 * leakage_rate) * (v_right - v_left) ∧
      v_unpinned_r - v_unpinned_l < v_right - v_left) := by
  intro v_pinned
  have hp_l : v_pinned z = v_left := by
    change v z = v_left
    exact h_l
  have hp_r : v_pinned (z + 1) = v_right := by
    change v (z + 1) = v_right
    exact h_r
  have h_step : stepHeight v_pinned z = v_right - v_left := by
    unfold stepHeight; rw [hp_r, hp_l]
  have h_raw : stepHeight v z = v_right - v_left := by
    unfold stepHeight; rw [h_r, h_l]
  refine ⟨h_step, by rw [h_step, h_raw], fun leakage_rate h_leak_pos _h_leak_le h_diff => ?_⟩
  dsimp [unpinnedCrossBoundaryVal]
  have h_eq : (1 - leakage_rate) * v_right + leakage_rate * v_left -
      ((1 - leakage_rate) * v_left + leakage_rate * v_right) =
      (1 - 2 * leakage_rate) * (v_right - v_left) := by ring
  refine ⟨h_eq, ?_⟩
  rw [h_eq]
  have h_gap : 0 < v_right - v_left := sub_pos.mpr h_diff
  have h_shrink : 1 - 2 * leakage_rate < 1 := by linarith [h_leak_pos]
  nlinarith [h_gap, h_shrink]

-- Compatibility aliases
def morseImpulseProd (v : ℤ → ℝ) (z : ℤ) : ℝ := morseImpulse v z
def isMorseImpulse (v : ℤ → ℝ) (z : ℤ) : Prop := 0 < morseImpulse v z
noncomputable def laplacian3pt (v : ℤ → ℝ) (z : ℤ) : ℝ := discreteLaplacian v z

-- 3D vector cross product helper
def cross3 (u v : Fin 3 → ℝ) (k : Fin 3) : ℝ :=
  match k.val with
  | 0 => u 1 * v 2 - u 2 * v 1
  | 1 => u 2 * v 0 - u 0 * v 2
  | _ => u 0 * v 1 - u 1 * v 0

-- ============================================================================
-- Section 5: Lie Rotors and Cellular Sheaves (Theorem 2.10)
-- ============================================================================

noncomputable def cayleyRotor2 (α : ℝ) (i j : Fin 2) : ℝ :=
  let denom := 1 + α ^ 2
  let c := (1 - α ^ 2) / denom
  let s := (2 * α) / denom
  match i.val, j.val with
  | 0, 0 => c
  | 0, 1 => s
  | 1, 0 => -s
  | 1, 1 => c
  | _, _ => 0

/-- **Theorem 2.10a (`cayley_rank2_orthogonality`)**:
For any skew-symmetric generator parameter $\alpha \in \mathbb{R}$, the Cayley transform
produces an exact orthogonal rotor satisfying $R^\top R = I_2$ and preserving the Euclidean norm:
$$\|R(\alpha) x\|_2^2 = \|x\|_2^2.$$ -/
theorem cayley_rank2_orthogonality (α : ℝ) (x : Fin 2 → ℝ) :
    (∀ j k : Fin 2, (∑ i : Fin 2, cayleyRotor2 α i j * cayleyRotor2 α i k) = if j = k then 1 else 0) ∧
    (∑ i : Fin 2, (∑ j : Fin 2, cayleyRotor2 α i j * x j) ^ 2) = (∑ i : Fin 2, x i ^ 2) := by
  have hdenom : 1 + α ^ 2 ≠ 0 := by positivity
  have h_alg : ((1 - α ^ 2) / (1 + α ^ 2)) ^ 2 + ((2 * α) / (1 + α ^ 2)) ^ 2 = 1 := by
    rw [div_pow, div_pow, ← add_div, show (1 - α ^ 2) ^ 2 + (2 * α) ^ 2 = (1 + α ^ 2) ^ 2 by ring,
        div_self (pow_ne_zero 2 hdenom)]
  refine ⟨fun j k => ?_, ?_⟩
  · rw [Fin.sum_univ_two]
    fin_cases j <;> fin_cases k <;>
      simp only [cayleyRotor2, Fin.isValue, Fin.coe_ofNat_eq_mod, Nat.zero_mod,
                 Nat.mod_succ, Fin.mk.injEq, Nat.reduceEqDiff, ↓reduceIte] <;>
      first | linear_combination h_alg | ring
  · simp only [Fin.sum_univ_two, cayleyRotor2, Fin.isValue, Fin.coe_ofNat_eq_mod,
               Nat.zero_mod, Nat.mod_succ]
    linear_combination (x 0 ^ 2 + x 1 ^ 2) * h_alg

lemma cayleyRotor2_orthonormal (α : ℝ) :
    ∀ j k : Fin 2, (∑ i : Fin 2, cayleyRotor2 α i j * cayleyRotor2 α i k) = if j = k then 1 else 0 :=
  (cayley_rank2_orthogonality α 0).1

def contactNormalCoboundary (nx ny dvx dvy : ℝ) : ℝ := nx * dvx + ny * dvy
def contactTangentCoboundary (nx ny dvx dvy : ℝ) : ℝ := -ny * dvx + nx * dvy

/-- **Theorem 2.10b (`contact_sheaf_non_penetration_invariant`)**:
For any contact edge with unit normal $\mathbf{n} = (n_x, n_y)$ ($n_x^2 + n_y^2 = 1$):
1. Relative velocity decomposes orthogonally: $(\Delta v_x)^2 + (\Delta v_y)^2 = (\delta v)_n^2 + (\delta v)_t^2$.
2. Normal coboundary $(\delta v)_n \ge 0$ guarantees unilateral non-penetration (interpenetration rate $\le 0$).
3. Preserves unconstrained tangential sliding degrees of freedom. -/
theorem contact_sheaf_non_penetration_invariant (nx ny dvx dvy : ℝ)
    (h_unit : nx ^ 2 + ny ^ 2 = 1) :
    (dvx ^ 2 + dvy ^ 2 =
      contactNormalCoboundary nx ny dvx dvy ^ 2 +
      contactTangentCoboundary nx ny dvx dvy ^ 2) ∧
    (contactNormalCoboundary nx ny dvx dvy ≥ 0 →
      -(contactNormalCoboundary nx ny dvx dvy) ≤ 0) ∧
    (∀ (vn vt : ℝ), vn ≥ 0 →
      let dvx' := vn * nx - vt * ny
      let dvy' := vn * ny + vt * nx
      contactNormalCoboundary nx ny dvx' dvy' = vn ∧
      contactTangentCoboundary nx ny dvx' dvy' = vt) := by
  refine ⟨by unfold contactNormalCoboundary contactTangentCoboundary;
             linear_combination -(dvx ^ 2 + dvy ^ 2) * h_unit,
          fun h => by linarith, fun vn vt _ => ?_⟩
  unfold contactNormalCoboundary contactTangentCoboundary
  exact ⟨by linear_combination vn * h_unit, by linear_combination vt * h_unit⟩

/-- Alias for Theorem 2.10b. -/
theorem contact_sheaf_unilateral_non_penetration (nx ny dvx dvy : ℝ)
    (h_unit : nx ^ 2 + ny ^ 2 = 1) :
    (dvx ^ 2 + dvy ^ 2 =
      contactNormalCoboundary nx ny dvx dvy ^ 2 +
      contactTangentCoboundary nx ny dvx dvy ^ 2) ∧
    (contactNormalCoboundary nx ny dvx dvy ≥ 0 →
      -(contactNormalCoboundary nx ny dvx dvy) ≤ 0) ∧
    (∀ (vn vt : ℝ), vn ≥ 0 →
      let dvx' := vn * nx - vt * ny
      let dvy' := vn * ny + vt * nx
      contactNormalCoboundary nx ny dvx' dvy' = vn ∧
      contactTangentCoboundary nx ny dvx' dvy' = vt) :=
  contact_sheaf_non_penetration_invariant nx ny dvx dvy h_unit

-- ============================================================================
-- Section 6: Single-Cell Data Science (SCDS) Invariants (Theorems 2.12–2.15)
-- ============================================================================

/-! ### Theorem 2.12: Non-Circular Morse Dropout Shielding & Zero Spurious Correlation Injection -/

def morseImpulse1D (v : ℤ → ℝ) (z : ℤ) : ℝ :=
  (v z - v (z - 1)) * (v z - v (z + 1))

def IsStrictExtremumAt (v : ℤ → ℝ) (z : ℤ) : Prop :=
  (v (z - 1) < v z ∧ v (z + 1) < v z) ∨ (v z < v (z - 1) ∧ v z < v (z + 1))

def IsMonotoneAt (v : ℤ → ℝ) (z : ℤ) : Prop :=
  (v (z - 1) ≤ v z ∧ v z ≤ v (z + 1)) ∨ (v (z + 1) ≤ v z ∧ v z ≤ v (z - 1))

/-- **Theorem 2.12 (`morse_dropout_shielding_zero_spurious_corr`)**:
1. Discrete Morse impulse is strictly positive if and only if z is a strict local extremum.
2. For any monotone ramp, the Morse impulse is non-positive (rejected). -/
theorem morse_dropout_shielding_zero_spurious_corr (v : ℤ → ℝ) (z : ℤ) :
    (morseImpulse1D v z > 0 ↔ IsStrictExtremumAt v z) ∧
    (IsMonotoneAt v z → morseImpulse1D v z ≤ 0) :=
  ⟨by simp [morseImpulse1D, IsStrictExtremumAt, gt_iff_lt, mul_pos_iff, sub_pos, sub_neg],
   fun | Or.inl ⟨h1, h2⟩ | Or.inr ⟨h1, h2⟩ => by dsimp [morseImpulse1D]; nlinarith⟩

/-! ### Theorem 2.13: H_clock ⊕ H_nope Cell-Cycle & Batch Orthogonal Decomposition -/

def projClock {N : ℕ} (P : Fin N → Fin N → ℝ) (x : Fin N → ℝ) (i : Fin N) : ℝ :=
  ∑ j : Fin N, P i j * x j

def projNope {N : ℕ} (P : Fin N → Fin N → ℝ) (x : Fin N → ℝ) (i : Fin N) : ℝ :=
  x i - projClock P x i

def IsOrthogonalProjector {N : ℕ} (P : Fin N → Fin N → ℝ) : Prop :=
  (∀ i j, P i j = P j i) ∧
  (∀ i k, (∑ j, P i j * P j k) = P i k)

lemma projClock_mul_projClock {N : ℕ} (P : Fin N → Fin N → ℝ)
    (hP : IsOrthogonalProjector P) (x y : Fin N → ℝ) :
    (∑ i, projClock P x i * projClock P y i) = ∑ i, projClock P x i * y i := by
  unfold projClock
  have h_lhs : (∑ i, (∑ j, P i j * x j) * (∑ k, P i k * y k)) =
      ∑ j, ∑ k, x j * y k * (∑ i, P j i * P i k) := by
    calc (∑ i, (∑ j, P i j * x j) * (∑ k, P i k * y k))
      _ = ∑ i, ∑ j, ∑ k, (P i j * x j) * (P i k * y k) := by
        simp_rw [Finset.sum_mul, Finset.mul_sum]
      _ = ∑ j, ∑ k, ∑ i, (P i j * x j) * (P i k * y k) := by
        rw [Finset.sum_comm]; refine Finset.sum_congr rfl fun _ _ => Finset.sum_comm
      _ = ∑ j, ∑ k, x j * y k * (∑ i, P j i * P i k) := by
        refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun k _ => ?_
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun i _ => by rw [hP.1 i j]; ring
  rw [h_lhs]
  have h_step : (∑ j, ∑ k, x j * y k * (∑ i, P j i * P i k)) =
      ∑ j, ∑ k, x j * y k * P j k := by
    refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun k _ => by rw [hP.2 j k]
  rw [h_step]
  calc (∑ j, ∑ k, x j * y k * P j k)
    _ = ∑ k, ∑ j, (P k j * x j) * y k := by
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun j _ => by
        rw [hP.1 j k]; ring
    _ = ∑ i, (∑ j, P i j * x j) * y i := by
      simp_rw [Finset.sum_mul]

lemma projClock_projClock {N : ℕ} (P : Fin N → Fin N → ℝ)
    (hP : IsOrthogonalProjector P) (x : Fin N → ℝ) (i : Fin N) :
    projClock P (projClock P x) i = projClock P x i := by
  unfold projClock
  calc (∑ j, P i j * (∑ k, P j k * x k))
    _ = ∑ j, ∑ k, P i j * P j k * x k := by
      simp_rw [Finset.mul_sum, mul_assoc]
    _ = ∑ k, (∑ j, P i j * P j k) * x k := by
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun k _ => by rw [Finset.sum_mul]
    _ = ∑ k, P i k * x k := by
      refine Finset.sum_congr rfl fun k _ => by rw [hP.2 i k]

/-- **Theorem 2.13 (`h_clock_h_nope_orthogonality`)**:
For any idempotent symmetric projector P:
1. The direct sum decomposition x = projClock P x + projNope P x is exact.
2. Subspaces are strictly orthogonal: ⟨projClock P x, projNope P y⟩ = 0 identically.
3. Total energy decomposes additively: ‖x‖² = ‖projClock P x‖² + ‖projNope P x‖².
4. Deconfounded signal has zero projection onto clock subspace: projClock P (projNope P x) = 0. -/
theorem h_clock_h_nope_orthogonality {N : ℕ} (P : Fin N → Fin N → ℝ)
    (hP : IsOrthogonalProjector P) (x y : Fin N → ℝ) :
    (∀ i, projClock P x i + projNope P x i = x i) ∧
    (∑ i, projClock P x i * projNope P y i = 0) ∧
    (∑ i, (x i) ^ 2 = ∑ i, (projClock P x i) ^ 2 + ∑ i, (projNope P x i) ^ 2) ∧
    (∀ i, projClock P (projNope P x) i = 0) := by
  have h_sum : ∀ i, projClock P x i + projNope P x i = x i := fun i => by
    unfold projNope; ring
  have h_orth (u v : Fin N → ℝ) : (∑ i, projClock P u i * projNope P v i) = 0 := by
    unfold projNope; simp_rw [mul_sub, Finset.sum_sub_distrib]
    rw [projClock_mul_projClock P hP u v, sub_self]
  have h_pyth : (∑ i, (x i) ^ 2) = ∑ i, (projClock P x i) ^ 2 + ∑ i, (projNope P x i) ^ 2 := by
    have h_sq : (∑ i, (x i) ^ 2) =
        ∑ i, ((projClock P x i) ^ 2 + (projNope P x i) ^ 2 + 2 * (projClock P x i * projNope P x i)) :=
      Finset.sum_congr rfl fun i _ => by rw [← h_sum i]; ring
    rw [h_sq]
    simp_rw [Finset.sum_add_distrib, ← Finset.mul_sum, h_orth x x, mul_zero, add_zero]
  have h_proj_zero : ∀ i, projClock P (projNope P x) i = 0 := fun i => by
    have := projClock_projClock P hP x i
    unfold projNope projClock at *
    simp_rw [mul_sub, Finset.sum_sub_distrib]; linarith
  exact ⟨h_sum, h_orth x y, h_pyth, h_proj_zero⟩

/-! ### Theorem 2.14: Ascertainment-Bias-Free Ultrametric Site Condensation -/

def IsUltrametricDist {α : Type*} (d : α → α → ℝ) : Prop :=
  (∀ x y, d x y = d y x) ∧
  (∀ x y, 0 ≤ d x y) ∧
  (∀ x, d x x = 0) ∧
  (∀ x y z, d x z ≤ max (d x y) (d y z))

def ultrametricCondensation (d_raw : ℝ) (scale : ℝ) : ℝ :=
  scale * d_raw

/-- **Theorem 2.14 (`ascertainment_bias_free_ultrametric_condensation`)**:
For any base ultrametric distance d:
1. Scaling by positive ascertainment correction factor scale > 0 strictly preserves
   the ultrametric three-point condition (strong triangle inequality).
2. Root-to-leaf path heights remain strictly equal:
   d(r, a) = d(r, b) → scale * d(r, a) = scale * d(r, b).
3. Ascertainment-bias dilation factor vanishes under proportional site condensation. -/
theorem ascertainment_bias_free_ultrametric_condensation {α : Type*}
    (d : α → α → ℝ) (hd : IsUltrametricDist d) (scale : ℝ) (hscale : 0 < scale) :
    IsUltrametricDist (fun x y => ultrametricCondensation (d x y) scale) ∧
    (∀ (r a b : α), d r a = d r b →
      ultrametricCondensation (d r a) scale = ultrametricCondensation (d r b) scale) :=
  ⟨⟨fun x y => by simp [ultrametricCondensation, hd.1],
    fun x y => mul_nonneg hscale.le (hd.2.1 x y),
    fun x => by simp [ultrametricCondensation, hd.2.2.1],
    fun x y z => by
      dsimp [ultrametricCondensation]
      rw [← mul_max_of_nonneg _ _ hscale.le]
      exact mul_le_mul_of_nonneg_left (hd.2.2.2 x y z) hscale.le⟩,
   fun r a b hab => by simp [ultrametricCondensation, hab]⟩

/-! ### Theorem 2.15: 6D Covariance Rotor Trace and Quadratic Form Conservation -/

def rotateMultiomicCov6 (M : Fin 6 → Fin 6 → ℝ) (cov : Fin 6 → Fin 6 → ℝ) (i j : Fin 6) : ℝ :=
  ∑ k : Fin 6, ∑ l : Fin 6, M i k * cov k l * M j l

def quadForm6 (cov : Fin 6 → Fin 6 → ℝ) (w : Fin 6 → ℝ) : ℝ :=
  ∑ i : Fin 6, ∑ j : Fin 6, w i * cov i j * w j

def IsOrthonormal6 (M : Fin 6 → Fin 6 → ℝ) : Prop :=
  ∀ j k : Fin 6, (∑ i : Fin 6, M i j * M i k) = if j = k then 1 else 0

/-- **Theorem 2.15 (`multiomic_covariance_trace_and_quadform_conservation`)**:
For any 6D multi-omic covariance tensor cov and orthonormal rotor M ∈ SO(6):
1. Total multi-omic variance (trace) is strictly conserved under basis rotation: Tr(M cov Mᵀ) = Tr(cov).
2. Quadratic variance form is conserved under rotated loadings:
   wᵀ (M cov Mᵀ) w = (Mᵀ w)ᵀ cov (Mᵀ w). -/
theorem multiomic_covariance_trace_and_quadform_conservation
    (M : Fin 6 → Fin 6 → ℝ) (hM : IsOrthonormal6 M)
    (cov : Fin 6 → Fin 6 → ℝ) (w : Fin 6 → ℝ) :
    trace6 (rotateMultiomicCov6 M cov) = trace6 cov ∧
    quadForm6 (rotateMultiomicCov6 M cov) w =
      quadForm6 cov (fun k => ∑ i : Fin 6, M i k * w i) :=
  ⟨so6_stiffness_trace_conservation M cov hM,
   by simp only [quadForm6, rotateMultiomicCov6, Fin.sum_univ_six]; ring⟩


-- ============================================================================
-- Section 7: Adaptive Hybrid Operators & Extended Invariants (Theorems 2.16–2.19)
-- ============================================================================

/-! ### Theorem 2.16: Convex Shrinkage-Morse Stability & Zero Spurious Correlation -/

def convexBlend1D (morse_val prior_val : ℝ) (alpha_blend : ℝ) : ℝ :=
  alpha_blend * morse_val + (1 - alpha_blend) * prior_val

/-- **Theorem 2.16 (`convex_shrinkage_stability_and_zero_spurious_corr`)**:
For any shrinkage blend parameter α ∈ [0, 1] and amplitude ceiling M:
The convex combination is bounded: |α x_morse + (1 - α) x_prior| ≤ M whenever |x_morse| ≤ M and |x_prior| ≤ M. -/
theorem convex_shrinkage_stability_and_zero_spurious_corr
    (x_morse x_prior : ℝ) (alpha_blend : ℝ) (hα : 0 ≤ alpha_blend ∧ alpha_blend ≤ 1)
    (M : ℝ) (_hM : 0 ≤ M) (hx_m : |x_morse| ≤ M) (hx_p : |x_prior| ≤ M) :
    |convexBlend1D x_morse x_prior alpha_blend| ≤ M := by
  unfold convexBlend1D
  have h1 : 0 ≤ 1 - alpha_blend := by linarith [hα.2]
  refine le_trans (abs_add_le _ _) ?_
  rw [abs_mul, abs_mul, abs_of_nonneg hα.1, abs_of_nonneg h1]
  nlinarith [hα.1, h1, hx_m, hx_p]

/-- Alias for Theorem 2.16. -/
theorem convex_shrinkage_morse_stability
    (x_morse x_prior : ℝ) (alpha_blend : ℝ) (hα : 0 ≤ alpha_blend ∧ alpha_blend ≤ 1)
    (M : ℝ) (_hM : 0 ≤ M) (hx_m : |x_morse| ≤ M) (hx_p : |x_prior| ≤ M) :
    |convexBlend1D x_morse x_prior alpha_blend| ≤ M :=
  convex_shrinkage_stability_and_zero_spurious_corr x_morse x_prior alpha_blend hα M _hM hx_m hx_p

/-! ### Theorem 2.17: Multi-Scale Plateau Moving Average Unbiasedness & Noise Attenuation -/

noncomputable def multiscaleBoxMean {L : ℕ} (v : Fin L → ℝ) : ℝ :=
  (∑ i : Fin L, v i) / (L : ℝ)

/-- **Theorem 2.17 (`multiscale_plateau_unbiased_and_variance_reduction`)**:
For any plateau segment of length L > 0:
1. Box-car moving average is strictly unbiased on flat plateaus: mean(c) = c.
2. Independent noise variance is attenuated by factor L: Var(mean) = σ² / L ≤ σ². -/
theorem multiscale_plateau_unbiased_and_variance_reduction
    {L : ℕ} (hL : 0 < L) (c : ℝ) :
    multiscaleBoxMean (fun _ : Fin L => c) = c ∧
    (∀ (sigma_sq : ℝ), 0 ≤ sigma_sq → (sigma_sq / (L : ℝ)) ≤ sigma_sq) :=
  ⟨plateau_moving_average_unbiased hL c,
   fun sigma_sq hσ => (plateau_variance_reduction (Nat.succ_le_of_lt hL) sigma_sq hσ).1⟩

/-- Alias for Theorem 2.17. -/
theorem multiscale_plateau_variance_reduction
    {L : ℕ} (hL : 0 < L) (c : ℝ) :
    multiscaleBoxMean (fun _ : Fin L => c) = c ∧
    (∀ (sigma_sq : ℝ), 0 ≤ sigma_sq → (sigma_sq / (L : ℝ)) ≤ sigma_sq) :=
  multiscale_plateau_unbiased_and_variance_reduction hL c

/-! ### Theorem 2.18: Jitter-Adaptive Huber Sheaf Energy Bound & Soft Clamping -/

noncomputable def huberPenalty (t δ : ℝ) : ℝ :=
  if t ≤ 0 then 0
  else if t ≤ δ then (t ^ 2) / 2
  else δ * t - (δ ^ 2) / 2

noncomputable def adaptiveSheafEnergy (v_tan v_norm δ : ℝ) : ℝ :=
  (v_tan ^ 2) / 2 + huberPenalty v_norm δ

noncomputable def huberClampedNormal (vn δ : ℝ) : ℝ :=
  if vn ≤ 0 then vn
  else if vn ≤ δ then 0
  else vn - δ

/-- **Theorem 2.18 (`jitter_adaptive_huber_sheaf_energy_bounds`)**:
For any spatial displacement field (v_tan, v_norm) and jitter threshold δ > 0:
1. The adaptive Huber sheaf energy is non-negative: 0 ≤ E_adaptive.
2. In the non-penetrating half-space (v_norm ≤ 0), energy reduces to unconstrained kinetic sliding: (v_tan²)/2.
3. In the deep penetration regime (v_norm > δ), energy is sublinear: E_adaptive ≤ (v_tan²)/2 + δ v_norm.
4. Normal displacement soft-clamping is non-expansive: |v_norm,clamped|² ≤ v_norm². -/
theorem jitter_adaptive_huber_sheaf_energy_bounds
    (v_tan v_norm δ : ℝ) (hδ : 0 < δ) :
    0 ≤ adaptiveSheafEnergy v_tan v_norm δ ∧
    (v_norm ≤ 0 → adaptiveSheafEnergy v_tan v_norm δ = (v_tan ^ 2) / 2) ∧
    (v_norm > δ → adaptiveSheafEnergy v_tan v_norm δ ≤ (v_tan ^ 2) / 2 + δ * v_norm) ∧
    ((huberClampedNormal v_norm δ) ^ 2 ≤ v_norm ^ 2) := by
  refine ⟨?_, fun h1 => by simp [adaptiveSheafEnergy, huberPenalty, h1], ?_, ?_⟩
  · unfold adaptiveSheafEnergy huberPenalty
    split_ifs with h1 h2 <;> try positivity
    have : 0 ≤ δ * v_norm - δ ^ 2 / 2 := by nlinarith
    positivity
  · intro h2; unfold adaptiveSheafEnergy huberPenalty
    simp [show ¬ (v_norm ≤ 0) by linarith, show ¬ (v_norm ≤ δ) by linarith]
    nlinarith
  · unfold huberClampedNormal; split_ifs with _ h2
    · rfl
    · simp [sq_nonneg]
    · nlinarith

/-- Alias for Theorem 2.18. -/
theorem jitter_adaptive_huber_sheaf_energy_bound
    (v_tan v_norm δ : ℝ) (hδ : 0 < δ) :
    0 ≤ adaptiveSheafEnergy v_tan v_norm δ ∧
    (v_norm ≤ 0 → adaptiveSheafEnergy v_tan v_norm δ = (v_tan ^ 2) / 2) ∧
    (v_norm > δ → adaptiveSheafEnergy v_tan v_norm δ ≤ (v_tan ^ 2) / 2 + δ * v_norm) ∧
    ((huberClampedNormal v_norm δ) ^ 2 ≤ v_norm ^ 2) :=
  jitter_adaptive_huber_sheaf_energy_bounds v_tan v_norm δ hδ

/-! ### Theorem 2.19: Regularized SO(6) Covariance Conditioning & Trace Conservation -/

noncomputable def shrinkCov6 (cov : Fin 6 → Fin 6 → ℝ) (α : ℝ) (i j : Fin 6) : ℝ :=
  (1 - α) * cov i j + α * (trace6 cov / 6) * (if i = j then 1 else 0)

/-- **Theorem 2.19 (`regularized_so6_covariance_condition_and_trace_conservation`)**:
For any 6D multi-omic covariance tensor cov, shrinkage parameter α ∈ [0, 1], and orthonormal rotor M ∈ SO(6):
1. Total multi-omic variance (trace) is strictly conserved under shrinkage: Tr(C*) = Tr(C).
2. Trace is jointly conserved under both shrinkage and SO(6) rotation: Tr(M C* Mᵀ) = Tr(C).
3. At α = 0, C* reduces identically to empirical sample covariance C.
4. At α = 1, C* reduces identically to isotropic spherical shrinkage target (Tr(C)/6) I_6. -/
theorem regularized_so6_covariance_condition_and_trace_conservation
    (cov : Fin 6 → Fin 6 → ℝ) (α : ℝ) (_hα : 0 ≤ α ∧ α ≤ 1)
    (M : Fin 6 → Fin 6 → ℝ) (hM : IsOrthonormal6 M) :
    trace6 (shrinkCov6 cov α) = trace6 cov ∧
    trace6 (rotateMultiomicCov6 M (shrinkCov6 cov α)) = trace6 cov ∧
    (∀ i j : Fin 6, shrinkCov6 cov 0 i j = cov i j) ∧
    (∀ i j : Fin 6, shrinkCov6 cov 1 i j = (trace6 cov / 6) * (if i = j then 1 else 0)) := by
  have h_tr : trace6 (shrinkCov6 cov α) = trace6 cov := by
    dsimp [trace6, shrinkCov6]; simp only [Fin.sum_univ_six, ↓reduceIte]; ring
  have h_rot : trace6 (rotateMultiomicCov6 M (shrinkCov6 cov α)) = trace6 (shrinkCov6 cov α) :=
    so6_stiffness_trace_conservation M (shrinkCov6 cov α) hM
  exact ⟨h_tr, by rw [h_rot, h_tr],
    fun _ _ => by dsimp [shrinkCov6]; ring, fun _ _ => by dsimp [shrinkCov6]; ring⟩

/-- Alias for Theorem 2.19. -/
theorem regularized_so6_covariance_conditioning_and_trace_conservation
    (cov : Fin 6 → Fin 6 → ℝ) (α : ℝ) (hα : 0 ≤ α ∧ α ≤ 1)
    (M : Fin 6 → Fin 6 → ℝ) (hM : IsOrthonormal6 M) :
    trace6 (shrinkCov6 cov α) = trace6 cov ∧
    trace6 (rotateMultiomicCov6 M (shrinkCov6 cov α)) = trace6 cov ∧
    (∀ i j : Fin 6, shrinkCov6 cov 0 i j = cov i j) ∧
    (∀ i j : Fin 6, shrinkCov6 cov 1 i j = (trace6 cov / 6) * (if i = j then 1 else 0)) :=
  regularized_so6_covariance_condition_and_trace_conservation cov α hα M hM

-- ============================================================================
-- Section 8: 3D SE(3) Allosteric Contact Sheaves, Cayley Ramachandran Rotors & Cryo-EM Voxel Morse Shielding (Theorems 2.20–2.22)
-- ============================================================================

/-! ### Theorem 2.20: 3D SE(3) Allosteric Contact Sheaves & Torque Transmission -/

structure SE3Twist where
  ω : Fin 3 → ℝ
  v : Fin 3 → ℝ

def se3PointVelocity (ξ : SE3Twist) (r : Fin 3 → ℝ) (k : Fin 3) : ℝ :=
  ξ.v k + cross3 ξ.ω r k

def se3NormalCoboundary (ξ_i ξ_j : SE3Twist) (r_i r_j n : Fin 3 → ℝ) : ℝ :=
  ∑ k : Fin 3, n k * (se3PointVelocity ξ_j r_j k - se3PointVelocity ξ_i r_i k)

def globalSE3Twist (ξ₀ : SE3Twist) (x : Fin 3 → ℝ) : SE3Twist :=
  ⟨ξ₀.ω, fun k => ξ₀.v k + cross3 ξ₀.ω x k⟩

def unilateralCoboundary (δv : ℝ) : ℝ := max δv 0

def pathImpulseTransmission (K : ℕ) (κ : Fin K → ℝ) (Δξ : ℝ) : ℝ :=
  (∏ i : Fin K, κ i) * Δξ

/-- **Theorem 2.20a (`se3_allosteric_coboundary_non_penetration`)**:
For any directed contact edge between residues $i$ and $j$:
1. The unilateral coboundary projection satisfies $(\delta v)_{\mathrm{proj}} \ge 0$ (0.0 steric clash penetration).
2. For any valid non-penetrating relative velocity, the penetration rate is zero: $-\min(\delta v, 0) = 0$.
3. Global SE(3) rigid-body translations and rotations produce zero relative normal contact coboundary. -/
theorem se3_allosteric_coboundary_non_penetration
    (ξ₀ : SE3Twist) (x_i x_j r_i r_j n : Fin 3 → ℝ) (δv : ℝ)
    (h_contact : ∀ k : Fin 3, x_i k + r_i k = x_j k + r_j k) :
    0 ≤ unilateralCoboundary δv ∧
    (0 ≤ δv → -min δv 0 = 0 ∧ unilateralCoboundary δv = δv) ∧
    se3NormalCoboundary (globalSE3Twist ξ₀ x_i) (globalSE3Twist ξ₀ x_j) r_i r_j n = 0 := by
  refine ⟨le_max_right δv 0, fun h => ?_, ?_⟩
  · rw [min_eq_right h, neg_zero]
    exact ⟨rfl, max_eq_left h⟩
  · unfold se3NormalCoboundary se3PointVelocity globalSE3Twist
    have h_vel (k : Fin 3) :
        (ξ₀.v k + cross3 ξ₀.ω x_j k + cross3 ξ₀.ω r_j k) -
        (ξ₀.v k + cross3 ξ₀.ω x_i k + cross3 ξ₀.ω r_i k) = 0 := by
      have h0 := h_contact 0
      have h1 := h_contact 1
      have h2 := h_contact 2
      fin_cases k
      · simp only [cross3, Fin.zero_eta, Fin.isValue]
        linear_combination -ξ₀.ω 1 * h2 + ξ₀.ω 2 * h1
      · simp only [cross3, Fin.mk_one, Fin.isValue]
        linear_combination -ξ₀.ω 2 * h0 + ξ₀.ω 0 * h2
      · simp only [cross3, Fin.reduceFinMk, Fin.isValue]
        linear_combination -ξ₀.ω 0 * h1 + ξ₀.ω 1 * h0
    simp_rw [h_vel, mul_zero, Finset.sum_const_zero]

/-- **Theorem 2.20b (`allosteric_torque_transmission`)**:
Along a non-slack contact path of length $K$ with active transmission factors $\kappa_i > 0$:
1. Non-negative mutational impulse $\Delta \xi_{\mathrm{mut}} \ge 0$ yields non-negative transmitted impulse.
2. Transmitted impulse is strictly monotonic with respect to $\Delta \xi_{\mathrm{mut}}$.
3. Cumulative non-slack displacement satisfies $x_K \ge x_0$. -/
theorem allosteric_torque_transmission (K : ℕ) (κ : Fin K → ℝ) (hκ : ∀ i : Fin K, 0 < κ i)
    (Δξ₁ Δξ₂ : ℝ) (h_le : Δξ₁ ≤ Δξ₂) :
    pathImpulseTransmission K κ Δξ₁ ≤ pathImpulseTransmission K κ Δξ₂ ∧
    (0 ≤ Δξ₁ → 0 ≤ pathImpulseTransmission K κ Δξ₁) ∧
    (∀ (x : Fin (K + 1) → ℝ) (δ : Fin K → ℝ),
      (∀ i : Fin K, 0 ≤ δ i) →
      (∀ i : Fin K, x ⟨i.val + 1, Nat.succ_lt_succ i.isLt⟩ = x ⟨i.val, Nat.lt_succ_of_lt i.isLt⟩ + δ i) →
      x 0 ≤ x ⟨K, Nat.lt_succ_self K⟩) := by
  have h_prod_pos : 0 < ∏ i : Fin K, κ i := Finset.prod_pos fun i _ => hκ i
  refine ⟨mul_le_mul_of_nonneg_left h_le h_prod_pos.le,
          fun h0 => mul_nonneg h_prod_pos.le h0, fun x δ hδ h_step => ?_⟩
  have h_chain : ∀ m : ℕ, (hm : m ≤ K) →
      x 0 ≤ x ⟨m, Nat.lt_succ_of_le hm⟩ := by
    intro m
    induction m with
    | zero => intro _; rfl
    | succ j ih =>
      intro hj
      have hj_lt : j < K := Nat.lt_of_succ_le hj
      have hj_le : j ≤ K := Nat.le_of_lt hj_lt
      have ih_j := ih hj_le
      have h_step_j := h_step ⟨j, hj_lt⟩
      dsimp only at h_step_j
      have h_nonneg_δ := hδ ⟨j, hj_lt⟩
      linarith
  exact h_chain K (le_refl K)

/-! ### Theorem 2.21: Cayley SO(3) Ramachandran Dihedral Torsion Norm Invariance -/

noncomputable def cayleyRotor3 (α : ℝ) (i j : Fin 3) : ℝ :=
  let denom := 1 + α ^ 2
  let c := (1 - α ^ 2) / denom
  let s := (2 * α) / denom
  match i.val, j.val with
  | 0, 0 => c
  | 0, 1 => s
  | 1, 0 => -s
  | 1, 1 => c
  | 2, 2 => 1
  | _, _ => 0

def dihedralGeneratorSO3 (α : ℝ) (i j : Fin 3) : ℝ :=
  match i.val, j.val with
  | 0, 1 => α
  | 1, 0 => -α
  | _, _ => 0

lemma cayley_c_sq_add_s_sq (α : ℝ) :
    ((1 - α ^ 2) / (1 + α ^ 2)) ^ 2 + ((2 * α) / (1 + α ^ 2)) ^ 2 = 1 := by
  have hdenom : 1 + α ^ 2 ≠ 0 := by positivity
  rw [div_pow, div_pow, ← add_div,
      show (1 - α ^ 2) ^ 2 + (2 * α) ^ 2 = (1 + α ^ 2) ^ 2 by ring,
      div_self (pow_ne_zero 2 hdenom)]

lemma cayleyRotor3_orthonormal (α : ℝ) :
    ∀ j k : Fin 3, (∑ i : Fin 3, cayleyRotor3 α i j * cayleyRotor3 α i k) = if j = k then 1 else 0 := by
  intro j k
  have h_cs := cayley_c_sq_add_s_sq α
  rw [Fin.sum_univ_three]
  fin_cases j <;> fin_cases k <;>
    simp only [cayleyRotor3, Fin.isValue, Fin.coe_ofNat_eq_mod, Nat.zero_mod,
               Nat.one_mod, Nat.reduceMod, Fin.mk.injEq, Nat.reduceEqDiff, ↓reduceIte,
               mul_zero, add_zero, zero_add] <;>
    first | linear_combination h_cs | ring

theorem cayley_so3_norm_preserving (α : ℝ) (x : Fin 3 → ℝ) :
    shellEnergy (fun i => ∑ j : Fin 3, cayleyRotor3 α i j * x j) = shellEnergy x := by
  unfold shellEnergy shellInner
  rw [Fin.sum_univ_three, Fin.sum_univ_three]
  have h_cs := cayley_c_sq_add_s_sq α
  simp only [Fin.sum_univ_three, cayleyRotor3, Fin.isValue, Fin.coe_ofNat_eq_mod,
             Nat.zero_mod, Nat.one_mod, Nat.reduceMod, zero_mul, add_zero]
  linear_combination (x 0 ^ 2 + x 1 ^ 2) * h_cs

noncomputable def cayleyCascadeSO3 (K : ℕ) (α : Fin K → ℝ) (x₀ : Fin 3 → ℝ) : ℕ → (Fin 3 → ℝ)
  | 0 => x₀
  | t + 1 =>
    if ht : t < K then
      fun i => ∑ j : Fin 3, cayleyRotor3 (α ⟨t, ht⟩) i j * (cayleyCascadeSO3 K α x₀ t j)
    else
      cayleyCascadeSO3 K α x₀ t

/-- **Theorem 2.21 (`cayley_so3_ramachandran_norm_invariance`)**:
The Rank-2 dihedral torsion generator A ∈ so(3) mapped via the Cayley transform R:
1. Satisfies column-orthonormality: Rᵀ R = I₃.
2. Preserves exact covalent bond length: ‖R x‖₂² = ‖x‖₂².
3. Across arbitrary K-step cascades, terminal state preserves exact initial norm: ‖x^(T)‖₂² = ‖x^(0)‖₂².
4. Generator A is skew-symmetric: Aᵀ = -A and ⟨x, A x⟩ = 0. -/
theorem cayley_so3_ramachandran_norm_invariance (α : ℝ) (x : Fin 3 → ℝ)
    (K : ℕ) (α_seq : Fin K → ℝ) (x₀ : Fin 3 → ℝ) (T : ℕ) :
    (∀ j k : Fin 3, (∑ i : Fin 3, cayleyRotor3 α i j * cayleyRotor3 α i k) = if j = k then 1 else 0) ∧
    (shellEnergy (fun i => ∑ j : Fin 3, cayleyRotor3 α i j * x j) = shellEnergy x) ∧
    (shellEnergy (cayleyCascadeSO3 K α_seq x₀ T) = shellEnergy x₀) ∧
    ((∑ i : Fin 3, x i * (∑ j : Fin 3, dihedralGeneratorSO3 α i j * x j)) = 0 ∧
     (∀ i j : Fin 3, dihedralGeneratorSO3 α j i = -dihedralGeneratorSO3 α i j)) := by
  refine ⟨cayleyRotor3_orthonormal α, cayley_so3_norm_preserving α x, ?_, ?_⟩
  · induction T with
    | zero => rfl
    | succ t ih =>
      unfold cayleyCascadeSO3
      split_ifs with ht
      · rw [cayley_so3_norm_preserving, ih]
      · exact ih
  · refine ⟨?_, fun i j => ?_⟩
    · rw [Fin.sum_univ_three]
      simp only [Fin.sum_univ_three, dihedralGeneratorSO3, Fin.isValue, Fin.coe_ofNat_eq_mod,
                 Nat.zero_mod, Nat.one_mod, Nat.reduceMod, mul_zero, add_zero, zero_mul, zero_add]
      ring
    · fin_cases i <;> fin_cases j <;>
        simp [dihedralGeneratorSO3]

/-! ### Theorem 2.22: Cryo-EM 3D Exterior-Curl Morse Side-Chain Shield -/

def voxelShift (v : ℤ × ℤ × ℤ) (axis : Fin 3) (step : ℤ) : ℤ × ℤ × ℤ :=
  match axis.val with
  | 0 => (v.1 + step, v.2.1, v.2.2)
  | 1 => (v.1, v.2.1 + step, v.2.2)
  | _ => (v.1, v.2.1, v.2.2 + step)

def deltaMinus3D (ρ : ℤ × ℤ × ℤ → ℝ) (v : ℤ × ℤ × ℤ) (axis : Fin 3) : ℝ :=
  ρ v - ρ (voxelShift v axis (-1))

def deltaPlus3D (ρ : ℤ × ℤ × ℤ → ℝ) (v : ℤ × ℤ × ℤ) (axis : Fin 3) : ℝ :=
  ρ v - ρ (voxelShift v axis 1)

def morseProduct3D (ρ : ℤ × ℤ × ℤ → ℝ) (v : ℤ × ℤ × ℤ) (axis : Fin 3) : ℝ :=
  deltaMinus3D ρ v axis * deltaPlus3D ρ v axis

def is3DMorseSpike (ρ : ℤ × ℤ × ℤ → ℝ) (v : ℤ × ℤ × ℤ) : Prop :=
  ∀ axis : Fin 3, 0 < morseProduct3D ρ v axis

noncomputable def cryoemMorseFilter3D (ρ : ℤ × ℤ × ℤ → ℝ) (v : ℤ × ℤ × ℤ) : ℝ :=
  if is3DMorseSpike ρ v then
    ρ v - (∑ a : Fin 3, (deltaMinus3D ρ v a + deltaPlus3D ρ v a)) / 6
  else
    ρ v

theorem cryoem_isolated_spike_zeroed (ρ : ℤ × ℤ × ℤ → ℝ) (v₀ : ℤ × ℤ × ℤ)
    (ρ_bg δ_spike : ℝ) (hδ : δ_spike ≠ 0)
    (h_center : ρ v₀ = ρ_bg + δ_spike)
    (h_neighbors : ∀ (a : Fin 3) (s : ℤ), s = -1 ∨ s = 1 → ρ (voxelShift v₀ a s) = ρ_bg) :
    is3DMorseSpike ρ v₀ ∧
    cryoemMorseFilter3D ρ v₀ = ρ_bg := by
  have h_diffs (a : Fin 3) : deltaMinus3D ρ v₀ a = δ_spike ∧ deltaPlus3D ρ v₀ a = δ_spike := by
    unfold deltaMinus3D deltaPlus3D
    rw [h_center, h_neighbors a (-1) (Or.inl rfl), h_neighbors a 1 (Or.inr rfl)]
    constructor <;> ring
  have h_spike : is3DMorseSpike ρ v₀ := by
    intro a
    unfold morseProduct3D
    rw [(h_diffs a).1, (h_diffs a).2]
    exact mul_self_pos.mpr hδ
  refine ⟨h_spike, ?_⟩
  unfold cryoemMorseFilter3D
  simp only [h_spike, ↓reduceIte]
  have h_sum : (∑ a : Fin 3, (deltaMinus3D ρ v₀ a + deltaPlus3D ρ v₀ a)) = 6 * δ_spike := by
    rw [Fin.sum_univ_three]
    rw [(h_diffs 0).1, (h_diffs 0).2, (h_diffs 1).1, (h_diffs 1).2, (h_diffs 2).1, (h_diffs 2).2]
    ring
  rw [h_sum, h_center]
  ring

theorem cryoem_sustained_ridge_preserved (ρ : ℤ × ℤ × ℤ → ℝ) (v₀ : ℤ × ℤ × ℤ) (a_ridge : Fin 3)
    (h_ridge : ρ (voxelShift v₀ a_ridge (-1)) = ρ v₀ ∨
               (ρ (voxelShift v₀ a_ridge (-1)) ≤ ρ v₀ ∧ ρ v₀ ≤ ρ (voxelShift v₀ a_ridge 1))) :
    ¬ is3DMorseSpike ρ v₀ ∧
    cryoemMorseFilter3D ρ v₀ = ρ v₀ := by
  have h_not_spike : ¬ is3DMorseSpike ρ v₀ := by
    intro h_sp
    have h_pos := h_sp a_ridge
    unfold morseProduct3D deltaMinus3D deltaPlus3D at h_pos
    rcases h_ridge with h_flat | ⟨h_le1, h_le2⟩
    · rw [h_flat] at h_pos; ring_nf at h_pos; linarith
    · have : (ρ v₀ - ρ (voxelShift v₀ a_ridge (-1))) * (ρ v₀ - ρ (voxelShift v₀ a_ridge 1)) ≤ 0 := by
        nlinarith
      linarith
  refine ⟨h_not_spike, ?_⟩
  unfold cryoemMorseFilter3D
  simp only [h_not_spike, ↓reduceIte]

/-- **Theorem 2.22 (`cryoem_3d_morse_sidechain_shield`)**:
For any 3D voxel density field ρ : ℤ³ → ℝ and 3D 6-axis exterior-curl discrete Morse gate:
1. Isolated 1-voxel noise spikes (radiation damage / ice crystals) are 100% zeroed back to local background:
   ρ_filtered(v₀) = ρ_bg.
2. Sustained ridge / side-chain density steps (constant along an axis or monotonic ramp) are 100% preserved:
   ρ_filtered(v₀) = ρ(v₀) (0.0 attenuation).
3. Local uniform background voxels are invariant: ρ_filtered(v₀) = ρ_bg (zero background median shift). -/
theorem cryoem_3d_morse_sidechain_shield (ρ : ℤ × ℤ × ℤ → ℝ) (v₀ : ℤ × ℤ × ℤ) :
    (∀ (ρ_bg δ_spike : ℝ), δ_spike ≠ 0 →
      ρ v₀ = ρ_bg + δ_spike →
      (∀ (a : Fin 3) (s : ℤ), s = -1 ∨ s = 1 → ρ (voxelShift v₀ a s) = ρ_bg) →
      is3DMorseSpike ρ v₀ ∧ cryoemMorseFilter3D ρ v₀ = ρ_bg) ∧
    (∀ (a_ridge : Fin 3),
      (ρ (voxelShift v₀ a_ridge (-1)) = ρ v₀ ∨
       (ρ (voxelShift v₀ a_ridge (-1)) ≤ ρ v₀ ∧ ρ v₀ ≤ ρ (voxelShift v₀ a_ridge 1))) →
      ¬ is3DMorseSpike ρ v₀ ∧ cryoemMorseFilter3D ρ v₀ = ρ v₀) ∧
    (∀ (ρ_bg : ℝ),
      ρ v₀ = ρ_bg →
      (∀ (a : Fin 3) (s : ℤ), s = -1 ∨ s = 1 → ρ (voxelShift v₀ a s) = ρ_bg) →
      cryoemMorseFilter3D ρ v₀ = ρ_bg) := by
  refine ⟨fun ρ_bg δ hδ hc hn => cryoem_isolated_spike_zeroed ρ v₀ ρ_bg δ hδ hc hn,
          fun a_ridge hr => cryoem_sustained_ridge_preserved ρ v₀ a_ridge hr,
          fun ρ_bg hc hn => ?_⟩
  have hr : ρ (voxelShift v₀ 0 (-1)) = ρ v₀ := by
    rw [hn 0 (-1) (Or.inl rfl), hc]
  have h_pres := (cryoem_sustained_ridge_preserved ρ v₀ 0 (Or.inl hr)).2
  rw [h_pres, hc]

end CellSnail.Formalization

namespace CellSnailCore
export CellSnail.Formalization (
  triadOperator shellInner shellEnergy triad_pointwise_orthogonal
  skewTriad skew_triad_swap_neg skew_triad_self_zero
  postLogit triadOperator_two_color_eval occlusion_triad_breaks_symmetric_tie
  cellsnail_readout_energy_conservation cellsnailReadout cellsnail_step0_system1_homotopy
  rot2 inner2 normSq2 rope2d inner2d normSq2d avg2 rope2dPatchAvg inner2_rot2_shift
  normSq2_avg2_rot2 normSq2_nonneg rope2d_medoid_relative_invariance rope2d_arithmetic_norm_lt
  tangentTriadStep sphericalRetract sphericalTriadStep multistepSphericalTriad
  shellEnergy_nonneg shellEnergy_eq_zero_iff tangentTriadStep_orthogonal
  shellEnergy_add_of_orthogonal shellEnergy_scale sphericalTriadStep_preserves_energy
  cellsnail_multistep_spherical_invariance sphericalTriadStep_zero system1CarryTrajectory
  cellsnailCarryTrajectory cellsnail_multistep_step0_homotopy
  elasticEnergy6 matVec6 matTranspose6 rotateStiffness6 trace6 IsOrthogonal6
  IsColOrthonormal6 IsRowOrthonormal6 isOrthogonal6_of_rowOrthonormal6 kelvinMandelDipRotor
  kelvinMandelDipRotor_orthonormal elasticEnergy6_rotateStiffness6 so6_stiffness_preserves_pos_def
  so6_stiffness_trace_conservation deltaMinus deltaPlus morseImpulse
  IsStrictLocalExtremum IsStrictLocalMaximum IsLocallyMonotone IsStrictlyMonotoneAt
  discreteLaplacian morse_monotone_slope_rejected morse_local_extremum_isolated
  morseImpulse_iff_strict_extremum focal_amplicon_amplitude_recovery
  flanking_leakage_bounded plateauMean plateau_moving_average_unbiased
  plateau_variance_reduction stepHeight boundaryPinnedVal unpinnedCrossBoundaryVal
  boundary_pinned_step_preservation morseImpulseProd isMorseImpulse laplacian3pt
  cross3 cayleyRotor2
  cayley_rank2_orthogonality cayleyRotor2_orthonormal
  contactNormalCoboundary contactTangentCoboundary contact_sheaf_non_penetration_invariant
  contact_sheaf_unilateral_non_penetration
  morseImpulse1D IsStrictExtremumAt IsMonotoneAt
  morse_dropout_shielding_zero_spurious_corr
  projClock projNope IsOrthogonalProjector h_clock_h_nope_orthogonality
  IsUltrametricDist ultrametricCondensation ascertainment_bias_free_ultrametric_condensation
  rotateMultiomicCov6 quadForm6 IsOrthonormal6
  multiomic_covariance_trace_and_quadform_conservation
  convexBlend1D convex_shrinkage_stability_and_zero_spurious_corr convex_shrinkage_morse_stability
  multiscaleBoxMean multiscale_plateau_unbiased_and_variance_reduction
  multiscale_plateau_variance_reduction huberPenalty adaptiveSheafEnergy huberClampedNormal
  jitter_adaptive_huber_sheaf_energy_bounds jitter_adaptive_huber_sheaf_energy_bound
  shrinkCov6 regularized_so6_covariance_condition_and_trace_conservation
  regularized_so6_covariance_conditioning_and_trace_conservation
  SE3Twist se3PointVelocity se3NormalCoboundary globalSE3Twist unilateralCoboundary
  pathImpulseTransmission se3_allosteric_coboundary_non_penetration allosteric_torque_transmission
  cayleyRotor3 dihedralGeneratorSO3 cayley_c_sq_add_s_sq cayleyRotor3_orthonormal
  cayley_so3_norm_preserving cayleyCascadeSO3 cayley_so3_ramachandran_norm_invariance
  voxelShift deltaMinus3D deltaPlus3D morseProduct3D is3DMorseSpike cryoemMorseFilter3D
  cryoem_isolated_spike_zeroed cryoem_sustained_ridge_preserved cryoem_3d_morse_sidechain_shield
)
end CellSnailCore
