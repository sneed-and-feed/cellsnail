"""scripts/adversarial_challenge_regimes.py
=========================================
Empirical Adversarial Challenge Suite for cellsnail v0.2.0:
Regime 1 (Count Thinning & Spurious Correlations) & Regime 2 (CNV Resolution & Low SNR).
Conducted by Challenger 1 (challenger_p2_1).
"""

import sys
import numpy as np
sys.path.insert(0, ".")

from cellsnail.data import load_sc_mixology, load_paired_cnv_cohort
from cellsnail.benchmark.mode_b_blind import (
    binomial_count_split,
    apply_snr_noise,
    _safe_corrcoef,
    _compute_bounded_edge_fidelity,
    alra_impute,
    hmmcopy_infercnv_smooth,
    benchmark_cnv_width_resolution_crossover,
)
from cellsnail.core.morse_filter import (
    morse_dropout_filter,
    morse_cnv_filter,
    beltrami_morse_gate,
)

def run_regime1_challenge():
    print("=" * 78)
    print("CHALLENGE REPORT — REGIME 1: EXTREME COUNT THINNING & SPURIOUS CORRELATIONS")
    print("=" * 78)
    
    seeds = [42, 1337, 2026, 7, 999]
    p_levels = [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
    
    failures_r2 = []
    failures_spurious = []
    
    for seed in seeds:
        mix = load_sc_mixology(n_cells=300, dropout_rate=0.0, random_state=seed)
        hum = np.where(~np.array(mix.var["is_ercc"]))[0]
        ercc = np.where(np.array(mix.var["is_ercc"]))[0]
        
        ref_mod = np.log1p(mix.X[:, hum[:20]])
        ref_cent = ref_mod - np.mean(ref_mod, axis=0, keepdims=True)
        
        def _r2(mat):
            m = np.log1p(np.maximum(0.0, mat[:, hum[:20]]))
            mc = m - np.mean(m, axis=0, keepdims=True)
            num = float(np.sum(mc * ref_cent))
            den = float(np.linalg.norm(mc) * np.linalg.norm(ref_cent) + 1e-12)
            return float(max(0.0, (num / den) ** 2))

        for p in p_levels:
            X_thin, _ = binomial_count_split(mix.X, p=p, random_state=seed)
            U_h, S_h, Vt_h = np.linalg.svd(X_thin[:, hum], full_matrices=False)
            k_p = min(5, len(S_h))
            prior_hum = np.maximum(0.0, (U_h[:, :k_p] * S_h[:k_p]) @ Vt_h[:k_p, :])
            
            X_cs_v020 = np.zeros_like(X_thin)
            for idx_e, j_e in enumerate(ercc):
                X_cs_v020[:, j_e] = morse_dropout_filter(X_thin[:, j_e], tau_mad=1.0, mode="canonical")["filtered"]
            for idx_h, j_h in enumerate(hum):
                X_cs_v020[:, j_h] = morse_dropout_filter(
                    X_thin[:, j_h],
                    prior_matrix=prior_hum[:, idx_h],
                    p_thin=p,
                    tau_mad=1.0,
                    mode="adaptive",
                )["filtered"]
            
            r2_val = _r2(X_cs_v020)
            corr_ercc = _safe_corrcoef(X_cs_v020[:, ercc])
            spur_cnt = int(np.sum(np.abs(corr_ercc) > 0.70) // 2)
            
            thresh_r2 = 0.20 if p <= 0.03 else 0.45
            if r2_val < thresh_r2:
                failures_r2.append((seed, p, r2_val))
            if spur_cnt > 0:
                max_r = float(np.max(np.abs(corr_ercc)))
                failures_spurious.append((seed, p, spur_cnt, max_r))
                
    print(f"Total Sweep Conditions Tested: {len(seeds) * len(p_levels)}")
    print(f"R^2 Violations (thresh=0.20 for p<=0.03, 0.45 for p>=0.05): {len(failures_r2)} / {len(seeds) * len(p_levels)}")
    for s, p, r2 in failures_r2[:8]:
        print(f"  - Seed {s}, p_thin={p:.2f}: R^2 = {r2:.4f}")
    print(f"Spurious ERCC Correlation Violations (|r| > 0.70): {len(failures_spurious)} incidents")
    for s, p, cnt, max_r in failures_spurious:
        print(f"  - Seed {s}, p_thin={p:.2f}: {cnt} spurious pair(s), max |r| = {max_r:.4f}")

    return len(failures_r2) == 0 and len(failures_spurious) == 0


def run_regime2_challenge():
    print("\n" + "=" * 78)
    print("CHALLENGE REPORT — REGIME 2: CNV RESOLUTION UNDER LOW SNR (5 dB)")
    print("=" * 78)

    seeds = [42, 100, 2026]
    widths = [1, 2, 3, 5, 10, 15, 20, 30]

    edge_fid_narrow_pass = []
    edge_fid_broad_pass = []
    rmse_broad_match_hmm = []
    rmse_broad_phys = []
    fracture_bugs = []

    for seed in seeds:
        res = benchmark_cnv_width_resolution_crossover(amplicon_widths_bins=widths, snr_db=5.0, random_state=seed)
        for r in res:
            w = r["amplicon_width_bins"]
            fid_hmm = r["bounded_edge_fidelity"]["hmmcopy_infercnv"]
            fid_v02 = r["bounded_edge_fidelity"]["cellsnail_v020_adaptive"]
            rmse_hmm = r["single_cell_plateau_rmse"]["hmmcopy_infercnv"]
            rmse_raw = r["single_cell_plateau_rmse"]["cellsnail_unsmoothed_interior"]
            rmse_v02 = r["single_cell_plateau_rmse"]["cellsnail_v020_adaptive"]

            if w in [1, 2, 3]:
                if fid_v02 > 0.96:
                    edge_fid_narrow_pass.append(True)
                else:
                    edge_fid_narrow_pass.append(False)
            elif w >= 10:
                if fid_v02 > 0.85:
                    edge_fid_broad_pass.append(True)
                else:
                    edge_fid_broad_pass.append(False)

                if abs(rmse_v02 - rmse_hmm) < 0.10 or rmse_v02 < rmse_hmm:
                    rmse_broad_match_hmm.append(True)
                else:
                    rmse_broad_match_hmm.append(False)

                if rmse_v02 < 0.70:
                    rmse_broad_phys.append(True)
                else:
                    rmse_broad_phys.append(False)

                if rmse_v02 > 1.5 * rmse_hmm and rmse_v02 > 1.5:
                    fracture_bugs.append((seed, w, rmse_v02, rmse_hmm, fid_v02))

    print(f"Narrow Amplicons (W in {{1, 2, 3}}): Edge Fidelity > 0.96: {sum(edge_fid_narrow_pass)} / {len(edge_fid_narrow_pass)} PASS")
    print(f"Broad Plateaus (W >= 10): Edge Fidelity > 0.85: {sum(edge_fid_broad_pass)} / {len(edge_fid_broad_pass)} PASS")
    print(f"Broad Plateaus (W >= 10): RMSE Matches/Beats HMMcopy: {sum(rmse_broad_match_hmm)} / {len(rmse_broad_match_hmm)} PASS")
    print(f"Broad Plateaus (W >= 10): RMSE < 0.70 (Physical Noise Limit at 5 dB): {sum(rmse_broad_phys)} / {len(rmse_broad_phys)} PASS")
    print(f"Plateau Fracture / Interior Edge Pinning Bugs Detected: {len(fracture_bugs)}")
    for s, w, v02, hmm, fid in fracture_bugs:
        print(f"  - CRITICAL BUG in Seed {s}, W={w}: cellsnail RMSE={v02:.4f} vs HMMcopy RMSE={hmm:.4f}")

    return (
        all(edge_fid_narrow_pass)
        and all(edge_fid_broad_pass)
        and all(rmse_broad_match_hmm)
        and all(rmse_broad_phys)
        and len(fracture_bugs) == 0
    )


if __name__ == "__main__":
    p1 = run_regime1_challenge()
    p2 = run_regime2_challenge()
    print("\n" + "=" * 78)
    print(f"FINAL VERDICT: {'APPROVE' if (p1 and p2) else 'REQUEST_CHANGES'}")
    print("=" * 78)
