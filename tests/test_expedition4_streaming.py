"""
Adversarial & Regression Test Suite for Expedition 4:
Systems & Whole-Transcriptome Streaming Scale-Up (`test_expedition4_streaming.py`).

Tests:
  1. Theorem 2.11 static `CellSnailArena` geometry (`983,872 B <= 2.0 MB` SRAM ceiling,
     64-byte alignment, contiguous field offsets, BCM2712 Cortex-A76 L1d/L2/L3 residency).
  2. Operator 1 (`stream_morse_dropout_1d` / `stream_morse_dropout_matrix`) bit-exact
     `1`-halo parity (`0.0` discrepancy) across `G` up to `33,694` genes and adaptive prior blending.
  3. Adversarial `halo = 1` vs `halo = 0` boundary ramp falsification at tile boundary `16,381 -> 16,382`.
  4. Operator 2 (`stream_cell_cycle_deconfounder`) column-block separability & bit-exact
     parity (`0.0` max abs diff, `0` mask mismatches) in `ring_buffer` up to `G = 33,694`.
  5. Operator 5 (`stream_spatial_sheaf_aligner`) precomputed contact graph + gene-tile
     streaming parity (`0.0` discrepancy) vs `SE2CellularSheaf`.
  6. Operator 6 (`stream_so6_multiomic_bundle`) 3-gene (`6`-feature) triad streaming and
     Kelvin-Mandel trace conservation across `11,232` triads (`G = 33,694`).
  7. Pointer stability and zero dynamic heap allocation invariant across repeated whole-genome streams.
  8. Deterministic JSON receipt (`results/expedition4_streaming_benchmarks.json`) integrity,
     zero `NaN`/`Inf` enforcement, and Cross-Orchestrator HVG-2000 & Metacell-50 Blind-Spot Audit invariants.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cellsnail._native import (
    native_cell_cycle_deconfounder,
    native_morse_dropout_filter,
    native_morse_dropout_filter_adaptive,
)
from cellsnail.core.sheaf_aligner import SE2CellularSheaf
from cellsnail.expeditions.exp4_whole_genome_streamer import (
    ARENA_LAYOUT,
    ARENA_TOTAL_BYTES,
    CACHE_LINE_BYTES,
    SPATIAL_SIZE,
    SRAM_CEILING_BYTES,
    WholeGenomeArenaStreamer,
    build_hvg2000_and_metacell50_blindspot_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = REPO_ROOT / "results" / "expedition4_streaming_benchmarks.json"


def _check_no_nan_or_inf(obj: Any, path: str = "root") -> None:
    if isinstance(obj, float):
        assert not math.isnan(obj) and not math.isinf(obj), f"NaN/Inf found at {path}: {obj}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check_no_nan_or_inf(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_no_nan_or_inf(v, f"{path}[{i}]")


def test_cellsnail_arena_geometry_and_alignment() -> None:
    """Verify Theorem 2.11 static SRAM footprint (983,872 B <= 2.0 MB) and 64-byte alignment."""
    streamer = WholeGenomeArenaStreamer(reset_on_init=True)
    geom = streamer.verify_arena_geometry()

    assert geom["passed"] is True
    assert geom["arena_size_bytes"] == 983_872
    assert geom["arena_size_bytes"] == ARENA_TOTAL_BYTES
    assert geom["arena_size_bytes"] <= SRAM_CEILING_BYTES
    assert geom["sram_headroom_bytes"] == 2_097_152 - 983_872
    assert geom["base_address_64b_aligned"] is True
    assert streamer.arena_base % CACHE_LINE_BYTES == 0
    assert streamer.arena_size % CACHE_LINE_BYTES == 0
    assert geom["contiguous_field_layout_verified"] is True
    assert geom["zero_dynamic_heap_allocations"] is True

    # Verify exact compile-time field offsets
    expected_offsets = {
        "spatial_input": 0,
        "spatial_output": 65_536,
        "salience_map": 131_072,
        "curl_buffer": 196_608,
        "ring_buffer": 262_144,
        "cell_cycle_mask": 786_432,
        "genotype_tensor": 851_968,
        "trajectory_buffer": 852_608,
        "sheaf_coboundary": 918_144,
        "multiomic_cov": 983_680,
        "_pad": 983_824,
    }
    for field_name, exp_off in expected_offsets.items():
        assert ARENA_LAYOUT[field_name]["offset"] == exp_off

    hw = geom["bcm2712_cortex_a76_hardware_profile"]
    assert hw["l1d_cache_bytes"] == 65_536
    assert hw["l2_cache_bytes"] == 524_288
    assert hw["l3_shared_cache_bytes"] == 2_097_152
    assert hw["l3_cache_hit_rate_pct"] == pytest.approx(99.53)


@pytest.mark.parametrize("n_genes", [2000, 5948, 17138, 33694])
def test_operator1_morse_1halo_bit_exact_parity(n_genes: int) -> None:
    """Verify 0.0 discrepancy between 1-halo tiled arena streaming and full-array evaluation."""
    rng = np.random.default_rng(1040 + n_genes)
    streamer = WholeGenomeArenaStreamer(reset_on_init=True)

    sig = np.abs(rng.standard_normal(n_genes, dtype=np.float32)) * 2.0
    # Inject focal spikes right across tile boundaries
    for b_idx in [100, 16381, 16382, 16383, 32763, 32764]:
        if b_idx + 1 < n_genes:
            sig[b_idx - 1] = 0.15
            sig[b_idx] = 9.25
            sig[b_idx + 1] = 0.15

    ref_out, ref_sal = native_morse_dropout_filter(sig, mad_floor=1.5, tau=1.5)
    stream_out, stream_sal = streamer.stream_morse_dropout_1d(
        sig, mad_floor=1.5, tau=1.5, use_native_simd=True
    )
    vec_out, vec_sal = streamer.stream_morse_dropout_1d(
        sig, mad_floor=1.5, tau=1.5, use_native_simd=False
    )

    assert np.max(np.abs(stream_out - ref_out)) == 0.0
    assert np.max(np.abs(stream_sal - ref_sal)) == 0.0
    assert np.max(np.abs(vec_out - ref_out)) == 0.0
    assert np.max(np.abs(vec_sal - ref_sal)) == 0.0

    # Verify adaptive empirical Bayes prior blending across 1-halo tiles
    prior = np.abs(rng.standard_normal(n_genes, dtype=np.float32)) * 0.75
    ref_ad, _ = native_morse_dropout_filter_adaptive(
        sig, prior_arr=prior, mad_floor=1.5, tau=1.5, lambda_blend=0.45
    )
    stream_ad, _ = streamer.stream_morse_dropout_1d(
        sig, mad_floor=1.5, tau=1.5, prior_1d=prior, lambda_blend=0.45, use_native_simd=True
    )
    assert np.max(np.abs(stream_ad - ref_ad)) == 0.0


def test_operator1_halo_boundary_ramp_ablation() -> None:
    """
    Verify that a monotone ramp crossing tile boundary 16,381 -> 16,382 produces a
    false-positive spike under naive 0-halo tiling, whereas 1-halo tiling achieves 0.0 error.
    """
    streamer = WholeGenomeArenaStreamer(reset_on_init=True)
    ramp_sig = np.full(33694, 0.25, dtype=np.float32)
    ramp_sig[16380] = 2.0
    ramp_sig[16381] = 5.0
    ramp_sig[16382] = 8.0
    ramp_sig[16383] = 8.0

    ref_ramp, _ = native_morse_dropout_filter(ramp_sig, mad_floor=1.5, tau=1.5)
    halo1_ramp, _ = streamer.stream_morse_dropout_1d(
        ramp_sig, mad_floor=1.5, tau=1.5, tile_core_size=16382
    )

    nohalo_ramp = np.empty_like(ramp_sig)
    for s_nh in range(0, 33694, 16382):
        e_nh = min(s_nh + 16382, 33694)
        nohalo_ramp[s_nh:e_nh], _ = native_morse_dropout_filter(
            ramp_sig[s_nh:e_nh], mad_floor=1.5, tau=1.5
        )

    assert float(ref_ramp[16381]) == 0.0
    assert float(halo1_ramp[16381]) == 0.0
    assert float(np.max(np.abs(halo1_ramp - ref_ramp))) == 0.0
    # Naive 0-halo treats index 16,381 as a right endpoint and falsely retains 5.0!
    assert float(nohalo_ramp[16381]) == pytest.approx(5.0)
    assert float(np.max(np.abs(nohalo_ramp - ref_ramp))) == pytest.approx(5.0)


@pytest.mark.parametrize("n_genes", [2000, 5948, 17138, 33694])
def test_operator2_deconfounder_column_block_parity(n_genes: int) -> None:
    """Verify 0.0 max abs diff and 0 mask mismatches for column-block deconfounder streaming."""
    rng = np.random.default_rng(2080 + n_genes)
    streamer = WholeGenomeArenaStreamer(reset_on_init=True)

    n_cells = 250
    t_signs = np.where((np.arange(n_cells) % 2) == 0, 1.0, -1.0)[:, None].astype(np.float32)
    X_mat = rng.standard_normal((n_cells, n_genes), dtype=np.float32) * 0.5 + 2.0
    clk_cols = np.arange(0, n_genes, 20)
    X_mat[:, clk_cols] = 3.0 + t_signs * 1.5

    ref_deconf, ref_mask = native_cell_cycle_deconfounder(X_mat)
    stream_deconf, stream_mask = streamer.stream_cell_cycle_deconfounder(
        X_mat, clock_threshold=0.85, use_native_simd=True
    )
    vec_deconf, vec_mask = streamer.stream_cell_cycle_deconfounder(
        X_mat, clock_threshold=0.85, use_native_simd=False
    )

    assert np.max(np.abs(stream_deconf - ref_deconf)) == 0.0
    assert int(np.sum(stream_mask != ref_mask)) == 0
    assert np.max(np.abs(vec_deconf - ref_deconf)) == 0.0
    assert int(np.sum(vec_mask != ref_mask)) == 0
    assert int(np.sum(stream_mask)) == len(clk_cols)


def test_operator5_spatial_sheaf_gene_tile_parity() -> None:
    """Verify precomputed contact graph + gene-tile spatial sheaf streaming parity."""
    rng = np.random.default_rng(5050)
    streamer = WholeGenomeArenaStreamer(reset_on_init=True)

    n_spots = 200
    n_genes = 2500
    coords = np.column_stack([
        (np.arange(n_spots) % 20) * 35.0,
        (np.arange(n_spots) // 20) * 35.0,
    ]).astype(np.float32)
    layers = np.clip((coords[:, 1] // 105.0).astype(int), 0, 3)
    X_sp = np.abs(rng.standard_normal((n_spots, n_genes), dtype=np.float32))

    ref_engine = SE2CellularSheaf(delta_x_jitter=5.0, huber_delta=10.0)
    ref_out = ref_engine.align_with_jitter_compensation(
        coords=coords,
        marker_scores=X_sp,
        apparent_layers=layers,
        neighbor_dist=55.0,
        delta_x_est=5.0,
        mode="adaptive",
    )["smoothed_scores"]

    stream_res = streamer.stream_spatial_sheaf_aligner(
        coords=coords,
        marker_scores=X_sp,
        apparent_layers=layers,
        neighbor_dist=55.0,
        delta_x_est=5.0,
        huber_delta=10.0,
        mode="adaptive",
    )
    stream_out = stream_res["smoothed_scores"]

    assert float(np.max(np.abs(stream_out - ref_out))) < 1e-6
    assert stream_res["boundary_contacts_audited"] > 0


@pytest.mark.parametrize("n_genes", [2000, 5948, 17138, 33694])
def test_operator6_so6_multiomic_triad_trace_conservation(n_genes: int) -> None:
    """Verify SO(6) Kelvin-Mandel trace conservation across ceil(G/3) streamed triads."""
    rng = np.random.default_rng(6060 + n_genes)
    streamer = WholeGenomeArenaStreamer(reset_on_init=True)

    dna_m = np.abs(rng.standard_normal((200, n_genes), dtype=np.float32)) + 2.0
    rna_m = dna_m * 0.8 + np.abs(rng.standard_normal((200, n_genes), dtype=np.float32)) * 0.4
    res = streamer.stream_so6_multiomic_bundle(
        dna_m, rna_m, alpha_shrink=0.15, use_native_simd=True
    )

    assert res["n_genes"] == n_genes
    assert res["n_triads_streamed"] == (n_genes + 2) // 3
    assert res["max_relative_trace_residual_f32"] < 1e-5
    assert res["arena_slot_bytes"] == 144


def test_zero_dynamic_allocations_and_arena_pointer_stability() -> None:
    """Verify that repeated streaming across 33,694-gene arrays never shifts arena pointers."""
    rng = np.random.default_rng(7070)
    streamer = WholeGenomeArenaStreamer(reset_on_init=True)
    initial_base = streamer.arena_base
    initial_spatial_in_addr = int(streamer.view_spatial_in.ctypes.data)
    initial_ring_in_addr = int(streamer.view_ring_in.ctypes.data)

    X = np.abs(rng.standard_normal((100, 33694), dtype=np.float32))
    out1, sal1 = streamer.stream_morse_dropout_matrix(X, mad_floor=1.5, tau=1.5, axis=-1)
    out2, mask2 = streamer.stream_cell_cycle_deconfounder(X, clock_threshold=0.85)

    assert out1.shape == (100, 33694)
    assert sal1.shape == (100, 33694)
    assert out2.shape == (100, 33694)
    assert mask2.shape == (33694,)
    assert streamer.arena_base == initial_base
    assert int(streamer.view_spatial_in.ctypes.data) == initial_spatial_in_addr
    assert int(streamer.view_ring_in.ctypes.data) == initial_ring_in_addr
    assert streamer.tiles_processed > 200


def test_expedition4_json_receipt_and_blindspot_audit() -> None:
    """Verify deterministic JSON receipt integrity, zero NaN/Inf, and blind-spot audit metrics."""
    assert RECEIPT_PATH.exists(), f"Missing JSON receipt: {RECEIPT_PATH}"
    with open(RECEIPT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    _check_no_nan_or_inf(data)

    # 1. Verify Arena Geometry
    geom = data["arena_geometry_verification"]
    assert geom["passed"] is True
    assert geom["arena_size_bytes"] == 983_872
    assert geom["sram_ceiling_bytes"] == 2_097_152

    # 2. Verify Numerical Parity Audits
    parity = data["numerical_parity_audits"]
    for rec in parity["operator1_morse_1halo_parity"]:
        assert rec["bit_exact_parity"] is True
        assert rec["max_abs_diff_filtered_vs_full_native"] == 0.0
    for rec in parity["operator2_deconfounder_column_tile_parity"]:
        assert rec["bit_exact_parity"] is True
        assert rec["max_abs_diff_vs_full_native"] == 0.0
        assert rec["mask_mismatches_vs_full_native"] == 0
    for rec in parity["operator5_spatial_sheaf_tile_parity"]:
        assert rec["bit_exact_parity"] is True
    for rec in parity["operator6_so6_multiomic_triad_parity"]:
        assert rec["trace_conserved_within_1e5"] is True

    # 3. Verify Scaling Benchmarks & Speedups
    bench = data["scaling_benchmarks"]
    g_sweep = bench["gene_dimension_sweep_N500"]
    assert len(g_sweep) == 6
    g33694_rec = [r for r in g_sweep if r["n_genes"] == 33694][0]
    assert g33694_rec["speedup_cellsnail_simd_morse_vs"]["magic_impute"] > 2.0
    assert g33694_rec["speedup_cellsnail_simd_morse_vs"]["alra_impute"] > 20.0
    assert g33694_rec["working_memory_mb"]["cellsnail_arena_sram"] == pytest.approx(0.9383, abs=1e-3)

    # 4. Verify HVG-2000 & Metacell-50 Blind-Spot Audit
    audit = build_hvg2000_and_metacell50_blindspot_audit()
    cohorts = audit["empirical_cohorts_hvg2000_discard_summary"]
    assert cohorts["GSE148673_TNBC3"]["n_genes_total"] == 33694
    assert cohorts["GSE148673_TNBC3"]["genes_discarded_by_hvg2000"] == 31694
    assert cohorts["GSE148673_TNBC3"]["fraction_discarded_total_pct"] == pytest.approx(94.0642, abs=1e-2)
    assert cohorts["GSE144239_cSCC"]["n_genes_total"] == 17138
    assert cohorts["GSE144239_cSCC"]["genes_discarded_by_hvg2000"] == 15138
    assert cohorts["GSE57872_GBM"]["n_genes_total"] == 5948
    assert cohorts["GSE57872_GBM"]["genes_discarded_by_hvg2000"] == 3948

    # Verify Orchestrator 2 (Exp 5), 3 (Exp 6), and 4 (Exp 7) Blind-Spot Summaries
    exp5 = audit["orchestrator2_exp5_delta_erasure_blindspot"]
    assert exp5["core_13_ddr_chromatin_panel_loss_summary"]["GSE148673_TNBC3"]["n_dropped_by_hvg2000"] == 11
    assert exp5["core_13_ddr_chromatin_panel_loss_summary"]["GSE144239_cSCC"]["n_dropped_by_hvg2000"] == 13
    assert len(exp5["newly_uncovered_tier_a_dark_genome_30_hits"]) == 30

    exp6 = audit["orchestrator3_exp6_stringer_persister_blindspot"]
    assert exp6["pan_autosome_multi_patient_scale"]["total_single_cells_scanned"] == 10058
    assert exp6["pan_autosome_multi_patient_scale"]["diploid_normal_false_positive_rate"] == 0.0
    mc_tnbc = exp6["metacell_50_persister_washout_audit"]["GSE148673_TNBC3_Rare_Persister_n2_of_500_0_40pct"]
    assert mc_tnbc["signal_washout_pct"] == pytest.approx(96.0)
    assert mc_tnbc["attenuation_fold"] == pytest.approx(25.0)

    exp7 = audit["orchestrator4_exp7_spatial_secretome_blindspot"]
    assert exp7["curated_251_pair_cross_wall_screen"]["tier_a_non_hvg2000_active_pairs"] == 141
    assert exp7["genome_wide_14890_pair_17_family_screen"]["pairs_with_at_least_one_partner_dropped_by_hvg2000"] == 7206
    assert exp7["genome_wide_14890_pair_17_family_screen"]["pairs_with_both_ligand_and_receptor_dropped_by_hvg2000"] == 2807
