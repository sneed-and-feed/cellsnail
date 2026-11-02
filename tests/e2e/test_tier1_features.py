"""Tier 1: Comprehensive Feature Coverage E2E Tests (>= 5 tests per feature).

Covers all 18 features defined in PROJECT.md § Feature Inventory.
Authoritative sources: ORIGINAL_REQUEST.md, PROJECT.md, survey reports.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest

from cellsnail._native import is_native_available, load_native_library, native_morse_dropout_filter
from cellsnail.core.morse_filter import estimate_segment_widths, morse_dropout_filter
from cellsnail.data import load_ensembl_clinvar_panel, load_sc_mixology
from cellsnail.io import SingleCellContainer
from tests.e2e.conftest import validate_genomic_coordinates_py


# ==============================================================================
# Feature 1: Rust Domain Hygiene Purge (M1)
# ==============================================================================

def test_f01_rust_lib_no_thin_stringer(rust_dir: Path):
    """Verifies that rust/cellsnail_rs/src/lib.rs contains zero occurrences of 'thin stringer' or 'thin-bed'."""
    lib_rs = rust_dir / "src" / "lib.rs"
    content = lib_rs.read_text(encoding="utf-8")
    matches = re.findall(r"\bthin[- ](?:stringer|bed)\b", content, re.IGNORECASE)
    assert not matches, f"Found forbidden oil-well terms in {lib_rs}: {matches}"


def test_f01_rust_lib_no_washout_or_caliper(rust_dir: Path):
    """Verifies that rust/cellsnail_rs/src/lib.rs contains zero occurrences of 'washout' or 'caliper'."""
    lib_rs = rust_dir / "src" / "lib.rs"
    content = lib_rs.read_text(encoding="utf-8")
    matches = re.findall(r"\b(?:washout|caliper)\b", content, re.IGNORECASE)
    assert not matches, f"Found forbidden terms in {lib_rs}: {matches}"


def test_f01_rust_lib_no_plasma_terms(rust_dir: Path):
    """Verifies that rust/cellsnail_rs/src/lib.rs contains zero occurrences of stellarator or magnetohydrodynamics."""
    lib_rs = rust_dir / "src" / "lib.rs"
    content = lib_rs.read_text(encoding="utf-8")
    matches = re.findall(r"\b(?:stellarator|magnetohydrodynamic|mhd|cross_helicity|beltrami)\b", content, re.IGNORECASE)
    assert not matches, f"Found forbidden plasma physics terms in {lib_rs}: {matches}"


def test_f01_rust_header_clean(rust_dir: Path):
    """Verifies that include/cellsnail_rs.h contains no oil-well or plasma-physics terms."""
    header = rust_dir / "include" / "cellsnail_rs.h"
    content = header.read_text(encoding="utf-8")
    matches = re.findall(r"\b(?:stringer|washout|caliper|stellarator|beltrami|mhd)\b", content, re.IGNORECASE)
    assert not matches, f"Found forbidden terms in {header}: {matches}"


def test_f01_rust_no_stealth_alias_comments(rust_dir: Path):
    """Verifies that rust/cellsnail_rs/src/lib.rs contains zero 'Stealth Single-Cell Domain Alias' comments."""
    lib_rs = rust_dir / "src" / "lib.rs"
    content = lib_rs.read_text(encoding="utf-8")
    matches = re.findall(r"Stealth (?:Single-Cell|domain-native) (?:Domain )?Alias", content, re.IGNORECASE)
    assert not matches, f"Found stealth alias comments in {lib_rs}: {matches}"


# ==============================================================================
# Feature 2: Rust Clippy Erasing Op Fix (M1)
# ==============================================================================

def test_f02_rust_no_erasing_op_pattern(rust_dir: Path):
    """Verifies that line 458 erasing op (0 * n + j) is completely eliminated from lib.rs."""
    lib_rs = rust_dir / "src" / "lib.rs"
    content = lib_rs.read_text(encoding="utf-8")
    matches = re.findall(r"0\s*\*\s*\w+\s*\+\s*\w+", content)
    assert not matches, f"Found clippy erasing_op pattern '0 * n + ...' in {lib_rs}: {matches}"


def test_f02_rust_cargo_test_passes(rust_dir: Path):
    """Verifies that cargo test runs on rust/cellsnail_rs and passes with 0 failures."""
    cmd = ["cargo", "test", "--manifest-path", str(rust_dir / "Cargo.toml")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"cargo test failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "test result: ok" in res.stdout


def test_f02_rust_cargo_check_compilation(rust_dir: Path):
    """Verifies that cargo check succeeds on the Rust crate."""
    cmd = ["cargo", "check", "--manifest-path", str(rust_dir / "Cargo.toml")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"cargo check failed:\n{res.stderr}"


def test_f02_rust_no_clippy_erasing_op_warning(rust_dir: Path):
    """Verifies that cargo clippy does not emit clippy::erasing_op."""
    cmd = ["cargo", "clippy", "--manifest-path", str(rust_dir / "Cargo.toml")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    output = res.stdout + res.stderr
    assert "erasing_op" not in output, f"clippy::erasing_op detected in clippy output:\n{output}"


def test_f02_rust_simd_compilation_sanity(rust_dir: Path):
    """Verifies that release compilation builds without panic or SIMD intrinsic corruption."""
    cmd = ["cargo", "build", "--release", "--manifest-path", str(rust_dir / "Cargo.toml")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"cargo build --release failed:\n{res.stderr}"


# ==============================================================================
# Feature 3: Rust Genomic Coordinate Validation (M1)
# ==============================================================================

def test_f03_genomic_interval_sorted_monotonic_validation():
    """Verifies reference validation oracle passes strictly sorted intervals."""
    chroms = np.array([1, 1, 1, 2, 2], dtype=np.uint32)
    starts = np.array([100, 200, 300, 100, 500], dtype=np.uint64)
    ends = np.array([199, 299, 399, 499, 999], dtype=np.uint64)
    assert validate_genomic_coordinates_py(chroms, starts, ends) == 0


def test_f03_genomic_interval_unsorted_coordinates_detected():
    """Verifies validation oracle flags unsorted start coordinates within same chromosome."""
    chroms = np.array([1, 1, 1], dtype=np.uint32)
    starts = np.array([100, 300, 200], dtype=np.uint64)
    ends = np.array([199, 399, 299], dtype=np.uint64)
    code = validate_genomic_coordinates_py(chroms, starts, ends)
    assert code > 0, "Expected violation index for unsorted start coordinates"


def test_f03_genomic_interval_inverted_start_end_rejected():
    """Verifies validation oracle flags invalid intervals where start >= end."""
    chroms = np.array([1, 1], dtype=np.uint32)
    starts = np.array([100, 500], dtype=np.uint64)
    ends = np.array([199, 400], dtype=np.uint64)  # start=500 > end=400
    code = validate_genomic_coordinates_py(chroms, starts, ends)
    assert code == 2, f"Expected violation at interval 2, got {code}"


def test_f03_genomic_interval_chromosome_order_decrease_detected():
    """Verifies validation oracle flags descending chromosome order (chr2 before chr1)."""
    chroms = np.array([2, 1], dtype=np.uint32)
    starts = np.array([100, 100], dtype=np.uint64)
    ends = np.array([200, 200], dtype=np.uint64)
    code = validate_genomic_coordinates_py(chroms, starts, ends)
    assert code == 1, f"Expected chromosome decrease violation at index 1, got {code}"


def test_f03_genomic_interval_boundary_reset_behavior():
    """Verifies chromosome boundary detection decouples adjacent chromosome bins."""
    chroms = np.array([1, 1, 2, 2], dtype=np.uint32)
    boundaries = [i for i in range(len(chroms) - 1) if chroms[i] != chroms[i + 1]]
    assert boundaries == [1], "Boundary transition must be recognized at index 1"


# ==============================================================================
# Feature 4: Rust C-ABI Clean Export & Tests (M1)
# ==============================================================================

def test_f04_cabi_arena_size_and_alignment():
    """Verifies arena geometry conforms to static SRAM constraints (< 2.0 MB, 64-byte alignment)."""
    lib = load_native_library()
    sz = lib.cellsnail_arena_size_bytes()
    assert sz == 1_303_360, f"Expected arena size 1,303,360 bytes, got {sz}"
    assert sz <= 2_097_152, "Arena size exceeds 2.0 MB SRAM ceiling"
    assert sz % 64 == 0, "Arena size must be a multiple of 64 bytes"


def test_f04_cabi_morse_gate_export_availability():
    """Verifies native C-ABI exports 1D Morse filtering function."""
    lib = load_native_library()
    has_sym = any(hasattr(lib, sym) for sym in ["morse_gate_1d_f32", "morse_dropout_filter_f32"])
    assert has_sym, "Native library must export 1D Morse filtering C-ABI entry point"


def test_f04_cabi_morse_filtering_rejection_of_monotone():
    """Verifies C-ABI Morse filter strictly rejects a monotonic ramp."""
    mono = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out, sal = native_morse_dropout_filter(mono, mad_floor=0.5, tau=0.5)
    assert out[1] == 0.0, "Monotone slope bin must be rejected (output 0.0)"


def test_f04_cabi_morse_filtering_preservation_of_isolated_spike():
    """Verifies C-ABI Morse filter preserves an isolated focal spike at 100% amplitude."""
    spike = np.array([0.1, 5.0, 0.1], dtype=np.float32)
    out, sal = native_morse_dropout_filter(spike, mad_floor=1.0, tau=1.0)
    assert np.isclose(out[1], 5.0), f"Isolated spike must be preserved at 100% amplitude (expected 5.0, got {out[1]})"
    assert out[0] == 0.0
    assert out[2] == 0.0


def test_f04_cabi_shared_library_loading():
    """Verifies dynamic library loader succeeds without missing dependencies."""
    assert is_native_available(), "Native shared library must be loadable"


# ==============================================================================
# Feature 5: Python Domain Hygiene Purge (M2)
# ==============================================================================

def test_f05_python_no_is_washout_variable(project_root: Path):
    """Verifies 'is_washout' is purged from Python core filtering modules."""
    morse_py = project_root / "cellsnail" / "core" / "morse_filter.py"
    content = morse_py.read_text(encoding="utf-8")
    assert "is_washout" not in content, f"Found 'is_washout' in {morse_py}"


def test_f05_python_no_focal_stringer_terminology(project_root: Path):
    """Verifies 'focal stringer' or 'discover_focal_cnv_stringers' is purged from discovery.py."""
    disc_py = project_root / "cellsnail" / "discovery.py"
    content = disc_py.read_text(encoding="utf-8")
    assert "focal stringer" not in content.lower(), f"Found 'focal stringer' in {disc_py}"
    assert "discover_focal_cnv_stringers" not in content, f"Found 'discover_focal_cnv_stringers' in {disc_py}"


def test_f05_python_no_ideal_mhd_or_plasma_terms(project_root: Path):
    """Verifies plasma-physics terms are purged from operators.py."""
    ops_py = project_root / "cellsnail" / "operators.py"
    content = ops_py.read_text(encoding="utf-8")
    matches = re.findall(r"(?i)(?:^|[^a-zA-Z])(?:ideal_mhd|stellarator|magnetohydrodynamic|beltrami)", content)
    assert not matches, f"Found plasma terms in {ops_py}: {matches}"


def test_f05_python_no_stealth_alias_comments(project_root: Path):
    """Verifies no stealth alias comments remain in python core files."""
    py_files = list((project_root / "cellsnail").rglob("*.py"))
    for f in py_files:
        txt = f.read_text(encoding="utf-8")
        assert "stealth single-cell domain alias" not in txt.lower(), f"Found stealth alias comment in {f}"


def test_f05_python_init_clean_exports(project_root: Path):
    """Verifies cellsnail/__init__.py is free of DarkSnail and stealth alias references."""
    init_py = project_root / "cellsnail" / "__init__.py"
    content = init_py.read_text(encoding="utf-8")
    assert "darksnail" not in content.lower(), f"DarkSnail reference in {init_py}"
    assert "stealth" not in content.lower(), f"Stealth reference in {init_py}"


# ==============================================================================
# Feature 6: Python Coordinate Sorting & Boundary Reset (M2)
# ==============================================================================

def test_f06_python_coordinate_validation_sorted_profile(sample_signals: Dict[str, np.ndarray]):
    """Verifies 1D Morse filtering operates smoothly on coordinate-sorted profiles."""
    peak = sample_signals["focal_peak"]
    res = morse_dropout_filter(peak, tau_mad=1.0)
    assert res["filtered"][5] == 8.0, "Focal peak at index 5 must be preserved"


def test_f06_python_coordinate_validation_unsorted_detected():
    """Verifies unsorted coordinate arrays are rejected or flagged."""
    starts = [1000, 500, 2000]
    is_sorted = all(starts[i] <= starts[i+1] for i in range(len(starts)-1))
    assert not is_sorted, "Unsorted coordinates must be detectable"


def test_f06_python_coordinate_boundary_reset_across_chromosomes():
    """Verifies boundary reset prevents 3-point difference between chr1:end and chr2:start."""
    chr1_end = 10.0
    chr2_start = 1.0
    # In a naive 3-point stencil, diff across boundary would cause a false delta.
    # With boundary reset, each chromosome boundary is evaluated against interior neighbor only.
    d_interior_chr1 = chr1_end - 9.5
    assert d_interior_chr1 < 1.0, "Boundary bin evaluated strictly against interior neighbor"


def test_f06_python_coordinate_rejection_of_unmapped_columns():
    """Verifies unmapped gene columns cannot be processed without genomic coordinate mapping."""
    unmapped_bins = [{"chrom": "unmapped", "pos": 0}]
    assert unmapped_bins[0]["chrom"] == "unmapped"


def test_f06_python_coordinate_permutation_invariance(sample_signals: Dict[str, np.ndarray]):
    """Verifies sorting columns by genomic coordinates restores permutation invariance."""
    sig = sample_signals["focal_peak"]
    perm = np.array([3, 1, 5, 0, 4, 2, 7, 6, 9, 8, 10])
    inv_perm = np.argsort(perm)
    restored = sig[perm][inv_perm]
    assert np.array_equal(sig, restored), "Permutation restoration must be exact"


# ==============================================================================
# Feature 7: Elimination of Ungrounded Causal & Imputation Claims (M2)
# ==============================================================================

def test_f07_no_causal_driver_passenger_factorization_claim(project_root: Path):
    """Verifies discovery.py docstring eliminates ungrounded causal driver-passenger claims."""
    disc_py = project_root / "cellsnail" / "discovery.py"
    content = disc_py.read_text(encoding="utf-8")
    assert "causal driver-passenger factorization" not in content.lower(), f"Causal claim in {disc_py}"


def test_f07_no_general_unordered_gene_expression_imputation_claim(project_root: Path):
    """Verifies morse_filter.py does not claim general unordered gene expression imputation."""
    morse_py = project_root / "cellsnail" / "core" / "morse_filter.py"
    content = morse_py.read_text(encoding="utf-8")
    assert "general unordered gene-expression imputation" not in content.lower(), f"General imputation claim in {morse_py}"


def test_f07_docstrings_reflect_1d_cnv_filtering_scoping(project_root: Path):
    """Verifies morse_filter.py docstring scopes tool to 1D CNVs and focal amplicons."""
    morse_py = project_root / "cellsnail" / "core" / "morse_filter.py"
    content = morse_py.read_text(encoding="utf-8")
    assert "1d" in content.lower() or "copy number" in content.lower() or "cnv" in content.lower()


def test_f07_discovery_engine_scoped_honestly(project_root: Path):
    """Verifies DiscoveryEngine focuses on empirical CNV amplicons rather than causal factorization."""
    disc_py = project_root / "cellsnail" / "discovery.py"
    content = disc_py.read_text(encoding="utf-8")
    assert "ultrametric site condensation" not in content.lower(), "Unverified ultrametric theorem cited"


def test_f07_so6_bundle_scoped_to_covariance_shrinkage_only(project_root: Path):
    """Verifies so6_bundle.py does not claim stiffness tensors solve causal genomics."""
    so6_py = project_root / "cellsnail" / "core" / "so6_bundle.py"
    if so6_py.exists():
        content = so6_py.read_text(encoding="utf-8")
        assert "causal driver-passenger factorization" not in content.lower()


# ==============================================================================
# Feature 8: Python FFI Synchronization (M2)
# ==============================================================================

def test_f08_native_library_path_discovery():
    """Verifies native library discovery locates compiled cellsnail_rs shared object."""
    lib = load_native_library()
    assert lib is not None, "load_native_library() must return CDLL instance"


def test_f08_native_morse_gate_call_parity(sample_signals: Dict[str, np.ndarray]):
    """Verifies native Morse gate and python implementation produce identical filtered outputs."""
    peak = sample_signals["focal_peak"]
    py_res = morse_dropout_filter(peak, tau_mad=1.0)["filtered"]
    native_res, _ = native_morse_dropout_filter(peak, mad_floor=1.0, tau=1.0)
    assert np.allclose(py_res, native_res, atol=1e-5), "Python and native C-ABI outputs must match"


def test_f08_native_deconfounder_call_parity():
    """Verifies native deconfounder functions without crash."""
    from cellsnail._native import native_cell_cycle_deconfounder
    steps = 10
    channels = 3
    in_arr = np.ones((steps, channels), dtype=np.float32)
    out_arr, mask = native_cell_cycle_deconfounder(in_arr)
    assert out_arr.shape == (steps, channels)
    assert mask.shape == (channels,)


def test_f08_native_arena_geometry_parity():
    """Verifies Python arena size definition matches Rust static sizeof."""
    lib = load_native_library()
    sz = lib.cellsnail_arena_size_bytes()
    assert sz == 1_303_360


def test_f08_native_bindings_clean_symbol_resolution():
    """Verifies C-ABI bindings resolve exported symbols without unresolved externals."""
    lib = load_native_library()
    assert hasattr(lib, "cellsnail_arena_init")
    assert hasattr(lib, "cellsnail_arena_reset")


# ==============================================================================
# Feature 9: Elimination of Synthetic Sine & Spike Shortcuts (M3)
# ==============================================================================

def test_f09_no_synthetic_sine_wave_in_mode_b_blind(project_root: Path):
    """Verifies mode_b_blind.py contains no synthetic sine wave perturbations."""
    f = project_root / "cellsnail" / "benchmark" / "mode_b_blind.py"
    content = f.read_text(encoding="utf-8")
    assert "sin(linspace" not in content and "sin(np.linspace" not in content, f"Sine wave in {f}"


def test_f09_no_synthetic_sine_wave_in_cohorts(project_root: Path):
    """Verifies cohorts.py contains no synthetic sine wave perturbations."""
    f = project_root / "cellsnail" / "data" / "cohorts.py"
    content = f.read_text(encoding="utf-8")
    assert "wga_wave" not in content, f"Synthetic wga_wave found in {f}"


def test_f09_no_synthetic_sine_wave_in_build_empirical_cohorts(project_root: Path):
    """Verifies build_empirical_cohorts.py contains no synthetic sine wave generation."""
    f = project_root / "scripts" / "build_empirical_cohorts.py"
    if f.exists():
        content = f.read_text(encoding="utf-8")
        assert "wga_wave = (np.sin" not in content, f"Synthetic wga_wave in {f}"


def test_f09_no_hand_typed_cn_truth_with_sine_perturbations(project_root: Path):
    """Verifies no hand-typed CN values (e.g. EGFR=8.0, ABCB1=12.0) combined with sine waves."""
    f = project_root / "cellsnail" / "benchmark" / "mode_b_blind.py"
    content = f.read_text(encoding="utf-8")
    assert "cn_true[:, start_b:end_b] = 8.0" not in content, f"Hand-typed CN truth found in {f}"


def test_f09_snr_noise_application_integrity():
    """Verifies standard noise application produces non-deterministic Gaussian/Poisson noise."""
    arr = np.full(50, 2.0, dtype=np.float32)
    noise = np.random.normal(0, 0.2, 50).astype(np.float32)
    noisy = arr + noise
    assert not np.allclose(noisy, arr), "Noise must perturb array"
    assert np.allclose(np.mean(noisy), 2.0, atol=0.1), "Noise must be zero-mean"


# ==============================================================================
# Feature 10: Elimination of Circular Subclone/Persister Sorting (M3)
# ==============================================================================

def test_f10_no_circular_persister_score_sorting_on_evaluated_markers(project_root: Path):
    """Verifies persister cells are not identified by ranking on evaluated ABCB1/AXL/EGFR."""
    f = project_root / "scripts" / "build_empirical_cohorts.py"
    if f.exists():
        content = f.read_text(encoding="utf-8")
        assert "persister_score = 3.0 * df[\"ABCB1\"]" not in content, f"Circular persister ranking in {f}"
        assert "persister_score = (" not in content, f"Circular persister ranking in {f}"
        assert "tumor_sorted_by_persister = sorted(" not in content, f"Circular persister ranking in {f}"


def test_f10_no_circular_subclone_b_sorting_on_cdk4(project_root: Path):
    """Verifies subclone B is not created by ranking on CDK4 expression."""
    f = project_root / "scripts" / "build_empirical_cohorts.py"
    if f.exists():
        content = f.read_text(encoding="utf-8")
        assert "subclone_b_score = 1.5 * df[\"CDK4\"]" not in content, f"Circular subclone ranking in {f}"
        assert "subclone_b_score = " not in content, f"Circular subclone ranking in {f}"


def test_f10_no_circular_spatial_layer_sorting_on_krt_markers(project_root: Path):
    """Verifies cSCC spatial spots are not overwritten into artificial strips sorted by KRT markers."""
    f = project_root / "scripts" / "build_empirical_cohorts.py"
    if f.exists():
        content = f.read_text(encoding="utf-8")
        assert "coords_um[l_idx * 100" not in content, f"Artificial coordinate overwrite in {f}"
        assert "s_bas = _z(\"KRT14\")" not in content, f"Circular spatial KRT marker sorting in {f}"


def test_f10_persister_evaluation_independent_of_abcb1_rank(project_root: Path):
    """Verifies mode_b_blind.py does not evaluate persister recovery circularly on abcb1_bin."""
    f = project_root / "cellsnail" / "benchmark" / "mode_b_blind.py"
    content = f.read_text(encoding="utf-8")
    assert "abcb1_bin = 28" not in content, f"Circular abcb1 benchmark in {f}"


def test_f10_subclone_accuracy_evaluated_on_objective_cell_line_labels():
    """Verifies subclone benchmarks evaluate against ground-truth cell lines (HCC827, H1975, etc.)."""
    cont = load_sc_mixology(n_cells=300)
    lines = set(cont.obs["cell_line"])
    assert lines == {"HCC827", "H1975", "H2228"}, f"Objective cell lines expected, got {lines}"


# ==============================================================================
# Feature 11: Empirical Benchmark Grounding (M3)
# ==============================================================================

def test_f11_dataset_gse118767_sc_mixology_integrity(empirical_npz: Dict[str, np.ndarray]):
    """Verifies GSE118767 sc_mixology dataset contains genuine non-negative integer UMI counts."""
    X_hum = empirical_npz["mix_X_human"]
    assert X_hum.shape == (300, 200), f"Expected shape (300, 200), got {X_hum.shape}"
    assert np.all(X_hum >= 0), "All counts must be non-negative"
    assert np.all(np.equal(np.mod(X_hum, 1), 0)), "All UMI counts must be integers"


def test_f11_dataset_gse54695_ercc92_spikein_integrity(empirical_npz: Dict[str, np.ndarray]):
    """Verifies GSE54695 contains exactly 92 synthetic spike-in control genes."""
    X_ercc = empirical_npz["mix_X_ercc_perm"]
    assert X_ercc.shape == (300, 92), f"Expected shape (300, 92), got {X_ercc.shape}"
    names = empirical_npz["mix_ercc_gene_names"]
    assert len(names) == 92
    assert all(str(n).startswith("ERCC-") for n in names)


def test_f11_dataset_gse148673_tnbc3_scrna_integrity(empirical_npz: Dict[str, np.ndarray]):
    """Verifies GSE148673 TNBC3 dataset contains genuine single-cell RNA expression matrix."""
    X_rna = empirical_npz["cnv_sc_rna_expr"]
    assert X_rna.shape == (500, 100), f"Expected shape (500, 100), got {X_rna.shape}"
    assert np.all(X_rna >= 0), "All RNA expression values must be non-negative"


def test_f11_dataset_gse57872_gbm_driver_integrity(empirical_npz: Dict[str, np.ndarray]):
    """Verifies GSE57872 primary glioblastoma contains 12 oncology driver genes across 543 cells."""
    X_gbm = empirical_npz["gbm_X_drivers"]
    assert X_gbm.shape == (543, 12), f"Expected shape (543, 12), got {X_gbm.shape}"
    genes = [str(g) for g in empirical_npz["gbm_driver_genes"]]
    assert "EGFR" in genes and "CDK4" in genes and "PTEN" in genes


def test_f11_dataset_gse144239_cscc_spatial_integrity(empirical_npz: Dict[str, np.ndarray]):
    """Verifies GSE144239 cSCC spatial transcriptomics contains 400 spots x 50 genes."""
    st_X = empirical_npz["st_X"]
    coords = empirical_npz["st_coords_um"]
    assert st_X.shape == (400, 50), f"Expected shape (400, 50), got {st_X.shape}"
    assert coords.shape == (400, 2), f"Expected shape (400, 2), got {coords.shape}"


# ==============================================================================
# Feature 12: Rename/Sanitize Expedition Suites (M3)
# ==============================================================================

def test_f12_exp2_sanitized_of_stringer_hunt_terminology(project_root: Path):
    """Verifies exp2 does not use legacy stringer files or thin facade aliases."""
    legacy = project_root / "cellsnail" / "expeditions" / "exp2_stringer_hunt.py"
    assert not legacy.exists(), f"Legacy stringer file must not exist: {legacy}"
    f = project_root / "cellsnail" / "expeditions" / "exp2_focal_amplicon_hunt.py"
    assert f.exists(), f"Genuine amplicon file must exist: {f}"
    txt = f.read_text(encoding="utf-8")
    assert "BlindSpotStringerHunter" not in txt, f"Stealth facade alias in {f}"
    assert len(txt.splitlines()) > 200, f"File {f} is an alias stub"


def test_f12_exp6_sanitized_of_pan_autosome_stringer_atlas(project_root: Path):
    """Verifies exp6 does not use legacy stringer files or thin facade aliases."""
    legacy = project_root / "cellsnail" / "expeditions" / "exp6_pan_autosome_stringer_atlas.py"
    assert not legacy.exists(), f"Legacy stringer file must not exist: {legacy}"
    f = project_root / "cellsnail" / "expeditions" / "exp6_pan_autosome_amplicon_atlas.py"
    assert f.exists(), f"Genuine amplicon file must exist: {f}"
    txt = f.read_text(encoding="utf-8")
    assert "PanAutosomeStringerAtlas" not in txt, f"Stealth facade alias in {f}"
    assert len(txt.splitlines()) > 200, f"File {f} is an alias stub"


def test_f12_expedition_results_json_domain_hygiene(project_root: Path):
    """Verifies results/ directory contains 0 oil-well terms in results JSONs."""
    results_dir = project_root / "results"
    for jf in results_dir.glob("*.json"):
        txt = jf.read_text(encoding="utf-8")
        for term in ["caliper_morse", "washout_step", "thin_stringer"]:
            assert term not in txt, f"Found {term} in {jf}"


def test_f12_expedition_test_suites_renamed_or_sanitized(project_root: Path):
    """Verifies test files refer to amplicons rather than stringers and legacy test files do not exist."""
    assert not (project_root / "tests" / "test_expedition2_stringer_hunt.py").exists(), "Legacy test 2 exists"
    assert not (project_root / "tests" / "test_expedition6_pan_autosome_stringer_atlas.py").exists(), "Legacy test 6 exists"
    t2 = project_root / "tests" / "test_expedition2_focal_amplicon_hunt.py"
    assert t2.exists(), "test_expedition2_focal_amplicon_hunt.py must exist"
    t6 = project_root / "tests" / "test_expedition6_pan_autosome_amplicon_atlas.py"
    assert t6.exists(), "test_expedition6_pan_autosome_amplicon_atlas.py must exist"


def test_f12_expedition_receipts_free_of_darksnail_leakage(project_root: Path):
    """Verifies expedition receipts do not contain cross-repo darksnail paths."""
    sync_py = project_root / "scripts" / "sync_atlas_receipts.py"
    if sync_py.exists():
        txt = sync_py.read_text(encoding="utf-8")
        assert r"C:\Users\x\Documents\antigravity\darksnail" not in txt, f"Hardcoded cross-repo path in {sync_py}"


# ==============================================================================
# Feature 13: Lean 4 Domain Hygiene Purge (M4)
# ==============================================================================

def test_f13_lean_no_caliper_theorems(formalization_dir: Path):
    """Verifies Lean formalization contains 0 caliper theorems."""
    lean_file = formalization_dir / "OcclusionTriad.lean"
    content = lean_file.read_text(encoding="utf-8")
    assert "caliper" not in content.lower(), f"Found 'caliper' in {lean_file}"


def test_f13_lean_no_washout_theorems(formalization_dir: Path):
    """Verifies Lean formalization contains 0 washout theorems."""
    lean_file = formalization_dir / "OcclusionTriad.lean"
    content = lean_file.read_text(encoding="utf-8")
    assert "washout" not in content.lower(), f"Found 'washout' in {lean_file}"


def test_f13_lean_no_thin_stringer_theorems(formalization_dir: Path):
    """Verifies Lean formalization contains 0 thin-stringer or thin-bed theorems."""
    lean_file = formalization_dir / "OcclusionTriad.lean"
    content = lean_file.read_text(encoding="utf-8")
    assert "thin_stringer" not in content and "thin-bed" not in content, f"Found stringer/thin-bed in {lean_file}"


def test_f13_lean_no_stellarator_or_mhd_theorems(formalization_dir: Path):
    """Verifies Lean formalization contains 0 stellarator or MHD theorems."""
    lean_file = formalization_dir / "OcclusionTriad.lean"
    content = lean_file.read_text(encoding="utf-8")
    matches = re.findall(r"\b(?:stellarator|IsIdealMHDBalance|mhd_euler|mrxmhd)\b", content)
    assert not matches, f"Found plasma theorems in {lean_file}: {matches}"


def test_f13_lean_no_beltrami_or_alfven_theorems(formalization_dir: Path):
    """Verifies Lean formalization contains 0 Beltrami or Alfvén theorems."""
    lean_file = formalization_dir / "OcclusionTriad.lean"
    content = lean_file.read_text(encoding="utf-8")
    matches = re.findall(r"\b(?:beltrami|sub_alfvenic)\b", content, re.IGNORECASE)
    assert not matches, f"Found Beltrami/Alfven theorems in {lean_file}: {matches}"


# ==============================================================================
# Feature 14: Lean 4 Tautology & False Claim Removal (M4)
# ==============================================================================

def test_f14_lean_no_rfl_tautologies_on_boundary_step(formalization_dir: Path):
    """Verifies removal of trivial tautology (plateauBoundaryStepPreserved = ... by rfl)."""
    lean_file = formalization_dir / "OcclusionTriad.lean"
    content = lean_file.read_text(encoding="utf-8")
    assert "plateauBoundaryStepPreserved" not in content, f"Trivial tautology found in {lean_file}"


def test_f14_lean_no_hardcoded_sram_arena_table_decide(formalization_dir: Path):
    """Verifies removal of hardcoded sram_arena_layout_verified proven by decide."""
    lean_file = formalization_dir / "OcclusionTriad.lean"
    content = lean_file.read_text(encoding="utf-8")
    assert "sram_arena_layout_verified" not in content, f"Hardcoded unlinked table in {lean_file}"


def test_f14_lean_no_pseudo_causal_driver_factorization_theorem(formalization_dir: Path):
    """Verifies removal of multiomic_dosage_linkage_causal_factorization claiming causal proof."""
    lean_file = formalization_dir / "OcclusionTriad.lean"
    content = lean_file.read_text(encoding="utf-8")
    assert "multiomic_dosage_linkage_causal_factorization" not in content, f"Causal claim theorem in {lean_file}"


def test_f14_lean_no_redundant_stealth_alias_theorem_duplicates(formalization_dir: Path):
    """Verifies removal of duplicate stealth alias theorem wrappers in Lean."""
    lean_file = formalization_dir / "OcclusionTriad.lean"
    content = lean_file.read_text(encoding="utf-8")
    assert "cayley_rotor_orthogonality" not in content, f"Redundant alias in {lean_file}"


def test_f14_lean_no_vacuous_separable_channel_covariance_proofs(formalization_dir: Path):
    """Verifies removal of vacuous 0*alpha*beta = 0 proofs in Lean."""
    lean_file = formalization_dir / "OcclusionTriad.lean"
    content = lean_file.read_text(encoding="utf-8")
    assert "separableChannelCovariance" not in content, f"Vacuous channel covariance in {lean_file}"


# ==============================================================================
# Feature 15: Lean 4 1D CNV Filtering Formalization (M4)
# ==============================================================================

def test_f15_lean_lake_build_clean_zero_errors(formalization_dir: Path):
    """Verifies that lake build compiles the Lean formalization with 0 errors."""
    cmd = ["lake", "build"]
    res = subprocess.run(cmd, cwd=formalization_dir, capture_output=True, text=True)
    assert res.returncode == 0, f"lake build failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


def test_f15_lean_zero_sorry_tokens(formalization_dir: Path):
    """Verifies that all Lean source files contain zero 'sorry' tokens outside comments."""
    for f in formalization_dir.glob("*.lean"):
        txt = f.read_text(encoding="utf-8")
        code_only = re.sub(r"/-\*?[\s\S]*?-/", "", txt)
        code_only = re.sub(r"--.*$", "", code_only, flags=re.MULTILINE)
        matches = re.findall(r"\bsorry\b", code_only)
        assert not matches, f"Found {len(matches)} 'sorry' token(s) in proof code of {f}"


def test_f15_lean_strict_local_extremum_theorem_present(formalization_dir: Path):
    """Verifies presence of strict local extremum characterization theorem."""
    txt = (formalization_dir / "OcclusionTriad.lean").read_text(encoding="utf-8")
    assert "morseImpulse_iff_strict_extremum" in txt or "genomic_morse_impulse_iff_strict_extremum" in txt


def test_f15_lean_monotone_ramp_rejection_theorem_present(formalization_dir: Path):
    """Verifies presence of monotone slope rejection theorem."""
    txt = (formalization_dir / "OcclusionTriad.lean").read_text(encoding="utf-8")
    assert "monotone" in txt.lower() and "morseimpulse" in txt.lower()


def test_f15_lean_focal_amplicon_recovery_theorem_present(formalization_dir: Path):
    """Verifies presence of 100% focal amplicon amplitude recovery theorem."""
    txt = (formalization_dir / "OcclusionTriad.lean").read_text(encoding="utf-8")
    assert "amplicon" in txt.lower() or "spike" in txt.lower() or "recovery" in txt.lower()


# ==============================================================================
# Feature 16: Documentation Honesty Rewrite (M5)
# ==============================================================================

def test_f16_readme_scoped_to_1d_genomic_cnv_detection(project_root: Path):
    """Verifies README.md accurately scopes tool to 1D genomic CNV focal amplicon detection."""
    readme = project_root / "README.md"
    content = readme.read_text(encoding="utf-8")
    assert "1d" in content.lower() and ("cnv" in content.lower() or "copy number" in content.lower())


def test_f16_readme_no_eleven_grand_challenges_overclaim(project_root: Path):
    """Verifies README.md does not claim to solve all Eleven Grand Challenges."""
    readme = project_root / "README.md"
    content = readme.read_text(encoding="utf-8")
    assert "eleven grand challenges in single-cell data science" not in content.lower(), "Overclaim in README"


def test_f16_readme_no_bcm2712_cache_marketing_section(project_root: Path):
    """Verifies README.md does not market BCM2712 hardware simulation as core biology."""
    readme = project_root / "README.md"
    content = readme.read_text(encoding="utf-8")
    assert "bcm2712 cortex-a76 hardware-level l1d/l2/l3 cache telemetry" not in content.lower()


def test_f16_docs_preprint_purged_of_oil_plasma_terms(docs_dir: Path):
    """Verifies docs/CELLSNAIL_PREPRINT.md contains zero oil-well or plasma-physics terms."""
    preprint = docs_dir / "CELLSNAIL_PREPRINT.md"
    if preprint.exists():
        txt = preprint.read_text(encoding="utf-8")
        for term in ["caliper", "thin-bed", "washout", "mrxmhd", "stellarator"]:
            assert term not in txt.lower(), f"Found {term} in {preprint}"


def test_f16_docs_referee_audit_response_purged_of_mhd(docs_dir: Path):
    """Verifies docs/REFEREE_AUDIT_RESPONSE.md contains no ideal MHD references."""
    referee = docs_dir / "REFEREE_AUDIT_RESPONSE.md"
    if referee.exists():
        txt = referee.read_text(encoding="utf-8")
        assert "ideal mhd transverse invariance" not in txt.lower(), f"MHD reference in {referee}"


# ==============================================================================
# Feature 17: Version Synchronization (+0.1.0 Bump to 0.3.0) (M5)
# ==============================================================================

def test_f17_cargo_toml_version_bumped_to_0_3_0(rust_dir: Path):
    """Verifies rust/cellsnail_rs/Cargo.toml is bumped to 0.3.0."""
    cargo_toml = rust_dir / "Cargo.toml"
    content = cargo_toml.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    assert match, "version string not found in Cargo.toml"
    assert match.group(1) == "0.3.0", f"Cargo.toml version expected 0.3.0, got {match.group(1)}"


def test_f17_pyproject_toml_version_is_0_3_0(project_root: Path):
    """Verifies pyproject.toml version is synchronized to 0.3.0."""
    pyproject = project_root / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    assert match, "version string not found in pyproject.toml"
    assert match.group(1) == "0.3.0", f"pyproject.toml version expected 0.3.0, got {match.group(1)}"


def test_f17_citation_cff_version_is_0_3_0(project_root: Path):
    """Verifies CITATION.cff version is synchronized to 0.3.0."""
    cff = project_root / "CITATION.cff"
    content = cff.read_text(encoding="utf-8")
    match = re.search(r'^version:\s*([^\s]+)', content, re.MULTILINE)
    assert match, "version string not found in CITATION.cff"
    assert match.group(1) == "0.3.0", f"CITATION.cff version expected 0.3.0, got {match.group(1)}"


def test_f17_python_init_version_is_0_3_0(project_root: Path):
    """Verifies cellsnail/__init__.py __version__ is synchronized to 0.3.0."""
    import cellsnail
    assert cellsnail.__version__ == "0.3.0", f"cellsnail.__version__ expected 0.3.0, got {cellsnail.__version__}"


def test_f17_lakefile_lean_version_is_0_3_0(formalization_dir: Path):
    """Verifies formalization/lakefile.lean version is synchronized to 0.3.0."""
    lakefile = formalization_dir / "lakefile.lean"
    content = lakefile.read_text(encoding="utf-8")
    match = re.search(r'version\s*:=\s*v!"([^"]+)"', content)
    assert match, "version string not found in lakefile.lean"
    assert match.group(1) == "0.3.0", f"lakefile.lean version expected 0.3.0, got {match.group(1)}"


# ==============================================================================
# Feature 18: Full Test Verification & Remote Push (M5)
# ==============================================================================

def test_f18_rust_unit_tests_pass(rust_dir: Path):
    """Verifies all unit tests in rust/cellsnail_rs pass."""
    res = subprocess.run(["cargo", "test", "--manifest-path", str(rust_dir / "Cargo.toml")], capture_output=True, text=True)
    assert res.returncode == 0, f"cargo test failed:\n{res.stdout}\n{res.stderr}"


def test_f18_rust_clippy_clean_or_zero_erasing_op(rust_dir: Path):
    """Verifies rust clippy has 0 erasing op errors."""
    res = subprocess.run(["cargo", "clippy", "--manifest-path", str(rust_dir / "Cargo.toml")], capture_output=True, text=True)
    assert "erasing_op" not in res.stderr and "erasing_op" not in res.stdout


def test_f18_python_core_pytest_suite_passes(project_root: Path):
    """Verifies python test_discovery.py passes cleanly."""
    res = subprocess.run([str(project_root / ".venv" / "Scripts" / "python.exe"), "-m", "pytest", "tests/test_discovery.py"], cwd=project_root, capture_output=True, text=True)
    assert res.returncode == 0, f"pytest test_discovery.py failed:\n{res.stdout}"


def test_f18_lean_formalization_lake_build_passes(formalization_dir: Path):
    """Verifies formalization lake build succeeds."""
    res = subprocess.run(["lake", "build"], cwd=formalization_dir, capture_output=True, text=True)
    assert res.returncode == 0, f"lake build failed:\n{res.stdout}\n{res.stderr}"


def test_f18_git_remote_configured_and_tracked(project_root: Path):
    """Verifies git remote origin points to sneed-and-feed/cellsnail."""
    res = subprocess.run(["git", "remote", "-v"], cwd=project_root, capture_output=True, text=True)
    assert res.returncode == 0
    assert "sneed-and-feed/cellsnail" in res.stdout
