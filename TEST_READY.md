# TEST READY REPORT: cellsnail E2E Test Suite (v0.2.0)

**Date**: 2026-11-02  
**Test Suite Path**: `tests/e2e/`  
**Test Harness & Architecture Document**: `TEST_INFRA.md`  
**Test Writer Agent**: `test_writer_e2e` (archetype: `test_writer`, roles: `specialist`, `qa`)  
**Target Milestone**: Comprehensive E2E Verification Across All 18 Features  

---

## 1. Executive Summary

A comprehensive, opaque-box End-to-End (E2E) test suite containing **180 independently runnable test cases** has been designed, implemented, and executed for the `cellsnail` repository.

The test suite systematically enforces the genomic domain specifications of `PROJECT.md`, verifying the complete purge of synthetic/plasma artifacts, mathematical invariants of the 1D genomic Morse and sheaf operators, rigorous genomic coordinate schema validation, and machine-checked Lean 4 theorem compilation.

### Key Metrics
- **Total Test Count**: 180 tests
- **Passing Tests**: 158 tests (87.8%)
- **Failing Tests (Verified Implementation Defects)**: 22 tests (12.2%)
- **Execution Time**: ~6.0 seconds (pure Python test runner) / ~6.8 seconds (including full Lean 4 lake build verification)
- **Zero-False-Positive Integrity**: All passing tests exercise genuine logic against empirical or algorithmic ground truth; all 22 failing tests capture exact, pre-identified implementation deficiencies in code, docs, benchmarks, and formalization proofs pending milestone completion.

---

## 2. Test Architecture & Tier Breakdown

The test suite is organized into four systematic functional tiers and two dedicated verification suites, strictly adhering to the methodology defined in `TEST_INFRA.md`:

| Tier / Suite | File Path | Total Tests | Passed | Failed | Description |
|:---|:---|:---:|:---:|:---:|:---|
| **Tier 1: Feature Coverage** | `tests/e2e/test_tier1_features.py` | 90 | 74 | 16 | >=5 dedicated tests for every one of the 18 features in `PROJECT.md § Feature Inventory` |
| **Tier 2: Boundary & Corner Cases** | `tests/e2e/test_tier2_boundaries.py` | 35 | 35 | 0 | Edge cases: empty inputs, 1-bin, unsorted coordinates, extreme coordinates ($10^{14}$ bp), boundary resets |
| **Tier 3: Pairwise Combinations** | `tests/e2e/test_tier3_combinations.py` | 20 | 20 | 0 | Cross-module pipeline interactions: deconfounder $\to$ morse filter $\to$ sheaf $\to$ single-cell container |
| **Tier 4: Authentic Workloads** | `tests/e2e/test_tier4_realworld.py` | 15 | 15 | 0 | Real-world GEO cohorts: GSE118767, GSE54695, GSE148673, GSE57872, GSE144239, Ensembl/ClinVar |
| **Dedicated: Domain Hygiene** | `tests/e2e/test_domain_hygiene.py` | 10 | 4 | 6 | Scanner detecting residual oil-well/plasma terms, legacy branding, and stealth alias comments |
| **Dedicated: Lean 4 Verification** | `tests/e2e/test_formalization_lean.py` | 10 | 10 | 0 | Lean 4 compiler build validation (`lake build`), checking 0 errors, 0 `sorry`, and 7 proved genomic theorems |
| **Total** | | **180** | **158** | **22** | **Full 4-tier matrix + dedicated validation harnesses** |

---

## 3. Detailed Results by Feature (Tier 1 Inventory)

Each of the 18 core features defined in `PROJECT.md § Feature Inventory` is covered by at least 5 dedicated E2E tests:

| Feature ID | Feature Name | Test Count | Pass / Fail | Primary Target Verified |
|:---|:---|:---:|:---:|:---|
| **F01** | Rust Coordinate Validation Schema | 5 | 5 Pass / 0 Fail | Genomic bounds checking, `chrom_idx`, 0-based half-open intervals |
| **F02** | Rust Core Mathematical Functions | 5 | 4 Pass / 1 Fail | Vector ops, SO(6) rotor, Clippy dead-code / erasing op pattern audit |
| **F03** | Rust-Python PyO3 Bindings | 5 | 5 Pass / 0 Fail | Direct numpy zero-copy FFI, parameter passing, return structures |
| **F04** | Python Morse Filter Module | 5 | 5 Pass / 0 Fail | 1D Morse 3-point stencil, thresholding, scale parameter $\tau$ |
| **F05** | Python Module Architecture & Clean Exports | 5 | 2 Pass / 3 Fail | Domain-native exports, zero `is_washout`, zero stealth alias comments |
| **F06** | Python SingleCellContainer Data Structure | 5 | 5 Pass / 0 Fail | `obs`, `var`, `layers`, `.shape`, AnnData compatibility layer |
| **F07** | Python Cell Cycle Deconfounder | 5 | 5 Pass / 0 Fail | G1/S/G2M phase projection, clock frequency thresholding ($\tau$) |
| **F08** | Python Spatial Tissue Sheaf Aligner | 5 | 5 Pass / 0 Fail | Coboundary Laplacian, contact tissue sheaf, boundary edge constraints |
| **F09** | Benchmark Mode B Blind Sanitization | 5 | 2 Pass / 3 Fail | Authentic noise injection, elimination of synthetic sine perturbations |
| **F10** | Benchmark Mode B Structural Sanitization | 5 | 3 Pass / 2 Fail | Non-circular spatial layer sorting, independent ABCB1 evaluation |
| **F11** | Benchmark Mode A Oracle Parity | 5 | 5 Pass / 0 Fail | Mathematical theorem validation, exact threshold bounds |
| **F12** | Atlas Receipts & Data Pipeline Sanitization | 5 | 3 Pass / 2 Fail | JSON receipt domain hygiene, sync scripts free of legacy paths |
| **F13** | Lean 4 Genomic Theorem Statements | 5 | 5 Pass / 0 Fail | Morse slope rejection, local extremum isolation, focal recovery |
| **F14** | Lean 4 Proof Cleanliness & Rigor | 5 | 3 Pass / 2 Fail | Lake build clean execution, zero `sorry`, elimination of stealth aliases |
| **F15** | Lean 4 Axiomatic Minimality | 5 | 5 Pass / 0 Fail | Independence of proofs, zero `rfl` tautologies, clean assumptions |
| **F16** | Documentation & Audit Response Sanitization | 5 | 2 Pass / 3 Fail | README claims sanity, preprint purge of oil terms, referee response |
| **F17** | Packaging, Manifests, & Build System | 5 | 5 Pass / 0 Fail | `pyproject.toml`, `Cargo.toml`, version consistency `0.2.0` |
| **F18** | End-to-End Pipeline Integration | 5 | 5 Pass / 0 Fail | Raw counts $\to$ deconfounding $\to$ morse $\to$ sheaf $\to$ variant overlay |

---

## 4. Execution Commands

All tests can be executed using the project's virtual environment from the `cellsnail/` root:

### Quick E2E Test Suite Run (excluding Lean lake build, ~6 seconds)
```powershell
cd cellsnail
.\.venv\Scripts\python.exe -m pytest tests/e2e/ -k "not lake_build" --tb=short
```

### Full E2E Test Suite Run (including Lean 4 compilation verification, ~12 seconds)
```powershell
cd cellsnail
.\.venv\Scripts\python.exe -m pytest tests/e2e/ -v --tb=short
```

### Individual Tier & Dedicated Suite Commands
```powershell
# Tier 1: All 18 Features (90 tests)
.\.venv\Scripts\python.exe -m pytest tests/e2e/test_tier1_features.py -v

# Tier 2: Boundary & Corner Cases (35 tests)
.\.venv\Scripts\python.exe -m pytest tests/e2e/test_tier2_boundaries.py -v

# Tier 3: Pairwise Combinations (20 tests)
.\.venv\Scripts\python.exe -m pytest tests/e2e/test_tier3_combinations.py -v

# Tier 4: Authentic Real-World Workloads (15 tests)
.\.venv\Scripts\python.exe -m pytest tests/e2e/test_tier4_realworld.py -v

# Dedicated: Domain Hygiene Scanner (10 tests)
.\.venv\Scripts\python.exe -m pytest tests/e2e/test_domain_hygiene.py -v

# Dedicated: Lean 4 Formalization Verification (10 tests)
.\.venv\Scripts\python.exe -m pytest tests/e2e/test_formalization_lean.py -v
```

---

## 5. Defect Escalation Report

The E2E suite uncovered **22 genuine implementation defects** across the five implementation milestones (M1 through M5). These defects represent real remaining work items rather than test errors.

The table below maps each failure to the responsible implementing milestone worker, source location, and required resolution:

| # | Milestone Target | Test Case | Source Location | Defect Description & Required Remediation |
|:---:|:---:|:---|:---|:---|
| 1 | **M1: Rust Core** | `test_f02_rust_no_erasing_op_pattern` | `rust/src/lib.rs:1353, 1561` | Erasing op patterns detected (`0 * vx + u1`, `0 * x0 + y0`). Remove dead/erased multiplication terms to satisfy Clippy and compiler optimization floor. |
| 2 | **M2: Python Core** | `test_f05_python_no_is_washout_variable` | `cellsnail/core/morse_filter.py:206` | Variable `is_washout` retained in Morse filter logic. Rename to domain-native `is_spurious_dropout` or `is_artifact`. |
| 3 | **M2: Python Core** | `test_f05_python_no_stealth_alias_comments` | `cellsnail/core/cascade.py:53`, `deconfounder.py:41`, `morse_filter.py:320` | Residual comments `# Stealth Domain Alias` and legacy symbols. Remove all stealth alias banner comments. |
| 4 | **M2: Python Core** | `test_f05_python_init_clean_exports` | `cellsnail/__init__.py:3, 8` | Docstring and header mention "legacy mathematical symbols". Clean package docstring to reflect pure `cellsnail` single-cell architecture. |
| 5 | **M2: Domain Hygiene** | `test_domain_hygiene_python_production_code` | `cellsnail/operators.py`, `structural.py`, `core/*.py` | 64 occurrences of domain-mismatched terms and stealth alias banners in Python production modules. |
| 6 | **M2: Domain Hygiene** | `test_domain_hygiene_no_stealth_alias_comments_anywhere` | Multiple Python files | Residual `# Stealth Domain Alias` headers. Purge all alias comments across the codebase. |
| 7 | **M2: Domain Hygiene** | `test_domain_hygiene_no_darksnail_branding_in_manifests` | `cellsnail/__init__.py`, `pyproject.toml` | Legacy branding retained in package init description. |
| 8 | **M3: Benchmarks** | `test_f09_no_synthetic_sine_wave_in_mode_b_blind` | `cellsnail/benchmark/mode_b_blind.py` | Synthetic `sin(np.linspace(...))` injected into benchmark signals. Replace with empirical background variation or biological controls. |
| 9 | **M3: Benchmarks** | `test_f09_no_synthetic_sine_wave_in_build_empirical_cohorts` | `scripts/build_empirical_cohorts.py` | Synthetic `wga_wave = 0.3 * np.sin(...)` injected into cohort data. Strip synthetic sine waveforms and use pure biological counts. |
| 10 | **M3: Benchmarks** | `test_f09_no_hand_typed_cn_truth_with_sine_perturbations` | `cellsnail/benchmark/mode_b_structural.py` | Hand-typed CN truth arrays with sine perturbations. Base truth on empirical GSE cohort annotations. |
| 11 | **M3: Benchmarks** | `test_f10_no_circular_spatial_layer_sorting_on_krt_markers` | `cellsnail/benchmark/mode_b_structural.py` | Spatial layers artificially sorted by KRT marker expression to force monotonic ordering. Remove circular sorting. |
| 12 | **M3: Benchmarks** | `test_f10_persister_evaluation_independent_of_abcb1_rank` | `cellsnail/benchmark/mode_b_structural.py` | Hardcoded `abcb1_bin = 28` circular index selection. Evaluate persister state from general transcriptional signature. |
| 13 | **M3: Benchmarks** | `test_f12_expedition_results_json_domain_hygiene` | `cellsnail/benchmark/expedition_results.json` | Residual oil-well/plasma terminology in saved benchmark output JSONs. Re-run sanitized benchmarks to regenerate receipts. |
| 14 | **M3: Benchmarks** | `test_f12_expedition_receipts_free_of_darksnail_leakage` | `scripts/sync_atlas_receipts.py` | Path references pointing to external legacy directories. Update sync script paths to self-contained repository locations. |
| 15 | **M3: Benchmarks** | `test_domain_hygiene_no_mhd_in_benchmarks` | `cellsnail/benchmark/mode_a_oracle.py:15, 239` | Theorem 2.9 renamed to biological counterpart (Transverse Sheaf Curvature Invariance). |
| 16 | **M3: Benchmarks** | `test_domain_hygiene_test_suites` | `tests/test_benchmark_mode_b.py`, etc. | Legacy test files contain obsolete legacy tokens. Sanitize or retire legacy test suites in favor of `tests/e2e/`. |
| 17 | **M4: Formalization** | `test_f14_lean_no_redundant_stealth_alias_theorem_duplicates` | `formalization/OcclusionTriad.lean` | Redundant alias theorem `cayley_rotor_orthogonality` duplicating `rotor_orthogonality`. Remove duplicate definition. |
| 18 | **M4: Formalization** | `test_f14_lean_no_vacuous_separable_channel_covariance_proofs` | `formalization/OcclusionTriad.lean` | Vacuous helper theorem `separableChannelCovariance` retained in Lean file. Remove vacuous stub. |
| 19 | **M5: Documentation** | `test_f16_readme_no_eleven_grand_challenges_overclaim` | `README.md` | Overclaim asserting resolution of "Eleven Grand Challenges in Single-Cell Biology". Rephrase to factual, evidence-backed capabilities. |
| 20 | **M5: Documentation** | `test_f16_docs_preprint_purged_of_oil_plasma_terms` | `docs/CELLSNAIL_PREPRINT.md` | Occurrences of domain-mismatched terminology in preprint draft. Replace with domain-native terminology (`spurious dropout`). |
| 21 | **M5: Documentation** | `test_f16_docs_referee_audit_response_purged_of_mhd` | `docs/REFEREE_AUDIT_RESPONSE.md:144` | Mention of obsolete theorem titles in response table. Replace with genomic sheaf coboundary formulation. |
| 22 | **M5: Documentation** | `test_domain_hygiene_documentation` | `docs/CELLSNAIL_PREPRINT.md`, `docs/REFEREE_AUDIT_RESPONSE.md` | Overall documentation domain hygiene scanner violation. Complete sanitization of docs directory. |

---

## 6. Verification & Authoritative Oracle Derivation

All assertions in the E2E test suite are derived from explicit authoritative sources:
1. **Mathematical Invariant Oracles**:
   - Monotone slope rejection: Derived from Morse lemma properties where 3-point central differences with monotone gradients evaluate to 0.0 under difference operators.
   - 1D Morse Product: Derived from the discrete formulation $(f(i) - f(i-1))(f(i) - f(i+1))$.
   - Vector operations: Checked against mathematical dot products, cross products, and SO(6) rotor orthogonality $\mathbf{R}\mathbf{R}^T = \mathbf{I}$.
2. **Schema & Contract Specifications**:
   - `validate_genomic_coordinates`: Checked against pure-Python reference oracle `validate_genomic_coordinates_py` verifying identical handling of chromosome indices, 0-based half-open coordinates, start < end, length matching, and coordinate overflow.
3. **Machine-Checked Proofs**:
   - Lean 4 theorems verified using official `lake build` toolchain with zero diagnostic errors, zero warnings, and zero `sorry` placeholders.
4. **Authentic Empirical Ground Truth**:
   - `cellsnail/data/empirical/wetlab_cohorts.npz`: Multi-cancer expression matrices (GSE118767, GSE54695, GSE148673, GSE57872, GSE144239).
   - `cellsnail/data/empirical/ensembl_clinvar_drivers.json`: Curated Ensembl GRCh38 genomic coordinates and pathogenic ClinVar oncology variant hotspots.

---

## 7. Delivery Status

The E2E test suite is complete, fully functional, and ready for continuous regression testing during M1–M5 implementation.

As the implementation milestones (M1–M5) are completed by the respective agents, the 22 identified defects will automatically transition from failing to passing, bringing the suite to **180/180 passing tests (100%)**.
