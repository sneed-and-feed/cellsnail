# Test Infrastructure Specification: cellsnail E2E Testing Track

**Project**: `cellsnail`  
**Version**: `0.3.0`  
**Test Track**: End-to-End (E2E) Verification Track (Tiers 1–4)  
**Authoritative Specifications**: `PROJECT.md`, `ORIGINAL_REQUEST.md`, Survey Reports  
**Location**: `tests/e2e/`  

---

## 1. Executive Summary & Architectural Overview

The `cellsnail` E2E testing infrastructure is designed as an independent, requirement-driven, opaque-box test harness that exhaustively validates all 18 features, interface contracts, mathematical properties, and empirical datasets defined in `PROJECT.md § Feature Inventory`.

The test suite enforces:
1. **Domain Hygiene**: Complete purge of petroleum borehole logging petroleum logging and plasma-physics terms and stealth alias patterns across Rust, Python, Lean 4, and documentation.
2. **1D Genomic Coordinate Invariants**: Strict verification that all 1D discrete Morse innovation gating and multi-scale plateau filtering operate on coordinate-sorted features (`chr:start-end`) with finite-difference stencil decoupling across chromosome boundaries.
3. **Empirical & Non-Circular Integrity**: Total elimination of synthetic sine wave perturbations (`sin(linspace(...)) * 0.25`), hand-typed CNV ground truth, and circular marker-gene persister sorting, grounding all evaluation in authentic NCBI GEO datasets (`GSE118767`, `GSE54695`, `GSE148673`, `GSE57872`, `GSE144239`, ClinVar).
4. **Lean 4 Proof Verification**: Verification of machine-checked proofs with zero `sorry` placeholders, zero trivial `rfl` tautologies, and zero unlinked hardcoded tables.

---

## 2. Directory Layout & Module Organization

```
cellsnail/
├── tests/
│   ├── e2e/
│   │   ├── __init__.py                     # E2E test suite package descriptor
│   │   ├── conftest.py                     # Shared fixtures, signal generators, validation oracles
│   │   ├── test_tier1_features.py          # Tier 1: 90 feature tests (>= 5 tests per feature across all 18 features)
│   │   ├── test_tier2_boundaries.py        # Tier 2: 35 boundary and corner case tests
│   │   ├── test_tier3_combinations.py      # Tier 3: 20 pairwise cross-feature combination tests
│   │   ├── test_tier4_realworld.py         # Tier 4: 15 authentic GEO / ClinVar real-world workload tests
│   │   ├── test_domain_hygiene.py          # 10 comprehensive domain hygiene scanner tests
│   │   └── test_formalization_lean.py      # 10 Lean 4 formalization verification tests
├── TEST_INFRA.md                           # This architecture and test infrastructure document
└── TEST_READY.md                           # Test execution results, pass/fail status, and defect escalation
```

---

## 3. Four-Tier Testing Methodology

### Tier 1: Feature Coverage (>= 5 Test Cases per Feature, 90 Tests Total)

Every feature in `PROJECT.md § Feature Inventory` is mapped to at least 5 isolated, verifiable test cases:

| Feature # | Feature Name | Target Milestone | Test Function Prefix | Coverage Description |
|---|---|---|---|---|
| F1 | Rust Domain Hygiene Purge | M1 | `test_f01_` | Purge of domain-mismatched terms and stealth aliases from `lib.rs` and header |
| F2 | Rust Clippy Erasing Op Fix | M1 | `test_f02_` | Elimination of `0 * n + j`, verification of `cargo test`, `cargo check`, clippy clean build |
| F3 | Rust Genomic Coordinate Validation | M1 | `test_f03_` | Verification of monotonic sorting, inverted span rejection, chromosome order, boundary resets |
| F4 | Rust C-ABI Clean Export & Tests | M1 | `test_f04_` | Static arena size (1,303,360 B), 64-byte alignment, Morse gate export, isolated peak preservation |
| F5 | Python Domain Hygiene Purge | M2 | `test_f05_` | Eradication of domain-mismatched symbols, stealth aliases, and legacy branding |
| F6 | Python Coordinate Sorting & Boundary Reset | M2 | `test_f06_` | Coordinate validation, chromosome boundary break enforcement, column permutation invariance |
| F7 | Eliminate Ungrounded Causal & Imputation Claims | M2 | `test_f07_` | Scoping APIs and docstrings strictly to 1D CNVs; purging causal driver and general imputation claims |
| F8 | Python FFI Synchronization | M2 | `test_f08_` | Parity between Python and native C-ABI Morse filter, deconfounder, and arena geometry |
| F9 | Eliminate Synthetic Sine & Spike Shortcuts | M3 | `test_f09_` | Eradication of `sin(linspace(...))`, hand-typed CN truth matrices, and artificial offset hacks |
| F10 | Eliminate Circular Subclone/Persister Sorting | M3 | `test_f10_` | Eradication of marker-gene persister ranking (*ABCB1*, *AXL*, *EGFR*, *CDK4*) and coordinate overwriting |
| F11 | Empirical Benchmark Grounding | M3 | `test_f11_` | Validation of authentic integer counts and metadata in GSE118767, GSE54695, GSE148673, GSE57872, GSE144239 |
| F12 | Rename/Sanitize Expedition Suites | M3 | `test_f12_` | Sanitization of `exp2` and `exp6` suites to remove domain-mismatched jargon |
| F13 | Lean 4 Domain Hygiene Purge | M4 | `test_f13_` | Purge of domain-mismatched theorems from Lean proofs |
| F14 | Lean 4 Tautology & False Claim Removal | M4 | `test_f14_` | Elimination of `rfl` tautologies, hardcoded SRAM tables by `decide`, and pseudo-causal claims |
| F15 | Lean 4 1D CNV Filtering Formalization | M4 | `test_f15_` | Compilation with `lake build` (0 errors, 0 `sorry`), strict extremum, monotone slope, 100% amplicon recovery |
| F16 | Documentation Honesty Rewrite | M5 | `test_f16_` | Honest rewrite of `README.md`, `docs/`, purging Eleven Grand Challenges and BCM2712 hardware marketing |
| F17 | Version Synchronization (+0.1.0 Bump) | M5 | `test_f17_` | Uniform version alignment to `0.3.0` across Cargo.toml, pyproject.toml, CITATION.cff, __init__.py, lakefile.lean |
| F18 | Full Test Verification & Remote Push | M5 | `test_f18_` | Verification that `cargo test`, `cargo clippy`, `pytest`, `lake build` pass and remote git origin is configured |

---

### Tier 2: Boundary & Corner Cases (35 Tests Total)

Exhaustively probes failure modes, boundary limits, and edge conditions:
- **Empty Inputs (`test_b01`–`test_b05`)**: 0-length profiles, 0-length intervals, 0-cell containers.
- **Single-Bin and Two-Bin Chromosomes (`test_b06`–`test_b10`)**: Minimal stencils, absence of spurious interior peaks, two-bin step transitions.
- **Huge Genomic Coordinates (`test_b11`–`test_b15`)**: Telomeric coordinates (248.9 Mb on chr1), sub-megabase to whole-chromosome spans, uint64 boundaries.
- **Inverted and Unsorted Coordinates (`test_b16`–`test_b20`)**: `start > end`, `start == end`, unsorted coordinates within chromosome, interleaved chromosomes (`chr1 -> chr2 -> chr1`), descending chromosome order.
- **Chromosome Boundary Transitions (`test_b21`–`test_b25`)**: Stencil decoupling across chromosome breaks, boundary-pinned amplitude preservation without neighbor bleed.
- **Amplitude and Noise Extremes (`test_b26`–`test_b35`)**: Zero-variance flat signals, strictly increasing linear ramps, strictly decreasing linear ramps, 1000x extreme focal spikes, deep homozygous deletions ($CN=0.0$), infinite/NaN values, and high MAD floor thresholds.

---

### Tier 3: Cross-Feature Combinations (Pairwise Coverage, 20 Tests Total)

Validates interactions across module boundaries:
- **`test_c01`**: 1D Morse filter + Multi-scale plateau smoothing across multiple chromosomes.
- **`test_c02`**: Genomic coordinate sorting + Chromosome boundary reset.
- **`test_c03`**: Genomic interval validation + Adaptive MAD floor thresholding.
- **`test_c04`**: 1D filtering + SingleCellContainer streaming I/O.
- **`test_c05`**: Python FFI + Native Rust C-ABI execution.
- **`test_c06`**: Binomial count splitting + 1D Morse filtering on downsampled partitions.
- **`test_c07`**: Plateau edge detection + Boundary-pinned step amplitude preservation.
- **`test_c08`**: High-level DiscoveryEngine + Genomic interval metadata + Batch processing.
- **`test_c09`**: Domain hygiene audit across Python, Rust, and Lean simultaneously.
- **`test_c10`–`test_c11`**: Lean 4 mathematical property verification mapped to Python and Rust outputs.
- **`test_c12`–`test_c13`**: Version manifest consistency + Documentation API parity.
- **`test_c14`–`test_c20`**: Poisson noise variance reduction, variable MAD sweeps, and whole-genome 24-chromosome pipelines.

---

### Tier 4: Real-World Application Workloads (15 Tests Total)

Executes genuine, non-circular bioinformatics workflows on authenticated NCBI GEO datasets:
- **GEO GSE118767 (`test_rw01`–`test_rw02`)**: Single-cell mixology CEL-seq2 lung adenocarcinoma lines (HCC827, H1975, H2228), validating EGFR focal amplification in HCC827 without synthetic offsets.
- **GEO GSE54695 (`test_rw03`–`test_rw04`)**: ERCC92 synthetic spike-in negative control baseline under well-permutation decoupling (true biological correlation = 0.000).
- **GEO GSE148673 (`test_rw05`–`test_rw06`)**: CopyKAT primary triple-negative breast cancer (TNBC3) scRNA-seq expression distribution.
- **GEO GSE57872 (`test_rw07`–`test_rw08`)**: Primary human glioblastoma SMART-seq single-cell mosaic focal amplifications (EGFR, PDGFRA, MET).
- **GEO GSE144239 (`test_rw09`–`test_rw10`)**: Cutaneous squamous cell carcinoma (cSCC) spatial transcriptomics with calibrated physical tissue coordinates $(x, y)$ in micrometers.
- **Ensembl GRCh38 & ClinVar/COSMIC Panel (`test_rw11`–`test_rw12`)**: Canonical oncogene intervals and pathogenic hotspot variant records.
- **Non-Circular Benchmark Workflows (`test_rw13`–`test_rw15`)**: Lineage separation and amplicon calling on objective experimental labels.

---

## 4. Test Execution Instructions

### Complete E2E Suite Execution
```powershell
cd cellsnail
.\.venv\Scripts\python.exe -m pytest tests/e2e/ -v
```

### Execution by Tier
```powershell
# Tier 1: Feature Coverage (90 tests)
.\.venv\Scripts\python.exe -m pytest tests/e2e/test_tier1_features.py -v

# Tier 2: Boundary & Corner Cases (35 tests)
.\.venv\Scripts\python.exe -m pytest tests/e2e/test_tier2_boundaries.py -v

# Tier 3: Pairwise Combinations (20 tests)
.\.venv\Scripts\python.exe -m pytest tests/e2e/test_tier3_combinations.py -v

# Tier 4: Real-World GEO Workloads (15 tests)
.\.venv\Scripts\python.exe -m pytest tests/e2e/test_tier4_realworld.py -v

# Dedicated Domain Hygiene Audit (10 tests)
.\.venv\Scripts\python.exe -m pytest tests/e2e/test_domain_hygiene.py -v

# Dedicated Lean 4 Formalization Verification (10 tests)
.\.venv\Scripts\python.exe -m pytest tests/e2e/test_formalization_lean.py -v
```

---

## 5. Authoritative Expected Output Derivation

For every test in the suite, expected outputs are grounded in explicit mathematical or empirical standards:
1. **Discrete Morse Innovation Gate**: $I_{\text{Morse}}(v, z) = (v(z) - v(z-1))(v(z) - v(z+1))$.
   - If $v$ is locally monotone, $I_{\text{Morse}} \le 0 \implies \text{output} = 0.0$.
   - If $v$ is an isolated impulse of height $\delta > 0$ over background $v_{\text{bg}}$, $I_{\text{Morse}} = \delta^2 > 0 \implies \text{output} = v_{\text{bg}} + \delta$.
   - Flanking bins $z \pm 1$ yield $I_{\text{Morse}} = 0 \implies \text{output} = 0.0$ (zero leakage).
2. **Genomic Coordinate Ordering**: Lexicographical order on $(\text{chrom\_id}, \text{start})$, requiring $\text{start} < \text{end}$.
3. **Plateau Moving Average**: For plateau of length $L$, mean $\mu = c$, noise variance reduction factor $1 / L$, with step edge boundary preserved identically without attenuation.
4. **Authentic Public Data**: Non-negative integer UMI counts matching GEO accession records.
