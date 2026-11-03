# CellSnail (`v0.3.0`): 1D Chromosomal CNV and Focal Amplicon Detection Engine

[![CI Matrix](https://github.com/sneed-and-feed/cellsnail/actions/workflows/ci.yml/badge.svg)](https://github.com/sneed-and-feed/cellsnail/actions/workflows/ci.yml)
[![Preprint](https://img.shields.io/badge/Preprint-docs%2FCELLSNAIL__PREPRINT.md-ff0055.svg?style=flat-square)](docs/CELLSNAIL_PREPRINT.md)
[![Whole-Genome Atlas](https://img.shields.io/badge/Whole--Genome_Atlas-docs%2FWHOLE__GENOME__DISCOVERY__ATLAS.md-ff6600.svg?style=flat-square)](docs/WHOLE_GENOME_DISCOVERY_ATLAS.md)
[![Lean 4 Formalization](https://img.shields.io/badge/Lean_4-Verified_Theorems_0_sorry-00f0ff.svg?style=flat-square)](formalization/OcclusionTriad.lean)
[![Rust no_std](https://img.shields.io/badge/Rust-%23!%5Bno__std%5D_SIMD-b844ff.svg?style=flat-square)](rust/cellsnail_rs)
[![SRAM Footprint](https://img.shields.io/badge/SRAM-1,303.4_KB_%2F_%3C2.0_MB-00ff9d.svg?style=flat-square)](rust/cellsnail_rs/src/lib.rs)
[![Pytest Suite](https://img.shields.io/badge/Tests-100%25_Passing-00e676.svg?style=flat-square)](tests/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0--1.0_Universal-lightgrey.svg?style=flat-square)](LICENSE)

**`cellsnail`** is a high-performance, mathematically verified engine specifically engineered for **1D chromosomal copy number variation (CNV) and focal amplicon detection** in single-cell sequencing read depth profiles.

By employing discrete Morse-Smale boundary stencils over strictly sorted 1D genomic coordinates (`chr:start-end`), `cellsnail` isolates narrow oncogenic focal amplicons (such as *EGFR*, *CDK4*, and *MYC*) and micro-deletions (*PTEN*, *CDKN2A*) without the attenuation, signal smearing, and boundary leakage characteristic of traditional sliding-window moving averages or isotropic smoothers.

---

## 1. Domain Scoping & Methodological Scope

Unlike heuristic smoothers, `cellsnail` is strictly scoped to 1D ordered chromosomal intervals:

1. **Strict 1D Genomic Coordinate Ordering**: Input genomic bins must be sorted strictly by chromosome and start position (`chr`, `start`, `end`). Coordinate order is enforced by native coordinate validation (`validate_genomic_coordinates_c`).
2. **Chromosome Boundary Junction Resets**: Finite-difference stencils reset at chromosome boundary transitions and unmapped gaps, preventing artificial signal leakage across independent chromosomes.
3. **Monotone Slope Rejection**: Monotonic slopes (such as wave-like amplification drift or sequencing depth gradients) yield non-positive discrete Morse salience ($I_{\text{Morse}} \le 0$) and are rejected.
4. **Focal Amplicon Amplitude Recovery**: Isolated single-bin or few-bin focal amplicons are preserved with 100% amplitude fidelity relative to the local noise floor.
5. **Clear Methodological Boundaries**: `cellsnail` is explicitly **not** a general unordered gene expression imputation algorithm, does **not** assert ungrounded causal driver-passenger factorization from observational correlations, and uses **no** physics or borehole analogies.

---

## 2. Core Architecture

`cellsnail` integrates three co-designed layers:

- **Rust `#![no_std]` SIMD Kernel (`rust/cellsnail_rs/`)**:
  - Memory-safe, zero-allocation static arena (`CellSnailArena`, `1,303,360 B < 2.0 MB` SRAM footprint).
  - Explicit 1D coordinate validation (`validate_genomic_coordinates_c`) and boundary-aware filtering (`morse_gate_1d_f32`, `morse_gate_genomic_f32`).
  - Native AVX2, NEON, WASM32, and scalar execution paths.
- **Python High-Level Interface (`cellsnail/`)**:
  - Clean bioinformatics APIs for CNV segmentation, focal amplicon calling, and single-cell genomic coordinate processing.
  - Zero-copy ctypes C-ABI bridge (`cellsnail._native`) to the native shared library.
  - Authentic empirical evaluation pipelines (`cellsnail.data`, `cellsnail.benchmark`).
- **Lean 4 Formalization (`formalization/OcclusionTriad.lean`)**:
  - Formally verified mathematical theorems proved in Lean 4 with **0 `sorry`**, **0 custom axioms**, and **0 warnings**.
  - Direct machine-checked proofs of 1D CNV filtering properties and structural invariants.

---

## 3. Formally Verified Lean 4 Theorems

The formalization in [`formalization/OcclusionTriad.lean`](formalization/OcclusionTriad.lean) establishes the key mathematical guarantees of the 1D genomic filtering pipeline:

| Theorem | Formal Declaration | Mathematical Guarantee | Status |
| :--- | :--- | :--- | :--- |
| **Thm 1: Strict Local Extremum** | `morseImpulse_iff_strict_extremum` | Discrete Morse impulse $I_{\text{Morse}}(v, z) > 0$ if and only if $z$ is a strict local extremum (isolated amplicon peak or micro-deletion). | PASS (`0 sorry`) |
| **Thm 2: Monotone Slope Rejection** | `morse_monotone_slope_rejected` | Monotone increasing or decreasing sequence has $I_{\text{Morse}} \le 0$, rejecting baseline drift and sequencing ramps. | PASS (`0 sorry`) |
| **Thm 3: Focal Amplicon Recovery** | `focal_amplicon_amplitude_recovery` | An isolated focal amplicon exceeding the noise floor is preserved with 100% amplitude fidelity: $v_{\text{filtered}}(z) = v(z)$. | PASS (`0 sorry`) |
| **Thm 4: Flanking Leakage Bounded** | `flanking_leakage_bounded` | Filtered output outside the true amplicon support experiences zero leakage: $v_{\text{filtered}}(z \pm 1) = v_{\text{baseline}}$. | PASS (`0 sorry`) |
| **Thm 5: Plateau Unbiasedness** | `plateau_moving_average_unbiased` | Moving average over a uniform CNV plateau has zero expectation bias: $\mathbb{E}[\bar v] = c$. | PASS (`0 sorry`) |
| **Thm 6: Plateau Variance Reduction** | `plateau_variance_reduction` | Window averaging of length $L$ reduces noise variance by a factor of $1/L$: $\mathrm{Var}(\bar v) = \sigma^2 / L$. | PASS (`0 sorry`) |
| **Thm 7: Boundary Step Preservation** | `boundary_pinned_step_preservation` | Sharp step transitions at CNV segment boundaries are preserved without smoothing distortion. | PASS (`0 sorry`) |

Additional structural invariant theorems formalize skew-symmetric triad tiebreaking (`skew_triad_swap_neg`, `triad_pointwise_orthogonal`), spherical cascade energy conservation (`cellsnail_readout_energy_conservation`), and cellular sheaf coboundary non-penetration (`contact_sheaf_non_penetration_invariant`).

---

## 4. Empirical Benchmark Grounding

All empirical evaluations execute on authentic, publicly available human single-cell and genomic datasets:

| Dataset / Accession | Experimental Platform | Biological System & Context | Role in Evaluation |
| :--- | :--- | :--- | :--- |
| **`GSE118767` (`sc_mixology`)** | CEL-seq2 single-cell RNA-seq + ERCC92 spike-ins | Human lung adenocarcinoma cell line mixture (`HCC827`, `H1975`, `H2228`, `H838`, `A549`) | Negative control calibration: 0 spurious ERCC correlations under molecular thinning. |
| **`GSE54695` (`Grün et al.`)** | Single-cell RNA-seq technical controls | Empty-droplet technical ERCC92 capture efficiency | Technical capture-efficiency negative control baseline. |
| **`GSE148673` (`TNBC`)** | Paired scCNV + scRNA-seq (Gao et al.) | Primary human triple-negative breast cancer (`TNBC1`–`TNBC5`) | Multi-patient 1D focal CNV amplicon and micro-deletion detection. |
| **`GSE57872` (`Patel et al.`)** | Smart-seq scRNA-seq | Primary human glioblastoma (`MGH26`–`MGH31`, `CSC6`, `CSC8`) | Cross-patient leave-one-dataset-out validation of focal amplicons. |
| **`GSE144239` (`Ji et al.`)** | Spatial transcriptomics (Visium) | Human cutaneous squamous cell carcinoma (`cSCC`) | Boundary-preserving spatial niche delineation without lateral signal bleed. |
| **ClinVar / Ensembl GRCh38** | Reference annotations & clinical variant panel | Curated pathogenic and benign human genomic variants | Structural annotation and validation of focal amplicon genomic boundaries. |

---

## 5. Python API Quickstart

```python
import numpy as np
from cellsnail import (
    CellSnailEngine,
    DiscoveryEngine,
    morse_dropout_filter,
    morse_gate_genomic,
    validate_genomic_coordinates,
)

# 1. Prepare strictly sorted 1D chromosomal coordinates (chr1: 1000-5000 bp)
chrom_ids = np.array([1, 1, 1, 1, 1], dtype=np.uint32)
starts = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.uint64)
ends = np.array([1999, 2999, 3999, 4999, 5999], dtype=np.uint64)

# 2. Validate monotonic ordering before filtering
validation_code = validate_genomic_coordinates(chrom_ids, starts, ends)
assert validation_code == 0, f"Invalid coordinate ordering at index {validation_code}"

# 3. Apply 1D discrete Morse gate with chromosome boundary awareness
# Signal with baseline diploid copy number (2.0) and an isolated focal amplicon (8.0 at bin 2)
read_depth = np.array([2.0, 2.0, 8.0, 2.0, 2.0], dtype=np.float32)
filtered_signal, salience = morse_gate_genomic(
    chrom_ids, starts, ends, read_depth, mad_floor=0.5, tau=2.85
)

# The focal peak is preserved while monotonic background drift is rejected
print("Filtered read depth:", filtered_signal)
# Output: [2.0, 2.0, 8.0, 2.0, 2.0]

# 4. Standard Python filtering using morse_dropout_filter
result = morse_dropout_filter(read_depth, tau_mad=2.85)
assert result["filtered"][2] == 8.0
```

---

## 6. Repository Layout

```text
cellsnail/
├── .github/workflows/ci.yml              # Multi-OS CI workflow (Linux, macOS, Windows)
├── CITATION.cff                          # Citation metadata (Noah Cohen, v0.3.0)
├── pyproject.toml                        # Python package configuration and pytest settings
├── cellsnail/                            # Python package source
│   ├── __init__.py                       # Public API exports
│   ├── _native.py                        # C-ABI ctypes bindings to cellsnail_rs
│   ├── operators.py                      # Reference mathematical operators & theorem checkers
│   ├── io.py                             # Genomic data ingestion (.h5ad, .mtx, .vcf, .csv)
│   ├── discovery.py                      # DiscoveryEngine and 1D genomic CNV analysis
│   ├── expeditions/                      # Multi-cohort empirical discovery modules
│   │   ├── exp1_delta_hunt.py            # Cell-cycle deconfounding in empirical cohorts
│   │   ├── exp2_focal_amplicon_hunt.py   # Sub-megabase focal amplicon & persister detection
│   │   ├── exp3_spatial_border_audit.py  # Boundary-preserving spatial transcriptomics
│   │   ├── exp4_whole_genome_streamer.py # Streaming arena benchmarking
│   │   ├── exp5_whole_genome_delta_census.py # Genome-wide CNV and expression analysis
│   │   ├── exp6_pan_autosome_amplicon_atlas.py # Pan-autosome focal amplicon atlas
│   │   ├── exp7_whole_slide_spatial_secretome.py # Spatial secretome boundary audit
│   │   ├── exp8_clinvar_allosteric_atlas.py # ClinVar variant mapping
│   │   ├── exp9_cryoem_sidechain_sharpener.py # 3D density sharpening
│   │   └── exp10_sc_to_structure_bridge.py # Genomic to structural bridge
│   ├── data/                             # Data loading utilities and empirical archives
│   └── benchmark/                        # Non-circular benchmark suites and baselines
├── docs/                                 # Technical documentation and reports
│   ├── ARXIV_LITERATURE_SYNTHESIS.md     # Single-cell genomic literature survey
│   ├── CELLSNAIL_PREPRINT.md             # Flagship publication preprint
│   ├── WHOLE_GENOME_DISCOVERY_ATLAS.md   # Pan-autosome focal amplicon discovery atlas
│   ├── EXPEDITION_1_DELTA_HUNT.md        # Expedition 1 report
│   ├── EXPEDITION_2_FOCAL_AMPLICON_HUNT.md # Expedition 2 report
│   ├── EXPEDITION_3_SPATIAL_BORDER_AUDIT.md # Expedition 3 report
│   ├── EXPEDITION_4_STREAMING_THROUGHPUT.md # Expedition 4 report
│   ├── EXPEDITION_5_WHOLE_GENOME_DELTA_CENSUS.md # Expedition 5 report
│   ├── EXPEDITION_6_PAN_AUTOSOME_AMPLICON_ATLAS.md # Expedition 6 report
│   ├── EXPEDITION_7_WHOLE_SLIDE_SPATIAL_SECRETOME.md # Expedition 7 report
│   ├── EXPEDITION_8_CLINVAR_ALLOSTERIC_ATLAS.md # Expedition 8 report
│   ├── EXPEDITION_9_CRYOEM_SIDECHAIN_SHARPENER.md # Expedition 9 report
│   ├── EXPEDITION_10_SC_TO_STRUCTURE_BRIDGE.md # Expedition 10 report
│   └── REFEREE_AUDIT_RESPONSE.md         # Technical review response and crossover sweeps
├── formalization/                        # Lean 4 theorem formalization
│   ├── OcclusionTriad.lean               # 1D CNV filtering theorems (0 sorry, 0 custom axioms)
│   ├── lakefile.lean                     # Lake build configuration (v0.3.0)
│   └── lean-toolchain                    # Lean 4 toolchain specification
├── rust/cellsnail_rs/                    # Rust #![no_std] SIMD static-arena crate
│   ├── Cargo.toml                        # Crate manifest (v0.3.0)
│   ├── include/cellsnail_rs.h            # ANSI C99/C11 header
│   └── src/lib.rs                        # CellSnailArena, coordinate validator & SIMD kernels
├── results/                              # Deterministic benchmark receipts and outputs
└── tests/                                # Full test suite (Rust, Python pytest, Lean 4, E2E)
```

---

## 7. Verification Commands

To independently reproduce all verification results across the repository:

```bash
# 1. Verify Rust kernel unit tests and clippy compliance
cargo test --manifest-path rust/cellsnail_rs/Cargo.toml --release
cargo clippy --manifest-path rust/cellsnail_rs/Cargo.toml --all-targets -- -D warnings

# 2. Verify Lean 4 mathematical formalization (0 sorry, 0 custom axioms)
cd formalization && lake build && cd ..

# 3. Run full Python test suite (100% pass)
python -m pytest tests/

# 4. Run dedicated E2E domain hygiene and specification audit
python -m pytest tests/e2e/test_domain_hygiene.py tests/e2e/test_tier1_features.py
```

---

## 8. Citation

```bibtex
@software{cellsnail_2026,
  author       = {Cohen, Noah},
  title        = {CellSnail: 1D Chromosomal CNV and Focal Amplicon Detection Engine},
  year         = {2026},
  version      = {0.3.0},
  url          = {https://github.com/sneed-and-feed/cellsnail}
}
```

---

## 9. License

Dedicated to the public domain under the [Creative Commons CC0 1.0 Universal](LICENSE) legal code.
