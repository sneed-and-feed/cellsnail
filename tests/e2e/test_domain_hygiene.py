"""Dedicated Domain Hygiene E2E Test Suite.

Audits the entire codebase for complete eradication of:
1. Oil-well borehole logging terminology:
   - "caliper", "washout", "thin stringer", "thin-bed", "Stringer Hunt", "Pan-Autosome Stringer Atlas"
2. Fusion-reactor / plasma-physics terminology:
   - "stellarator", "magnetohydrodynamic", "mhd", "beltrami", "mrxmhd", "cross-helicity", "alfven"
3. Recycled darksnail / triad-core branding and stealth aliases.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple
import pytest

from tests.e2e.conftest import (
    OIL_WELL_TERMS,
    PLASMA_PHYSICS_TERMS,
    PROJECT_ROOT,
    RECYCLED_ECOSYSTEM_TERMS,
)


def _scan_files_for_terms(
    directory: Path,
    extensions: List[str],
    forbidden_terms: List[str],
    excludes: List[str] = None,
) -> List[Tuple[Path, int, str, str]]:
    import os
    base_excludes = {".git", ".venv", "__pycache__", "target", ".agents", ".lake", "e2e", "test_domain_hygiene.py", "conftest.py"}
    if excludes:
        base_excludes.update(excludes)

    ext_set = {f".{ext.lstrip('.')}" for ext in extensions}
    violations = []

    # Compile a single combined regex for 20x performance boost over large JSON artifacts
    combined_pattern = re.compile(
        rf"(?i)(?:^|[^a-zA-Z])({'|'.join(re.escape(t) for t in forbidden_terms)})(?:$|[^a-zA-Z])"
    )

    for root, dirs, files in os.walk(directory):
        # In-place directory pruning prevents entering excluded directories
        dirs[:] = [d for d in dirs if d not in base_excludes]
        for f in files:
            if f in base_excludes:
                continue
            if any(f.endswith(ext) for ext in ext_set):
                file_path = Path(root) / f
                try:
                    content = file_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, PermissionError):
                    continue

                for line_no, line in enumerate(content.splitlines(), start=1):
                    m = combined_pattern.search(line)
                    if m:
                        violations.append((file_path, line_no, m.group(1), line.strip()))

    return violations


def test_domain_hygiene_rust_codebase():
    """Audits rust/cellsnail_rs/ for oil-well, plasma-physics, and darksnail terms."""
    rust_dir = PROJECT_ROOT / "rust" / "cellsnail_rs"
    forbidden = OIL_WELL_TERMS + PLASMA_PHYSICS_TERMS + RECYCLED_ECOSYSTEM_TERMS
    violations = _scan_files_for_terms(rust_dir, ["rs", "h", "toml"], forbidden)
    msg = "\n".join(f"{p}:{ln} [{t}] {line}" for p, ln, t, line in violations)
    assert not violations, f"Domain hygiene violations in Rust codebase ({len(violations)} found):\n{msg}"


def test_domain_hygiene_python_production_code():
    """Audits cellsnail/ production code for oil-well, plasma-physics, and darksnail terms."""
    src_dir = PROJECT_ROOT / "cellsnail"
    forbidden = OIL_WELL_TERMS + PLASMA_PHYSICS_TERMS + RECYCLED_ECOSYSTEM_TERMS
    violations = _scan_files_for_terms(src_dir, ["py"], forbidden)
    msg = "\n".join(f"{p}:{ln} [{t}] {line}" for p, ln, t, line in violations)
    assert not violations, f"Domain hygiene violations in Python production code ({len(violations)} found):\n{msg}"


def test_domain_hygiene_lean_formalization():
    """Audits formalization/ for oil-well and plasma-physics terms."""
    lean_dir = PROJECT_ROOT / "formalization"
    forbidden = OIL_WELL_TERMS + PLASMA_PHYSICS_TERMS + RECYCLED_ECOSYSTEM_TERMS
    violations = _scan_files_for_terms(lean_dir, ["lean"], forbidden)
    msg = "\n".join(f"{p}:{ln} [{t}] {line}" for p, ln, t, line in violations)
    assert not violations, f"Domain hygiene violations in Lean formalization ({len(violations)} found):\n{msg}"


def test_domain_hygiene_documentation():
    """Audits README.md and docs/ for oil-well and plasma-physics terms."""
    docs_dir = PROJECT_ROOT / "docs"
    forbidden = OIL_WELL_TERMS + PLASMA_PHYSICS_TERMS
    violations = _scan_files_for_terms(docs_dir, ["md"], forbidden)
    msg = "\n".join(f"{p}:{ln} [{t}] {line}" for p, ln, t, line in violations)
    assert not violations, f"Domain hygiene violations in documentation ({len(violations)} found):\n{msg}"


def test_domain_hygiene_test_suites():
    """Audits tests/ for oil-well and plasma-physics terms (excluding E2E tests scanner)."""
    tests_dir = PROJECT_ROOT / "tests"
    forbidden = OIL_WELL_TERMS + PLASMA_PHYSICS_TERMS
    violations = _scan_files_for_terms(tests_dir, ["py"], forbidden, excludes=["e2e"])
    msg = "\n".join(f"{p}:{ln} [{t}] {line}" for p, ln, t, line in violations)
    assert not violations, f"Domain hygiene violations in tests/ ({len(violations)} found):\n{msg}"


def test_domain_hygiene_no_stealth_alias_comments_anywhere():
    """Verifies that the phrase 'Stealth Single-Cell Domain Alias' is completely purged."""
    forbidden = ["Stealth Single-Cell Domain Alias", "Stealth domain-native alias"]
    violations = _scan_files_for_terms(PROJECT_ROOT, ["rs", "py", "lean", "h", "md"], forbidden)
    msg = "\n".join(f"{p}:{ln} [{t}] {line}" for p, ln, t, line in violations)
    assert not violations, f"Found stealth alias comments ({len(violations)} found):\n{msg}"


def test_domain_hygiene_no_washout_in_core_filtering():
    """Verifies that 'washout' is completely absent from cellsnail/core/."""
    core_dir = PROJECT_ROOT / "cellsnail" / "core"
    violations = _scan_files_for_terms(core_dir, ["py"], ["washout"])
    msg = "\n".join(f"{p}:{ln} [{t}] {line}" for p, ln, t, line in violations)
    assert not violations, f"Found 'washout' in core filtering ({len(violations)} found):\n{msg}"


def test_domain_hygiene_no_caliper_in_formalization():
    """Verifies that 'caliper' is completely absent from formalization/."""
    lean_dir = PROJECT_ROOT / "formalization"
    violations = _scan_files_for_terms(lean_dir, ["lean"], ["caliper"])
    msg = "\n".join(f"{p}:{ln} [{t}] {line}" for p, ln, t, line in violations)
    assert not violations, f"Found 'caliper' in Lean formalization ({len(violations)} found):\n{msg}"


def test_domain_hygiene_no_mhd_in_benchmarks():
    """Verifies that 'mhd' or 'ideal_mhd' is completely absent from cellsnail/benchmark/."""
    bench_dir = PROJECT_ROOT / "cellsnail" / "benchmark"
    violations = _scan_files_for_terms(bench_dir, ["py"], ["mhd", "ideal_mhd"])
    msg = "\n".join(f"{p}:{ln} [{t}] {line}" for p, ln, t, line in violations)
    assert not violations, f"Found MHD in benchmarks ({len(violations)} found):\n{msg}"


def test_domain_hygiene_no_darksnail_branding_in_manifests():
    """Verifies that Cargo.toml and pyproject.toml have no DarkSnail references."""
    for manifest_path in [PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "rust" / "cellsnail_rs" / "Cargo.toml"]:
        txt = manifest_path.read_text(encoding="utf-8")
        assert "darksnail" not in txt.lower(), f"DarkSnail reference found in {manifest_path}"


def test_domain_hygiene_scripts():
    """Audits scripts/ for oil-well, plasma-physics, and darksnail terms."""
    scripts_dir = PROJECT_ROOT / "scripts"
    forbidden = OIL_WELL_TERMS + PLASMA_PHYSICS_TERMS + RECYCLED_ECOSYSTEM_TERMS
    violations = _scan_files_for_terms(scripts_dir, ["py", "sh"], forbidden)
    msg = "\n".join(f"{p}:{ln} [{t}] {line}" for p, ln, t, line in violations)
    assert not violations, f"Domain hygiene violations in scripts/ ({len(violations)} found):\n{msg}"


def test_domain_hygiene_studio():
    """Audits studio/ for oil-well, plasma-physics, and darksnail terms."""
    studio_dir = PROJECT_ROOT / "studio"
    forbidden = OIL_WELL_TERMS + PLASMA_PHYSICS_TERMS + RECYCLED_ECOSYSTEM_TERMS
    violations = _scan_files_for_terms(studio_dir, ["html", "js", "css"], forbidden)
    msg = "\n".join(f"{p}:{ln} [{t}] {line}" for p, ln, t, line in violations)
    assert not violations, f"Domain hygiene violations in studio/ ({len(violations)} found):\n{msg}"


def test_domain_hygiene_results():
    """Audits results/ for oil-well, plasma-physics, and darksnail terms."""
    results_dir = PROJECT_ROOT / "results"
    forbidden = OIL_WELL_TERMS + PLASMA_PHYSICS_TERMS + RECYCLED_ECOSYSTEM_TERMS
    violations = _scan_files_for_terms(results_dir, ["json"], forbidden)
    msg = "\n".join(f"{p}:{ln} [{t}] {line}" for p, ln, t, line in violations)
    assert not violations, f"Domain hygiene violations in results/ ({len(violations)} found):\n{msg}"


def test_domain_hygiene_readme():
    """Audits README.md for oil-well and plasma-physics terms."""
    readme = PROJECT_ROOT / "README.md"
    forbidden = OIL_WELL_TERMS + PLASMA_PHYSICS_TERMS + RECYCLED_ECOSYSTEM_TERMS
    assert readme.exists()
    content = readme.read_text(encoding="utf-8")
    violations = []
    pattern = re.compile(rf"(?i)(?:^|[^a-zA-Z])({'|'.join(re.escape(t) for t in forbidden)})(?:$|[^a-zA-Z])")
    for line_no, line in enumerate(content.splitlines(), start=1):
        m = pattern.search(line)
        if m:
            violations.append((readme, line_no, m.group(1), line.strip()))
    msg = "\n".join(f"{p}:{ln} [{t}] {line}" for p, ln, t, line in violations)
    assert not violations, f"Domain hygiene violations in README.md ({len(violations)} found):\n{msg}"


def test_domain_hygiene_filenames_and_paths():
    """Verifies that no file or directory anywhere in the repository contains forbidden terms in its name."""
    import os
    base_excludes = {".git", ".venv", "__pycache__", "target", ".agents", ".lake"}
    forbidden = [
        "caliper", "calipers", "washout", "washouts", "stringer", "stringers",
        "thin-bed", "thin-beds", "thin_bed", "thin_beds",
        "stellarator", "stellarators", "mhd", "mhds", "beltrami", "beltramis",
        "mrxmhd", "mrxmhds", "darksnail", "darksnails",
        "triad-core", "triad-cores", "triad_core", "triad_cores", "triadcore",
    ]
    violations = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in base_excludes]
        for name in dirs + files:
            lower = name.lower()
            for term in forbidden:
                if term in lower:
                    violations.append((Path(root) / name, term))
    msg = "\n".join(f"{p} [contains '{t}']" for p, t in violations)
    assert not violations, f"Found forbidden terms in repository file/directory names ({len(violations)} found):\n{msg}"


def test_domain_hygiene_empirical_wetlab_cohorts_continuous_distribution():
    """Verifies cellsnail/data/empirical/wetlab_cohorts.npz:
    1. Exists and is tracked by git.
    2. Has > 1,000 unique continuous copy-number values in cnv_truth (0 integer shortcuts).
    3. Has 0 integer-only bins across all 100 genomic bins.
    4. Residual between observed DNA and truth exhibits no synthetic sine wave correlation (r < 0.20).
    """
    import subprocess
    import numpy as np
    npz_path = PROJECT_ROOT / "cellsnail" / "data" / "empirical" / "wetlab_cohorts.npz"
    assert npz_path.exists(), f"wetlab_cohorts.npz missing at {npz_path}"

    with np.load(npz_path, allow_pickle=False) as data:
        assert "cnv_truth" in data, "cnv_truth missing from wetlab_cohorts.npz"
        assert "cnv_sc_dna_obs" in data, "cnv_sc_dna_obs missing from wetlab_cohorts.npz"
        truth = data["cnv_truth"]
        obs = data["cnv_sc_dna_obs"]

    uniques = np.unique(truth)
    assert len(uniques) > 1000, f"Expected >1000 unique continuous values in cnv_truth, got {len(uniques)}"
    integer_shortcuts = {0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0}
    assert not set(uniques).issubset(integer_shortcuts), "cnv_truth consists entirely of discrete integer shortcuts"

    n_bins = truth.shape[1]
    integer_bins = [b for b in range(n_bins) if np.all(np.equal(truth[:, b], np.round(truth[:, b])))]
    assert len(integer_bins) == 0, f"Found {len(integer_bins)} integer-only bins in cnv_truth: {integer_bins}"

    residual = obs - truth
    sine_wave = (np.sin(np.linspace(0, 4 * np.pi, n_bins)) * 0.25).astype(np.float32)
    mean_profile_r = abs(np.corrcoef(np.mean(residual, axis=0), sine_wave)[0, 1])
    assert mean_profile_r < 0.20, f"Residual exhibits synthetic sine wave correlation: r = {mean_profile_r:.4f} >= 0.20"

    res = subprocess.run(
        ["git", "status", "--porcelain", "--", str(npz_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert not res.stdout.startswith("??"), f"wetlab_cohorts.npz is untracked in git:\n{res.stdout}"

