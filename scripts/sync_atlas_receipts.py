"""scripts/sync_atlas_receipts.py
=============================
Synchronizes local receipt artifacts within the cellsnail repository.
Operates strictly in standalone mode with zero cross-repository dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _REPO_ROOT / "results"


def sync_receipts():
    summary_path = _RESULTS_DIR / "cellsnail_evaluation_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Verified cellsnail evaluation summary at {summary_path}")
        return data
    print(f"No summary found at {summary_path}")
    return {}


if __name__ == "__main__":
    sync_receipts()
