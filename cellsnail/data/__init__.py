"""cellsnail.data
==============
Curated wet-lab single-cell cohorts and benchmark datasets.
"""

from cellsnail.data.cohorts import (
    load_sc_mixology,
    load_paired_cnv_cohort,
    load_merfish_spatial,
)

__all__ = [
    "load_sc_mixology",
    "load_paired_cnv_cohort",
    "load_merfish_spatial",
]
