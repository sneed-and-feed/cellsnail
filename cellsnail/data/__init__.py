"""cellsnail.data
==============
Empirical wet-lab single-cell cohorts and benchmark datasets.
"""

from cellsnail.data.cohorts import (
    load_ensembl_clinvar_panel,
    load_gbm_cohort,
    load_merfish_spatial,
    load_paired_cnv_cohort,
    load_sc_mixology,
)

__all__ = [
    "load_sc_mixology",
    "load_paired_cnv_cohort",
    "load_merfish_spatial",
    "load_ensembl_clinvar_panel",
    "load_gbm_cohort",
]
