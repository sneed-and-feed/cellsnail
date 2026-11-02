"""End-to-end (E2E) testing track for cellsnail.

Tiers:
  - Tier 1: Feature coverage (>=5 tests per feature across all 18 features)
  - Tier 2: Boundary & corner cases (empty, single-bin, huge coords, unsorted, resets)
  - Tier 3: Cross-feature combinations (pairwise matrix across modules)
  - Tier 4: Real-world application scenarios (GSE118767, GSE148673, GSE57872, GSE144239, ClinVar)
  - Dedicated: Domain hygiene and Lean 4 formalization verification
"""
