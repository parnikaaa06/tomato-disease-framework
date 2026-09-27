# Dataset report

This folder contains the verified Phase 1 dataset lineage summaries and
annotation-feasibility checks.

## Authoritative near-duplicate result

The authoritative final near-duplicate result is **65 candidate pairs**. Its
distance distribution is:

- distance 0: 14
- distance 1: 2
- distance 2: 7
- distance 3: 10
- distance 4: 32

[near_duplicate_review.csv](near_duplicate_review.csv) contains this
authoritative 65-pair result. An earlier 66-pair result in the executed
notebook is historical/intermediate and must not be treated as the final
result. Historical notebook outputs are preserved as research records.

## Split-generation warning

The verified files in `data/splits/` are the current source of truth.
Existing split-generation scripts must not be rerun without first validating
the duplicate-aware reproducibility procedure.
