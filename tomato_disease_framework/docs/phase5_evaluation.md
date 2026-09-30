# Phase 5: Evaluation and Comparison

Phase 5 compares the already-completed test-set evaluations for the Phase 3
EfficientNetB0 baseline and the Phase 4 EfficientNetB0 with Dynamic
Self-Attention. The comparison reads saved YAML metrics only; it does not train
models or rerun inference.

## Evaluation scope

The results are from the official 2,725-sample controlled PlantVillage test
split. The comparison reports accuracy, macro precision, macro recall, macro
F1, weighted F1, per-class precision/recall/F1/support, confusion matrices,
correct and incorrect prediction totals, and misclassification counts by
true/predicted class pair. Metric deltas are dynamic attention minus baseline.

This controlled PlantVillage comparison does not establish real-world field
reliability.

## Inputs and output

- Baseline results: `results/phase3/efficientnet_b0_baseline/test_results.yaml`
- Dynamic-attention results:
  `results/phase3/efficientnet_b0_dynamic_attention/test_results.yaml`
- Comparison report: `results/phase3/phase3_model_comparison.yaml`

The script validates that both inputs describe the same test split, contain
2,725 samples, and share exactly the same ordered list of ten classes before
comparing per-class metrics or confusion matrices.

## Run

From the repository root, run:

```powershell
python -m evaluation.compare_phase3_models
```

Both input YAML files must be present at the paths listed above. The report is
written to the output path above, and a concise aggregate/per-class F1 table
and error totals are printed to the terminal.
