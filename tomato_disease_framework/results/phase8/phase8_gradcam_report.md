# Phase 8 Grad-CAM Report

## Objective and role in the pipeline

Phase 8 adds a visual explanation step to the tomato classification pipeline.
Grad-CAM uses gradients of a selected class score with respect to a model's
convolutional feature maps to produce a coarse, class-conditioned heatmap. The
visualizations provide a qualitative view of image regions that influenced the
classifier's output. They are not lesion masks, disease measurements, or
validation that the model relies on disease-specific evidence.

## Implementation

The evaluation uses the existing Phase 3 EfficientNet-B0 baseline and its
trained checkpoint. The recorded checkpoint path for this evaluation is:

`/content/drive/MyDrive/UROP_Plant_Disease/results/phase3/efficientnet_b0_baseline/best_checkpoint.pt`

Checkpoint weights are loaded from the `model_state_dict` entry using the
existing checkpoint-loading helper. The Grad-CAM target layer is
`model.features[-1]`. Images use the existing Phase 3 test preprocessing:
resize to 224 × 224, convert to tensor, and apply ImageNet normalization. No
training or model-weight updates are part of this phase.

## Evaluation method

Evaluation used only the official Phase 2 test split at
`data/splits/test.csv` (2,725 rows). The split was validated to contain all 10
configured classes. Five images per class were selected without replacement
using the Phase 3 configured seed, 42. The selected sample therefore contains
50 images total, with five images from each class. The saved CSV's image order
matches the deterministic selection reproduced from the current split and
configuration.

For each selected image, the evaluation records the true and predicted class,
whether the prediction is correct, and the generated visualization path. Each
PNG contains the original image, the Grad-CAM heatmap, and an overlay.

## Measured results

The per-image CSV and JSON summary report:

| Measure | Result |
|---|---:|
| Balanced sample size | 50 |
| Correct predictions | 50 |
| Accuracy on this balanced sample | 100% |
| Images per class | 5 |
| Heatmap PNGs present | 50 of 50 |
| Missing heatmap files | 0 |

These figures describe **only this deterministic 50-image balanced sample**.
They are not full-test-set metrics and should not be interpreted as an
estimate with established statistical precision. The zero-missing-file check
matched all 50 heatmap basenames in the CSV against the 50 PNGs in the local
mirrored output directory. The CSV retains the original Colab Drive paths.

## Qualitative observations

The 10-class contact sheet was assembled from the 50 generated
original/heatmap/overlay visualizations. As a preliminary visual observation,
heatmap patterns vary across images: many show broad activation over part of
the leaf blade, while some include activation near leaf edges or extend into
surrounding image regions. The patterns are not uniformly shaped or positioned
across all examples.

This is descriptive review only. It does not establish that highlighted
regions correspond to disease lesions or that the maps consistently isolate
disease-specific features. No quantitative localization comparison was
performed.

## Output artifacts

Evaluation outputs are under `results/phase8/balanced_gradcam/`:

- `gradcam_evaluation.csv` — one row per selected image, including image path,
  true class, predicted class, correctness, and heatmap path.
- `gradcam_evaluation_summary.json` — sample size, per-class counts, correct
  predictions, sample accuracy, and checkpoint and test-split paths.
- 50 per-image PNG visualizations, each showing the original image, heatmap,
  and overlay.

## Reproducibility

From the repository root in the Colab runtime, mount Drive, ensure the test
split is available in the checkout, and run:

```bash
python -m evaluation.evaluate_gradcam \
  --config configs/phase3_baseline.yaml \
  --dataset-root "$TDF_DATASET_ROOT" \
  --checkpoint "/content/drive/MyDrive/UROP_Plant_Disease/results/phase3/efficientnet_b0_baseline/best_checkpoint.pt" \
  --output-dir "results/phase8/balanced_gradcam" \
  --samples-per-class 5
```

The evaluation's recorded test split is
`/content/tomato-disease-framework/tomato_disease_framework/data/splits/test.csv`
(repository-relative path: `data/splits/test.csv`). The configured seed is
42. Reproduction requires the same official test CSV, compatible Phase 3
checkpoint, and accessible PlantVillage image root.

## Limitations and relation to prior feasibility audits

- The 100% accuracy is for a selected, balanced sample of 50 test images, not
  for all 2,725 official test images.
- Grad-CAM is a coarse attribution visualization, not lesion segmentation,
  proof of disease-specific attention, or a measure of lesion area.
- The Phase 6 feasibility report did not establish verified disease-lesion
  masks; its documented leaf/background segmentation evidence is not lesion
  ground truth.
- The Phase 7 audit did not establish defensible severity grades, disease-area
  measurements, or temporal progression targets, and did not justify
  severity/progression model training.
- These controlled-dataset results do not establish reliability or
  generalization to field images, other acquisition conditions, or deployment.

## Conclusion and remaining work

The existing Phase 3 classifier generated Grad-CAM visualizations for a
reproducible 50-image, 10-class sample from the official test split. The
recorded predictions were correct for all 50 sampled images, and all 50
visualization files were present. This is a limited demonstration of the
workflow, not evidence of full-test-set performance, lesion localization, or
field reliability.

Remaining work includes evaluating the full official test split for
classification metrics, defining a validated protocol and ground truth before
any lesion-localization or severity claims, and testing generalization on
appropriate external or field data. Any such work should be reported
separately from this balanced-sample result.
