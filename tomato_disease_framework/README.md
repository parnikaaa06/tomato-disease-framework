# Explainable Vision-Language and Augmented Reality Framework for Early Plant Disease Detection

This project is an empty, scientific baseline scaffold for a Level 1 prototype focused on the tomato subset of the PlantVillage dataset. The goal is to build a reproducible pipeline for disease classification, segmentation feasibility checks, severity estimation, explainability, and knowledge-grounded recommendations without claiming real-world generalization.

## Scope

- Level 1 only
- Controlled PlantVillage tomato subset
- No AR, no temporal progression, no real-world deployment claims
- Scientific validity preserved through dataset inspection, checks, and explicit future-stage roadmap

## Project structure

- `data/`: dataset storage, processed data, split files, and dataset reports
- `models/`: classifier and segmentation model definitions
- `preprocessing/`: transforms, image quality checks, and preprocessing utilities
- `training/`: training scripts for classifiers and segmentation models
- `evaluation/`: metrics and evaluation logic
- `explainability/`: Grad-CAM / Score-CAM implementations
- `vlm/`: VLM explanation templates and prompts
- `knowledge_base/`: disease knowledge retrieval source and recommendations
- `app/`: Streamlit interface
- `results/`: experiments and reports
- `configs/`: configuration files
- `scripts/`: operational scripts
- `tests/`: validation tests

## Phase roadmap

- Phase 0: project initialization
- Phase 1: dataset acquisition and inspection
- Phase 2: data splitting and preprocessing
- Phase 3: EfficientNet-B0 baseline
- Phase 4: EfficientNet-B0 + dynamic self-attention
- Phase 5: evaluation and comparison
- Phase 6: segmentation feasibility only if valid annotations exist
- Phase 7: severity/progression feasibility audit; estimation only if validated targets are available
- Phase 8: Grad-CAM / Score-CAM
- Phase 9: structured VLM explanation
- Phase 10: knowledge-grounded recommendation
- Phase 11: Streamlit app
- Phase 12: research dashboard and evaluation
- Phase 13: documentation and report

## Environment

Python and PyTorch are used, with a configurable device selection:

- `cuda` if available
- otherwise `cpu`

## Phase 8: Grad-CAM demonstration

Run the demonstration from the repository root so Python resolves the project
packages as modules:

```bash
python -m evaluation.visualize_gradcam \
  --dataset-root "/path/to/PlantVillage/tomato" \
  --checkpoint "results/phase3/efficientnet_b0_baseline/best_checkpoint.pt" \
  --output "results/phase8/gradcam_example.png"
```

The dataset root must be the parent directory referenced by paths in
`data/splits/test.csv`. The script uses that official test split and the
Phase 3 resize, tensor conversion, and ImageNet normalization; it does not
create a new split or train a model. The checkpoint is not expected to be
versioned with the repository. Paths can be supplied explicitly as above, or
the dataset root can be provided through the configured `TDF_DATASET_ROOT`
environment variable. The output path can point to a mounted Google Drive.

In Colab, mount Drive and run the module from the cloned repository root. Use
the actual dataset and checkpoint locations in your Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/tomato_disease_framework
```

```bash
!python -m evaluation.visualize_gradcam \
  --dataset-root "/content/drive/MyDrive/PlantVillage/tomato" \
  --checkpoint "/content/drive/MyDrive/tomato-disease/results/phase3/efficientnet_b0_baseline/best_checkpoint.pt" \
  --output "/content/drive/MyDrive/tomato-disease/phase8/gradcam_example.png"
```

Grad-CAM visualizes model sensitivity for the selected class; its heatmap is
not a lesion segmentation mask or evidence of disease localization.

### Balanced-sample evaluation

To create Grad-CAM visualizations and per-image evaluation metadata for the
official Phase 2 test split, run from the repository root:

```bash
python -m evaluation.evaluate_gradcam \
  --dataset-root "/path/to/PlantVillage/tomato" \
  --checkpoint "results/phase3/efficientnet_b0_baseline/best_checkpoint.pt" \
  --output-dir "results/phase8/gradcam_evaluation"
```

`--config` defaults to `configs/phase3_baseline.yaml`; `--dataset-root`
defaults to the configured environment variable (`TDF_DATASET_ROOT`);
`--output-dir` defaults to `results/phase8/gradcam_evaluation`; and
`--samples-per-class` defaults to 5. The `--checkpoint` argument is required
because the Phase 3 checkpoint may be stored outside GitHub. The deterministic
sample uses the Phase 3 configured seed unless the sampling helper is called
directly with another seed. The command requires the official test CSV to
contain exactly 2,725 rows and all 10 configured classes; it never samples
from training or validation data.

The output directory contains one original/heatmap/overlay PNG per selected
image, `gradcam_evaluation.csv` with the image paths, true/predicted classes,
correctness, and visualization path, and `gradcam_evaluation_summary.json`
with the balanced sample counts, correct predictions, accuracy, and input
artifact paths. Reported accuracy applies only to this balanced sample, not
the complete test set. Grad-CAM heatmaps do not validate lesion segmentation.

## Important notes

- Do not add PlantVillage images to version control.
- This project starts from an empty folder and requires controlled dataset validation before model implementation.
- The prototype is a rigorous baseline for future real-world generalization work.
