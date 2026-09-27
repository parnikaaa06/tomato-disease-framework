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
- Phase 7: severity estimation
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

## Important notes

- Do not add PlantVillage images to version control.
- This project starts from an empty folder and requires controlled dataset validation before model implementation.
- The prototype is a rigorous baseline for future real-world generalization work.
