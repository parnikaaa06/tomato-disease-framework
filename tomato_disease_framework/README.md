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

## Phase 9: structured Gemini vision explanations

The provider-neutral orchestrator can be explicitly given a Gemini backend.
The Phase 3 classifier remains the source of the predicted class and
uncalibrated softmax score; Gemini generates only visual observations and
evidence-linked explanation claims. The application validates the response
locally and always withholds recommendations because no verified
recommendation source is currently configured. Recommendations are handled
only by the separate Phase 10 application policy, never by Gemini. Human review
remains necessary: schema-valid model output is not proof that a natural-language
claim is factually correct.

### Install and configure

Install the project dependencies, including Google's official `google-genai`
SDK (`>=2.12.1,<3.0.0`):

```bash
python -m pip install -r requirements.txt
```

Create an API key in [Google AI Studio](https://aistudio.google.com/apikey).
Keep it out of source files, notebooks, YAML, Git, and logs. The backend reads
`GEMINI_API_KEY` and uses `GEMINI_MODEL` when set; the default model identifier
is `gemini-3.6-flash` (availability can change). For a local PowerShell
session, set values without writing them to the repository:

```powershell
$secureKey = Read-Host "Gemini API key" -AsSecureString
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
Remove-Variable secureKey
$env:GEMINI_MODEL = "gemini-3.6-flash"
```

In Colab, save the key under **Secrets** (for example, `GEMINI_API_KEY`) and
load it into the runtime environment without printing it:

```python
import os
from google.colab import userdata

os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
os.environ["GEMINI_MODEL"] = "gemini-3.6-flash"
```

### Offline tests and opt-in API smoke test

The default test run is offline and excludes the separately marked real-API
smoke test:

```bash
python -m pytest -v
```

To deliberately send one request with synthetic images and verify the live
Gemini structured-output configuration, set the key and opt-in flag, then run:

```bash
RUN_GEMINI_INTEGRATION=1 python -m pytest -o addopts= \
  -m gemini_integration -q tests/test_gemini_integration.py
```

In Colab, after loading the key from Secrets as above:

```python
os.environ["RUN_GEMINI_INTEGRATION"] = "1"
```

```bash
!python -m pytest -o addopts= -m gemini_integration -q tests/test_gemini_integration.py
```

This smoke test consumes Gemini quota and may incur charges. The configured
backend uses a 20-second Google SDK HTTP timeout by default, overridable with
`GEMINI_TIMEOUT_MS` (1–60,000 milliseconds). The SDK's own
retry count is set to one attempt (no SDK-level retry); the application makes
at most two attempts by default, with one 0.5-second backoff. Thus the default
configured timeout budget is about 40.5 seconds across two attempts, plus
connection and processing overhead. `GeminiBackend(max_attempts=3)` can
explicitly permit three application attempts (about 61.5 seconds at the
default timeout). These are approximate budgets, not hard wall-clock
deadlines: the SDK timeout applies to HTTP operations and upstream behavior
can add latency. Only transient network/timeouts and HTTP 408, 425, 429, 500,
502, 503, and 504 are retried. Authentication, permission, malformed-request,
and response-schema errors are not retried. Quotas, pricing,
supported model identifiers, and availability depend on the Google account
and can change; check current Google AI Studio/API terms and billing before
use.

### One-image Phase 9 experiment

Run a single real image listed in the official Phase 2 test CSV through the
existing Phase 3 EfficientNet-B0 checkpoint, Phase 8 Grad-CAM, and Phase 9
Gemini backend. From Colab, mount Drive, clone/pull the project, load
`GEMINI_API_KEY` from Colab Secrets as shown above, and set `GEMINI_MODEL` if
you want a non-default model. Then run from the repository root:

```bash
!python -m evaluation.run_phase9_experiment \
  --config configs/phase3_baseline.yaml \
  --dataset-root "/content/drive/MyDrive/PlantVillage/tomato" \
  --checkpoint "/content/drive/MyDrive/tomato-disease/results/phase3/efficientnet_b0_baseline/best_checkpoint.pt" \
  --output-dir "/content/drive/MyDrive/tomato-disease/phase9/single_image" \
  --row-index 0
```

`--row-index` is a zero-based row index in the configured official `test.csv`
and defaults to `0`. Instead, `--image-path` may name a path present in that
CSV (relative paths are resolved from `--dataset-root`). The two selectors
cannot be used together. The command rejects an image not listed in the
official test split. It saves one Grad-CAM visualization and a JSON result
with the split row/image identity, checkpoint/model identity, true class,
prediction, explicitly uncalibrated softmax score, and locally validated
explanation. If Gemini or local explanation validation fails, the visualization
and a failure-status JSON are still saved; no explanation is fabricated and
the command exits with a nonzero status.

This is a single-image demonstration, not evidence of disease severity,
progression, lesion segmentation, or field reliability. The original image
and Grad-CAM heatmap are sent to Google for inference.

For an application call, load the existing Phase 3 model and checkpoint,
prepare `input_tensor` with Phase 3 test transforms, and retain the original
PIL image. Then inject the backend explicitly:

```python
from vlm.explanation import generate_structured_explanation
from vlm.gemini_backend import GeminiBackend

backend = GeminiBackend()
try:
    result = generate_structured_explanation(
        model, input_tensor, original_image, class_names, backend
    )
finally:
    backend.close()
```

The original image and a grayscale Grad-CAM heatmap, plus classifier context,
are sent to Google's cloud service for inference. Do not send images unless
you have permission and accept the provider's applicable data handling terms.
The heatmap is attribution evidence, not a lesion mask. Outputs are not
validated against disease-specific ground truth, do not provide treatments,
and do not establish field reliability; review every explanation before use.

## Phase 10: knowledge-grounded recommendations

`knowledge_base/tomato_diseases.json` now provides deterministic exact-class
records for the ten PlantVillage tomato labels. It contains disease display
names only: there are currently no source records, background claims, reported
symptom claims, or management guidance in the file. **No real source has been
verified and recommendations remain withheld for every class.**

Retrieval uses the Phase 3 classifier's predicted class as an exact lookup key;
it does not substitute another record and does not treat the prediction as a
confirmed diagnosis. The application-owned policy can emit only management
statements stored in the local knowledge base that are explicitly marked as
sufficient, cite sources explicitly marked verified after a curator's manual
content review, and carry full source provenance. A URL or valid JSON alone
does not establish verification. Missing, unverified, irrelevant, insufficient,
or contradictory evidence produces an explicit withholding result. Healthy
class predictions never produce disease-treatment guidance. Gemini has no
access to or authority over recommendation records or policy decisions.

If verified guidance is added later, it must remain general educational
information—not a diagnosis or individualized treatment instruction. This
policy does not support pesticide-specific advice, dosage, concentrations, or
application rates. Any future source entries need human review for source
authority, relevance, current applicability, and consistency; model/schema
validation is not evidence verification.

Run the local retrieval and recommendation policy tests, with no network or
Gemini API access:

```bash
python -m pytest tests/test_knowledge_base.py tests/test_vlm_explanation.py -q
python -m pytest -q
```

## Important notes

- Do not add PlantVillage images to version control.
- This project starts from an empty folder and requires controlled dataset validation before model implementation.
- The prototype is a rigorous baseline for future real-world generalization work.
