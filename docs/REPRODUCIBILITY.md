# Reproducibility guide

This guide describes how to install and run the released code and data, along
with the resources required beyond the repository.

## 1. Install

Python 3.10 or newer is recommended. The environment may have any name.
The core inference/evaluation environment, annotation interface, and OCR
preprocessing environment are specified separately; this avoids installing
the platform-sensitive PaddleOCR stack when only inference is needed.

```bash
git clone https://github.com/chenqiuchan1-code/Emo-FOLIO.git
cd Emo-FOLIO
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Verify the release without an API key

```bash
python scripts/check_environment.py --scope core
python scripts/validate_release.py
python -m unittest discover -s tests -v
```

These checks verify the installed runtime, the alignment of the 34 released
books with their annotations and 603 page images, and a deterministic
post-processing/evaluation test.

Request construction can also be inspected without sending a model request:

```bash
DRY_RUN=1 bash scripts/inference/run_baselines.sh

# Preview MOSAIC Step A for the configured book:
bash scripts/inference/run_mosaic.sh --dry-run-stepA --force
```

## 3. Configure a model service

The inference code uses model APIs; model weights are not bundled. Copy
`configs/example.env` to `.env`, add only the credentials required by the
selected provider, and load the variables into the current shell.

```bash
cp configs/example.env .env
set -a; source .env; set +a
```

The default `gpt-4o` configuration requires `OPENAI_API_KEY`. Gemini requires
`GEMINI_API_KEY`, and Qwen requires `DASHSCOPE_API_KEY`. InternVL requires a
user-accessible OpenAI-compatible inference endpoint. It uses the released
local images by default (`INTERNVL_IMAGE_SOURCE=local`). If that endpoint
requires public image URLs, the user must upload the released images to a
repository under their own control, set `INTERNVL_IMAGE_SOURCE=remote`, and
fill in the `HF_IMAGE_*` variables. This release does not contain or fall back
to an author-owned image-hosting repository.

Service identifiers and API availability may change after the February 2026
access period reported in the paper.

## 4. Run one released book end to end

The default entry point selects `book_1.json` to limit API cost and runs
MOSAIC, post-processing, and evaluation:

```bash
bash scripts/run_pipeline.sh
```

Set `MODE=e2e`, `MODE=cot`, or `MODE=all` to select other inference modes.
The lower-level commands remain available when individual stages need to be
inspected:

```bash
# Full-book E2E (set USE_COT=1 in the script for CoT)
bash scripts/inference/run_baselines.sh

# MOSAIC: Step A -> Step B -> Step C -> merge
bash scripts/inference/run_mosaic.sh
```

MOSAIC writes merged predictions to
`results/mosaic/<model>/merged/`. Post-process and evaluate them with:

```bash
python -m scripts.evaluation.postprocess \
  --in_dir results/mosaic/gpt-4o/merged \
  --out_dir results/postprocessed/mosaic-gpt-4o \
  --secondary_threshold 1

python -m scripts.evaluation.evaluate \
  --gold_dir data/annotations \
  --pred_dir results/postprocessed/mosaic-gpt-4o \
  --out_dir results/evaluation/mosaic-gpt-4o \
  --trend_equal_tol 0 \
  --trend_min_intensity 0 \
  --trend_exclude_eq \
  --trend_eval_mode primary_involved \
  --segment_iou_thresh 0.3 \
  --boundary_tol 0
```

To process all released books, prefix either inference command with
`BOOK_GLOB=book_1-book_34`. Model calls may incur substantial provider costs.

## 5. Ablations

Ablations reuse outputs from the complete MOSAIC run. Produce those outputs
first, then select an experiment identifier:

```bash
bash scripts/inference/run_ablation.sh C2
```

The identifiers and their configuration differences are listed at the top of
`scripts/inference/run_ablation.sh`.

## 6. Annotation and preprocessing tools

Install and verify the annotation dependencies, then launch the interface:

```bash
pip install -r tools/annotation/requirements.txt
python scripts/check_environment.py --scope annotation
streamlit run tools/annotation/annotator.py
```

It reads `data/books/` and `data/images/` and writes new work to
`outputs/annotations/`; the released gold files are not overwritten.

The OCR cleanup tool has a separate setup script. The environment name is
chosen by the user; the script pins the PaddleOCR 2.7.0.3 and OpenCV 4.10.0
runtime used by the released preprocessing utility:

```bash
bash tools/data_processing/setup_ocr_env.sh my-ocr-env
conda activate my-ocr-env
python scripts/check_environment.py --scope ocr
bash tools/data_processing/run_ocr.sh
```

Raw scans are not distributed. Before running the tool, use the **User
configuration** block at the top of `tools/data_processing/run_ocr.sh` to set
the input/output paths, optional book selection, and inspection controls. The
OCR parameters retain the released defaults and normally need no adjustment.
PaddleOCR downloads its recognition models on the first real OCR run.

## Reproducibility boundary

The repository is sufficient to run the released 34-book subset through the
baseline, MOSAIC, post-processing, and evaluation code when the user supplies
access to a supported model service. It also supports the released annotation
interface and preprocessing utilities under their documented dependencies.

It is not, by itself, sufficient to reproduce the exact numbers in the paper:
the reported experiments use a separate 200-book test set, whereas this
repository releases 34 books. Exact multi-backbone reproduction additionally
depends on access to the commercial model APIs or user-deployed endpoints used
by each backbone. The public subset supports the complete released pipeline,
inspection of the annotation format, and evaluation on the released material.
