# Connecting Emotion Dynamics and Narrative Organization in Picture Books

This repository contains the public data subset and implementation accompanying the paper **“Connecting Emotion Dynamics and Narrative Organization in Picture Books.”** It includes:

- preprocessing utilities for corrected page text and text-free page images;
- the annotation interface used to create page-, transition-, and stage-level labels;
- Full-book E2E and CoT baselines;
- **MOSAIC** (*Multilevel Organization of Story-Aware Information for Cross-level Reasoning*);
- post-processing, evaluation, ablation, and length-analysis scripts; and
- a public subset of 50 picture books with aligned source records, annotations, and page images.

MOSAIC is a structured inference framework that organizes reusable story-aware information and connects book-, stage-, and page-level reasoning through controlled, structure-guided information flow.

## Repository layout

```text
.
├── configs/                 # environment-variable template
├── data/
│   ├── books/               # corrected page text and image references
│   ├── annotations/         # human annotations
│   └── images/              # text-free page images (books 1-50)
├── mosaic/                  # Steps A, B, and C of MOSAIC
├── examples/                # real one-book predictions and evaluation outputs
├── scripts/
│   ├── inference/           # baseline, MOSAIC, and ablation runners
│   └── evaluation/          # post-processing and evaluation
├── tools/
│   ├── annotation/          # Streamlit annotation interface
│   └── data_processing/     # OCR cleanup and JSON construction
└── tests/                   # lightweight release checks
```

## Installation

Python 3.10 or newer is recommended.
The repository deliberately separates three dependency scopes so that the
platform-sensitive OCR stack is not forced into the inference environment:

| Scope | Used for | Dependency specification |
|---|---|---|
| Core | baseline/MOSAIC inference, post-processing, evaluation, and analysis | `requirements.txt` |
| Annotation | Streamlit annotation interface | `tools/annotation/requirements.txt` |
| OCR | OpenCV/PaddleOCR preprocessing | `tools/data_processing/setup_ocr_env.sh` and `requirements-runtime.txt` |

Create an environment with any name, activate it, and install the core
dependencies. For example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Validate the public subset

This check does not call any model API:

```bash
python scripts/check_environment.py --scope core
python scripts/validate_release.py
python -m unittest discover -s tests -v
```

These commands verify the installed runtime, the alignment of books 1-50 with
their annotation files and page images, and the post-processing/evaluation
pipeline. For detailed setup, execution, and reproducibility limits, see
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

A precomputed, no-API example for Full-book E2E, CoT, and MOSAIC is available
under [`examples/book_1/gpt-4o/`](examples/book_1/gpt-4o/).

To run its complete post-processing and evaluation path without an API key:

```bash
USE_PRECOMPUTED=1 MODE=all bash scripts/run_pipeline.sh
```

## Run inference

### Configure a model service

Copy the environment template, fill in only the credentials required by the
selected provider, and load it into the current shell:

```bash
cp configs/example.env .env
set -a; source .env; set +a
```

Alternatively, export the required variables with any environment manager.
Never commit API keys. The complete variable list and the local/remote image
options for InternVL are documented in `configs/example.env`.

### Run the complete pipeline

After configuring `.env`, the following command runs MOSAIC on the default
one-book example, followed by post-processing and evaluation:

```bash
bash scripts/run_pipeline.sh
```

The same entry point can run another inference mode, model, or book selection
without editing source files:

```bash
MODE=e2e MODEL=gpt-4o BOOK_GLOB=book_1.json bash scripts/run_pipeline.sh
MODE=cot MODEL=gpt-4o BOOK_GLOB=book_1.json bash scripts/run_pipeline.sh
MODE=all MODEL=gpt-4o BOOK_GLOB=book_1-book_50 bash scripts/run_pipeline.sh
```

`MODE` accepts `e2e`, `cot`, `mosaic`, or `all`. Set `RUN_POSTPROCESS=0` or
`RUN_EVALUATION=0` only when the corresponding stage should be skipped. Model
calls may incur provider charges.

The shell wrappers are the recommended entry points on macOS and Linux. They
keep the editable parameters in one block and call the corresponding Python
modules internally.
Run all commands from the repository root after activating the environment in
which `requirements.txt` was installed.

### Lower-level entry points

The editable block at the top of each file contains the defaults. The same
settings can be overridden for one command without editing the script. The
default selection processes `book_1.json` to limit API cost.

| Task | Main settings to edit | Command |
|---|---|---|
| Full-book E2E | `MODEL`, `BOOK_GLOB`, `USE_COT=0` | `bash scripts/inference/run_baselines.sh` |
| CoT | `MODEL`, `BOOK_GLOB`, `USE_COT=1` | `bash scripts/inference/run_baselines.sh` |
| MOSAIC | `MODEL_STEPA/B/C`, `BOOK_GLOB`, reuse controls | `bash scripts/inference/run_mosaic.sh` |
| Ablation | `EXP_ID` or the first command argument, `BOOK_GLOB` | `bash scripts/inference/run_ablation.sh C2` |

```bash
# Full-book E2E; enable --cot in the shell file for CoT
bash scripts/inference/run_baselines.sh

# CoT without editing the shell file
USE_COT=1 bash scripts/inference/run_baselines.sh

# Step A -> Step B -> Step C -> merge
bash scripts/inference/run_mosaic.sh

# Run the released 50-book subset
BOOK_GLOB=book_1-book_50 bash scripts/inference/run_mosaic.sh

# Example ablation
bash scripts/inference/run_ablation.sh C2
```

Baseline outputs are written under `results/baselines/<model>/`. MOSAIC
stores step outputs under `results/mosaic/<model>/finals/` and merged
predictions under `results/mosaic/<model>/merged/`.

The checked-in paper-aligned defaults use a 512-pixel maximum image side,
temperature 0.2, no explicit reasoning-effort or output-token limit, ordered
page inputs, the same backbone across the three MOSAIC steps, and the final
Step B/Step C information configuration described in the paper. The paper
reports the arithmetic mean of two independent runs on the non-public
200-book test set. The released 50-book subset supports execution and
evaluation of the complete pipeline but cannot reproduce the full reported
scores exactly.

The Python modules remain available as lower-level interfaces. Use
`python -m scripts.inference.run_baselines --help` or
`python -m scripts.inference.run_mosaic --help` when a command-line override
is needed.

To inspect request construction without a paid API call:

```bash
DRY_RUN=1 bash scripts/inference/run_baselines.sh
bash scripts/inference/run_mosaic.sh --dry-run-stepA --force
```

Available ablation identifiers and their configuration differences are
documented at the beginning of `scripts/inference/run_ablation.sh`.

## Post-process and evaluate

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

`scripts/evaluation/compare_results.py` provides the integrated comparison and
length-analysis procedure used in the experiments. Set its configuration block
before execution.

## Annotation interface

Install the annotation dependencies and launch the interface from the repository root:

```bash
pip install -r tools/annotation/requirements.txt
python scripts/check_environment.py --scope annotation
streamlit run tools/annotation/annotator.py
```

The interface reads the released books and images but writes new annotations
to `outputs/annotations/`, leaving the released gold annotations unchanged.

## Data processing

The OCR cleanup utility has a separate environment because PaddleOCR
installations are platform-sensitive. The setup script accepts any environment
name and installs the paper-aligned PaddleOCR 2.7.0.3 and OpenCV 4.10.0 stack:

```bash
bash tools/data_processing/setup_ocr_env.sh my-ocr-env
conda activate my-ocr-env
python scripts/check_environment.py --scope ocr
bash tools/data_processing/run_ocr.sh
```

Set the input and output paths in `run_ocr.sh` before execution.
Raw scans are not distributed in this repository; the OCR utility therefore
runs on user-supplied images and writes to `outputs/ocr/` by default. On its
first real OCR run, PaddleOCR downloads the required recognition models.

## Public data subset and copyright

The full Emo-FOLIO benchmark contains 234 picture books. This repository contains a 50-book public subset with corrected page text, text-free page images, and human annotations. The code license does **not** grant rights to third-party picture-book content. See [DATA_NOTICE.md](DATA_NOTICE.md) before redistributing or reusing any data files.

Additional research materials may be made available upon reasonable request, subject to copyright and other applicable restrictions.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). Publication
venue, DOI, and final bibliographic details will be added after publication.

## License

The source code is released under the MIT License. The dataset and page images are excluded from that license; see [DATA_NOTICE.md](DATA_NOTICE.md).
