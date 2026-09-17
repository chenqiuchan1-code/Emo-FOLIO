#!/usr/bin/env bash
set -euo pipefail
# set -x  # Uncomment to print expanded commands.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! "${PYTHON_BIN}" -c "import openai, PIL" >/dev/null 2>&1; then
  echo "Missing inference dependencies. Activate any Python 3.10+ environment and run: pip install -r requirements.txt"
  exit 1
fi

# ===================== Experiment selection =====================
# Runnable experiment IDs:
# A1: full MOSAIC
#     - Reuses the existing Step A, Step B, Step C, and merged outputs
#     - Does not call a model again
#
# A2: w/o Step A
#     - Runs Step B, Step C, and merge without Step A inputs
#
# A3: A + fixed-window C
#     - Reuses Step A, skips Step B, and runs Step C with fixed windows
#
# A4: A + full-book C
#     - Reuses Step A, skips Step B, and runs Step C on the full book
#
# B2: w/o Summary Context (Step B)
#     - Reuses Step A and removes book and page summaries from Step B
#
# B3: w/o Evidence Cues (Step B)
#     - Reuses Step A and removes text and visual cues from Step B
#
# B4: w/o Emotion Candidates (Step B)
#     - Reuses Step A and removes emotion candidates from Step B
#
# B5: w/o Rationale Output (Step B)
#     - Reuses Step A and removes rationale output from Step B
#
# C2: w/o Evidence Cues (Step C)
#     - Reuses Step A and Step B, then removes text and visual cues from Step C
#
# C3: w/o Anchor Hint (Step C)
#     - Reuses Step A and Step B, then removes the continuity anchor from Step C
#
# C4: w/o Evidence + Anchor (Step C)
#     - Reuses Step A and Step B, then removes both inputs from Step C
#
# A5 in Table 3 is the Full-book E2E reference. Generate it with
# run_baselines.sh rather than this ablation wrapper.
#
# Select an experiment:
EXP_ID="${1:-C4}"


# ===================== Full MOSAIC output =====================
FULL_MOSAIC_ROOT="./results/mosaic/gpt-4o"

# ===================== Model configuration =====================
# The paper's ablations use GPT-4o for all three steps.
MODEL_STEPA="gpt-4o"
MODEL_STEPB="gpt-4o"
MODEL_STEPC="gpt-4o"

# ===================== A3 fixed-window configuration =====================
# Only A3 uses this value; the other experiments ignore it.
WINDOW_SIZE_STEPC=4

# ===================== Book selection =====================
# Glob:
#   --book-glob 'book_*.json'
# One book:
#   --book-glob 'book_4.json'
#   --book-glob 'book_4'
# Range:
#   --book-glob 'book_1-book_50'
# Explicit list (comma- or space-separated):
#   --book-glob 'book_1,book_6,book_50'
# The default is a one-book smoke test.
BOOKS_DIR="./data/books"
BOOK_GLOB='book_1'
BATCH_LIMIT=0

# ===================== Payload and execution controls =====================
# Save request payloads for newly executed stages.
DUMP_PAYLOAD=1

# Reuse existing artifacts when available.
REUSE_EXISTING=1

# Force stages to run again.
FORCE_RUN=0

# The paper does not set these optional generation controls.
REASONING_EFFORT=""
TEXT_VERBOSITY=""

args=(
  --exp-id "${EXP_ID}"
  --full-mosaic-root "${FULL_MOSAIC_ROOT}"

  --books-dir "${BOOKS_DIR}"
  --book-glob "${BOOK_GLOB}"
  --batch-limit "${BATCH_LIMIT}"

  --model-stepA "${MODEL_STEPA}"
  --model-stepB "${MODEL_STEPB}"
  --model-stepC "${MODEL_STEPC}"

  --window-size-stepC "${WINDOW_SIZE_STEPC}"
)

if [[ "${DUMP_PAYLOAD}" == "1" ]]; then
  args+=( --dump-payload )
fi

if [[ "${REUSE_EXISTING}" == "1" ]]; then
  args+=( --reuse-existing )
fi

if [[ "${FORCE_RUN}" == "1" ]]; then
  args+=( --force )
fi

if [[ -n "${REASONING_EFFORT}" ]]; then
  args+=( --reasoning-effort "${REASONING_EFFORT}" )
fi

if [[ -n "${TEXT_VERBOSITY}" ]]; then
  args+=( --text-verbosity "${TEXT_VERBOSITY}" )
fi

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m scripts.inference.run_ablation "${args[@]}" "${@:2}"
