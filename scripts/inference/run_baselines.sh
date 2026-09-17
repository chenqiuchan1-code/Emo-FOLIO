#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! "${PYTHON_BIN}" -c "import openai, PIL" >/dev/null 2>&1; then
  echo "Missing inference dependencies in: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)' 2>/dev/null || echo "${PYTHON_BIN}")"
  echo "Activate any Python 3.10+ environment and run: pip install -r requirements.txt"
  exit 1
fi

# ===== Model selection =====
# MODEL="gpt-4o"
# MODEL="gpt-4o-mini"
# MODEL="gemini-2.5-pro"
# MODEL="gemini-2.5-flash"
# MODEL="qwen3-vl-8b-instruct"
# MODEL="qwen3-vl-32b-instruct"
# MODEL="internvl35-8b"
# MODEL="internvl35-38b"
MODEL="${MODEL:-gpt-4o}"

# Data and input configuration. The one-book default is a low-cost smoke test.
BOOK_GLOB="${BOOK_GLOB:-book_1.json}"
IMAGE_DETAIL="${IMAGE_DETAIL:-auto}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-512}"

# Mode and diagnostics: 0=off, 1=on.
USE_COT="${USE_COT:-0}"
DUMP_PAYLOAD="${DUMP_PAYLOAD:-0}"
DRY_RUN="${DRY_RUN:-0}"

# The paper does not set reasoning effort, output-token limits, or text verbosity.
REASONING_EFFORT="${REASONING_EFFORT:-}"
TEXT_VERBOSITY="${TEXT_VERBOSITY:-}"

# Output directory, grouped by model.
OUT_ROOT="${OUT_ROOT:-./results/baselines}"
OUT_DIR="${OUT_ROOT}/${MODEL}"

args=(
  ./data/books
  --book-glob "${BOOK_GLOB}"

  # Output control
  --out "${OUT_DIR}"

  # Model service
  --model "${MODEL}"

  # Input diagnostics and image sizing
  --trace-tokens
  --image-detail "${IMAGE_DETAIL}"
  --max-image-side "${MAX_IMAGE_SIDE}"
  # --no-inject-images  # Text-only diagnostic setting

  # Optional prompt overrides
  # --prompt-head-file ./prompt_head.txt
  # --prompt-tail-file ./prompt_tail.txt
  # --instructions-file ./instructions.txt
)

if [[ "${USE_COT}" == "1" ]]; then args+=( --cot ); fi
if [[ "${DUMP_PAYLOAD}" == "1" ]]; then args+=( --dump-payload ); fi
if [[ "${DRY_RUN}" == "1" ]]; then args+=( --dry-run --dump-payload ); fi

# Add provider-specific optional parameters.
case "${MODEL}" in
  gpt-5*|gemini-*)
    if [[ -n "${REASONING_EFFORT}" ]]; then
      args+=( --reasoning-effort "${REASONING_EFFORT}" )
    fi
    ;;
esac

case "${MODEL}" in
  gpt-5*)
    if [[ -n "${TEXT_VERBOSITY}" ]]; then
      args+=( --text-verbosity "${TEXT_VERBOSITY}" )
    fi
    ;;
esac

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m scripts.inference.run_baselines "${args[@]}" "$@"
