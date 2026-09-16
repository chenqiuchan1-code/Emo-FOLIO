#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Load repository-local credentials when available.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-mosaic}"                  # e2e / cot / mosaic / all
MODEL="${MODEL:-gpt-4o}"
BOOK_GLOB="${BOOK_GLOB:-book_1.json}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_POSTPROCESS="${RUN_POSTPROCESS:-1}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
SECONDARY_THRESHOLD="${SECONDARY_THRESHOLD:-1}"
USE_PRECOMPUTED="${USE_PRECOMPUTED:-0}"

require_env() {
  local variable_name="$1"
  if [[ -z "${!variable_name:-}" ]]; then
    echo "Missing required environment variable for MODEL=${MODEL}: ${variable_name}"
    echo "Copy configs/example.env to .env and configure the selected provider."
    exit 2
  fi
}

check_provider_configuration() {
  case "${MODEL}" in
    gemini-*)
      require_env GEMINI_API_KEY
      ;;
    qwen*)
      require_env DASHSCOPE_API_KEY
      ;;
    internvl35-8b)
      require_env HF_ENDPOINT_API_KEY
      require_env HF_INTERNVL35_8B_BASE_URL
      ;;
    internvl35-38b)
      require_env HF_ENDPOINT_API_KEY
      require_env HF_INTERNVL35_38B_BASE_URL
      ;;
    *)
      require_env OPENAI_API_KEY
      ;;
  esac

  if [[ "${MODEL}" == internvl35-* && "${INTERNVL_IMAGE_SOURCE:-local}" == "remote" ]]; then
    require_env HF_IMAGE_REPO_ID
  fi
}

if [[ "${USE_PRECOMPUTED}" == "1" ]]; then
  if [[ "${MODEL}" != "gpt-4o" || "${BOOK_GLOB}" != "book_1.json" ]]; then
    echo "The bundled precomputed example is available only for MODEL=gpt-4o and BOOK_GLOB=book_1.json."
    exit 2
  fi
  RUN_INFERENCE=0
fi

if [[ "${RUN_INFERENCE}" == "1" ]]; then
  check_provider_configuration
fi

if ! "${PYTHON_BIN}" -c "import openai, PIL, matplotlib" >/dev/null 2>&1; then
  echo "Missing runtime dependencies. Install them with: pip install -r requirements.txt"
  exit 1
fi

case "${MODE}" in
  e2e|cot|mosaic) variants=("${MODE}") ;;
  all) variants=(e2e cot mosaic) ;;
  *) echo "MODE must be one of: e2e, cot, mosaic, all"; exit 2 ;;
esac

run_variant() {
  local variant="$1"
  local raw_dir=""
  local result_tag="${variant}-${MODEL}"

  echo
  echo "========== ${variant} | ${MODEL} | ${BOOK_GLOB} =========="

  case "${variant}" in
    e2e)
      raw_dir="results/baselines/${MODEL}"
      if [[ "${RUN_INFERENCE}" == "1" ]]; then
        PYTHON_BIN="${PYTHON_BIN}" MODEL="${MODEL}" BOOK_GLOB="${BOOK_GLOB}" \
          USE_COT=0 bash scripts/inference/run_baselines.sh
      fi
      ;;
    cot)
      raw_dir="results/cot/${MODEL}"
      if [[ "${RUN_INFERENCE}" == "1" ]]; then
        PYTHON_BIN="${PYTHON_BIN}" MODEL="${MODEL}" BOOK_GLOB="${BOOK_GLOB}" \
          USE_COT=1 bash scripts/inference/run_baselines.sh
      fi
      ;;
    mosaic)
      raw_dir="results/mosaic/${MODEL}/merged"
      if [[ "${RUN_INFERENCE}" == "1" ]]; then
        PYTHON_BIN="${PYTHON_BIN}" MODEL_STEPA="${MODEL}" MODEL_STEPB="${MODEL}" \
          MODEL_STEPC="${MODEL}" BOOK_GLOB="${BOOK_GLOB}" \
          bash scripts/inference/run_mosaic.sh
      fi
      ;;
  esac

  if [[ "${USE_PRECOMPUTED}" == "1" ]]; then
    raw_dir="examples/book_1/gpt-4o/${variant}/raw"
  fi

  local post_dir="results/postprocessed/${result_tag}"
  local eval_dir="results/evaluation/${result_tag}"

  if [[ "${RUN_POSTPROCESS}" == "1" ]]; then
    if [[ ! -d "${raw_dir}" ]]; then
      echo "Raw prediction directory not found: ${raw_dir}"
      exit 3
    fi
    rm -rf "${post_dir}"
    "${PYTHON_BIN}" -m scripts.evaluation.postprocess \
      --in_dir "${raw_dir}" \
      --out_dir "${post_dir}" \
      --secondary_threshold "${SECONDARY_THRESHOLD}"
  fi

  if [[ "${RUN_EVALUATION}" == "1" ]]; then
    if [[ ! -d "${post_dir}" ]]; then
      echo "Post-processed prediction directory not found: ${post_dir}"
      exit 4
    fi
    rm -rf "${eval_dir}"
    local gold_subset="${post_dir}/.gold_subset"
    rm -rf "${gold_subset}"
    mkdir -p "${gold_subset}"
    local pred_file pred_name book_id gold_file
    for pred_file in "${post_dir}"/*_normalized.json; do
      [[ -e "${pred_file}" ]] || continue
      pred_name="$(basename "${pred_file}")"
      book_id="${pred_name%%_WHOLE*}"
      book_id="${book_id%%_MOSAIC*}"
      gold_file="data/annotations/${book_id}_annotated.json"
      if [[ -f "${gold_file}" ]]; then
        cp "${gold_file}" "${gold_subset}/"
      fi
    done
    if ! compgen -G "${gold_subset}/*_annotated.json" >/dev/null; then
      echo "No matching gold annotations were found for predictions in: ${post_dir}"
      exit 5
    fi
    "${PYTHON_BIN}" -m scripts.evaluation.evaluate \
      --gold_dir "${gold_subset}" \
      --pred_dir "${post_dir}" \
      --out_dir "${eval_dir}" \
      --trend_equal_tol 0 \
      --trend_min_intensity 0 \
      --trend_exclude_eq \
      --trend_eval_mode primary_involved \
      --segment_iou_thresh 0.3 \
      --boundary_tol 0
    echo "Evaluation summary: ${eval_dir}/overall_summary.csv"
  fi
}

for variant in "${variants[@]}"; do
  run_variant "${variant}"
done

echo
echo "Pipeline completed successfully."
