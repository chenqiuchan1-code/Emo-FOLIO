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

# ===================== Editable configuration =====================
# The paper uses the same backbone in all three MOSAIC steps.
MODEL_STEPA="${MODEL_STEPA:-gpt-4o}"
MODEL_STEPB="${MODEL_STEPB:-gpt-4o}"
MODEL_STEPC="${MODEL_STEPC:-gpt-4o}"

# Data and execution configuration. The one-book default is a low-cost test.
BOOK_GLOB="${BOOK_GLOB:-book_1.json}"
BATCH_LIMIT="${BATCH_LIMIT:-0}"
FROM_STEP="${FROM_STEP:-auto}"       # auto / stepA / stepB / stepC / merge
REUSE_EXISTING="${REUSE_EXISTING:-1}" # 1 skips completed artifacts; set FORCE_RUN=1 to overwrite
FORCE_RUN="${FORCE_RUN:-0}"

# The paper does not set reasoning effort, output-token limits, or text verbosity.
REASONING_EFFORT="${REASONING_EFFORT:-}"
TEXT_VERBOSITY="${TEXT_VERBOSITY:-}"

# ===================== Output directory =====================
# 统一放在 results/mosaic 下，并按模型自动区分子文件夹，避免覆盖。
#
# 规则：
# - 如果三个步骤模型相同：
#     results/mosaic/gpt-4o
#     results/mosaic/gemini-2.5-flash
#
# - 如果三个步骤模型不同：
#     results/mosaic/A_gemini-2.5-flash__B_gemini-2.5-pro__C_gpt-4o
if [[ "${MODEL_STEPA}" == "${MODEL_STEPB}" && "${MODEL_STEPB}" == "${MODEL_STEPC}" ]]; then
  OUT_TAG="${MODEL_STEPA}"
else
  OUT_TAG="A_${MODEL_STEPA}__B_${MODEL_STEPB}__C_${MODEL_STEPC}"
fi
OUT_ROOT="results/mosaic/${OUT_TAG}"

# ===================== Qwen / 百炼环境检查 =====================
if [[ "${MODEL_STEPA}" == qwen* || "${MODEL_STEPB}" == qwen* || "${MODEL_STEPC}" == qwen* ]]; then
  if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
    echo "❌ StepA/StepB/StepC 中至少有一个模型是 Qwen，但未检测到 DASHSCOPE_API_KEY。"
    echo '请先 export DASHSCOPE_API_KEY="你的百炼API Key"'
    exit 1
  fi
fi

# ===================== InternVL35-8B 环境检查 =====================
if [[ "${MODEL_STEPA}" == "internvl35-8b" || "${MODEL_STEPB}" == "internvl35-8b" || "${MODEL_STEPC}" == "internvl35-8b" ]]; then
  if [[ -z "${HF_ENDPOINT_API_KEY:-}" ]]; then
    echo "❌ StepA/StepB/StepC 中至少有一个模型是 internvl35-8b，但未检测到 HF_ENDPOINT_API_KEY。"
    echo '请先 export HF_ENDPOINT_API_KEY="你的 Hugging Face token"'
    exit 1
  fi
  if [[ -z "${HF_INTERNVL35_8B_BASE_URL:-}" ]]; then
    echo "❌ StepA/StepB/StepC 中至少有一个模型是 internvl35-8b，但未检测到 HF_INTERNVL35_8B_BASE_URL。"
    echo '请先 export HF_INTERNVL35_8B_BASE_URL="你的 8B Endpoint URL/v1"'
    exit 1
  fi
fi

# ===================== InternVL35-38B 环境检查 =====================
if [[ "${MODEL_STEPA}" == "internvl35-38b" || "${MODEL_STEPB}" == "internvl35-38b" || "${MODEL_STEPC}" == "internvl35-38b" ]]; then
  if [[ -z "${HF_ENDPOINT_API_KEY:-}" ]]; then
    echo "❌ StepA/StepB/StepC 中至少有一个模型是 internvl35-38b，但未检测到 HF_ENDPOINT_API_KEY。"
    echo '请先 export HF_ENDPOINT_API_KEY="你的 Hugging Face token"'
    exit 1
  fi
  if [[ -z "${HF_INTERNVL35_38B_BASE_URL:-}" ]]; then
    echo "❌ StepA/StepB/StepC 中至少有一个模型是 internvl35-38b，但未检测到 HF_INTERNVL35_38B_BASE_URL。"
    echo '请先 export HF_INTERNVL35_38B_BASE_URL="你的 38B Endpoint URL/v1"'
    exit 1
  fi
fi

args=(
  # Data scope. The one-book default is a low-cost smoke test.
  --books-dir ./data/books
  --book-glob "${BOOK_GLOB}"
  --batch-limit "${BATCH_LIMIT}"
  --out-root "${OUT_ROOT}"

  # ========================= 执行控制 ==========================
  --from-step "${FROM_STEP}"

  # ===================== 通用推理参数 ==========================
  # --trace-tokens

  # ===================== Step A =====================
  --model-stepA "${MODEL_STEPA}"
  --allow-missing-images-stepA
  --dump-stepA-payload
  --image-detail-stepA auto
  --max-image-side-stepA 512

  # StepA 注入消融
  # --no-visual-style-guide-stepA

  # StepA 调试
  # --dry-run-stepA

  # ===================== Step B =====================
  --model-stepB "${MODEL_STEPB}"
  --dump-stepB-payload
  --max-retries-stepB 2
  --image-detail-stepB auto
  --max-image-side-stepB 512

  # Paper configuration: ordered page text with page-aligned Step A information;
  # page images are not re-injected in Step B.
  --inject-page-text-stepB
  --inline-stepA-per-page-stepB

  # StepB 调试
  # --emit-missing-image-placeholders-stepB

  # ==================== Step C ====================
  --model-stepC "${MODEL_STEPC}"
  --allow-missing-images-stepC
  --dump-stepC-payload
  --max-retries-stepC 2
  --image-detail-stepC auto
  --max-image-side-stepC 512
  --chunk-mode-stepC stepb_windows          # whole_book / stepb_windows

  # StepC 注入控制：软规则指令
  --no-include-prompt-head-stepC

  # StepC 注入控制：原始页面内容
  # --no-inject-page-text-stepC          # 不注入窗口页原始文本
  # --no-inject-page-images-stepC        # 不注入窗口页原始图片（做 no-image 实验时开启）

  # StepC 注入控制：anchor（仅 stepb_windows 模式有意义）
  --include-anchor-hint-stepC            # 注入跨窗口锚点（通常 stepb_windows 建议开启）
  --no-anchor-images-stepC               # 不注入锚点原始图片
  --no-anchor-s1-cues-stepC              # 不注入锚点页的 StepA 线索

  # Final Step C configuration keeps text/visual cues and excludes summaries
  # and emotion candidates.
  --no-inject-book-summary-stepC       # 关闭全书摘要注入
  --no-inject-s1-summary-stepC         # 关闭页级语义摘要注入
  --no-inject-s1-candidates-stepC        # 关闭候选情感注入
  # --no-inject-s1-text-cues-stepC       # 关闭文本线索注入
  # --no-inject-s1-visual-cues-stepC     # 关闭图像线索注入

  # StepC 输出/调试
  # --include-window-intent-stepC        # 让模型额外输出 window intent（debug/分析用）
  # --emit-missing-image-placeholders-stepC
  # --dry-run-stepC                      # 正式跑时不要开；开了只生成 payload，不真正请求模型
)

if [[ "${REUSE_EXISTING}" == "1" ]]; then args+=( --reuse-existing ); fi
if [[ "${FORCE_RUN}" == "1" ]]; then args+=( --force ); fi

# ===================== Optional provider parameters =====================
if [[ ${MODEL_STEPA} == gpt-5* || ${MODEL_STEPB} == gpt-5* || ${MODEL_STEPC} == gpt-5* || \
      ${MODEL_STEPA} == gemini-* || ${MODEL_STEPB} == gemini-* || ${MODEL_STEPC} == gemini-* ]]; then
  if [[ -n "${REASONING_EFFORT}" ]]; then
    args+=( --reasoning-effort "${REASONING_EFFORT}" )
  fi
fi

if [[ ${MODEL_STEPA} == gpt-5* || ${MODEL_STEPB} == gpt-5* || ${MODEL_STEPC} == gpt-5* ]]; then
  if [[ -n "${TEXT_VERBOSITY}" ]]; then
    args+=( --text-verbosity "${TEXT_VERBOSITY}" )
  fi
fi

mkdir -p "${OUT_ROOT}"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m scripts.inference.run_mosaic "${args[@]}" "$@"
