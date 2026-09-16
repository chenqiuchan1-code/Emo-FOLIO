#!/usr/bin/env bash
set -euo pipefail
# set -x  # 需要看完整展开命令时再打开

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! "${PYTHON_BIN}" -c "import openai, PIL" >/dev/null 2>&1; then
  echo "Missing inference dependencies. Activate any Python 3.10+ environment and run: pip install -r requirements.txt"
  exit 1
fi

# ===================== 实验选择 =====================
# 可用实验ID：
# A2: w/o Step A
#     - 不跑 StepA
#     - 跑 StepB + StepC + merge
#     - StepB / StepC 都不读取 StepA
#
# A3: A + fixed-window C
#     - 复用完整方法的 StepA
#     - 不跑 StepB
#     - 只跑 StepC + merge
#     - StepC 使用 fixed-window
#
# A4: A + full-book C
#     - 复用完整方法的 StepA
#     - 不跑 StepB
#     - 只跑 StepC + merge
#     - StepC 使用 whole-book
#
# A5: A→B→C（完整方法）
#     - 直接复用完整方法已有结果
#     - 不重新调用模型
#
# B2: w/o Summary Context（StepB）
#     - 复用完整方法的 StepA
#     - 只跑 StepB
#     - 关闭 StepB 的 book summary + page summaries
#
# B3: w/o Evidence Cues（StepB）
#     - 复用完整方法的 StepA
#     - 只跑 StepB
#     - 关闭 StepB 的 text cues + visual cues
#
# B4: w/o Emotion Candidates（StepB）
#     - 复用完整方法的 StepA
#     - 只跑 StepB
#     - 关闭 StepB 的 emotion candidates
#
# B5: w/o Rationale Output（StepB）
#     - 复用完整方法的 StepA
#     - 只跑 StepB
#     - 关闭 StepB 的 “依据” 输出
#
# C2: w/o Evidence Cues (Step C)
#     - 复用完整方法的 StepA + StepB
#     - 只跑 StepC + merge
#     - Step C configuration: No Summary + No Emotion Candidates + Evidence + Anchor
#     - 该组在此基础上关闭 Evidence（text cues + visual cues）
#
# C3: w/o Anchor Hint (Step C)
#     - 复用完整方法的 StepA + StepB
#     - 只跑 StepC + merge
#     - Step C configuration: No Summary + No Emotion Candidates + Evidence + Anchor
#     - 该组在此基础上关闭 Anchor Hint
#
# C4: w/o Evidence + Anchor (Step C)
#     - 复用完整方法的 StepA + StepB
#     - 只跑 StepC + merge
#     - Step C configuration: No Summary + No Emotion Candidates + Evidence + Anchor
#     - 该组在此基础上同时关闭 Evidence 和 Anchor Hint
#
# 选择实验：
EXP_ID="${1:-C4}"


# ===================== Full MOSAIC 基准结果位置 =====================
# 这里固定指向你已经跑好的完整方法结果目录
FULL_MOSAIC_ROOT="./results/mosaic/gpt-4o"

# ===================== 模型配置 =====================
# 本轮消融统一使用 gpt-4o
MODEL_STEPA="gpt-4o"
MODEL_STEPB="gpt-4o"
MODEL_STEPC="gpt-4o"

# ===================== A3 fixed-window 配置 =====================
# 仅 A3 使用；A2/A4/B2-B5/C2-C4 会自动忽略这个参数
WINDOW_SIZE_STEPC=4

# ===================== 选书方式 =====================
# 方式1：通配
#   --book-glob 'book_*.json'
# 方式2：单本
#   --book-glob 'book_4.json'
#   --book-glob 'book_4'
# 方式3：区间
#   --book-glob 'book_1-book_50'
# 方式4：点名多本（中英文逗号/顿号/空格均可）
#   --book-glob 'book_1,book_6,book_50'
# 默认使用区间：
BOOKS_DIR="./data/books"
BOOK_GLOB='book_1'
BATCH_LIMIT=0

# ===================== Payload / 推理控制 =====================
# 打开后：新跑的阶段会写 payload，便于检查实验配置是否正确
DUMP_PAYLOAD=1

# 是否优先复用已存在产物
REUSE_EXISTING=1

# 是否强制重跑
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
