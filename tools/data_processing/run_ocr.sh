#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# ===================== 参数区（按需改）=====================

PY="${PYTHON_BIN:-python}"
SCRIPT="./tools/data_processing/pb_ocr_cleanup_to_json.py"

# 输入/输出根目录
INITIAL_ROOT="./inputs/raw_books"        # user-supplied scans: category/book/pages
PICTURE_ROOT="./outputs/ocr/images"      # text-free page images
BOOKS_ROOT="./outputs/ocr/books"         # corrected book JSON files

# 选择器（留空=全部）
TYPES=""                          # 例如：情景教育类,益智类
BOOKS="1..50"                             # 例如：book_8、8..12 或书名关键词

# 只列出/只预演（true/false）
DO_LIST=false                     # true: 仅列出将要处理的书，不执行
DRY_RUN=false                     # true: 仅预览，不写文件

# OCR and text-removal parameters
SCALE=1.8
MIN_PROB=0.80
TEXT_BAND_Y=0.55
DILATE=8
RADIUS=3
INPAINT="ns"                      # ns / telea
MASK_MIN_PROB=0.80
MAX_MASK_AREA_FRAC=0.12
MAX_MASK_H_FACTOR=3.5

# 段落合并参数
X_OVERLAP=0.35
VGAP_FACTOR=1.30
LINE_MERGE_FACTOR=0.60

# ===================== 执行区（一般不改）=====================

cmd=(
  "$PY" "$SCRIPT"
  --initial_root "$INITIAL_ROOT"
  --picture_root "$PICTURE_ROOT"
  --books_root "$BOOKS_ROOT"
  --scale "$SCALE"
  --min_prob "$MIN_PROB"
  --text_band_y "$TEXT_BAND_Y"
  --dilate "$DILATE"
  --radius "$RADIUS"
  --inpaint "$INPAINT"
  --mask_min_prob "$MASK_MIN_PROB"
  --max_mask_area_frac "$MAX_MASK_AREA_FRAC"
  --max_mask_h_factor "$MAX_MASK_H_FACTOR"
  --x_overlap "$X_OVERLAP"
  --vgap_factor "$VGAP_FACTOR"
  --line_merge_factor "$LINE_MERGE_FACTOR"
)

[[ -n "$TYPES" ]] && cmd+=( --types "$TYPES" )
[[ -n "$BOOKS" ]] && cmd+=( --books "$BOOKS" )
[[ "$DO_LIST" == "true" ]] && cmd+=( --list )
[[ "$DRY_RUN" == "true" ]] && cmd+=( --dry_run )

echo "[RUN] ${cmd[*]}"
exec "${cmd[@]}"
