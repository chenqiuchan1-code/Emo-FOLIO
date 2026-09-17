#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# ===================== User configuration =====================

PY="${PYTHON_BIN:-python}"
SCRIPT="./tools/data_processing/pb_ocr_cleanup_to_json.py"

# Input and output roots
INITIAL_ROOT="./inputs/raw_books"        # user-supplied scans: category/book/pages
PICTURE_ROOT="./outputs/ocr/images"      # text-free page images
BOOKS_ROOT="./outputs/ocr/books"         # OCR-derived book JSON files

# Optional selectors; leave empty to process all discovered books.
TYPES=""                                 # Comma-separated category names.
BOOKS="1..34"                            # Example: book_8, 8..12, or a title keyword.

# Inspection controls (true/false)
DO_LIST=false                            # List selected books without processing.
DRY_RUN=false                            # Preview processing without writing files.

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

# OCR paragraph-merging parameters
X_OVERLAP=0.35
VGAP_FACTOR=1.30
LINE_MERGE_FACTOR=0.60

# ===================== Execution (normally unchanged) =====================

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
