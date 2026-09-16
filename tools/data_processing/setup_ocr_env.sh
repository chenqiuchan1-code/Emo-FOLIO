#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${1:-emo-folio-ocr}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required to create the OCR environment."
  exit 1
fi

conda create --name "${ENV_NAME}" python=3.10 pip -y
conda run --name "${ENV_NAME}" python -m pip install \
  -r "${REPO_ROOT}/tools/data_processing/requirements-runtime.txt"
conda run --name "${ENV_NAME}" python -m pip install --no-deps paddleocr==2.7.0.3
conda run --name "${ENV_NAME}" python \
  "${REPO_ROOT}/scripts/check_environment.py" --scope ocr

echo "OCR environment '${ENV_NAME}' is ready."
echo "Activate it with: conda activate ${ENV_NAME}"
