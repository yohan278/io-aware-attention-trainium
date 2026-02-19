#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ioattn-trn2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_ENV_FILE="${CONDA_ENV_FILE:-${ROOT_DIR}/conda/environment.trainium.yml}"

echo "[bootstrap] Repository root: ${ROOT_DIR}"
echo "[bootstrap] Conda env: ${CONDA_ENV_NAME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[bootstrap] Error: 'conda' command not found."
  echo "[bootstrap] Install Miniconda/Anaconda on the host, then rerun."
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
  echo "[bootstrap] Reusing existing conda env."
else
  echo "[bootstrap] Creating conda env with python=${PYTHON_VERSION}..."
  conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda activate "${CONDA_ENV_NAME}"

if [[ -f "${CONDA_ENV_FILE}" ]]; then
  echo "[bootstrap] Updating env from ${CONDA_ENV_FILE}..."
  conda env update -n "${CONDA_ENV_NAME}" -f "${CONDA_ENV_FILE}" --prune
else
  echo "[bootstrap] Error: missing conda environment file ${CONDA_ENV_FILE}"
  exit 1
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${ROOT_DIR}"

echo "[bootstrap] Validating environment..."
python "${ROOT_DIR}/scripts/validate_trainium_env.py"
echo "[bootstrap] Done."
