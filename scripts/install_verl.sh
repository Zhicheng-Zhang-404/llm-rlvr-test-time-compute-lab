#!/usr/bin/env bash
set -euo pipefail

# Run inside the supplied pedagogy conda environment. v0.4.1 is intentionally
# pinned: its declared extras support torch 2.6 and vLLM 0.8.2--0.8.5.
INSTALL_DIR="${VERL_DIR:-${PWD}/work/verl}"
VERL_REF="${VERL_REF:-v0.4.1}"
if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
  git clone --branch "${VERL_REF}" --depth 1 https://github.com/verl-project/verl.git "${INSTALL_DIR}"
else
  git -C "${INSTALL_DIR}" fetch --depth 1 origin "${VERL_REF}"
  git -C "${INSTALL_DIR}" checkout --detach FETCH_HEAD
fi

python -m pip install -e "${INSTALL_DIR}[vllm,math]"
python -m pip install "tensordict==0.6.2"
python scripts/check_environment.py --require-gpu --require-verl
