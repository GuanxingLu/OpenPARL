#!/usr/bin/env bash
# Install OpenPARL + its pinned miles fork.
# Requirements: Python 3.10+, pip, git, CUDA-capable GPU for training
# (tests in tests/ are CPU-only).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/2] Installing miles@v0.1-openparl (PARL framework hooks on radixark/miles@5d11fe2f0)..."
pip install "git+https://github.com/GuanxingLu/miles.git@v0.1-openparl"

echo "[2/2] Installing OpenPARL in editable mode..."
pip install -e "${HERE}"

cat <<'EOF'

OpenPARL installed. Next steps:
  1. (Optional) Launch the local RAG server:
       bash scripts/launch_rag_server.sh
  2. Run the WideSearch training:
       bash scripts/run-qwen3-4B-parl.sh
  See docs/reproducibility.md for hardware / environment details.
EOF
