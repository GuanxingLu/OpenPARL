#!/usr/bin/env bash
# Install OpenPARL inside the official miles container.
#
# Prereq: you are inside a running `radixark/miles:latest` container (or a
# pinned tag). This script:
#   1. Overlays the 4 PARL framework hooks on top of the image's miles
#      install via a `pip install --no-deps --force-reinstall` from the
#      GuanxingLu/miles@v0.1-openparl tag. `--no-deps` keeps the image's
#      sglang / megatron / ray versions untouched.
#   2. Installs OpenPARL in editable mode so `python -m openparl.run`
#      resolves to this checkout.
#
# If you are NOT inside the miles container, run it first:
#   docker run --gpus all -it --shm-size=32g --privileged \
#     -v $(pwd):/workspace/OpenPARL \
#     radixark/miles:latest /bin/bash
#   cd /workspace/OpenPARL && ./install.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/2] Overlaying PARL hooks on top of the image's miles..."
pip install --no-deps --force-reinstall \
    "git+https://github.com/GuanxingLu/miles.git@v0.1-openparl"

echo "[2/2] Installing OpenPARL in editable mode..."
pip install -e "${HERE}"

cat <<'EOF'

OpenPARL installed. Next steps:
  1. (Optional) Launch the local RAG server:
       bash scripts/launch_rag_server.sh
  2. Run the WideSearch training:
       bash scripts/run-qwen3-4B-parl.sh
EOF
