#!/bin/bash
# Launches the E5 + Qdrant retrieval server for the widesearch benchmark.
#
# The server code lives under ./rag_server/ (vendored from RLinf's
# examples/agent/tools/search_local_server_qdrant/, Apache-2.0; see the
# license headers inside each file). miles no longer depends on an external
# RLinf checkout.
#
# Prereqs (one-time):
#   uv pip install qdrant-client==1.16.2
#   Start Qdrant on :6333 (one tmux window):
#     cd ${DATA_ROOT}/wiki-2018-corpus/qdrant && ./qdrant
#
# This script starts the E5 encoder HTTP server on :8000 (another tmux window),
# which proxies /retrieve -> Qdrant + /access -> wiki_webpages.jsonl.
#
# Env overrides:
#   DATA_ROOT   miles DATA root (default ../../../DATA from this script)
#   MODEL_ROOT  miles MODEL root (default ../../../MODEL from this script)
#   PORT        HTTP port to bind (default 8000)
#   QDRANT_URL  Qdrant URL (default http://localhost:6333)

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

DATA_ROOT=${DATA_ROOT:-${REPO_DIR}/DATA}
MODEL_ROOT=${MODEL_ROOT:-${REPO_DIR}/MODEL}
PORT=${PORT:-8000}
QDRANT_URL=${QDRANT_URL:-http://localhost:6333}

SERVER_DIR="${SCRIPT_DIR}/rag_server"
if [ ! -f "${SERVER_DIR}/local_retrieval_server.py" ]; then
   echo "ERROR: ${SERVER_DIR}/local_retrieval_server.py not found." >&2
   exit 1
fi

WIKI_DIR="${DATA_ROOT}/wiki-2018-corpus"
PAGES_FILE="${WIKI_DIR}/wiki_webpages.jsonl"
RETRIEVER_PATH="${MODEL_ROOT}/e5-base-v2"

for p in "${PAGES_FILE}" "${RETRIEVER_PATH}"; do
   if [ ! -e "${p}" ]; then
      echo "ERROR: missing required path: ${p}" >&2
      exit 1
   fi
done

cd "${SERVER_DIR}"
python3 -u local_retrieval_server.py \
   --pages_path "${PAGES_FILE}" \
   --topk 3 \
   --retriever_name e5 \
   --retriever_model "${RETRIEVER_PATH}" \
   --qdrant_collection_name wiki_collection_m32_cef512 \
   --qdrant_url "${QDRANT_URL}" \
   --qdrant_search_param '{"hnsw_ef":256}' \
   --port "${PORT}"
