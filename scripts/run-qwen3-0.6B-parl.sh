#!/bin/bash

# PARL v2 widesearch smoke test on Qwen3-0.6B (4x small GPUs).
# Thin wrapper around examples/parl_v2/run_parl_v2.py with --env widesearch.
#
# Prereq: local RAG server running on :8000. Launch it in a separate tmux:
#   bash examples/parl_v2/widesearch/launch_rag_server.sh
# and verify:
#   curl -XPOST localhost:8000/retrieve -d '{"queries":["test"],"topk":3}'

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
# Targeted python kill so we don't clobber the long-running RAG server
# (examples/agent/tools/search_local_server_qdrant/local_retrieval_server.py).
# Matches ray workers + miles train.py + parl_v2 launcher processes only.
pkill -9 -f 'ray::\|train\.py\|parl_v2\|run_parl_v2' || true
sleep 3
pkill -9 ray
pkill -9 -f 'ray::\|train\.py\|parl_v2\|run_parl_v2' || true

set -ex

export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Strip dev-host proxies so httpx/wandb don't route through a missing socksio backend.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export WANDB_API_KEY=${WANDB_API_KEY:-local-82cbbacfe8e3c0c527da528160bd76a1e85c9fea}
export WANDB_BASE_URL=${WANDB_BASE_URL:-http://33.180.4.104}
export WANDB_MODE=${WANDB_MODE:-offline}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

DEV_REPO_DIR=${DEV_REPO_DIR:-${REPO_DIR}}
DATA_ROOT=${DATA_ROOT:-${DEV_REPO_DIR}/DATA}
MODEL_ROOT=${MODEL_ROOT:-${DEV_REPO_DIR}/MODEL}
MODE=${MODE:-normal}  # debug_minimal | normal
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}

# widesearch-specific env vars consumed by widesearch/assign_task.py.
export MILES_PARL_V2_RAG_SERVER=${MILES_PARL_V2_RAG_SERVER:-localhost:8000}
export MILES_PARL_V2_SUBAGENT_MAX_TURNS=${MILES_PARL_V2_SUBAGENT_MAX_TURNS:-4}
export MILES_PARL_V2_SUBAGENT_MAX_TOOLCALLS=${MILES_PARL_V2_SUBAGENT_MAX_TOOLCALLS:-6}

MODEL_ARGS=(
   --env widesearch
   --model qwen3-0.6B
   --hf-checkpoint "${MODEL_ROOT}/Qwen3-0.6B"
   --ref-load "${MODEL_ROOT}/Qwen3-0.6B_torch_dist"
)

RUN_ARGS=(
   --mode "${MODE}"
   --run-id "${RUN_ID}"
   --dev-repo-dir "${DEV_REPO_DIR}"
   --save-path "${DEV_REPO_DIR}/saves/Qwen3-0.6B-parl-v2-widesearch/${RUN_ID}"
   --rollout-batch-size 2
   --n-samples-per-prompt 2
   --global-batch-size 4
   --num-rollout 10
   --rollout-max-response-len 8192
   --rollout-max-critical-steps 32
)

PARALLEL_ARGS=(
   --num-gpus-per-node "${NUM_GPUS}"
   --tensor-model-parallel-size 1
   --rollout-num-gpus-per-engine 1
)

# Smoke slice: first 128 rows of the hybrid training split (miles supports
# the @[start:end] suffix on prompt paths via its generalized-path parser).
DATA_ARGS=(
   --prompt-data "${DATA_ROOT}/wideseek-r1-train/hybrid_20k.miles.jsonl@[:128]"
)

GENERATE_ARGS=(
   --generate-max-turns 4
)

# Multi-eval through --extra-args. 4 sets: widesearch (primary, OOD), plus
# two multi-hop QA sets + one single-hop for sanity. Keep to small subsets
# in smoke via @[:n] slices.
EVAL_EXTRA_ARGS=(
   --eval-interval 5
   --n-samples-per-eval-prompt 2
   --eval-max-response-len 8192
   --eval-max-context-len 32768
   --eval-top-p 1
   --log-passrate
   --eval-prompt-data
   widesearch "${DATA_ROOT}/widesearch-test/test.miles.jsonl@[:16]"
   hotpotqa   "${DATA_ROOT}/asearcher-test/HotpotQA_rand1000/test.miles.jsonl@[:32]"
   2wiki      "${DATA_ROOT}/asearcher-test/2WikiMultihopQA_rand1000/test.miles.jsonl@[:32]"
   bamboogle  "${DATA_ROOT}/asearcher-test/Bamboogle/test.miles.jsonl@[:32]"
)

cd "${REPO_DIR}"

RAY_GCS_PORT=${RAY_GCS_PORT:-26379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-28265}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --port ${RAY_GCS_PORT} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASHBOARD_PORT}
export RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
export MILES_SCRIPT_EXTERNAL_RAY=1

SGLANG_ROUTER_IP=${MILES_SGLANG_ROUTER_IP:-127.0.0.1}
SGLANG_ROUTER_PORT=${MILES_SGLANG_ROUTER_PORT:-18765}

SUBAGENT_MODE=${SUBAGENT_MODE:-frozen}
if [ "$SUBAGENT_MODE" = "frozen" ]; then
   SGLANG_EXTRA_ARGS=(--sglang-config examples/parl_v2/sglang_config_0.6B.yaml)
elif [ "$SUBAGENT_MODE" = "shared" ]; then
   SGLANG_EXTRA_ARGS=()
else
   echo "ERROR: SUBAGENT_MODE must be 'frozen' or 'shared', got '$SUBAGENT_MODE'" >&2
   exit 1
fi

python examples/parl_v2/run_parl_v2.py \
   ${MODEL_ARGS[@]} \
   ${RUN_ARGS[@]} \
   ${PARALLEL_ARGS[@]} \
   ${DATA_ARGS[@]} \
   ${GENERATE_ARGS[@]} \
   --sglang-router-ip ${SGLANG_ROUTER_IP} \
   --sglang-router-port ${SGLANG_ROUTER_PORT} \
   "${SGLANG_EXTRA_ARGS[@]}" \
   --extra-args "${EVAL_EXTRA_ARGS[*]}"
