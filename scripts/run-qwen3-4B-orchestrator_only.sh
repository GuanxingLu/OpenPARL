#!/bin/bash

# PARL v2 widesearch run — paper-faithful swarm (Arm B).
#
# Orchestrator gets direct search/access in addition to create_subagent /
# assign_task. Frozen-subagent topology and everything else matches
# run-qwen3-4B-widesearch.sh (Arm A, swarm-strict); only --agent-mode and
# the save-subdir differ. Prereq: local RAG server running on :8765.

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
# Targeted python kill so we don't clobber the long-running RAG server.
pkill -9 -f 'ray::\|train\.py\|parl_v2\|run_parl_v2' || true
sleep 3
pkill -9 ray
pkill -9 -f 'ray::\|train\.py\|parl_v2\|run_parl_v2' || true

set -ex

export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export WANDB_API_KEY=${WANDB_API_KEY:-local-82cbbacfe8e3c0c527da528160bd76a1e85c9fea}
export WANDB_BASE_URL=${WANDB_BASE_URL:-http://33.180.4.104}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

DEV_REPO_DIR=${DEV_REPO_DIR:-${REPO_DIR}}
DATA_ROOT=${DATA_ROOT:-${DEV_REPO_DIR}/DATA}
MODEL_ROOT=${MODEL_ROOT:-${DEV_REPO_DIR}/MODEL}
MODE=${MODE:-normal}
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}

# widesearch-specific env vars consumed by widesearch/assign_task.py AND
# widesearch/orchestrator_tools.py (both read MILES_PARL_V2_RAG_SERVER).
export MILES_PARL_V2_RAG_SERVER=${MILES_PARL_V2_RAG_SERVER:-localhost:8000}
export MILES_PARL_V2_SUBAGENT_MAX_TURNS=${MILES_PARL_V2_SUBAGENT_MAX_TURNS:-8}
export MILES_PARL_V2_SUBAGENT_MAX_TOOLCALLS=${MILES_PARL_V2_SUBAGENT_MAX_TOOLCALLS:-10}
export MILES_PARL_V2_SUBAGENT_CONCURRENCY=${MILES_PARL_V2_SUBAGENT_CONCURRENCY:-32}

MODEL_ARGS=(
   --env widesearch
   --model qwen3-4B
   --hf-checkpoint "${MODEL_ROOT}/Qwen3-4B"
   --ref-load "${MODEL_ROOT}/Qwen3-4B_torch_dist"
)

RUN_ARGS=(
   --mode "${MODE}"
   --run-id "${RUN_ID}"
   --dev-repo-dir "${DEV_REPO_DIR}"
   --save-path "${DEV_REPO_DIR}/saves/Qwen3-4B-parl-v2-paper-widesearch/${RUN_ID}"
   --rollout-batch-size 64
   --global-batch-size 512
   --rollout-max-response-len 28672
   --rollout-max-critical-steps 48
   --entropy-coef 0
)

PARALLEL_ARGS=(
   --num-gpus-per-node "${NUM_GPUS}"
   --tensor-model-parallel-size 2
   --rollout-num-gpus-per-engine 2
)

DATA_ARGS=(
   --prompt-data "${DATA_ROOT}/wideseek-r1-train/hybrid_20k.miles.jsonl"
)

GENERATE_ARGS=(
   --generate-max-turns 10
)

EVAL_EXTRA_ARGS=(
   --eval-interval 20
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 28672
   --eval-max-context-len 32768
   --eval-top-p 1
   --log-passrate
   --eval-prompt-data
   widesearch "${DATA_ROOT}/widesearch-test/test.miles.jsonl"
   hotpotqa   "${DATA_ROOT}/asearcher-test/HotpotQA_rand1000/test.miles.jsonl"
   2wiki      "${DATA_ROOT}/asearcher-test/2WikiMultihopQA_rand1000/test.miles.jsonl"
   bamboogle  "${DATA_ROOT}/asearcher-test/Bamboogle/test.miles.jsonl"
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

OPTIM_OVERRIDE_ARGS=(
   --weight-decay 0.01
   --adam-beta2 0.999
   --disable-entropy-computation
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

# swarm-paper keeps the frozen-subagent topology: Orchestrator's direct
# search/access hit the RAG server; its assign_task spawns still go to
# the frozen-SGLang pool declared in sglang_config_4B.yaml.
SUBAGENT_MODE=${SUBAGENT_MODE:-frozen}
if [ "$SUBAGENT_MODE" = "frozen" ]; then
   SGLANG_EXTRA_ARGS=(--sglang-config examples/parl_v2/sglang_config_4B.yaml)
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
   "${SGLANG_EXTRA_ARGS[@]}" \
   --agent-mode swarm-paper \
   --extra-args "${EVAL_EXTRA_ARGS[*]} ${PERF_ARGS[*]} ${OPTIM_OVERRIDE_ARGS[*]}"
