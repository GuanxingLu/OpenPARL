"""OpenPARL launcher.

Wraps miles's `train.py` with OpenPARL args (Orchestrator with
`create_subagent` + `assign_task` agent-swarm tools). Launchers under
`scripts/` are thin shells that set env vars and invoke this module via
`python -m openparl.run`.
"""

import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Literal

import miles.utils.external_utils.command_utils as U
import typer

WANDB_PROJECT = "openparl"

DEFAULT_DEV_REPO_DIR = os.environ.get(
    "OPENPARL_DEV_REPO_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)),
)

_MODEL_DEFAULTS = {
    "qwen3-4B": {
        "hf_checkpoint": "MODEL/Qwen3-4B",
        "ref_load": "MODEL/Qwen3-4B_torch_dist",
        "megatron_model_type": "qwen3-4B",
        "tensor_model_parallel_size": 2,
        "rollout_num_gpus_per_engine": 2,
    },
    "qwen3-30B-A3B": {
        "hf_checkpoint": "MODEL/Qwen3-30B-A3B",
        "ref_load": "MODEL/Qwen3-30B-A3B_torch_dist",
        "megatron_model_type": "qwen3-30B-A3B",
        "tensor_model_parallel_size": 4,
        "rollout_num_gpus_per_engine": 4,
    },
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = field(default_factory=U.create_run_id)
    hardware: Literal["H100", "GB200", "GB300"] = "H100"
    num_gpus_per_node: int | None = None
    model: Literal["qwen3-4B", "qwen3-30B-A3B"] = "qwen3-4B"
    dev_repo_dir: str = DEFAULT_DEV_REPO_DIR
    save_path: str = ""
    prompt_data: str = ""
    generate_max_turns: int = 6
    rollout_max_context_len: int = 32768
    rollout_max_response_len: int = 4096
    # Episode-length budget in TURN units (not tokens). Default 2x generate_max_turns
    # is a loose cap; the real turn cap is --generate-max-turns.
    rollout_max_critical_steps: int = 0
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    num_rollout: int = 500
    entropy_coef: float = 0.001
    sglang_router_ip: str = "127.0.0.1"
    sglang_router_port: int = 18765
    sglang_router_prometheus_port: int = 14444
    hf_checkpoint: str = ""
    ref_load: str = ""
    megatron_model_type: str = ""
    tensor_model_parallel_size: int = 0
    rollout_num_gpus_per_engine: int = 0
    # Optional miles --sglang-config YAML for multi-model pools (frozen subagent).
    sglang_config: str = ""
    # Orchestrator tool surface (see BLOG):
    #                     direct tools   subagent tools
    #   single-agent           ✓               ✗        (Single)
    #   delegate-only          ✗               ✓        (Delegate-only)
    #   parl                   ✓               ✓        (PARL)
    agent_mode: Literal["delegate-only", "parl", "single-agent"] = "delegate-only"
    extra_args: str = ""

    def __post_init__(self):
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]
        defaults = _MODEL_DEFAULTS[self.model]
        self.hf_checkpoint = self.hf_checkpoint or f"{self.dev_repo_dir}/{defaults['hf_checkpoint']}"
        self.ref_load = self.ref_load or f"{self.dev_repo_dir}/{defaults['ref_load']}"
        self.megatron_model_type = self.megatron_model_type or defaults["megatron_model_type"]
        self.tensor_model_parallel_size = self.tensor_model_parallel_size or defaults["tensor_model_parallel_size"]
        self.rollout_num_gpus_per_engine = self.rollout_num_gpus_per_engine or defaults["rollout_num_gpus_per_engine"]
        self.prompt_data = self.prompt_data or f"{self.dev_repo_dir}/DATA/wideseek-r1-train/hybrid_20k.miles.jsonl"
        self.rollout_max_critical_steps = self.rollout_max_critical_steps or (2 * self.generate_max_turns)
        if not self.save_path:
            self.save_path = f"{self.dev_repo_dir}/saves/{os.path.basename(self.hf_checkpoint)}-{self.agent_mode}/{self.run_id}"


def _get_wandb_args(args: ScriptArgs) -> str:
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
    return (
        "--use-wandb "
        f"--wandb-project {WANDB_PROJECT} "
        f"--wandb-group {args.model}-{args.agent_mode} "
        f"--wandb-key {WANDB_API_KEY} "
    )


def prepare(args: ScriptArgs):
    hf_dir = os.path.dirname(args.hf_checkpoint)
    U.convert_checkpoint(
        model_name=os.path.basename(args.hf_checkpoint),
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        hf_checkpoint=args.hf_checkpoint,
        dir_dst=hf_dir,
    )


_TOOL_SPECS_PATH = {
    "delegate-only": "openparl.tool.tool_specs",
    "parl": "openparl.widesearch.orchestrator_tools.tool_specs_parl",
    "single-agent": "openparl.widesearch.orchestrator_tools.tool_specs_single",
}

_ORCHESTRATOR_PROMPT_PATH = {
    # delegate-only: empty → generate.py uses ORCHESTRATOR_SYSTEM_PROMPT_DELEGATE_ONLY.
    "delegate-only": "",
    "parl": "openparl.prompts.ORCHESTRATOR_SYSTEM_PROMPT_PARL",
    "single-agent": "openparl.prompts.ORCHESTRATOR_SYSTEM_PROMPT_SINGLE",
}

_DIRECT_TOOLS_DISPATCH = "openparl.widesearch.orchestrator_tools.dispatch"


def execute(args: ScriptArgs):
    megatron_model_type = args.megatron_model_type

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--load {args.save_path} "
        f"--save {args.save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 50} "
    )

    custom_args = (
        "--custom-generate-function-path openparl.generate.generate "
        f"--generate-tool-specs-path {_TOOL_SPECS_PATH[args.agent_mode]} "
        "--generate-tool-call-parser qwen25 "
        f"--generate-max-turns {args.generate_max_turns} "
        "--assign-task-impl-path openparl.widesearch.assign_task.call "
        "--log-multi-turn "
        "--custom-rm-path openparl.widesearch.reward.reward_func "
        "--custom-rollout-log-function-path openparl.rollout_log.log_rollout_data "
        "--custom-eval-rollout-log-function-path openparl.rollout_log.log_eval_rollout_data "
        # --group-rm: reward_func needs the full group to normalize per-turn
        # rewards and populate sample.per_token_advantages (turn-level credit).
        "--group-rm "
    )
    prompt_path = _ORCHESTRATOR_PROMPT_PATH[args.agent_mode]
    if prompt_path:
        custom_args += f"--orchestrator-prompt-path {prompt_path} "
    if args.agent_mode in ("parl", "single-agent"):
        custom_args += f"--orchestrator-direct-tools-path {_DIRECT_TOOLS_DISPATCH} "

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--label-key label "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-context-len {args.rollout_max_context_len} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
        f"--sglang-router-ip {args.sglang_router_ip} "
        f"--sglang-router-port {args.sglang_router_port} "
        "--reward-key score "
        f"--rollout-max-critical-steps {args.rollout_max_critical_steps} "
    )

    eval_args = ""
    if args.mode != "debug_minimal":
        # Multi-set --eval-prompt-data is passed via --extra-args.
        eval_args = (
            "--eval-interval 20 "
            "--n-samples-per-eval-prompt 4 "
            f"--eval-max-response-len {args.rollout_max_response_len} "
            f"--eval-max-context-len {args.rollout_max_context_len} "
            "--eval-top-p 1 "
            "--log-passrate "
        )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        f"--entropy-coef {args.entropy_coef} "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-tis "
        "--custom-tis-function-path miles.backends.training_utils.loss.icepop_function "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} " "--sglang-mem-fraction-static 0.7 "
    )
    if args.sglang_config:
        sglang_args += f"--sglang-config {args.sglang_config} "

    # --sequence-parallel avoids OOM in loss (logits.clone) under long rollouts.
    # --recompute-* avoids holding all layers live through backward. --extra-args
    # can override any of these (argparse last-wins).
    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
    )

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{_get_wandb_args(args)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{custom_args} "
        f"{args.extra_args} "
    )

    # assign_task calls back into SGLang during rollouts, so the router must
    # exist before the ray job starts. miles skips its own router launch when
    # --sglang-router-ip is set, so we pre-launch it in before_ray_job_submit
    # (runs after execute_train's sglang cleanup, before ray submit).
    def _launch_router():
        log_dir = f"{args.dev_repo_dir}/logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = f"{log_dir}/sglang_router.log"
        with open(log_path, "ab") as log_f:
            proc = subprocess.Popen(
                [
                    "python3",
                    "-m",
                    "sglang_router.launch_router",
                    "--host",
                    args.sglang_router_ip,
                    "--port",
                    str(args.sglang_router_port),
                    "--prometheus-port",
                    str(args.sglang_router_prometheus_port),
                    "--log-level",
                    "warn",
                ],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        # Brief wait for the port to bind so engine /workers POSTs succeed.
        for _ in range(30):
            if proc.poll() is not None:
                raise RuntimeError(f"sglang_router exited early (rc={proc.returncode}); see {log_path}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex((args.sglang_router_ip, args.sglang_router_port)) == 0:
                    print(
                        f"sglang_router pid={proc.pid} listening on "
                        f"{args.sglang_router_ip}:{args.sglang_router_port}"
                    )
                    return
            time.sleep(1)
        raise RuntimeError(f"sglang_router pid={proc.pid} did not bind {args.sglang_router_port} in time")

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        # Absolute path so sys.path[0] points at the dev tree, not /root/miles.
        train_script=f"{args.dev_repo_dir}/train.py",
        before_ray_job_submit=_launch_router,
        extra_env_vars={
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            "MILES_SGLANG_ROUTER_IP": args.sglang_router_ip,
            "MILES_SGLANG_ROUTER_PORT": str(args.sglang_router_port),
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            # WANDB_BASE_URL has no CLI equivalent; without forwarding it,
            # remote actors default to wandb.ai and fail in egress-restricted envs.
            **{k: os.environ[k] for k in ("WANDB_BASE_URL", "WANDB_API_KEY") if k in os.environ},
            # Multi-node NCCL transport tuning. Without these the first cross-node
            # collective (Megatron _get_param_groups) fails with ncclRemoteError.
            # NCCL_IB_HCA prefix selects all RoCE HCAs on the cluster.
            **{k: os.environ[k] for k in ("TP_SOCKET_IFNAME",) if k in os.environ},
            "NCCL_IB_HCA": "mlx5_bond",
            "NCCL_CUMEM_ENABLE": "0",
            "NVTE_BWD_LAYERNORM_SM_MARGIN": "20",
            "NCCL_IB_TC": "160",
            "NCCL_PXN_DISABLE": "0",
            "NCCL_IB_GID_INDEX": "3",
            "NCCL_NET_GDR_LEVEL": "4",
            "NCCL_IB_RETRY_CNT": "7",
            "NCCL_IB_TIMEOUT": "32",
            "NCCL_IB_QPS_PER_CONNECTION": "8",
            "NCCL_P2P_LEVEL": "NVL",
            "NCCL_MIN_CTAS": "4",
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
