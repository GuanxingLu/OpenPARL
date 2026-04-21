"""Multi-turn rollout with create_subagent / assign_task.

critical_steps advances by 1 per orchestrator turn, plus max_i S_sub_i
on turns with >=1 assign_task (paper: S_main + max_i S_sub,i).
"""

import argparse
import asyncio
import json
import logging
import uuid
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import create_tool_call_parser, update_sample_with_tool_responses
from miles.rollout.sglang_rollout import get_model_url
from miles.utils.http_utils import post
from miles.utils.misc import load_function
from miles.utils.types import Sample
from openai.types.chat import ChatCompletionMessageToolCall
from sglang.srt.function_call.core_types import ToolCallItem

from .prompts import ORCHESTRATOR_SYSTEM_PROMPT
from .tool import _create_subagent

logger = logging.getLogger(__name__)
_logged_endpoint = False

MAX_CONCURRENT_ASSIGN = 8
MAX_CONCURRENT_DIRECT = 16


def _with_system_prompt(prompt, system_content: str):
    if isinstance(prompt, str):
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
    if isinstance(prompt, list) and (not prompt or prompt[0].get("role") != "system"):
        return [{"role": "system", "content": system_content}] + list(prompt)
    return prompt


def _normalize_tool_call(call) -> tuple[str, dict, str]:
    """Return (name, params, tool_call_id). Mirrors miles' _execute_tool_call.

    ``params`` is always a dict; malformed / non-object arguments (model emits
    ``"arguments": "<string>"`` or invalid JSON that still slipped past the
    detector) are coerced to ``{}`` so per-tool isinstance checks produce
    actionable error strings instead of crashing the whole rollout.
    """
    if isinstance(call, ChatCompletionMessageToolCall):
        name = call.function.name
        raw = call.function.arguments
        tool_call_id = call.id
    elif isinstance(call, ToolCallItem):
        name = call.name
        raw = call.parameters
        tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
    else:
        raise TypeError(f"Unsupported tool call type: {type(call)}")

    try:
        params = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        params = {}
    if not isinstance(params, dict):
        params = {}
    return name, params, tool_call_id


async def _execute_tool_calls_parallel(
    tool_calls,
    *,
    registry: dict[str, str],
    tokenizer,
    router_url: str,
    assign_task_impl,
    direct_dispatch,
) -> tuple[list[dict], dict]:
    """Two-phase execution:

    - Phase 1 (sync): every ``create_subagent`` runs inline (registry write,
      no inference).
    - Phase 2 (async, parallel): ``assign_task`` calls (cap
      ``MAX_CONCURRENT_ASSIGN``) and direct-dispatch calls (cap
      ``MAX_CONCURRENT_DIRECT``) are gathered together.

    ``direct_dispatch`` is an optional ``async (name, params) -> str | None``
    callable; ``None`` return means "unknown tool" and falls through to
    the error path. Set to ``None`` (the arg, not the return) for
    swarm-strict, where the Orchestrator only owns subagent tools.

    Preserves original tool_call order so response-text ↔ tool_response
    mapping stays intact. Returns ``(tool_messages, per_turn_stats)``.
    """
    normalized = [_normalize_tool_call(c) for c in tool_calls]
    results: list[str | None] = [None] * len(normalized)

    for i, (name, params, _) in enumerate(normalized):
        if name == "create_subagent":
            results[i] = _create_subagent(params, registry=registry)

    assign_indices = [i for i, (n, _, _) in enumerate(normalized) if n == "assign_task"]
    allowed_assign = assign_indices[:MAX_CONCURRENT_ASSIGN]
    denied_assign = assign_indices[MAX_CONCURRENT_ASSIGN:]

    direct_indices: list[int] = []
    if direct_dispatch is not None:
        for i, (n, _, _) in enumerate(normalized):
            if n in ("create_subagent", "assign_task"):
                continue
            if results[i] is not None:
                continue
            direct_indices.append(i)
    allowed_direct = direct_indices[:MAX_CONCURRENT_DIRECT]
    denied_direct = direct_indices[MAX_CONCURRENT_DIRECT:]

    async def run_assign(i: int):
        _, params, _ = normalized[i]
        text, is_valid, sub_steps = await assign_task_impl(
            params, registry=registry, tokenizer=tokenizer, router_url=router_url
        )
        return i, text, is_valid, sub_steps

    async def run_direct(i: int):
        name, params, _ = normalized[i]
        text = await direct_dispatch(name, params)
        return i, name, text

    assign_outputs, direct_outputs = await asyncio.gather(
        asyncio.gather(*[run_assign(i) for i in allowed_assign]),
        asyncio.gather(*[run_direct(i) for i in allowed_direct]),
    )

    n_valid = 0
    max_sub_steps = 0
    for i, text, is_valid, sub_steps in assign_outputs:
        results[i] = text
        if is_valid:
            n_valid += 1
        if sub_steps > max_sub_steps:
            max_sub_steps = sub_steps

    n_search = 0
    n_access = 0
    for i, name, text in direct_outputs:
        if text is None:
            # dispatcher did not recognize this tool; leave None so the
            # "unknown tool" error path below fires.
            continue
        results[i] = text
        if name == "search":
            n_search += 1
        elif name == "access":
            n_access += 1

    for i in denied_assign:
        results[i] = (
            f"Error: too many assign_task calls in this turn " f"(cap={MAX_CONCURRENT_ASSIGN}). Retry in a later turn."
        )
    for i in denied_direct:
        results[i] = (
            f"Error: too many direct tool calls in this turn " f"(cap={MAX_CONCURRENT_DIRECT}). Retry in a later turn."
        )

    for i, (name, _, _) in enumerate(normalized):
        if results[i] is None:
            results[i] = f"Error: unknown tool '{name}'"

    tool_messages = [
        {"role": "tool", "tool_call_id": tool_call_id, "content": result, "name": name}
        for (name, _, tool_call_id), result in zip(normalized, results, strict=True)
    ]

    stats = {
        "n_create": sum(1 for n, _, _ in normalized if n == "create_subagent"),
        "n_assign": len(allowed_assign),
        "n_valid": n_valid,
        "n_search": n_search,
        "n_access": n_access,
        "max_sub_steps": max_sub_steps,
    }
    return tool_messages, stats


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = deepcopy(input.sample)

    orch_prompt = ORCHESTRATOR_SYSTEM_PROMPT
    if getattr(args, "orchestrator_prompt_path", None):
        orch_prompt = load_function(args.orchestrator_prompt_path)
    sample.prompt = _with_system_prompt(sample.prompt, orch_prompt)

    tokenizer = input.state.tokenizer
    subagent_router_url = get_model_url(args, "subagent")
    live_router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    global _logged_endpoint
    if not _logged_endpoint:
        mode = "frozen" if subagent_router_url != live_router_url else "shared (ablation)"
        logger.info(f"[openparl] subagent mode: {mode}")
        logger.info(f"[openparl] subagent router: {subagent_router_url}")
        logger.info(f"[openparl] live router:     {live_router_url}")
        _logged_endpoint = True
    assert not args.partial_rollout, "Partial rollout is not supported"
    assert not args.generate_multi_samples, "generate_multi_samples is not supported in openparl custom multi-turn"

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    tool_specs = load_function(args.generate_tool_specs_path)
    tool_call_parser = create_tool_call_parser(tool_specs, args.generate_tool_call_parser)
    assign_task_impl = load_function(args.assign_task_impl_path)

    direct_dispatch = None
    if getattr(args, "orchestrator_direct_tools_path", None):
        direct_dispatch = load_function(args.orchestrator_direct_tools_path)

    max_cs_raw = getattr(args, "rollout_max_critical_steps", None)
    max_cs = int(max_cs_raw) if max_cs_raw is not None else 2 * int(args.generate_max_turns)

    registry: dict[str, str] = {}
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["critical_steps"] = 0
    sample.metadata["turns"] = []

    prompt_tokens_ids = compute_prompt_ids_from_sample(input.state, sample, tools=tool_specs)
    sample.tokens = prompt_tokens_ids.copy()

    for _turn in range(args.generate_max_turns):
        if sample.metadata["critical_steps"] >= max_cs:
            sample.status = Sample.Status.TRUNCATED
            break

        payload, halt_status = compute_request_payload(args, sample.tokens, input.sampling_params)
        if payload is None:
            sample.status = halt_status
            break

        output = await post(url, payload)
        await update_sample_from_response(args, sample, payload=payload, output=output, update_loss_mask=True)

        if output["meta_info"]["finish_reason"]["type"] in ("abort", "length"):
            sample.metadata["turns"].append({"n_create": 0, "n_assign": 0, "n_valid": 0, "final": False})
            sample.metadata["critical_steps"] += 1
            break

        raw_text = output["text"]
        raw_tool_call_count = raw_text.count(tool_call_parser.detector.bot_token)
        _, tool_calls = tool_call_parser.parse_non_stream(raw_text)
        failed = raw_tool_call_count - len(tool_calls)
        if failed > 0:
            sample.metadata["tool_call_parse_failures"] = sample.metadata.get("tool_call_parse_failures", 0) + failed
            sample.metadata["tool_call_raw_count"] = (
                sample.metadata.get("tool_call_raw_count", 0) + raw_tool_call_count
            )

        if len(tool_calls) == 0:
            sample.metadata["turns"].append({"n_create": 0, "n_assign": 0, "n_valid": 0, "final": False})
            sample.metadata["critical_steps"] += 1
            break

        tool_messages, stats = await _execute_tool_calls_parallel(
            tool_calls,
            registry=registry,
            tokenizer=tokenizer,
            router_url=subagent_router_url,
            assign_task_impl=assign_task_impl,
            direct_dispatch=direct_dispatch,
        )
        sample.metadata["turns"].append({**stats, "final": False})
        # critical_steps += 1 + (max_i S_sub,i if n_assign > 0 else 0)
        sample.metadata["critical_steps"] += 1 + (stats["max_sub_steps"] if stats["n_assign"] > 0 else 0)

        update_sample_with_tool_responses(sample, tool_messages, tokenizer=tokenizer)

    if sample.metadata["turns"]:
        sample.metadata["turns"][-1]["final"] = True
    sample.metadata["registry_size"] = len(registry)

    return GenerateFnOutput(samples=sample)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-max-turns", type=int, default=16)
    parser.add_argument("--generate-tool-specs-path", type=str)
    parser.add_argument("--generate-tool-call-parser", type=str)
    parser.add_argument("--generate-execute-tool-function-path", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true")
    parser.add_argument(
        "--assign-task-impl-path",
        type=str,
        default="openparl.widesearch.assign_task.call",
        help=(
            "Importable path to an async `call(params, *, registry, tokenizer, router_url)` "
            "function implementing the assign_task subagent inference."
        ),
    )
    parser.add_argument(
        "--orchestrator-prompt-path",
        type=str,
        default=None,
        help=(
            "Importable path to the Orchestrator system prompt string. Unset "
            "= the swarm-strict default in openparl.prompts. "
            "Swarm-paper sets ORCHESTRATOR_SYSTEM_PROMPT_PAPER; single-agent "
            "baseline sets ORCHESTRATOR_SYSTEM_PROMPT_SINGLE."
        ),
    )
    parser.add_argument(
        "--orchestrator-direct-tools-path",
        type=str,
        default=None,
        help=(
            "Importable path to an async `dispatch(name, params) -> str | None` "
            "coroutine that handles Orchestrator-side direct tool calls (e.g., "
            "widesearch search/access). Unset = swarm-strict: Orchestrator "
            "holds only create_subagent / assign_task. For widesearch use "
            "openparl.widesearch.orchestrator_tools.dispatch."
        ),
    )
    parser.add_argument(
        "--rollout-max-critical-steps",
        type=int,
        default=None,
        help=(
            "K2.5 PARL episode-length budget in TURN units: phase_cost = 1 per "
            "orchestrator turn (final/create-only/length) or 2 per spawn turn "
            "(≥1 executed assign_task). Defaults to 2 * --generate-max-turns."
        ),
    )


generate.add_arguments = _add_arguments
