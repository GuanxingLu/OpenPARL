"""Widesearch assign_task impl: multi-turn ReAct subagent worker.

Topology Y from the design discussion: orchestrator keeps the single
``create_subagent``/``assign_task`` tool pair (context stays sharded);
each ``assign_task`` spawns one subagent here that runs a ReAct loop
against the local RAG server (``search`` + ``access``) until it either
produces a ``<result>…</result>`` block or exhausts its turn / tool-call
budget.

Environment variables drive the RAG endpoint and the subagent budgets:

- ``MILES_PARL_V2_RAG_SERVER``         default ``localhost:8000``
- ``MILES_PARL_V2_SUBAGENT_MAX_TURNS`` default ``8``   (generation calls per subagent)
- ``MILES_PARL_V2_SUBAGENT_MAX_TOOLCALLS`` default ``10`` (total tool_calls per subagent)
- ``MILES_PARL_V2_SUBAGENT_MAX_NEW_TOKENS`` default ``1024`` (per generation call)
- ``MILES_PARL_V2_SUBAGENT_CONCURRENCY`` default ``16`` (semaphore across subagents)

They're read lazily so the launch script can set them without touching
miles' CLI schema.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid

from openai.types.chat import ChatCompletionMessageToolCall
from sglang.srt.function_call.core_types import ToolCallItem

from miles.rollout.generate_utils.tool_call_utils import create_tool_call_parser
from miles.utils.http_utils import post

from ..tool import _RESULT_RE, SUBAGENT_OUTPUT_SUFFIX, extract_subagent_result
from . import search_client
from .subagent_prompts import SUBAGENT_REACT_SUFFIX
from .subagent_prompts import tool_specs as subagent_tool_specs

logger = logging.getLogger(__name__)

_solver_semaphore: asyncio.Semaphore | None = None
_tool_call_parser = None  # lazy


def _get_semaphore() -> asyncio.Semaphore:
    global _solver_semaphore
    if _solver_semaphore is None:
        limit = int(os.environ.get("MILES_PARL_V2_SUBAGENT_CONCURRENCY", "16"))
        _solver_semaphore = asyncio.Semaphore(limit)
    return _solver_semaphore


def _get_parser():
    global _tool_call_parser
    if _tool_call_parser is None:
        _tool_call_parser = create_tool_call_parser(subagent_tool_specs, "qwen25")
    return _tool_call_parser


def _normalize_tool_call(tc) -> tuple[str, dict, str]:
    """Return (name, params, tool_call_id). Mirrors generate.py's helper."""
    if isinstance(tc, ChatCompletionMessageToolCall):
        name = tc.function.name
        params = json.loads(tc.function.arguments) if tc.function.arguments else {}
        tool_call_id = tc.id
    elif isinstance(tc, ToolCallItem):
        name = tc.name
        params = json.loads(tc.parameters) if tc.parameters else {}
        tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
    else:
        raise TypeError(f"Unsupported tool call type: {type(tc)}")
    return name, params, tool_call_id


async def _dispatch_tool(name: str, params: dict, server_addr: str, access_max_chars: int) -> str:
    """Run one search/access call and return a string result."""
    if name == "search":
        query = params.get("query", "")
        topk = int(params.get("topk", 3) or 3)
        return await search_client.search(server_addr, query, topk=topk)
    if name == "access":
        url = params.get("url", "")
        return await search_client.access(server_addr, url, max_chars=access_max_chars)
    return f"Error: unknown tool '{name}'."


async def call(params: dict, *, registry: dict[str, str], tokenizer, router_url: str) -> tuple[str, bool, int]:
    """Multi-turn ReAct subagent. Returns (body, is_valid, sub_steps).

    - ``body``: content of the last ``<result>…</result>`` block if emitted,
      else the raw last-turn text (or an ``__SOLVER_ERROR__`` marker).
    - ``is_valid``: True iff the subagent (a) produced non-empty output,
      (b) wrapped it in a ``<result>`` block, and (c) used ``search`` or
      ``access`` at least once. Untool'd answers are treated as invalid —
      they correspond to the model hallucinating instead of researching.
    - ``sub_steps``: number of SGLang generation calls made (ReAct loop
      iterations). Feeds K2.5 critical-steps accounting upstream.
    """
    agent = params.get("agent")
    prompt = params.get("prompt")
    if not isinstance(agent, str) or not agent.strip():
        return "Error: 'agent' must be a non-empty string.", False, 0
    if not isinstance(prompt, str) or not prompt.strip():
        return "Error: 'prompt' must be a non-empty string.", False, 0
    if agent not in registry:
        return f"Error: agent '{agent}' not found. Call create_subagent first.", False, 0

    server_addr = os.environ.get("MILES_PARL_V2_RAG_SERVER", "localhost:8000")
    max_turns = int(os.environ.get("MILES_PARL_V2_SUBAGENT_MAX_TURNS", "8"))
    max_tool_calls = int(os.environ.get("MILES_PARL_V2_SUBAGENT_MAX_TOOLCALLS", "10"))
    max_new_tokens = int(os.environ.get("MILES_PARL_V2_SUBAGENT_MAX_NEW_TOKENS", "1024"))
    access_max_chars = int(os.environ.get("MILES_PARL_V2_SUBAGENT_ACCESS_CHARS", "5000"))
    temperature = float(os.environ.get("MILES_PARL_V2_SUBAGENT_TEMPERATURE", "1.0"))

    system_prompt = registry[agent] + SUBAGENT_OUTPUT_SUFFIX + SUBAGENT_REACT_SUFFIX
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    parser = _get_parser()
    tool_calls_used = 0
    body = ""
    sub_steps = 0

    async with _get_semaphore():
        for _turn in range(max_turns):
            text = tokenizer.apply_chat_template(
                messages, tools=subagent_tool_specs, tokenize=False, add_generation_prompt=True
            )
            payload = {
                "text": text,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": 1.0,
                },
            }
            try:
                output = await post(router_url, payload)
            except Exception as e:
                return f"__SOLVER_ERROR__: {e}", False, max(sub_steps, 1)

            body = output.get("text", "") or ""
            sub_steps += 1

            # Parse tool_calls out of the freshly generated text.
            _, tool_calls = parser.parse_non_stream(body)
            if not tool_calls:
                # No tool call → subagent decided to answer.
                break
            if tool_calls_used + len(tool_calls) > max_tool_calls:
                # Truncate to budget; next turn will see the tool responses
                # and the subagent can decide to stop.
                tool_calls = tool_calls[: max(0, max_tool_calls - tool_calls_used)]
                if not tool_calls:
                    break

            # Execute all tool_calls concurrently.
            dispatched = [_normalize_tool_call(tc) for tc in tool_calls]
            results = await asyncio.gather(
                *[_dispatch_tool(name, p, server_addr, access_max_chars) for name, p, _ in dispatched],
                return_exceptions=True,
            )
            tool_call_messages = []
            for (name, _, tc_id), result in zip(dispatched, results, strict=True):
                if isinstance(result, Exception):
                    logger.warning("subagent tool %s crashed: %s", name, result)
                    content = f"Tool execution error: {result}"
                else:
                    content = result
                tool_call_messages.append({"role": "tool", "tool_call_id": tc_id, "name": name, "content": content})

            # Append assistant turn + tool responses, iterate.
            messages.append({"role": "assistant", "content": body})
            messages.extend(tool_call_messages)
            tool_calls_used += len(dispatched)

    is_valid = (
        bool(body.strip())
        and not body.startswith("__SOLVER_ERROR__")
        and bool(_RESULT_RE.search(body))
        and tool_calls_used > 0
    )
    if is_valid:
        body = extract_subagent_result(body)
    return body, is_valid, max(sub_steps, 1)
