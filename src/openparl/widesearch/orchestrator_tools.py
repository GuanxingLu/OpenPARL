"""Widesearch orchestrator-side direct-tool layer.

Paper-faithful PARL v2 gives the Orchestrator ``search`` / ``access`` in
addition to ``create_subagent`` / ``assign_task``. This module composes
those tool-spec sets and exposes the single ``dispatch`` coroutine that
``generate.py`` routes "direct" (non-subagent) tool calls through.

Semantically, Orchestrator-side ``search``/``access`` is the same
capability as a Subagent's ``search``/``access``; the reason to delegate
is context offloading (a Subagent's tool outputs stay in the Subagent's
context, only ``<result>…</result>`` returns to the Orchestrator).

Three tool-spec sets:
  - ``tool_specs_swarm``       : [create_subagent, assign_task]
                                 (current swarm-strict default; forces
                                 delegation by construction).
  - ``tool_specs_swarm_paper`` : [create_subagent, assign_task,
                                  search, access]
                                 (paper-faithful; Orchestrator chooses
                                 when to delegate vs direct-call).
  - ``tool_specs_single``      : [search, access]
                                 (single-agent baseline; no delegation).

``dispatch(name, params)`` is an ``(str, dict) -> str | None`` coroutine;
returns the tool-response string, or ``None`` for any name this
dispatcher does not handle (so ``generate.py`` falls through to its
unknown-tool error path). The RAG server address is read from
``MILES_PARL_V2_RAG_SERVER`` (matching ``widesearch/assign_task.py``),
so ``generate.py`` stays env-agnostic and the launcher just exports the
env var once.
"""

from __future__ import annotations

import os

from ..tool import tool_specs as _orch_swarm_tool_specs
from . import search_client
from .subagent_prompts import tool_specs as _search_access_tool_specs

DEFAULT_ACCESS_MAX_CHARS = 5000
_RAG_SERVER_ENV = "MILES_PARL_V2_RAG_SERVER"
_RAG_SERVER_DEFAULT = "localhost:8000"

tool_specs_swarm = list(_orch_swarm_tool_specs)
tool_specs_swarm_paper = list(_orch_swarm_tool_specs) + list(_search_access_tool_specs)
tool_specs_single = list(_search_access_tool_specs)


def _server_addr() -> str:
    return os.environ.get(_RAG_SERVER_ENV, _RAG_SERVER_DEFAULT)


async def dispatch(name: str, params: dict) -> str | None:
    """Async direct-tool handler for Orchestrator-side search/access.

    Returns the tool-response string, or ``None`` if ``name`` is not a
    direct tool this dispatcher knows about.
    """
    if not isinstance(params, dict):
        params = {}
    if name == "search":
        query = params.get("query", "")
        topk_raw = params.get("topk", 3)
        try:
            topk = int(topk_raw)
        except (TypeError, ValueError):
            topk = 3
        return await search_client.search(_server_addr(), query, topk)
    if name == "access":
        url = params.get("url", "")
        return await search_client.access(_server_addr(), url, max_chars=DEFAULT_ACCESS_MAX_CHARS)
    return None
