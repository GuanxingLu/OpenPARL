"""Shared orchestrator tool layer for PARL v2 (env-agnostic).

Holds:
- ``tool_specs``: the two orchestrator tools ``create_subagent`` and
  ``assign_task`` exposed to SGLang tool-call parsing.
- ``_create_subagent``: pure registry write; no inference.
- ``SUBAGENT_OUTPUT_SUFFIX`` + ``extract_subagent_result``: context-sharding
  contract — every subagent appends this to its system prompt and must wrap
  its final output in <result>…</result>; orchestrator receives only that
  block.

The actual subagent inference (``assign_task``) is environment-specific and
lives under ``examples/parl_v2/<env>/assign_task.py::call``. The impl path
is loaded by ``generate.py`` at startup via ``--assign-task-impl-path``
(default resolved in ``run_parl_v2.py`` from ``--env``).
"""

import re

MAX_REGISTRY_SIZE = 8

# ── Context-sharding: subagent output extraction ──────────────────────
# Appended to every subagent's system prompt so it always emits a
# <result>…</result> block.  The extractor returns only that block to
# the orchestrator, keeping its context bounded.
SUBAGENT_OUTPUT_SUFFIX = (
    "\n\n# Output Format\n"
    "After completing your work, you MUST wrap your final answer or key "
    "findings in a <result>…</result> block at the end of your response. "
    "The orchestrator will ONLY see the content inside <result>…</result>. "
    "Put all essential information there — anything outside will be discarded."
)

_RESULT_RE = re.compile(r"<result>(.*?)</result>", re.DOTALL)


def extract_subagent_result(body: str) -> str:
    """Return the last <result>…</result> content, or the full body as fallback."""
    matches = _RESULT_RE.findall(body)
    if matches:
        return matches[-1].strip()
    return body


tool_specs = [
    {
        "type": "function",
        "function": {
            "name": "create_subagent",
            "description": (
                "Register a specialized subagent with a unique name and a "
                "custom system prompt for later reuse. Returns a short "
                "confirmation. No inference runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique name for this agent configuration.",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "System prompt defining the agent's role, capabilities, and boundaries.",
                    },
                },
                "required": ["name", "system_prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assign_task",
            "description": (
                "Launch a previously-created subagent on a single task and "
                "return its candidate solution. You can emit multiple "
                "assign_task calls in the same turn; they run concurrently."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "description": "Which created agent to use.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task for the agent to perform.",
                    },
                },
                "required": ["agent", "prompt"],
            },
        },
    },
]


def _create_subagent(params: dict, *, registry: dict[str, str]) -> str:
    name = params.get("name")
    system_prompt = params.get("system_prompt")
    if not isinstance(name, str) or not name.strip():
        return "Error: 'name' must be a non-empty string."
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        return "Error: 'system_prompt' must be a non-empty string."
    if name not in registry and len(registry) >= MAX_REGISTRY_SIZE:
        return (
            f"Error: subagent registry full (max {MAX_REGISTRY_SIZE}). "
            "Reuse an existing agent or avoid creating more."
        )
    registry[name] = system_prompt
    return f"Registered subagent '{name}'."
