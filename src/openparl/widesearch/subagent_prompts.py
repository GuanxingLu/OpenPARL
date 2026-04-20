"""Subagent-side prompt + tool-spec bundle for the widesearch environment.

The orchestrator spawns subagents via ``assign_task``. Each subagent then
runs a multi-turn ReAct loop inside ``widesearch/assign_task.py`` with
``search`` and ``access`` tools pointed at the local RAG server. The
subagent's system prompt combines:

1. The ``system_prompt`` the orchestrator registered via ``create_subagent``.
2. ``SUBAGENT_REACT_SUFFIX`` below — ReAct-style instructions describing
   the two tools and the mandatory ``<result>…</result>`` output block.
3. ``SUBAGENT_OUTPUT_SUFFIX`` from root tool.py (already appended to (1)
   by the shared convention, but we explicitly add it here too so the
   subagent can't skip it).

The tool_specs are what SGLang's tool-call parser sees when it decodes
the subagent's response; names/parameters are kept close to the
wideseek_r1 tools.py in RLinf (which we vendored the RAG server from)
so prompts can be reused.
"""

SUBAGENT_REACT_SUFFIX = (
    "\n\n# ReAct Workflow\n"
    "You are a research subagent. You have two tools and must use them to answer the user's task.\n\n"
    "## Tools\n"
    "- `search(query: str, topk: int = 3)` — runs a web-style search against a local knowledge base.\n"
    "  Returns a markdown-formatted list of result snippets with URLs. Use this to discover URLs\n"
    "  and gather high-level context. Issue a focused single-query call per invocation; you can\n"
    "  emit multiple search tool_calls in the same turn to cover multiple sub-questions in parallel.\n"
    "- `access(url: str)` — fetches the content of a URL from the local knowledge base.\n"
    "  Returns the page body, truncated to a few thousand characters. Use this after `search` to\n"
    "  read promising documents in depth. Again, multiple `access` tool_calls can be emitted in\n"
    "  parallel in a single turn.\n\n"
    "## Loop\n"
    "1. Think briefly about what sub-questions you need to answer.\n"
    "2. Emit one or more tool_calls (search and/or access) to gather evidence.\n"
    "3. Read the tool results. If you have enough evidence, STOP and produce your final answer.\n"
    "4. Otherwise, iterate: refine queries, follow up on URLs, and re-call tools.\n"
    "5. You MUST stop after at most 10 tool_call invocations in total. Budget conservatively.\n"
)


tool_specs = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Run a search query against the local knowledge base. Returns a markdown-formatted "
                "list of the top-k result snippets with URLs. You can invoke multiple search calls "
                "concurrently in the same turn by emitting multiple tool_calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string (max ~2000 chars).",
                    },
                    "topk": {
                        "type": "integer",
                        "description": "Number of top results to return (default 3).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "access",
            "description": (
                "Fetch the contents of a specific URL from the local knowledge base. Returns the "
                "page body, truncated to the first few thousand characters. Use after `search` to "
                "read promising pages in depth. Multiple access calls can run concurrently in the "
                "same turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch.",
                    },
                },
                "required": ["url"],
            },
        },
    },
]
