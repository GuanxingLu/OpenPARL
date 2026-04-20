"""Async HTTP client for the local RAG server (Qdrant + E5 + webpage store).

The server is the one launched by
``examples/parl_v2/widesearch/launch_rag_server.sh``; its sources live in
``examples/parl_v2/widesearch/rag_server/`` (vendored from RLinf's
``examples/agent/tools/search_local_server_qdrant/``, Apache-2.0). It exposes:

- ``POST /retrieve``: `{queries: [str], topk: int}` →
  `{result: [[{contents, url}, ...], ...]}`
- ``POST /access``: `{urls: [str]}` → `{result: [{contents}, ...]}`

Everything returned to the subagent is a single plain string (formatted
markdown for /retrieve, char-truncated body for /access). Failures are
swallowed and surface as a textual ``"No ... found"`` string — subagents
can keep working across intermittent RAG-server hiccups without the
whole rollout dying.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

_SEARCH_TIMEOUT_S = 60
_ACCESS_TIMEOUT_S = 120
_RETRY_LIMIT = 3
_RETRY_SLEEP_S = 2.0

_session: aiohttp.ClientSession | None = None
_session_lock = asyncio.Lock()


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        async with _session_lock:
            if _session is None or _session.closed:
                conn = aiohttp.TCPConnector(
                    limit=1000, limit_per_host=500, ttl_dns_cache=600, enable_cleanup_closed=True
                )
                _session = aiohttp.ClientSession(
                    connector=conn,
                    timeout=aiohttp.ClientTimeout(total=300, sock_connect=30),
                    trust_env=False,
                )
    return _session


async def _post_with_retry(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any] | None:
    session = await _get_session()
    last_exc: Exception | None = None
    for attempt in range(_RETRY_LIMIT):
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            last_exc = e
            if attempt < _RETRY_LIMIT - 1:
                await asyncio.sleep(_RETRY_SLEEP_S * (attempt + 1))
    logger.warning("RAG server %s failed after %d attempts: %s", url, _RETRY_LIMIT, last_exc)
    return None


def _format_search_results(result_list: list[dict[str, Any]]) -> str:
    docs = [r.get("contents", "") for r in result_list]
    urls = [r.get("url", "") for r in result_list]
    if not docs:
        return "No search results are found."
    lines = []
    for k, (doc, url) in enumerate(zip(docs, urls, strict=False), start=1):
        lines.append(f"[Doc {k}]({url}):\n{str(doc)[:5000]}")
    return "\n\n".join(lines)


async def search(server_addr: str, query: str, topk: int = 3) -> str:
    """Run one /retrieve query against the local RAG server."""
    if not query or not query.strip():
        return "Error: empty search query."
    resp = await _post_with_retry(
        f"http://{server_addr}/retrieve",
        {"queries": [query[:2000]], "topk": topk, "return_scores": False},
        timeout_s=_SEARCH_TIMEOUT_S,
    )
    if not resp or "result" not in resp or not resp["result"]:
        return "No search results are found."
    return _format_search_results(resp["result"][0])


async def access(server_addr: str, url: str, max_chars: int = 5000) -> str:
    """Fetch one URL's contents via /access; hard-truncated to max_chars."""
    if not url or not url.strip():
        return "Error: empty URL."
    resp = await _post_with_retry(
        f"http://{server_addr}/access",
        {"urls": [url]},
        timeout_s=_ACCESS_TIMEOUT_S,
    )
    if not resp or "result" not in resp or not resp["result"]:
        return "No More Information is Found for this URL."
    page = resp["result"][0]
    contents = (page.get("contents") or "") if isinstance(page, dict) else (page or "")
    if not contents.strip():
        return "No More Information is Found for this URL."
    return contents[:max_chars]
