"""Widesearch data preparation — label compaction, no downloads.

Assumes the raw datasets are already on disk under ``DATA/`` (user fetched
from HF separately). Produces per-file ``*.miles.jsonl`` siblings with:

- ``prompt``: the question / query text. Name matches run_parl_v2.py's
  hardcoded ``--input-key prompt``; widesearch-test's ``query`` and
  wideseek-r1-train's ``question`` both land here.
- ``label`` = ``json.dumps({"answer": ..., "unique_columns": ... | None,
  "required_columns": ... | None})``. ``required_columns`` is lifted from
  widesearch-test's ``evaluation.required`` (matches the paper's item-F1
  column list); absent for QA / train sources, in which case item-F1
  defaults to the full GT header. The reward's ``_decode_label`` reads
  this back.

Usage::

    python -m examples.parl_v2.widesearch.prepare_data
    python -m examples.parl_v2.widesearch.prepare_data --force   # rebuild

All paths resolve from the current working dir's ``DATA/`` unless
``--data-root`` is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# (src_relpath, prompt_key, has_unique_columns)
_TARGETS = [
    ("wideseek-r1-train/hybrid_20k.jsonl", "question", True),
    ("widesearch-test/test.jsonl", "query", True),
    ("asearcher-test/HotpotQA_rand1000/test.jsonl", "question", False),
    ("asearcher-test/2WikiMultihopQA_rand1000/test.jsonl", "question", False),
    ("asearcher-test/PopQA_rand1000/test.jsonl", "question", False),
    ("asearcher-test/Musique_rand1000/test.jsonl", "question", False),
    ("asearcher-test/TriviaQA_rand1000/test.jsonl", "question", False),
    ("asearcher-test/NQ_rand1000/test.jsonl", "question", False),
    ("asearcher-test/Bamboogle/test.jsonl", "question", False),
]


def _convert_file(src: Path, dst: Path, prompt_key: str, has_uc: bool) -> int:
    """Write dst from src; returns row count."""
    n = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("skip %s:%d invalid JSON: %s", src, line_num, e)
                continue
            prompt = row.get(prompt_key)
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            answer = row.get("answer", "")
            unique_columns = row.get("unique_columns") if has_uc else None
            # widesearch-test stashes `{unique_columns, required, eval_pipeline}`
            # as a JSON string under `evaluation`. `required` is the item-F1
            # column list the paper evaluates against; wideseek-r1 train rows
            # don't carry one, in which case item-F1 defaults to the full GT
            # header (handled downstream).
            required_columns = None
            if has_uc:
                ev = row.get("evaluation")
                if isinstance(ev, str):
                    try:
                        ev = json.loads(ev)
                    except (ValueError, TypeError):
                        ev = None
                if isinstance(ev, dict):
                    req = ev.get("required")
                    if isinstance(req, (list, tuple)) and req:
                        required_columns = [str(c) for c in req]
            label = {
                "answer": answer if isinstance(answer, str) else json.dumps(answer),
                "unique_columns": list(unique_columns) if unique_columns else None,
                "required_columns": required_columns,
            }
            out = {"prompt": prompt, "label": json.dumps(label, ensure_ascii=False)}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "DATA"))
    p.add_argument("--force", action="store_true", help="rebuild .miles.jsonl even if present")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"DATA root not found: {data_root}")

    any_processed = False
    for rel, prompt_key, has_uc in _TARGETS:
        src = data_root / rel
        dst = src.with_name(src.stem + ".miles.jsonl")
        if not src.exists():
            logger.warning("skip missing source: %s", src)
            continue
        if dst.exists() and not args.force:
            logger.info("skip (already exists): %s", dst)
            continue
        n = _convert_file(src, dst, prompt_key, has_uc)
        logger.info("wrote %s (%d rows)", dst, n)
        any_processed = True

    if not any_processed:
        logger.info("no files converted (use --force to rebuild).")


if __name__ == "__main__":
    main()
