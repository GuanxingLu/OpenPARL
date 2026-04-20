"""Rule-based metrics for the widesearch PARL v2 environment.

Two reward heads, picked per-sample by reward.py:

- ``item_f1_from_markdown(response, answer, unique_columns, required_columns)``:
  the main signal on WideSeek-R1 / WideSearch data. Both ``response`` and
  ``answer`` carry a markdown table. Rows are aligned by the ``unique_columns``
  row-key (set inner-join); for each matched row, every column in
  ``required_columns`` is compared cell-by-cell via ``cell_equal`` and the F1
  is computed over the (row × column) cell grid. ``required_columns`` defaults
  to the full GT header. This mirrors the paper's item-F1 formula but keeps
  non-key cell comparison rule-based — URL/multi-value canonicalizers cover
  the bulk of format jitter; semantic equivalence that would require an LLM
  judge is deliberately NOT attempted here (low FP > high recall).

- ``em_score(response, answer)``: normalized exact-match used on the
  ASearcher QA benchmarks (HotpotQA / 2Wiki / NQ / …). The model's final
  answer is expected either inside ``\\boxed{}`` or inside a
  ``<answer>…</answer>`` block; we search both, then EM against the GT.

Both helpers are sync / pure / side-effect-free so they can run inside the
reward hot path without async plumbing.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from urllib.parse import urlparse

_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$", re.MULTILINE)
_SEPARATOR_ROW_RE = re.compile(r"^\s*:?-{2,}:?\s*$")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# Split cell values on <br> tags and newlines. Widesearch GT uses <br> as a
# line-break inside a single cell (e.g. multi-line application fees, lists of
# player names); models tend to emit \n for the same structure. Both should
# be treated as multi-value separators.
_MULTIVALUE_SEP = re.compile(r"<br\s*/?>|\n", re.IGNORECASE)
# Quick-reject check before attempting URL parse. Anything not starting with
# http:// or https:// (after strip) is not a URL for our purposes.
_URL_PREFIX_RE = re.compile(r"^https?://", re.IGNORECASE)


def _norm_column(s: str) -> str:
    """Strip + lowercase + collapse whitespace, for column-name alignment."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_cell(s: str) -> str:
    """Strip + lowercase + drop surrounding punctuation for cell comparison.

    Kept simple on purpose — too aggressive a normalization (e.g. stripping
    all punctuation) would collapse semantically distinct values like
    "2020" vs "$2020". Instead we only strip trailing punctuation and
    collapse internal whitespace.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = s.strip("`*_ ").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _canonicalize_url(s: str) -> str | None:
    """Return canonical URL form, or None if ``s`` does not look like a URL.

    Canonicalization:
      - lowercase scheme + netloc (hostnames are case-insensitive)
      - strip trailing ``/`` from path (``https://x/`` and ``https://x`` match)
      - preserve query + fragment verbatim (case-sensitive)

    Returns None on malformed URLs so the caller can fall through to generic
    cell comparison.
    """
    if not s:
        return None
    s = s.strip()
    if not _URL_PREFIX_RE.match(s):
        return None
    try:
        parsed = urlparse(s)
    except ValueError:
        return None
    if not parsed.netloc:
        return None
    path = parsed.path.rstrip("/")
    out = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"
    if parsed.query:
        out += f"?{parsed.query}"
    if parsed.fragment:
        out += f"#{parsed.fragment}"
    return out


def _cell_set(s: str) -> frozenset[str]:
    """Split on ``<br>`` / newline, normalize each piece, return frozen set.

    Single-value cells become 1-element sets (so set equality still works in
    the plain case). Empty / whitespace-only fragments are dropped — this
    intentionally treats ``"A"`` and ``"A\\n"`` as the same multi-value cell.
    """
    if not s:
        return frozenset()
    parts = [p for p in _MULTIVALUE_SEP.split(s) if p and p.strip()]
    return frozenset(_norm_cell(p) for p in parts)


def cell_equal(pred: str, gt: str) -> bool:
    """Composite per-cell equality used by item-F1.

    Pipeline (first branch that decides wins):
      1. Raw string equality (fast path).
      2. Both sides parse as URLs → compare canonical URL forms.
      3. Multi-value set equality on ``<br>`` / newline splits, with each
         element normalized via ``_norm_cell``. Single-value cells degrade
         to 1-element sets, so this also covers the plain string case.

    Deliberately conservative: no substring matching, no token-F1 partial
    credit, no semantic / synonym handling. Those categories are high-FP and
    require an LLM judge to do safely.
    """
    if pred is None:
        pred = ""
    if gt is None:
        gt = ""
    if pred == gt:
        return True
    pred_url = _canonicalize_url(pred)
    gt_url = _canonicalize_url(gt)
    if pred_url is not None and gt_url is not None:
        return pred_url == gt_url
    if pred_url is not None or gt_url is not None:
        # One side is a URL, the other isn't — they can't match meaningfully.
        # Fall through to set compare anyway; will return False unless the
        # non-URL text happens to contain the exact same string, which is
        # fine (edge case, very low probability).
        pass
    return _cell_set(pred) == _cell_set(gt)


def _extract_markdown_tables(text: str) -> list[list[dict[str, str]]]:
    """Return every markdown table found in ``text`` as a list of row-dicts.

    A table = consecutive lines each matching ``| a | b | ...|``. The second
    row is allowed to be a separator (``| --- | --- |``) which we skip.
    Tables without at least one data row are dropped.
    """
    tables: list[list[dict[str, str]]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _TABLE_ROW_RE.match(lines[i])
        if not m:
            i += 1
            continue
        # collect consecutive pipe-rows
        block = []
        while i < len(lines):
            bm = _TABLE_ROW_RE.match(lines[i])
            if not bm:
                break
            block.append([c.strip() for c in bm.group(1).split("|")])
            i += 1
        if len(block) < 2:
            continue
        header = [_norm_column(c) for c in block[0]]
        data_rows = block[1:]
        if data_rows and all(_SEPARATOR_ROW_RE.match(c) for c in data_rows[0] if c):
            data_rows = data_rows[1:]
        rows: list[dict[str, str]] = []
        for row in data_rows:
            if len(row) != len(header):
                # truncate or pad so keys align; drop malformed silently
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                else:
                    row = row[: len(header)]
            rows.append({h: v for h, v in zip(header, row, strict=False) if h})
        if rows:
            tables.append(rows)
    return tables


def _pick_table_with_columns(tables: list[list[dict[str, str]]], required: list[str]) -> list[dict[str, str]] | None:
    """Return the first table that contains every required column, else None."""
    required_set = set(required)
    for t in tables:
        if t and required_set.issubset(t[0].keys()):
            return t
    return None


def _row_key(row: dict[str, str], unique_cols: list[str]) -> tuple[str, ...]:
    return tuple(_norm_cell(row.get(c, "")) for c in unique_cols)


def _index_rows_by_key(rows: list[dict[str, str]], unique_cols: list[str]) -> dict[tuple[str, ...], dict[str, str]]:
    """Dedup rows on ``unique_cols`` row-key (last-wins) and index by key.

    Rows whose entire key projection is empty are dropped — markdown parse
    occasionally picks up blank/filler rows which would otherwise collide on
    the empty key and poison the join.
    """
    out: dict[tuple[str, ...], dict[str, str]] = {}
    for r in rows:
        if not any(r.get(c) for c in unique_cols):
            continue
        out[_row_key(r, unique_cols)] = r
    return out


def item_f1_from_markdown(
    response: str,
    answer: str,
    unique_columns: list[str] | tuple[str, ...],
    required_columns: list[str] | tuple[str, ...] | None = None,
) -> float:
    """Rule-based item-F1 over ``required_columns`` × matched rows.

    Rows are aligned by the ``unique_columns`` row-key; each cell in
    ``required_columns`` of a matched row contributes a binary TP via
    ``cell_equal``. Unique-column cells are always TP for matched rows (they
    were the join key). Precision denominator is all pred rows × required
    cols; recall denominator is all GT rows × required cols, so over- and
    under-generation are both penalized.

    ``required_columns`` defaults to the full GT header when None — matches
    the widesearch source ``evaluation.required`` when prepare_data emits it,
    gracefully falls back otherwise.

    Returns 0.0 on any parse failure (no table, missing columns, empty join)
    instead of raising — bad formatting must not crash a training step.
    """
    if not unique_columns:
        return 0.0
    uc = [_norm_column(c) for c in unique_columns]
    gt_tables = _extract_markdown_tables(answer or "")
    pred_tables = _extract_markdown_tables(response or "")
    gt = _pick_table_with_columns(gt_tables, uc)
    pred = _pick_table_with_columns(pred_tables, uc)
    if gt is None or pred is None:
        return 0.0

    if required_columns:
        rc = [_norm_column(c) for c in required_columns]
        if not set(rc).issubset(gt[0].keys()):
            # GT doesn't actually have all declared required columns —
            # prepare_data bug or schema drift. Degrade to GT header.
            rc = list(gt[0].keys())
    else:
        rc = list(gt[0].keys())
    if not rc:
        return 0.0

    gt_by = _index_rows_by_key(gt, uc)
    pred_by = _index_rows_by_key(pred, uc)
    if not gt_by or not pred_by:
        return 0.0

    uc_set = set(uc)
    tp = 0.0
    for key in gt_by.keys() & pred_by.keys():
        g_row = gt_by[key]
        p_row = pred_by[key]
        for col in rc:
            if col in uc_set:
                # Matched by row-key, so the unique-col cells are equal by
                # construction (modulo _norm_cell which _row_key already applied).
                tp += 1.0
            elif cell_equal(p_row.get(col, ""), g_row.get(col, "")):
                tp += 1.0

    num_pred = len(pred_by) * len(rc)
    num_gt = len(gt_by) * len(rc)
    if num_pred == 0 or num_gt == 0:
        return 0.0
    precision = tp / num_pred
    recall = tp / num_gt
    if precision + recall <= 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def row_f1_from_markdown(
    response: str,
    answer: str,
    unique_columns: list[str] | tuple[str, ...],
) -> float:
    """Paper's "Row F1" metric — set-F1 over the ``unique_columns`` row-key.

    Retained as a side-metric alongside ``item_f1_from_markdown`` so eval
    reporting can mirror the paper's table 1b (Row F1 Avg@N / Max@N). Row-F1
    ignores non-key cells entirely, so a prediction that gets the row
    identities right but fills non-key columns with garbage still scores 1.0.
    Returns 0.0 on any parse failure.
    """
    if not unique_columns:
        return 0.0
    uc = [_norm_column(c) for c in unique_columns]
    gt_tables = _extract_markdown_tables(answer or "")
    pred_tables = _extract_markdown_tables(response or "")
    gt = _pick_table_with_columns(gt_tables, uc)
    pred = _pick_table_with_columns(pred_tables, uc)
    if gt is None or pred is None:
        return 0.0
    gt_keys = {_row_key(r, uc) for r in gt if any(r.get(c) for c in uc)}
    pred_keys = {_row_key(r, uc) for r in pred if any(r.get(c) for c in uc)}
    if not gt_keys or not pred_keys:
        return 0.0
    tp = len(gt_keys & pred_keys)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_keys)
    recall = tp / len(gt_keys)
    return 2 * precision * recall / (precision + recall)


def compute_eval_metrics(
    response: str,
    answer: str | list[str],
    unique_columns: list[str] | tuple[str, ...] | None,
    required_columns: list[str] | tuple[str, ...] | None,
) -> dict[str, float]:
    """Paper-aligned per-sample side-metrics, keyed by metric name.

    Returns different metric sets by sample type:
      - ``unique_columns`` present (WideSeek-R1 train / WideSearch eval):
        ``{item_f1, row_f1, is_success}``. Mirrors WideSearch table 1b.
      - ``unique_columns`` absent (ASearcher QA benchmarks): ``{em, cover_em,
        token_f1}``. Gives the strict / lenient / partial triplet used in
        the WideSeek-R1 paper's standard QA benchmarks.

    Meant to be stashed in ``sample.metadata["eval_metrics"]`` by reward.py
    and aggregated downstream by the eval logger into Avg@N / Max@N / Pass@N.
    """
    if unique_columns:
        item_f1 = float(item_f1_from_markdown(response, answer, unique_columns, required_columns))
        row_f1 = float(row_f1_from_markdown(response, answer, unique_columns))
        return {
            "item_f1": item_f1,
            "row_f1": row_f1,
            "is_success": float(item_f1 == 1.0),
        }
    em = float(em_score(response, answer))
    return {
        "em": em,
        "cover_em": float(cover_em_score(response, answer)),
        "token_f1": float(token_f1_score(response, answer)),
    }


def _extract_final_answer(response: str) -> str:
    """Pull the model's final answer string out of a response.

    Priority: last ``\\boxed{...}`` match, then last ``<answer>…</answer>``
    block, then fall back to the response tail (last 200 chars) stripped of
    think-trace noise.
    """
    if not response:
        return ""
    boxes = _BOXED_RE.findall(response)
    if boxes:
        return boxes[-1].strip()
    tags = _ANSWER_TAG_RE.findall(response)
    if tags:
        return tags[-1].strip()
    return response[-200:].strip()


def _normalize_em(s: str) -> str:
    """QA-style normalization: lowercase, strip articles + punctuation + ws."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def em_score(response: str, answer: str | list[str]) -> float:
    """Normalized exact-match against GT (which may be a list of aliases)."""
    pred = _normalize_em(_extract_final_answer(response))
    if not pred:
        return 0.0
    if isinstance(answer, (list, tuple)):
        gts = [_normalize_em(str(a)) for a in answer]
    else:
        gts = [_normalize_em(str(answer))]
    return 1.0 if any(g and g == pred for g in gts) else 0.0


def cover_em_score(response: str, answer: str | list[str]) -> float:
    """Cover-EM: 1.0 if any normalized GT alias appears as a substring of pred.

    Matches the Search-R1 / FlashRAG community convention used to score
    open-domain QA benchmarks (NQ / TriviaQA / HotpotQA / 2Wiki / Bamboogle /
    MuSiQue / PopQA). More lenient than strict EM — a prediction that wraps
    the answer in extra context ("former president Barack Obama" vs
    "Barack Obama") still scores. Prediction shorter than GT never scores.
    """
    pred = _normalize_em(_extract_final_answer(response))
    if not pred:
        return 0.0
    gts = answer if isinstance(answer, (list, tuple)) else [answer]
    for gt in gts:
        g = _normalize_em(str(gt))
        if g and g in pred:
            return 1.0
    return 0.0


def token_f1_score(response: str, answer: str | list[str]) -> float:
    """SQuAD-style token-level F1 against GT (max over aliases).

    Pred and GT both go through ``_normalize_em`` (lowercase, strip
    articles / punctuation) before token-splitting on whitespace. Uses
    multiset intersection so duplicate tokens are counted with their real
    multiplicities (matches the HuggingFace ``squad`` metric).
    """
    pred_toks = _normalize_em(_extract_final_answer(response)).split()
    if not pred_toks:
        return 0.0
    pred_counter = Counter(pred_toks)
    gts = answer if isinstance(answer, (list, tuple)) else [answer]
    best = 0.0
    for gt in gts:
        gt_toks = _normalize_em(str(gt)).split()
        if not gt_toks:
            continue
        common = pred_counter & Counter(gt_toks)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        precision = n_common / len(pred_toks)
        recall = n_common / len(gt_toks)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1
    return best
