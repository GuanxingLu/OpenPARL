"""Unit tests for the widesearch rule-based reward helpers.

Focus: make sure the Tier-1 upgrade (row-F1 → item-F1 + URL canonicalize +
<br>/newline split) changes behavior in the intended direction and does not
introduce high-FP shortcuts.
"""

from __future__ import annotations

import pytest

from examples.parl_v2.widesearch.reward_utils import (
    _canonicalize_url,
    _cell_set,
    cell_equal,
    cover_em_score,
    em_score,
    item_f1_from_markdown,
    row_f1_from_markdown,
    token_f1_score,
)


class TestCanonicalizeUrl:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://www.harvard.edu/", "https://www.harvard.edu"),
            ("https://www.harvard.edu", "https://www.harvard.edu"),
            ("https://www.Harvard.EDU/", "https://www.harvard.edu"),
            ("HTTPS://example.com/path/", "https://example.com/path"),
            (
                "https://example.com/a/b/?q=1#frag",
                "https://example.com/a/b?q=1#frag",
            ),
            ("http://x.y/z", "http://x.y/z"),
        ],
    )
    def test_canonicalize(self, raw, expected):
        assert _canonicalize_url(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "not a url",
            "ftp://example.com/",
            "January 1",
            "$85",
            None,
        ],
    )
    def test_non_url_returns_none(self, raw):
        assert _canonicalize_url(raw) is None


class TestCellSet:
    def test_empty(self):
        assert _cell_set("") == frozenset()
        assert _cell_set(None) == frozenset()

    def test_single_value(self):
        assert _cell_set("Harvard") == frozenset({"harvard"})

    def test_br_split(self):
        assert _cell_set("A<br>B<br>C") == frozenset({"a", "b", "c"})

    def test_br_order_invariant(self):
        assert _cell_set("A<br>B") == _cell_set("B<br>A")

    def test_newline_split(self):
        assert _cell_set("A\nB") == frozenset({"a", "b"})

    def test_br_and_newline_mixed(self):
        assert _cell_set("A<br>B\nC") == frozenset({"a", "b", "c"})

    def test_whitespace_fragments_dropped(self):
        # trailing <br> / blank lines must not introduce empty set elements
        assert _cell_set("A<br>") == frozenset({"a"})
        assert _cell_set("A\n\nB") == frozenset({"a", "b"})


class TestCellEqual:
    @pytest.mark.parametrize(
        "pred,gt",
        [
            ("Harvard University", "Harvard University"),
            ("Harvard University", "harvard university"),  # case insensitive
            ("  Harvard  ", "Harvard"),  # whitespace
            ("https://www.harvard.edu/", "https://www.harvard.edu"),  # URL slash
            ("https://Harvard.edu", "https://harvard.edu"),  # URL host case
            ("A<br>B<br>C", "C<br>B<br>A"),  # multi-value order
            ("A<br>B", "A\nB"),  # <br> vs newline
            ("40.0%", "40.0%"),
            ("", ""),
        ],
    )
    def test_equal(self, pred, gt):
        assert cell_equal(pred, gt), f"expected {pred!r} == {gt!r}"

    @pytest.mark.parametrize(
        "pred,gt",
        [
            ("Harvard", "Yale"),  # different entity
            ("https://harvard.edu", "https://yale.edu"),  # different URL
            ("https://harvard.edu/a", "https://harvard.edu/b"),  # path differs
            ("A<br>B", "A<br>B<br>C"),  # subset != set equal
            ("A<br>B", "A"),  # multi vs single
            ("40.0%", "40%"),  # percent format not canonicalized (Tier 2)
            ("$85", "85"),  # currency symbol matters
            ("8 miles", "8 km"),  # unit matters (Tier 2)
            ("January 1", "Jan 1"),  # date format not canonicalized (Tier 2)
        ],
    )
    def test_not_equal(self, pred, gt):
        assert not cell_equal(pred, gt), f"expected {pred!r} != {gt!r}"


def _md_table(rows: list[list[str]]) -> str:
    """Helper: build a markdown table string from a list of rows (including header)."""
    sep = ["---"] * len(rows[0])
    lines = [rows[0], sep] + rows[1:]
    return "\n".join("| " + " | ".join(r) + " |" for r in lines)


class TestItemF1:
    def test_perfect_match(self):
        table = _md_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard University", "1"],
                ["Arts", "Oxford", "2"],
            ]
        )
        # required defaults to full GT header, all cells match.
        assert item_f1_from_markdown(table, table, ["subject", "university"]) == 1.0

    def test_one_cell_wrong(self):
        gt = _md_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "1"],
                ["Arts", "Oxford", "2"],
            ]
        )
        pred = _md_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "99"],  # wrong rank
                ["Arts", "Oxford", "2"],
            ]
        )
        # 2 rows × 3 cols = 6 total cells; 5 TPs (one wrong rank).
        # precision = recall = 5/6 → f1 = 5/6.
        result = item_f1_from_markdown(pred, gt, ["subject", "university"])
        assert abs(result - 5 / 6) < 1e-9

    def test_missing_row(self):
        gt = _md_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "1"],
                ["Arts", "Oxford", "2"],
            ]
        )
        pred = _md_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "1"],
            ]
        )
        # matched: 1 row × 3 cols = 3 TPs; pred has 1 row, gt has 2 rows.
        # precision = 3/3 = 1.0; recall = 3/6 = 0.5 → f1 = 2/3.
        result = item_f1_from_markdown(pred, gt, ["subject", "university"])
        assert abs(result - 2 / 3) < 1e-9

    def test_extra_row_hurts_precision(self):
        gt = _md_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "1"],
            ]
        )
        pred = _md_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "1"],
                ["Arts", "Oxford", "2"],  # fabricated row
            ]
        )
        # matched rows: 1; pred has 2 rows; gt has 1.
        # precision = 3/6 = 0.5; recall = 3/3 = 1.0 → f1 = 2/3.
        result = item_f1_from_markdown(pred, gt, ["subject", "university"])
        assert abs(result - 2 / 3) < 1e-9

    def test_url_trailing_slash_counts_as_match(self):
        gt = _md_table(
            [
                ["Subject", "University", "Home Page"],
                ["Arts", "Harvard", "https://www.harvard.edu/"],
            ]
        )
        pred = _md_table(
            [
                ["Subject", "University", "Home Page"],
                ["Arts", "Harvard", "https://www.harvard.edu"],
            ]
        )
        # All 3 cells now match thanks to URL canonicalization.
        assert item_f1_from_markdown(pred, gt, ["subject", "university"]) == 1.0

    def test_multivalue_cell_order_invariant(self):
        gt = _md_table(
            [
                ["Date", "Event", "Top 3"],
                ["2018-02", "Halfpipe", "Hansen<br>Irving<br>Gu"],
            ]
        )
        pred = _md_table(
            [
                ["Date", "Event", "Top 3"],
                ["2018-02", "Halfpipe", "Gu<br>Irving<br>Hansen"],
            ]
        )
        assert item_f1_from_markdown(pred, gt, ["date", "event"]) == 1.0

    def test_required_columns_restricts_denominator(self):
        gt = _md_table(
            [
                ["Subject", "University", "Rank", "Notes"],
                ["Arts", "Harvard", "1", "foo"],
            ]
        )
        pred = _md_table(
            [
                ["Subject", "University", "Rank", "Notes"],
                ["Arts", "Harvard", "1", "bar"],  # notes differ
            ]
        )
        # Without required_columns, denominator is 4 cols, 1 wrong → f1 = 3/4 × 2 / (3/4 + 3/4) = 0.75
        assert item_f1_from_markdown(pred, gt, ["subject", "university"]) == 0.75
        # With required_columns excluding `notes`, all 3 cells match → f1 = 1.0
        assert (
            item_f1_from_markdown(
                pred, gt, ["subject", "university"], required_columns=["subject", "university", "rank"]
            )
            == 1.0
        )

    def test_no_table_returns_zero(self):
        pred = "no markdown table here"
        gt = _md_table([["A", "B"], ["x", "y"]])
        assert item_f1_from_markdown(pred, gt, ["a"]) == 0.0

    def test_missing_unique_columns_returns_zero(self):
        gt = _md_table([["Subject", "University"], ["Arts", "Harvard"]])
        pred = _md_table([["Subject", "University"], ["Arts", "Harvard"]])
        assert item_f1_from_markdown(pred, gt, []) == 0.0

    def test_pred_missing_required_column_drops_score(self):
        gt = _md_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "1"],
            ]
        )
        # pred is missing the Rank column entirely.
        pred = _md_table(
            [
                ["Subject", "University"],
                ["Arts", "Harvard"],
            ]
        )
        # GT table still picked via unique columns; pred table also picked
        # (subset covers unique cols). Missing Rank cell → `""` which won't
        # match "1". 2 TPs (subject+university) out of 3 req cols for matched
        # row. precision = 2/3, recall = 2/3 → f1 = 2/3.
        result = item_f1_from_markdown(pred, gt, ["subject", "university"])
        assert abs(result - 2 / 3) < 1e-9


class TestEmScore:
    @pytest.mark.parametrize(
        "response,gt,expected",
        [
            ("The answer is \\boxed{Paris}.", "Paris", 1.0),
            ("\\boxed{Paris}", "paris", 1.0),
            ("<answer>Paris</answer>", "Paris", 1.0),
            ("\\boxed{A Soldier's Oath}", "A Soldier'S Oath", 1.0),  # case corruption
            ("\\boxed{yes}", "yes", 1.0),
            ("\\boxed{Paris}", "London", 0.0),
            ("\\boxed{Paris, France}", "Paris", 0.0),  # must be strict
            # alias list
            ("\\boxed{Paris}", ["Paris", "Paris, France"], 1.0),
            # empty prediction
            ("", "anything", 0.0),
        ],
    )
    def test_em_score(self, response, gt, expected):
        assert em_score(response, gt) == expected


class TestCoverEmScore:
    @pytest.mark.parametrize(
        "response,gt,expected",
        [
            # Exact (post-normalization) match still counts as cover.
            ("\\boxed{Paris}", "Paris", 1.0),
            ("\\boxed{paris}", "Paris", 1.0),
            # GT is a substring of a longer prediction — the central win vs strict EM.
            ("\\boxed{Paris, France}", "Paris", 1.0),
            ("\\boxed{former president Barack Obama}", "Barack Obama", 1.0),
            # _normalize_em strips articles, so "the Eiffel Tower" normalizes to
            # "eiffel tower" and still covers "Eiffel Tower".
            ("\\boxed{the Eiffel Tower is in Paris}", "Eiffel Tower", 1.0),
            # Prediction shorter than GT — cover-EM must NOT count this.
            ("\\boxed{Obama}", "Barack Obama", 0.0),
            # Token appears but inside a different phrase: normalized substring
            # "paris" still occurs in "in paris", so cover still matches. That's
            # the documented behavior of cover-EM; leave it explicit.
            ("\\boxed{in Paris}", "Paris", 1.0),
            # No overlap at all.
            ("\\boxed{London}", "Paris", 0.0),
            # Alias list: any alias covered → 1.0.
            ("\\boxed{Paris, France}", ["London", "Paris"], 1.0),
            ("\\boxed{Madrid}", ["London", "Paris"], 0.0),
            # Empty prediction / empty GT.
            ("", "Paris", 0.0),
            ("\\boxed{Paris}", "", 0.0),
            # <answer> tag extraction (same code path as em_score).
            ("<answer>Paris, France</answer>", "Paris", 1.0),
        ],
    )
    def test_cover_em(self, response, gt, expected):
        assert cover_em_score(response, gt) == expected


class TestTokenF1Score:
    def test_exact_tokens_f1_one(self):
        assert token_f1_score("\\boxed{Barack Obama}", "Barack Obama") == 1.0

    def test_partial_overlap(self):
        # pred tokens: {barack, obama, jr}; gt tokens: {barack, obama}
        # common = {barack, obama} → p=2/3, r=2/2 → f1 = 2 * 2/3 * 1 / (2/3 + 1) = 4/5
        got = token_f1_score("\\boxed{Barack Obama Jr}", "Barack Obama")
        assert abs(got - 0.8) < 1e-9

    def test_zero_overlap(self):
        assert token_f1_score("\\boxed{Madrid}", "Paris") == 0.0

    def test_empty_pred(self):
        assert token_f1_score("", "Paris") == 0.0

    def test_empty_gt(self):
        assert token_f1_score("\\boxed{Paris}", "") == 0.0

    def test_articles_stripped(self):
        # _normalize_em removes "the" → both sides reduce to "eiffel tower".
        assert token_f1_score("\\boxed{The Eiffel Tower}", "Eiffel Tower") == 1.0

    def test_duplicate_tokens_multiset(self):
        # pred normalized: "the the eiffel tower" → after article strip: "eiffel tower"
        # that removes the duplicate before counting — use a non-stopword duplicate.
        # pred tokens: [ha, ha, ha]; gt tokens: [ha]
        # common multiset = {ha: 1}; p=1/3, r=1/1 → f1 = 2*(1/3)*1/(1/3+1) = 0.5
        got = token_f1_score("\\boxed{ha ha ha}", "ha")
        assert abs(got - 0.5) < 1e-9

    def test_alias_takes_max(self):
        # First alias: zero overlap. Second alias: exact. Max wins.
        got = token_f1_score("\\boxed{Barack Obama}", ["Donald Trump", "Barack Obama"])
        assert got == 1.0

    def test_case_insensitive(self):
        assert token_f1_score("\\boxed{BARACK OBAMA}", "barack obama") == 1.0


def _md_kv_table(rows: list[list[str]]) -> str:
    """Alias for _md_table, re-used by row-F1 tests for readability."""
    return _md_table(rows)


class TestRowF1:
    def test_perfect_key_match(self):
        # Row-F1 only cares about the unique_columns projection, non-key cells
        # (like Rank here) don't move the score.
        gt = _md_kv_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "1"],
                ["Arts", "Oxford", "2"],
            ]
        )
        pred = _md_kv_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "99"],  # rank wrong but key matches
                ["Arts", "Oxford", "2"],
            ]
        )
        assert row_f1_from_markdown(pred, gt, ["subject", "university"]) == 1.0

    def test_missing_row(self):
        gt = _md_kv_table(
            [
                ["Subject", "University"],
                ["Arts", "Harvard"],
                ["Arts", "Oxford"],
            ]
        )
        pred = _md_kv_table([["Subject", "University"], ["Arts", "Harvard"]])
        # tp=1, pred_keys=1, gt_keys=2 → p=1, r=0.5 → f1 = 2/3
        got = row_f1_from_markdown(pred, gt, ["subject", "university"])
        assert abs(got - 2 / 3) < 1e-9

    def test_extra_row(self):
        gt = _md_kv_table([["Subject", "University"], ["Arts", "Harvard"]])
        pred = _md_kv_table(
            [
                ["Subject", "University"],
                ["Arts", "Harvard"],
                ["Arts", "Yale"],
            ]
        )
        # tp=1, pred_keys=2, gt_keys=1 → p=0.5, r=1 → f1 = 2/3
        got = row_f1_from_markdown(pred, gt, ["subject", "university"])
        assert abs(got - 2 / 3) < 1e-9

    def test_no_table_returns_zero(self):
        pred = "no table here"
        gt = _md_kv_table([["Subject", "University"], ["Arts", "Harvard"]])
        assert row_f1_from_markdown(pred, gt, ["subject", "university"]) == 0.0

    def test_empty_unique_columns_returns_zero(self):
        gt = _md_kv_table([["Subject", "University"], ["Arts", "Harvard"]])
        pred = _md_kv_table([["Subject", "University"], ["Arts", "Harvard"]])
        assert row_f1_from_markdown(pred, gt, []) == 0.0

    def test_case_insensitive_key_match(self):
        gt = _md_kv_table([["Subject", "University"], ["Arts", "Harvard"]])
        pred = _md_kv_table([["Subject", "University"], ["ARTS", "harvard"]])
        # _norm_cell lowercases → keys match.
        assert row_f1_from_markdown(pred, gt, ["subject", "university"]) == 1.0
