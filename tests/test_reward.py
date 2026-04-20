"""Tests for the widesearch reward wiring.

Focus: ``compute_eval_metrics`` returns the right bag of side-metrics for
the two sample types (widesearch table vs QA boxed-answer). r_perf preserves
backwards-compatible training signal.
"""

from __future__ import annotations

import pytest

from examples.parl_v2.widesearch.reward_utils import compute_eval_metrics


def _md(rows: list[list[str]]) -> str:
    sep = ["---"] * len(rows[0])
    lines = [rows[0], sep] + rows[1:]
    return "\n".join("| " + " | ".join(r) + " |" for r in lines)


class TestComputeEvalMetricsWidesearch:
    def test_keys_for_widesearch_sample(self):
        gt = _md([["Subject", "University"], ["Arts", "Harvard"]])
        pred = _md([["Subject", "University"], ["Arts", "Harvard"]])
        out = compute_eval_metrics(pred, gt, ["subject", "university"], None)
        assert set(out.keys()) == {"item_f1", "row_f1", "is_success"}

    def test_perfect_match_all_metrics_one(self):
        gt = _md([["Subject", "University"], ["Arts", "Harvard"]])
        pred = _md([["Subject", "University"], ["Arts", "Harvard"]])
        out = compute_eval_metrics(pred, gt, ["subject", "university"], None)
        assert out == {"item_f1": 1.0, "row_f1": 1.0, "is_success": 1.0}

    def test_row_key_hit_but_cell_wrong_collapses_item_f1_and_success(self):
        gt = _md(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "1"],
            ]
        )
        pred = _md(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "99"],  # row-key ok, non-key wrong
            ]
        )
        out = compute_eval_metrics(pred, gt, ["subject", "university"], None)
        assert out["row_f1"] == 1.0
        assert out["item_f1"] < 1.0
        assert out["is_success"] == 0.0

    def test_is_success_requires_exact_item_f1_one(self):
        # Two rows, all cells exact match → item_f1 == 1.0 → success.
        gt = _md([["Subject", "University"], ["Arts", "Harvard"], ["Arts", "Yale"]])
        pred = _md([["Subject", "University"], ["Arts", "Harvard"], ["Arts", "Yale"]])
        out = compute_eval_metrics(pred, gt, ["subject", "university"], None)
        assert out["is_success"] == 1.0


class TestComputeEvalMetricsQa:
    def test_keys_for_qa_sample(self):
        out = compute_eval_metrics("\\boxed{Paris}", "Paris", None, None)
        assert set(out.keys()) == {"em", "cover_em", "token_f1"}

    @pytest.mark.parametrize(
        "response,answer,em,cover_em,token_f1",
        [
            ("\\boxed{Paris}", "Paris", 1.0, 1.0, 1.0),
            # EM=0 because strict, cover-EM=1 because GT substring, token-F1 > 0.
            ("\\boxed{Paris, France}", "Paris", 0.0, 1.0, pytest.approx(2 / 3)),
            # Pure miss on all three.
            ("\\boxed{London}", "Paris", 0.0, 0.0, 0.0),
        ],
    )
    def test_metric_values(self, response, answer, em, cover_em, token_f1):
        out = compute_eval_metrics(response, answer, None, None)
        assert out["em"] == em
        assert out["cover_em"] == cover_em
        assert out["token_f1"] == token_f1

    def test_alias_list_supported(self):
        out = compute_eval_metrics("\\boxed{Paris}", ["London", "Paris"], None, None)
        assert out["em"] == 1.0
        assert out["cover_em"] == 1.0
        assert out["token_f1"] == 1.0
