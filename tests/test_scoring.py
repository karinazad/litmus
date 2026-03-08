"""Tests for scoring functions."""

import pytest

from litmus.scoring import score_classification, score_multilabel, score_regression


class TestScoreRegression:
    def test_perfect_correlation(self):
        targets = [1.0, 2.0, 3.0, 4.0, 5.0]
        predictions = [1.0, 2.0, 3.0, 4.0, 5.0]
        scores = score_regression(targets, predictions)
        assert scores["spearman"] == pytest.approx(1.0)
        assert scores["pearson"] == pytest.approx(1.0)
        assert scores["r2"] == pytest.approx(1.0)
        assert scores["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert scores["mae"] == pytest.approx(0.0, abs=1e-10)

    def test_inverse_correlation(self):
        targets = [1.0, 2.0, 3.0, 4.0, 5.0]
        predictions = [5.0, 4.0, 3.0, 2.0, 1.0]
        scores = score_regression(targets, predictions)
        assert scores["spearman"] == pytest.approx(-1.0)

    def test_with_offset(self):
        targets = [1.0, 2.0, 3.0]
        predictions = [2.0, 3.0, 4.0]
        scores = score_regression(targets, predictions)
        assert scores["spearman"] == pytest.approx(1.0)
        assert scores["mae"] == pytest.approx(1.0)


class TestScoreClassification:
    def test_perfect_accuracy(self):
        targets = ["high", "low", "high", "low"]
        predictions = ["high", "low", "high", "low"]
        scores = score_classification(targets, predictions, "binary", ["high", "low"])
        assert scores["accuracy"] == pytest.approx(1.0)
        assert scores["f1_macro"] == pytest.approx(1.0)

    def test_random_accuracy(self):
        targets = ["high", "high", "low", "low"]
        predictions = ["high", "low", "high", "low"]
        scores = score_classification(targets, predictions, "binary", ["high", "low"])
        assert scores["accuracy"] == pytest.approx(0.5)

    def test_multiclass(self):
        targets = ["a", "b", "c", "a"]
        predictions = ["a", "b", "c", "a"]
        scores = score_classification(targets, predictions, "multiclass")
        assert scores["accuracy"] == pytest.approx(1.0)


class TestScoreMultilabel:
    def test_perfect_match(self):
        targets = [["a", "b"], ["c"]]
        predictions = [["a", "b"], ["c"]]
        scores = score_multilabel(targets, predictions, ["a", "b", "c"])
        assert scores["exact_match"] == pytest.approx(1.0)
        assert scores["f1_macro"] == pytest.approx(1.0)

    def test_partial_match(self):
        targets = [["a", "b"]]
        predictions = [["a"]]
        scores = score_multilabel(targets, predictions, ["a", "b"])
        assert scores["exact_match"] == pytest.approx(0.0)

    def test_empty_predictions(self):
        targets = [["a"]]
        predictions = [[]]
        scores = score_multilabel(targets, predictions, ["a", "b"])
        assert scores["exact_match"] == pytest.approx(0.0)
