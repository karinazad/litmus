"""Tests for the results storage module."""

import json
from pathlib import Path

import pytest

from litmus.report import TaskResult
from litmus.results import (
    build_comparison_table,
    format_comparison_csv,
    format_comparison_markdown,
    load_runs,
    save_run,
)


def _make_result(task_name="peer:fluorescence", metric="spearman", score_val=0.5):
    return TaskResult(
        task_name=task_name,
        benchmark="peer",
        framing="regression",
        metric=metric,
        score={metric: score_val},
        n_examples=100,
        n_failed=5,
        predictions=None,
    )


class TestSaveAndLoad:
    def test_save_creates_file(self, tmp_path):
        results = [_make_result()]
        path = save_run(tmp_path, "gpt-4o", "api", results)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["metadata"]["model"] == "gpt-4o"
        assert data["metadata"]["backend"] == "api"
        assert len(data["results"]) == 1

    def test_load_empty_dir(self, tmp_path):
        assert load_runs(tmp_path) == []

    def test_load_nonexistent_dir(self, tmp_path):
        assert load_runs(tmp_path / "nope") == []

    def test_roundtrip(self, tmp_path):
        results = [_make_result(), _make_result("peer:stability", "spearman", 0.3)]
        save_run(tmp_path, "model-a", "api", results)
        runs = load_runs(tmp_path)
        assert len(runs) == 1
        assert len(runs[0]["results"]) == 2

    def test_multiple_models(self, tmp_path):
        save_run(tmp_path, "model-a", "api", [_make_result(score_val=0.5)])
        save_run(tmp_path, "model-b", "api", [_make_result(score_val=0.7)])
        runs = load_runs(tmp_path)
        assert len(runs) == 2


class TestBuildComparisonTable:
    def test_basic_comparison(self, tmp_path):
        save_run(tmp_path, "model-a", "api", [
            _make_result("task1", "spearman", 0.5),
            _make_result("task2", "accuracy", 0.8),
        ])
        save_run(tmp_path, "model-b", "api", [
            _make_result("task1", "spearman", 0.6),
            _make_result("task2", "accuracy", 0.7),
        ])
        runs = load_runs(tmp_path)
        tasks, models, scores, _ = build_comparison_table(runs)

        assert tasks == ["task1", "task2"]
        assert models == ["model-a", "model-b"]
        assert scores[0] == [0.5, 0.6]  # task1
        assert scores[1] == [0.8, 0.7]  # task2

    def test_missing_task(self, tmp_path):
        save_run(tmp_path, "model-a", "api", [_make_result("task1")])
        save_run(tmp_path, "model-b", "api", [_make_result("task2")])
        runs = load_runs(tmp_path)
        tasks, models, scores, _ = build_comparison_table(runs)

        assert len(tasks) == 2
        # model-a has task1 but not task2
        assert scores[0][0] == 0.5  # task1, model-a
        assert scores[0][1] is None  # task1, model-b
        assert scores[1][0] is None  # task2, model-a
        assert scores[1][1] == 0.5  # task2, model-b


class TestFormatComparison:
    def test_markdown_output(self):
        tasks = ["task1", "task2"]
        models = ["model-a", "model-b"]
        scores = [[0.5, 0.6], [0.8, 0.7]]
        text = format_comparison_markdown(tasks, models, scores, {"task1": "spearman", "task2": "accuracy"})
        assert "task1" in text
        assert "model-a" in text
        assert "**0.6000**" in text  # best for task1
        assert "**0.8000**" in text  # best for task2

    def test_csv_output(self):
        tasks = ["task1"]
        models = ["m1", "m2"]
        scores = [[0.5, 0.7]]
        text = format_comparison_csv(tasks, models, scores)
        lines = text.strip().split("\n")
        assert lines[0] == "task,m1,m2"
        assert "0.5000" in lines[1]
        assert "0.7000" in lines[1]

    def test_none_scores(self):
        text = format_comparison_markdown(["t1"], ["m1"], [[None]])
        assert "-" in text
