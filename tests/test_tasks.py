"""Tests for task registration and framing utilities."""

import numpy as np
import pytest

from litmus.tasks._base import SYSTEM_PROMPT, TaskConfig
from litmus.tasks._framing import make_binary_task, make_binned_task


def _make_dummy_task() -> TaskConfig:
    """Create a dummy regression task for testing."""
    return TaskConfig(
        name="test:dummy",
        benchmark="test",
        task_type="regression",
        framing="regression",
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template="Predict the value.\n\nProtein sequence:\n{sequence}",
        load_fn=lambda: [{"input": "ACGT", "target": 1.0}],
        metric="spearman",
    )


class TestTaskConfig:
    def test_frozen(self):
        task = _make_dummy_task()
        with pytest.raises(AttributeError):
            task.name = "new_name"  # type: ignore[misc]

    def test_fields(self):
        task = _make_dummy_task()
        assert task.name == "test:dummy"
        assert task.benchmark == "test"
        assert task.framing == "regression"
        assert task.choices is None
        assert task.target_formatter is None


class TestMakeBinaryTask:
    def test_name(self):
        base = _make_dummy_task()
        binary = make_binary_task(base, lambda: [1.0, 2.0, 3.0, 4.0])
        assert binary.name == "test:dummy:binary"

    def test_framing(self):
        base = _make_dummy_task()
        binary = make_binary_task(base, lambda: [1.0, 2.0, 3.0, 4.0])
        assert binary.framing == "binary"
        assert binary.task_type == "binary"
        assert binary.choices == ["high", "low"]

    def test_target_formatter(self):
        base = _make_dummy_task()
        targets = [1.0, 2.0, 3.0, 4.0]
        binary = make_binary_task(base, lambda: targets)
        median = np.median(targets)
        assert binary.target_formatter(median + 1) == "high"
        assert binary.target_formatter(median - 1) == "low"

    def test_metric(self):
        base = _make_dummy_task()
        binary = make_binary_task(base, lambda: [1.0, 2.0, 3.0])
        assert binary.metric == "accuracy"


class TestMakeBinnedTask:
    def test_name(self):
        base = _make_dummy_task()
        binned = make_binned_task(base, lambda: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        assert binned.name == "test:dummy:binned"

    def test_framing(self):
        base = _make_dummy_task()
        binned = make_binned_task(base, lambda: [1.0, 2.0, 3.0, 4.0])
        assert binned.framing == "binned"
        assert binned.task_type == "multiclass"
        assert binned.choices == ["very_low", "low", "high", "very_high"]

    def test_four_bins(self):
        base = _make_dummy_task()
        targets = list(range(100))
        binned = make_binned_task(base, lambda: targets, n_bins=4)
        assert len(binned.choices) == 4

    def test_target_formatter_assigns_bins(self):
        base = _make_dummy_task()
        targets = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        binned = make_binned_task(base, lambda: targets)
        formatter = binned.target_formatter
        # Very low values should map to "very_low"
        assert formatter(0.0) == "very_low"
        # Very high values should map to "very_high"
        assert formatter(100.0) == "very_high"

    def test_metric(self):
        base = _make_dummy_task()
        binned = make_binned_task(base, lambda: [1.0, 2.0, 3.0, 4.0])
        assert binned.metric == "accuracy"
