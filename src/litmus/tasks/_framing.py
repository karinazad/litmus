"""Auto-generate classification framings from regression tasks."""

from collections.abc import Callable

import numpy as np

from litmus.tasks._base import TaskConfig


def make_binary_task(
    base: TaskConfig,
    train_load_fn: Callable[[], list[float]],
) -> TaskConfig:
    """Create a binary classification variant of a regression task.

    Splits at the median of the training set targets. The threshold is
    computed lazily on first use to avoid eager dataset downloads.

    Parameters
    ----------
    base : TaskConfig
        The base regression TaskConfig.
    train_load_fn : Callable[[], list[float]]
        Callable returning training target values (used to compute threshold).

    Returns
    -------
    TaskConfig
        New TaskConfig with binary framing.
    """
    _cache: dict[str, float] = {}

    def _get_threshold() -> float:
        if "threshold" not in _cache:
            _cache["threshold"] = float(np.median(train_load_fn()))
        return _cache["threshold"]

    def format_target(target: float) -> str:
        return "high" if target > _get_threshold() else "low"

    prompt = (
        "Is this value high or low?\n"
        "High means above the median of known values.\n\n"
        "Choices: high, low\n\n"
        f"{_get_input_section(base)}"
    )

    return TaskConfig(
        name=f"{base.name}:binary",
        benchmark=base.benchmark,
        task_type="binary",
        framing="binary",
        system_prompt=base.system_prompt,
        user_prompt_template=prompt,
        load_fn=base.load_fn,
        metric="accuracy",
        choices=["high", "low"],
        target_formatter=format_target,
    )


def make_binned_task(
    base: TaskConfig,
    train_load_fn: Callable[[], list[float]],
    n_bins: int = 4,
) -> TaskConfig:
    """Create a binned (multiclass) classification variant of a regression task.

    Splits at quantiles of the training set targets. The bin edges are
    computed lazily on first use to avoid eager dataset downloads.

    Parameters
    ----------
    base : TaskConfig
        The base regression TaskConfig.
    train_load_fn : Callable[[], list[float]]
        Callable returning training target values (used to compute bin edges).
    n_bins : int
        Number of bins (default 4 for quartiles).

    Returns
    -------
    TaskConfig
        New TaskConfig with binned (multiclass) framing.
    """
    bin_labels = ["very_low", "low", "high", "very_high"]
    if n_bins != 4:
        bin_labels = [f"bin_{i}" for i in range(n_bins)]

    _cache: dict[str, list[float]] = {}

    def _get_edges() -> list[float]:
        if "edges" not in _cache:
            train_targets = train_load_fn()
            quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
            _cache["edges"] = [float(np.quantile(train_targets, q)) for q in quantiles]
        return _cache["edges"]

    def format_target(target: float) -> str:
        for i, edge in enumerate(_get_edges()):
            if target <= edge:
                return bin_labels[i]
        return bin_labels[-1]

    prompt = (
        "Which range does this value fall into?\n\n"
        f"Choices: {', '.join(bin_labels)}\n\n"
        f"{_get_input_section(base)}"
    )

    return TaskConfig(
        name=f"{base.name}:binned",
        benchmark=base.benchmark,
        task_type="multiclass",
        framing="binned",
        system_prompt=base.system_prompt,
        user_prompt_template=prompt,
        load_fn=base.load_fn,
        metric="accuracy",
        choices=bin_labels,
        target_formatter=format_target,
    )


def _get_input_section(task: TaskConfig) -> str:
    """Extract the input placeholder section from the base task's prompt template."""
    # Return the input placeholder portion - this preserves {sequence}, {protein}, etc.
    # We look for the input section in the original template
    lines = task.user_prompt_template.strip().split("\n")
    input_lines = []
    capture = False
    for line in lines:
        if any(placeholder in line for placeholder in ["{sequence}", "{protein", "{smiles}", "{ligand}"]):
            capture = True
        if capture:
            input_lines.append(line)

    if input_lines:
        return "\n".join(input_lines)
    # Fallback: return everything after last blank line
    return lines[-1] if lines else ""
