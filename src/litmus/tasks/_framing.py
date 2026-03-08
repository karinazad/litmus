"""Auto-generate classification framings from regression tasks."""

import numpy as np

from litmus.tasks._base import TaskConfig


def make_binary_task(base: TaskConfig, train_targets: list[float]) -> TaskConfig:
    """Create a binary classification variant of a regression task.

    Splits at the median of the training set targets.

    Parameters
    ----------
    base : TaskConfig
        The base regression TaskConfig.
    train_targets : list[float]
        Target values from the training split (used to compute threshold).

    Returns
    -------
    TaskConfig
        New TaskConfig with binary framing.
    """
    threshold = float(np.median(train_targets))

    def format_target(target: float) -> str:
        return "high" if target > threshold else "low"

    prompt = (
        f"Is this value high or low?\n"
        f"High means above {threshold:.4g} (median of known values).\n\n"
        f"Choices: high, low\n\n"
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
    train_targets: list[float],
    n_bins: int = 4,
) -> TaskConfig:
    """Create a binned (multiclass) classification variant of a regression task.

    Splits at quantiles of the training set targets.

    Parameters
    ----------
    base : TaskConfig
        The base regression TaskConfig.
    train_targets : list[float]
        Target values from the training split (used to compute bin edges).
    n_bins : int
        Number of bins (default 4 for quartiles).

    Returns
    -------
    TaskConfig
        New TaskConfig with binned (multiclass) framing.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = [float(np.quantile(train_targets, q)) for q in quantiles]

    bin_labels = ["very_low", "low", "high", "very_high"]
    if n_bins != 4:
        bin_labels = [f"bin_{i}" for i in range(n_bins)]

    def format_target(target: float) -> str:
        for i, edge in enumerate(edges):
            if target <= edge:
                return bin_labels[i]
        return bin_labels[-1]

    choices_desc = []
    for i, label in enumerate(bin_labels):
        if i == 0:
            choices_desc.append(f"- {label}: below {edges[0]:.4g}")
        elif i == len(bin_labels) - 1:
            choices_desc.append(f"- {label}: above {edges[-1]:.4g}")
        else:
            choices_desc.append(f"- {label}: {edges[i-1]:.4g} to {edges[i]:.4g}")
    choices_str = "\n".join(choices_desc)

    prompt = (
        f"Which range does this value fall into?\n\n"
        f"Choices:\n{choices_str}\n\n"
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
