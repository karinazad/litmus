"""Report generation for Litmus benchmark results."""

import json
from dataclasses import dataclass, asdict


@dataclass
class TaskResult:
    """Result for a single task evaluation.

    Parameters
    ----------
    task_name : str
        Name of the task.
    benchmark : str
        Parent benchmark name.
    framing : str
        How the task was framed for the LLM.
    metric : str
        Primary metric name.
    score : dict[str, float]
        All computed metric scores.
    n_examples : int
        Number of examples evaluated.
    n_failed : int
        Number of examples where parsing failed.
    predictions : list[dict] | None
        Per-example predictions (optional, for detailed output).
    """

    task_name: str
    benchmark: str
    framing: str
    metric: str
    score: dict[str, float]
    n_examples: int
    n_failed: int
    predictions: list[dict] | None = None


def format_markdown(results: list[TaskResult]) -> str:
    """Format results as a markdown table.

    Parameters
    ----------
    results : list[TaskResult]
        List of task results.

    Returns
    -------
    str
        Markdown-formatted table.
    """
    lines = []
    lines.append("| Task | Framing | Metric | Score | Examples | Failed |")
    lines.append("|------|---------|--------|-------|----------|--------|")

    for r in results:
        primary_score = r.score.get(r.metric, float("nan"))
        lines.append(
            f"| {r.task_name} | {r.framing} | {r.metric} | {primary_score:.4f} "
            f"| {r.n_examples} | {r.n_failed} |"
        )

    return "\n".join(lines)


def format_json(results: list[TaskResult], include_predictions: bool = False) -> str:
    """Format results as JSON.

    Parameters
    ----------
    results : list[TaskResult]
        List of task results.
    include_predictions : bool
        Whether to include per-example predictions.

    Returns
    -------
    str
        JSON-formatted string.
    """
    output = []
    for r in results:
        d = asdict(r)
        if not include_predictions:
            d.pop("predictions", None)
        output.append(d)

    return json.dumps(output, indent=2)


def format_csv(results: list[TaskResult]) -> str:
    """Format results as CSV.

    Parameters
    ----------
    results : list[TaskResult]
        List of task results.

    Returns
    -------
    str
        CSV-formatted string.
    """
    lines = ["task,benchmark,framing,metric,score,n_examples,n_failed"]
    for r in results:
        primary_score = r.score.get(r.metric, float("nan"))
        lines.append(
            f"{r.task_name},{r.benchmark},{r.framing},{r.metric},"
            f"{primary_score:.4f},{r.n_examples},{r.n_failed}"
        )
    return "\n".join(lines)
