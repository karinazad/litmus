"""Structured results storage for Litmus benchmark runs."""

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from litmus.report import TaskResult
from litmus.tasks._base import Metric

DEFAULT_RESULTS_DIR = Path("litmus_results")


def _sanitize_model_name(model: str) -> str:
    """Turn a model identifier into a filesystem-safe string."""
    return re.sub(r"[^\w\-.]", "_", model)


def save_run(
    results_dir: Path,
    model: str,
    backend: str,
    results: list[TaskResult],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_examples: int | None = None,
) -> Path:
    """Save a single eval run to a structured JSON file.

    Parameters
    ----------
    results_dir : Path
        Directory to store result files.
    model : str
        Model identifier.
    backend : str
        Backend used (api, azure, vllm).
    results : list[TaskResult]
        Evaluation results.
    temperature : float
        Sampling temperature used.
    max_tokens : int
        Max tokens setting.
    max_examples : int | None
        Max examples per task (None = full dataset).

    Returns
    -------
    Path
        Path to the saved result file.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = _sanitize_model_name(model)
    filename = f"{safe_name}_{timestamp}.json"

    run_data = {
        "metadata": {
            "model": model,
            "backend": backend,
            "timestamp": timestamp,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_examples": max_examples,
        },
        "results": [asdict(r) for r in results],
    }

    path = results_dir / filename
    path.write_text(json.dumps(run_data, indent=2))
    return path


def load_runs(results_dir: Path) -> list[dict]:
    """Load all saved runs from the results directory.

    Parameters
    ----------
    results_dir : Path
        Directory containing result files.

    Returns
    -------
    list[dict]
        List of run dicts, each with "metadata" and "results" keys.
    """
    if not results_dir.exists():
        return []

    runs = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            if "metadata" in data and "results" in data:
                runs.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return runs


def build_comparison_table(
    runs: list[dict],
    metric_override: Metric | None = None,
) -> tuple[list[str], list[str], list[list[float | None]], dict[str, str]]:
    """Build a model x task comparison matrix from saved runs.

    When multiple runs exist for the same model, uses the latest one.

    Parameters
    ----------
    runs : list[dict]
        List of run dicts from ``load_runs``.
    metric_override : Metric | None
        If set, use this metric for all tasks instead of each task's primary.

    Returns
    -------
    tuple[list[str], list[str], list[list[float | None]], dict[str, str]]
        (task_names, model_names, scores, task_metrics) where scores[i][j]
        is the score for task i, model j. None means the model wasn't
        evaluated on that task. task_metrics maps task name to primary metric.
    """
    # Deduplicate: keep latest run per model (by timestamp)
    latest_by_model: dict[str, dict] = {}
    for run in runs:
        model = run["metadata"]["model"]
        ts = run["metadata"].get("timestamp", "")
        if model not in latest_by_model or ts > latest_by_model[model]["metadata"].get("timestamp", ""):
            latest_by_model[model] = run

    model_names = sorted(latest_by_model.keys())

    # Collect all tasks across runs and build per-model index
    task_metrics: dict[str, str] = {}
    result_index: dict[str, dict[str, dict]] = {}
    for model, run in latest_by_model.items():
        result_index[model] = {}
        for r in run["results"]:
            task_metrics.setdefault(r["task_name"], r["metric"])
            result_index[model][r["task_name"]] = r
    task_names = sorted(task_metrics.keys())

    # Build score matrix: task_names x model_names
    scores: list[list[float | None]] = []
    for task in task_names:
        row: list[float | None] = []
        for model in model_names:
            task_result = result_index[model].get(task)
            if task_result is None:
                row.append(None)
            else:
                metric = metric_override or task_result["metric"]
                row.append(task_result["score"].get(metric))
        scores.append(row)

    return task_names, model_names, scores, task_metrics


def format_comparison_markdown(
    task_names: list[str],
    model_names: list[str],
    scores: list[list[float | None]],
    task_metrics: dict[str, str] | None = None,
) -> str:
    """Format a comparison matrix as a markdown table.

    Parameters
    ----------
    task_names : list[str]
        Row labels (tasks).
    model_names : list[str]
        Column labels (models).
    scores : list[list[float | None]]
        Score matrix [task][model].
    task_metrics : dict[str, str] | None
        Map of task_name -> metric name for display.

    Returns
    -------
    str
        Markdown table string.
    """
    # Shorten model names for display
    short_names = [m.rsplit("/", 1)[-1] for m in model_names]

    header = "| Task | Metric |" + " | ".join(f" {n} " for n in short_names) + " |"
    separator = "|------|--------|" + "|".join("-" * (len(n) + 2) for n in short_names) + "|"

    lines = [header, separator]
    for i, task in enumerate(task_names):
        metric_label = (task_metrics or {}).get(task, "")
        cells = []
        # Find best score in this row for bolding
        valid_scores = [s for s in scores[i] if s is not None]
        best = max(valid_scores) if valid_scores else None

        for j, s in enumerate(scores[i]):
            if s is None:
                cells.append("  -  ")
            else:
                formatted = f"{s:.4f}"
                if best is not None and s == best and len(valid_scores) > 1:
                    formatted = f"**{formatted}**"
                cells.append(f" {formatted} ")

        lines.append(f"| {task} | {metric_label} |" + "|".join(cells) + "|")

    return "\n".join(lines)


def format_comparison_csv(
    task_names: list[str],
    model_names: list[str],
    scores: list[list[float | None]],
) -> str:
    """Format a comparison matrix as CSV.

    Parameters
    ----------
    task_names : list[str]
        Row labels.
    model_names : list[str]
        Column labels.
    scores : list[list[float | None]]
        Score matrix.

    Returns
    -------
    str
        CSV string.
    """
    header = "task," + ",".join(model_names)
    lines = [header]
    for i, task in enumerate(task_names):
        cells = [f"{s:.4f}" if s is not None else "" for s in scores[i]]
        lines.append(f"{task}," + ",".join(cells))
    return "\n".join(lines)
