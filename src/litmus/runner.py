"""Main evaluation runner for Litmus benchmark."""

import asyncio
import logging
from dataclasses import dataclass

from litmus.model import BatchModel, Model
from litmus.parsing import extract_answer, parse_float, parse_label, parse_multilabel
from litmus.report import TaskResult
from litmus.scoring import score_classification, score_multilabel, score_regression
from litmus.tasks._base import TaskConfig

logger = logging.getLogger(__name__)


@dataclass
class ExampleResult:
    """Result for a single example.

    Parameters
    ----------
    input : str | dict
        The input given to the model.
    target : str | float | list
        The ground truth target.
    response : str
        The raw model response.
    parsed : str | float | list | None
        The parsed prediction.
    success : bool
        Whether parsing succeeded.
    """

    input: str | dict
    target: str | float | list
    response: str
    parsed: str | float | list | None
    success: bool


def _build_user_prompt(task: TaskConfig, example: dict) -> str:
    """Build the user prompt for a single example.

    Parameters
    ----------
    task : TaskConfig
        The task configuration.
    example : dict
        Example dict with "input" and "target" keys.

    Returns
    -------
    str
        Formatted user prompt.
    """
    inp = example["input"]
    if isinstance(inp, dict):
        return task.user_prompt_template.format(**inp)
    return task.user_prompt_template.format(sequence=inp)


def _parse_prediction(
    task: TaskConfig,
    response: str,
) -> tuple[str | float | list | None, bool]:
    """Parse the model response into a prediction.

    Parameters
    ----------
    task : TaskConfig
        The task configuration.
    response : str
        Raw model response.

    Returns
    -------
    tuple[str | float | list | None, bool]
        Parsed prediction and whether parsing succeeded.
    """
    answer = extract_answer(response)
    if answer is None:
        return None, False

    if task.framing == "regression":
        val = parse_float(answer)
        return val, val is not None

    if task.task_type == "multilabel":
        labels = parse_multilabel(answer, task.choices or [])
        return labels, len(labels) > 0

    if task.choices:
        label = parse_label(answer, task.choices)
        return label, label is not None

    return answer, True


async def _evaluate_single(
    model: Model,
    task: TaskConfig,
    example: dict,
) -> ExampleResult:
    """Evaluate a single example.

    Parameters
    ----------
    model : Model
        The model to use.
    task : TaskConfig
        The task configuration.
    example : dict
        Example dict with "input" and "target" keys.

    Returns
    -------
    ExampleResult
        The result for this example.
    """
    user_prompt = _build_user_prompt(task, example)
    messages = [
        {"role": "system", "content": task.system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = await model.complete(messages)
    except Exception as e:
        logger.error("Model error for task %s: %s", task.name, e)
        return ExampleResult(
            input=example["input"],
            target=example["target"],
            response="",
            parsed=None,
            success=False,
        )

    parsed, success = _parse_prediction(task, response)

    return ExampleResult(
        input=example["input"],
        target=example["target"],
        response=response,
        parsed=parsed,
        success=success,
    )


def _compute_score(task: TaskConfig, results: list[ExampleResult]) -> dict[str, float]:
    """Compute metrics from evaluation results.

    Parameters
    ----------
    task : TaskConfig
        The task configuration.
    results : list[ExampleResult]
        List of per-example results.

    Returns
    -------
    dict[str, float]
        Computed metric scores.
    """
    successful = [r for r in results if r.success]
    if not successful:
        return {task.metric: 0.0}

    if task.framing == "regression":
        targets = [float(r.target) for r in successful]
        predictions = [float(r.parsed) for r in successful]
        return score_regression(targets, predictions)

    if task.task_type == "multilabel":
        formatter = task.target_formatter or (lambda x: x)
        targets = [formatter(r.target) if not isinstance(r.target, list) else r.target for r in successful]
        predictions = [r.parsed for r in successful]
        return score_multilabel(targets, predictions, task.choices or [])

    # Classification (binary, multiclass, binned)
    formatter = task.target_formatter or str
    targets = [formatter(r.target) for r in successful]
    predictions = [str(r.parsed) for r in successful]
    return score_classification(targets, predictions, task.task_type, task.choices)


async def _evaluate_batch(
    model: BatchModel,
    task: TaskConfig,
    examples: list[dict],
) -> list[ExampleResult]:
    """Evaluate a batch of examples using batched inference.

    Parameters
    ----------
    model : BatchModel
        A model that supports batch_complete.
    task : TaskConfig
        The task configuration.
    examples : list[dict]
        List of example dicts.

    Returns
    -------
    list[ExampleResult]
        Results for each example.
    """
    messages_batch = []
    for ex in examples:
        user_prompt = _build_user_prompt(task, ex)
        messages_batch.append([
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": user_prompt},
        ])

    try:
        responses = await model.batch_complete(messages_batch)
    except Exception as e:
        logger.error("Batch model error for task %s: %s", task.name, e)
        return [
            ExampleResult(
                input=ex["input"],
                target=ex["target"],
                response="",
                parsed=None,
                success=False,
            )
            for ex in examples
        ]

    results = []
    for ex, response in zip(examples, responses):
        parsed, success = _parse_prediction(task, response)
        results.append(
            ExampleResult(
                input=ex["input"],
                target=ex["target"],
                response=response,
                parsed=parsed,
                success=success,
            )
        )
    return results


async def run_eval(
    model: Model | BatchModel,
    tasks: list[TaskConfig],
    max_examples: int | None = None,
) -> list[TaskResult]:
    """Run the full evaluation pipeline.

    If the model supports ``batch_complete``, uses batched inference
    for better throughput (e.g. vLLM). Otherwise, falls back to
    concurrent single-example calls (e.g. API models).

    Parameters
    ----------
    model : Model | BatchModel
        The model to evaluate.
    tasks : list[TaskConfig]
        List of tasks to evaluate on.
    max_examples : int | None
        Maximum number of examples per task (for testing).

    Returns
    -------
    list[TaskResult]
        Results for each task.
    """
    use_batch = hasattr(model, "batch_complete")
    results = []

    for task in tasks:
        logger.info("Evaluating task: %s", task.name)
        examples = task.load_fn()

        if max_examples:
            examples = examples[:max_examples]

        if use_batch:
            logger.info("Using batched inference (%d examples)", len(examples))
            example_results = await _evaluate_batch(model, task, examples)  # type: ignore[arg-type]
        else:
            example_results = list(
                await asyncio.gather(
                    *[_evaluate_single(model, task, ex) for ex in examples]
                )
            )

        n_failed = sum(1 for r in example_results if not r.success)
        score = _compute_score(task, example_results)

        logger.info(
            "Task %s: %s=%.4f (%d/%d parsed)",
            task.name,
            task.metric,
            score.get(task.metric, 0.0),
            len(example_results) - n_failed,
            len(example_results),
        )

        predictions = [
            {
                "input": r.input,
                "target": r.target,
                "parsed": r.parsed,
                "success": r.success,
            }
            for r in example_results
        ]

        results.append(
            TaskResult(
                task_name=task.name,
                benchmark=task.benchmark,
                framing=task.framing,
                metric=task.metric,
                score=score,
                n_examples=len(example_results),
                n_failed=n_failed,
                predictions=predictions,
            )
        )

    return results
