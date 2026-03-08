"""Core data structures for Litmus tasks."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

TaskType = Literal["regression", "binary", "multiclass", "multilabel"]
Framing = Literal["regression", "binary", "binned", "multiclass", "multilabel"]
Metric = Literal["spearman", "accuracy", "f1_macro"]

SYSTEM_PROMPT = (
    "You are an expert in biology and chemistry. You will be asked to predict "
    "properties of biological sequences or molecules.\n\n"
    "Provide your reasoning, then give your final answer within <answer></answer> tags.\n"
    "Inside the tags, provide ONLY the value — no units, no explanation."
)


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for a single evaluation task.

    Parameters
    ----------
    name : str
        Unique task identifier, e.g. "peer:fluorescence" or "peer:fluorescence:binary".
    benchmark : str
        Parent benchmark name: "peer", "calm", or "moleculeace".
    task_type : TaskType
        Native task type: "regression", "binary", "multiclass", or "multilabel".
    framing : Framing
        How the task is presented to the LLM: "regression", "binary", "binned",
        "multiclass", or "multilabel".
    system_prompt : str
        System prompt for the LLM.
    user_prompt_template : str
        Template string with placeholders like {sequence}, {choices}, etc.
    load_fn : Callable
        Callable returning list[dict] with keys "input" and "target".
    metric : Metric
        Primary metric name: "spearman", "accuracy", "f1_macro".
    choices : list[str] | None
        For classification tasks, the valid class labels.
    target_formatter : Callable | None
        Optional function to format raw target for display in the prompt.
    """

    name: str
    benchmark: str
    task_type: TaskType
    framing: Framing
    system_prompt: str
    user_prompt_template: str
    load_fn: Callable[[], list[dict]]
    metric: Metric
    choices: list[str] | None = None
    target_formatter: Callable | None = None
