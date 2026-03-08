"""Task registry for Litmus benchmark."""

from litmus.tasks._base import TaskConfig
from litmus.tasks.calm import register_calm_tasks
from litmus.tasks.moleculeace import register_moleculeace_tasks
from litmus.tasks.peer import register_peer_tasks

TASK_REGISTRY: dict[str, TaskConfig] = {}


def _register(tasks: list[TaskConfig]) -> None:
    for task in tasks:
        TASK_REGISTRY[task.name] = task


def build_registry() -> dict[str, TaskConfig]:
    """Build the full task registry by registering all benchmark tasks."""
    if not TASK_REGISTRY:
        _register(register_peer_tasks())
        _register(register_calm_tasks())
        _register(register_moleculeace_tasks())
    return TASK_REGISTRY


def get_tasks(
    filter_str: str | None = None,
    framing: str | None = None,
) -> list[TaskConfig]:
    """Get tasks matching optional filter and framing constraints.

    Parameters
    ----------
    filter_str : str | None
        Comma-separated task name prefixes, e.g. "peer:fluorescence,calm".
    framing : str | None
        If set, only return tasks with this framing (e.g. "regression", "binary", "binned").

    Returns
    -------
    list[TaskConfig]
        Matching tasks.
    """
    registry = build_registry()
    tasks = list(registry.values())

    if filter_str:
        prefixes = [p.strip() for p in filter_str.split(",")]
        tasks = [t for t in tasks if any(t.name.startswith(p) for p in prefixes)]

    if framing:
        tasks = [t for t in tasks if t.framing == framing]

    return tasks
