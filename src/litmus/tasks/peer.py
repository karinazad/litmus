"""PEER benchmark task definitions for Litmus."""

from collections.abc import Callable
from functools import partial

from litmus.tasks._base import SYSTEM_PROMPT, TaskConfig
from litmus.tasks._framing import make_binary_task, make_binned_task
from litmus.tasks._loader import load_peer_task


def _make_train_loader(load_fn: Callable, task_name: str) -> Callable[[], list[float]]:
    """Create a lazy loader that extracts training targets on first call."""
    def _load() -> list[float]:
        return [ex["target"] for ex in load_fn(task_name, "train")]
    return _load


# Single-sequence regression tasks
_REGRESSION_TASKS = {
    "fluorescence": {
        "desc": "log-fluorescence",
        "prompt": (
            "What is the log-fluorescence value of this protein?\n\n"
            "Protein sequence:\n{sequence}"
        ),
    },
    "stability": {
        "desc": "stability score",
        "prompt": (
            "What is the stability score of this protein?\n\n"
            "Protein sequence:\n{sequence}"
        ),
    },
    "betalactamase": {
        "desc": "beta-lactamase activity (scaled fitness)",
        "prompt": (
            "What is the scaled fitness (beta-lactamase activity) of this protein?\n\n"
            "Protein sequence:\n{sequence}"
        ),
    },
    "gb1": {
        "desc": "GB1 binding fitness",
        "prompt": (
            "What is the binding fitness of this GB1 protein variant?\n\n"
            "Protein sequence:\n{sequence}"
        ),
    },
    "aav": {
        "desc": "AAV packaging fitness",
        "prompt": (
            "What is the packaging fitness of this AAV capsid variant?\n\n"
            "Protein sequence:\n{sequence}"
        ),
    },
    "thermostability": {
        "desc": "melting temperature (Tm)",
        "prompt": (
            "What is the melting temperature (Tm) of this protein?\n\n"
            "Protein sequence:\n{sequence}"
        ),
    },
}

# Single-sequence binary tasks
_BINARY_TASKS = {
    "solubility": {
        "prompt": (
            "Is this protein soluble or insoluble?\n\n"
            "Choices: soluble, insoluble\n\n"
            "Protein sequence:\n{sequence}"
        ),
        "choices": ["soluble", "insoluble"],
    },
    "binary_localization": {
        "prompt": (
            "Is this protein membrane-bound or soluble?\n\n"
            "Choices: membrane, soluble\n\n"
            "Protein sequence:\n{sequence}"
        ),
        "choices": ["membrane", "soluble"],
    },
}

# Single-sequence multiclass tasks
_MULTICLASS_TASKS = {
    "subcellular_localization": {
        "prompt": (
            "Which subcellular compartment is this protein localized to?\n\n"
            "Choices: nucleus, cytoplasm, extracellular, mitochondrion, cell_membrane, "
            "endoplasmic_reticulum, plastid, golgi, lysosome, peroxisome\n\n"
            "Protein sequence:\n{sequence}"
        ),
        "choices": [
            "nucleus", "cytoplasm", "extracellular", "mitochondrion",
            "cell_membrane", "endoplasmic_reticulum", "plastid", "golgi",
            "lysosome", "peroxisome",
        ],
    },
}

# Paired regression tasks (protein-ligand)
_LIGAND_REGRESSION_TASKS = {
    "bindingdb": {
        "desc": "binding affinity (pKd)",
        "prompt": (
            "What is the binding affinity (pKd) between this protein and ligand?\n\n"
            "Protein sequence:\n{protein}\n\n"
            "Ligand SMILES:\n{ligand}"
        ),
    },
    "pdbbind": {
        "desc": "binding affinity (-log Kd/Ki)",
        "prompt": (
            "What is the binding affinity (-log Kd/Ki) between this protein and ligand?\n\n"
            "Protein sequence:\n{protein}\n\n"
            "Ligand SMILES:\n{ligand}"
        ),
    },
}

# Paired regression tasks (protein-protein)
_PPI_REGRESSION_TASKS = {
    "ppiaffinity": {
        "desc": "protein-protein binding affinity",
        "prompt": (
            "What is the binding affinity between these two proteins?\n\n"
            "Protein 1:\n{protein1}\n\n"
            "Protein 2:\n{protein2}"
        ),
    },
}

# Paired binary tasks (protein-protein)
_PPI_BINARY_TASKS = {
    "humanppi": {
        "prompt": (
            "Do these two human proteins interact?\n\n"
            "Choices: yes, no\n\n"
            "Protein 1:\n{protein1}\n\n"
            "Protein 2:\n{protein2}"
        ),
        "choices": ["yes", "no"],
    },
    "yeastppi": {
        "prompt": (
            "Do these two yeast proteins interact?\n\n"
            "Choices: yes, no\n\n"
            "Protein 1:\n{protein1}\n\n"
            "Protein 2:\n{protein2}"
        ),
        "choices": ["yes", "no"],
    },
}


def register_peer_tasks() -> list[TaskConfig]:
    """Register all PEER benchmark tasks.

    Returns
    -------
    list[TaskConfig]
        List of TaskConfig objects for all PEER tasks and their framing variants.
    """
    tasks: list[TaskConfig] = []

    # Single-sequence regression tasks + binary/binned variants
    for name, info in _REGRESSION_TASKS.items():
        base = TaskConfig(
            name=f"peer:{name}",
            benchmark="peer",
            task_type="regression",
            framing="regression",
            system_prompt=SYSTEM_PROMPT,
            user_prompt_template=info["prompt"],
            load_fn=partial(load_peer_task, name, "test"),
            metric="spearman",
        )
        tasks.append(base)

        train_load_fn = _make_train_loader(load_peer_task, name)
        tasks.append(make_binary_task(base, train_load_fn))
        tasks.append(make_binned_task(base, train_load_fn))

    # Single-sequence binary tasks
    for name, info in _BINARY_TASKS.items():
        formatter = {
            "solubility": lambda t: "soluble" if t == 1 else "insoluble",
            "binary_localization": lambda t: "membrane" if t == 1 else "soluble",
        }
        tasks.append(
            TaskConfig(
                name=f"peer:{name}",
                benchmark="peer",
                task_type="binary",
                framing="binary",
                system_prompt=SYSTEM_PROMPT,
                user_prompt_template=info["prompt"],
                load_fn=partial(load_peer_task, name, "test"),
                metric="accuracy",
                choices=info["choices"],
                target_formatter=formatter[name],
            )
        )

    # Single-sequence multiclass tasks
    for name, info in _MULTICLASS_TASKS.items():
        tasks.append(
            TaskConfig(
                name=f"peer:{name}",
                benchmark="peer",
                task_type="multiclass",
                framing="multiclass",
                system_prompt=SYSTEM_PROMPT,
                user_prompt_template=info["prompt"],
                load_fn=partial(load_peer_task, name, "test"),
                metric="accuracy",
                choices=info["choices"],
            )
        )

    # Ligand binding regression tasks + variants
    for name, info in _LIGAND_REGRESSION_TASKS.items():
        base = TaskConfig(
            name=f"peer:{name}",
            benchmark="peer",
            task_type="regression",
            framing="regression",
            system_prompt=SYSTEM_PROMPT,
            user_prompt_template=info["prompt"],
            load_fn=partial(load_peer_task, name, "test"),
            metric="spearman",
        )
        tasks.append(base)

        train_load_fn = _make_train_loader(load_peer_task, name)
        tasks.append(make_binary_task(base, train_load_fn))
        tasks.append(make_binned_task(base, train_load_fn))

    # PPI regression tasks + variants
    for name, info in _PPI_REGRESSION_TASKS.items():
        base = TaskConfig(
            name=f"peer:{name}",
            benchmark="peer",
            task_type="regression",
            framing="regression",
            system_prompt=SYSTEM_PROMPT,
            user_prompt_template=info["prompt"],
            load_fn=partial(load_peer_task, name, "test"),
            metric="spearman",
        )
        tasks.append(base)

        train_load_fn = _make_train_loader(load_peer_task, name)
        tasks.append(make_binary_task(base, train_load_fn))
        tasks.append(make_binned_task(base, train_load_fn))

    # PPI binary tasks
    for name, info in _PPI_BINARY_TASKS.items():
        tasks.append(
            TaskConfig(
                name=f"peer:{name}",
                benchmark="peer",
                task_type="binary",
                framing="binary",
                system_prompt=SYSTEM_PROMPT,
                user_prompt_template=info["prompt"],
                load_fn=partial(load_peer_task, name, "test"),
                metric="accuracy",
                choices=info["choices"],
                target_formatter=lambda t: "yes" if t == 1 else "no",
            )
        )

    return tasks
