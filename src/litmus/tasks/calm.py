"""CALM benchmark task definitions for Litmus."""

from functools import partial

from litmus.tasks._base import SYSTEM_PROMPT, TaskConfig
from litmus.tasks._framing import make_binary_task, make_binned_task
from litmus.tasks._loader import load_calm_task


_REGRESSION_TASKS = {
    "meltome": {
        "desc": "melting temperature",
        "prompt": (
            "What is the melting temperature of this protein?\n\n"
            "Protein sequence:\n{sequence}"
        ),
    },
    "solubility": {
        "desc": "solubility value",
        "prompt": (
            "What is the solubility value of this protein?\n\n"
            "Protein sequence:\n{sequence}"
        ),
    },
    "protein_abundance": {
        "desc": "protein abundance (log scale)",
        "prompt": (
            "What is the protein abundance (log scale) of this protein?\n\n"
            "Protein sequence:\n{sequence}"
        ),
    },
    "transcript_abundance": {
        "desc": "transcript abundance (log scale)",
        "prompt": (
            "What is the transcript abundance (log scale) of this protein?\n\n"
            "Protein sequence:\n{sequence}"
        ),
    },
}

_MULTILABEL_TASKS = {
    "localization": {
        "prompt": (
            "Which subcellular locations is this protein found in? "
            "A protein can be in multiple locations.\n\n"
            "Choices: cytoplasm, nucleus, cell_membrane, mitochondrion, "
            "endoplasmic_reticulum, extracellular, golgi, peroxisome, "
            "lysosome, plastid\n\n"
            "Respond with a comma-separated list of locations.\n\n"
            "Protein sequence:\n{sequence}"
        ),
        "choices": [
            "cytoplasm", "nucleus", "cell_membrane", "mitochondrion",
            "endoplasmic_reticulum", "extracellular", "golgi", "peroxisome",
            "lysosome", "plastid",
        ],
    },
    "function_bp": {
        "prompt": (
            "Which biological processes is this protein involved in?\n\n"
            "Choices: metabolism, signaling, transport, regulation, stress_response\n\n"
            "Respond with a comma-separated list of processes.\n\n"
            "Protein sequence:\n{sequence}"
        ),
        "choices": [
            "metabolism", "signaling", "transport", "regulation", "stress_response",
        ],
    },
    "function_cc": {
        "prompt": (
            "Which cellular components is this protein associated with?\n\n"
            "Choices: cytosol, membrane, nucleus, mitochondrion, extracellular\n\n"
            "Respond with a comma-separated list of components.\n\n"
            "Protein sequence:\n{sequence}"
        ),
        "choices": [
            "cytosol", "membrane", "nucleus", "mitochondrion", "extracellular",
        ],
    },
    "function_mf": {
        "prompt": (
            "Which molecular functions does this protein have?\n\n"
            "Choices: binding, catalysis, transport, signaling, structural\n\n"
            "Respond with a comma-separated list of functions.\n\n"
            "Protein sequence:\n{sequence}"
        ),
        "choices": [
            "binding", "catalysis", "transport", "signaling", "structural",
        ],
    },
}


def register_calm_tasks() -> list[TaskConfig]:
    """Register all CALM benchmark tasks.

    Returns
    -------
    list[TaskConfig]
        List of TaskConfig objects for all CALM tasks and their framing variants.
    """
    tasks: list[TaskConfig] = []

    # Regression tasks + binary/binned variants
    for name, info in _REGRESSION_TASKS.items():
        base = TaskConfig(
            name=f"calm:{name}",
            benchmark="calm",
            task_type="regression",
            framing="regression",
            system_prompt=SYSTEM_PROMPT,
            user_prompt_template=info["prompt"],
            load_fn=partial(load_calm_task, name, "test"),
            metric="spearman",
        )
        tasks.append(base)

        train_examples = load_calm_task(name, "train")
        train_targets = [ex["target"] for ex in train_examples]
        tasks.append(make_binary_task(base, train_targets))
        tasks.append(make_binned_task(base, train_targets))

    # Multilabel tasks
    for name, info in _MULTILABEL_TASKS.items():
        tasks.append(
            TaskConfig(
                name=f"calm:{name}",
                benchmark="calm",
                task_type="multilabel",
                framing="multilabel",
                system_prompt=SYSTEM_PROMPT,
                user_prompt_template=info["prompt"],
                load_fn=partial(load_calm_task, name, "test"),
                metric="f1_macro",
                choices=info["choices"],
            )
        )

    return tasks
