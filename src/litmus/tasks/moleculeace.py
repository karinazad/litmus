"""MoleculeACE benchmark task definitions for Litmus."""

from functools import partial

from litmus.tasks._base import SYSTEM_PROMPT, TaskConfig
from litmus.tasks._framing import make_binary_task, make_binned_task
from litmus.tasks._loader import load_moleculeace_task


# All 30 ChEMBL targets in MoleculeACE
CHEMBL_TARGETS = [
    "CHEMBL2034",
    "CHEMBL2047",
    "CHEMBL2147",
    "CHEMBL214",
    "CHEMBL218",
    "CHEMBL219",
    "CHEMBL228",
    "CHEMBL231",
    "CHEMBL233",
    "CHEMBL234",
    "CHEMBL235",
    "CHEMBL236",
    "CHEMBL237",
    "CHEMBL238",
    "CHEMBL239",
    "CHEMBL244",
    "CHEMBL2835",
    "CHEMBL287",
    "CHEMBL2971",
    "CHEMBL3979",
    "CHEMBL4005",
    "CHEMBL4203",
    "CHEMBL4616",
    "CHEMBL4792",
    "CHEMBL4822",
    "CHEMBL5077",
    "CHEMBL5112",
    "CHEMBL5118",
    "CHEMBL5137",
    "CHEMBL5608",
]

_PROMPT_TEMPLATE = (
    "What is the binding activity (pKi or pEC50) of this molecule "
    "against {target}?\n\n"
    "SMILES:\n{{sequence}}"
)


def register_moleculeace_tasks() -> list[TaskConfig]:
    """Register all MoleculeACE benchmark tasks.

    Returns
    -------
    list[TaskConfig]
        List of TaskConfig objects for all MoleculeACE tasks and their framing variants.
    """
    tasks: list[TaskConfig] = []

    for chembl_id in CHEMBL_TARGETS:
        prompt = _PROMPT_TEMPLATE.format(target=chembl_id)

        base = TaskConfig(
            name=f"moleculeace:{chembl_id}",
            benchmark="moleculeace",
            task_type="regression",
            framing="regression",
            system_prompt=SYSTEM_PROMPT,
            user_prompt_template=prompt,
            load_fn=partial(load_moleculeace_task, chembl_id, "test"),
            metric="spearman",
        )
        tasks.append(base)

        train_examples = load_moleculeace_task(chembl_id, "train")
        train_targets = [ex["target"] for ex in train_examples]
        tasks.append(make_binary_task(base, train_targets))
        tasks.append(make_binned_task(base, train_targets))

    return tasks
