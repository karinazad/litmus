"""HuggingFace dataset loading utilities for Litmus tasks."""

from datasets import load_dataset


def load_peer_task(task: str, split: str = "test") -> list[dict]:
    """Load a PEER benchmark task from HuggingFace.

    Parameters
    ----------
    task : str
        Task name, e.g. "fluorescence", "stability".
    split : str
        Dataset split: "train" or "test".

    Returns
    -------
    list[dict]
        List of dicts with "input" and "target" keys.
    """
    # PEER tasks are organized by category
    # Single-sequence tasks
    single_seq_tasks = {
        "fluorescence": "protein_property",
        "stability": "protein_property",
        "betalactamase": "protein_property",
        "gb1": "protein_property",
        "aav": "protein_property",
        "thermostability": "protein_property",
        "solubility": "protein_property",
        "binary_localization": "protein_property",
        "subcellular_localization": "protein_property",
    }
    # Paired tasks (protein-ligand or protein-protein)
    paired_tasks = {
        "bindingdb": "ligand_binding",
        "pdbbind": "ligand_binding",
        "ppiaffinity": "protein_protein",
        "humanppi": "protein_protein",
        "yeastppi": "protein_protein",
    }

    if task in single_seq_tasks:
        category = single_seq_tasks[task]
    elif task in paired_tasks:
        category = paired_tasks[task]
    else:
        raise ValueError(f"Unknown PEER task: {task}")

    ds = load_dataset(
        "taylor-joren/peer",
        data_files=f"{category}/{task}/{split}.parquet",
        split="train",  # load_dataset returns "train" when loading from data_files
    )

    examples = []
    for row in ds:
        if task in paired_tasks:
            # Paired tasks have two sequence columns
            if task in ("bindingdb", "pdbbind"):
                inp = {"protein": row["protein"], "ligand": row["ligand"]}
            else:
                inp = {"protein1": row["protein1"], "protein2": row["protein2"]}
        else:
            inp = row["sequence"]
        examples.append({"input": inp, "target": row["target"]})

    return examples


def load_calm_task(task: str, split: str = "test", species: str = "hsapiens") -> list[dict]:
    """Load a CALM benchmark task from HuggingFace.

    Parameters
    ----------
    task : str
        Task name, e.g. "meltome", "solubility".
    split : str
        Dataset split: "train" or "test".
    species : str
        Species identifier, default "hsapiens".

    Returns
    -------
    list[dict]
        List of dicts with "input" and "target" keys.
    """
    # Species-specific tasks
    species_tasks = {"meltome", "solubility", "protein_abundance", "transcript_abundance"}

    if task in species_tasks:
        filename = f"{species}_{split}"
    else:
        filename = split

    ds = load_dataset(
        "taylor-joren/calm-property",
        data_files=f"{task}/{filename}.parquet",
        split="train",
    )

    examples = []
    for row in ds:
        examples.append({"input": row["sequence"], "target": row["target"]})

    return examples


def load_moleculeace_task(chembl_id: str, split: str = "test") -> list[dict]:
    """Load a MoleculeACE benchmark task from HuggingFace.

    Parameters
    ----------
    chembl_id : str
        ChEMBL target ID, e.g. "CHEMBL2034".
    split : str
        Dataset split: "train" or "test".

    Returns
    -------
    list[dict]
        List of dicts with "input" (SMILES) and "target" (pKi/pEC50) keys.
    """
    ds = load_dataset(
        "joren/MoleculeACE",
        data_files=f"{chembl_id}/{split}.csv",
        split="train",
    )

    examples = []
    for row in ds:
        examples.append({"input": row["smiles"], "target": row["y"]})

    return examples
