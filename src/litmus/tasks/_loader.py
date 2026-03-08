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
    # Maps task name -> (category, hf_folder, target_col, input_type)
    # input_type: "single", "ligand", or "ppi"
    _TASK_META = {
        "fluorescence":            ("function_prediction",          "fluorescence",            "log_fluorescence", "single"),
        "stability":               ("function_prediction",          "stability",               "stability_score",  "single"),
        "betalactamase":           ("function_prediction",          "betalactamase",           "scaled_effect1",   "single"),
        "gb1":                     ("function_prediction",          "gb1",                     "target",           "single"),
        "aav":                     ("function_prediction",          "aav",                     "target",           "single"),
        "thermostability":         ("function_prediction",          "thermostability",          "target",           "single"),
        "solubility":              ("function_prediction",          "solubility",              "solubility",       "single"),
        "binary_localization":     ("localization_prediction",      "binarylocalization",      "localization",     "single"),
        "subcellular_localization":("localization_prediction",      "subcellularlocalization", "localization",     "single"),
        "bindingdb":               ("protein_ligand_interaction",   "bindingdb",               "affinity",         "ligand"),
        "pdbbind":                 ("protein_ligand_interaction",   "pdbbind",                 "affinity",         "ligand"),
        "ppiaffinity":             ("protein_protein_interaction",  "ppiaffinity",             "interaction",      "ppi"),
        "humanppi":                ("protein_protein_interaction",  "humanppi",                "interaction",      "ppi"),
        "yeastppi":                ("protein_protein_interaction",  "yeastppi",                "interaction",      "ppi"),
    }

    if task not in _TASK_META:
        raise ValueError(f"Unknown PEER task: {task}")

    category, folder, target_col, input_type = _TASK_META[task]

    ds = load_dataset(
        "taylor-joren/peer",
        data_files=f"{category}/{folder}/{split}.parquet",
        split="train",  # load_dataset returns "train" when loading from data_files
    )

    examples = []
    for row in ds:
        if input_type == "ligand":
            inp = {"protein": row["protein_sequence"], "ligand": row["ligand_smiles"]}
        elif input_type == "ppi":
            inp = {"protein1": row["protein1_sequence"], "protein2": row["protein2_sequence"]}
        else:
            inp = row["protein_sequence"]
        target = row[target_col]
        # Some columns store scalars in single-element lists
        if isinstance(target, list):
            target = target[0]
        examples.append({"input": inp, "target": target})

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
