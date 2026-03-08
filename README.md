# Litmus

LLM property prediction benchmark for biological sequences and small molecules.

Litmus evaluates how well language models can predict quantitative and categorical properties of proteins, protein pairs, and molecules. It aggregates tasks from three established benchmarks — **PEER**, **CALM**, and **MoleculeACE** — and presents them to LLMs through a standardized prompt-and-parse pipeline.

## Benchmarks and Tasks

### PEER

Protein property prediction tasks spanning single-sequence, protein-ligand, and protein-protein settings:

| Task | Type | Description |
|------|------|-------------|
| fluorescence | regression | Log-fluorescence of GFP variants |
| stability | regression | Protein stability score |
| betalactamase | regression | Beta-lactamase activity (scaled fitness) |
| gb1 | regression | GB1 binding fitness |
| aav | regression | AAV capsid packaging fitness |
| thermostability | regression | Melting temperature (Tm) |
| solubility | binary | Soluble vs. insoluble |
| binary_localization | binary | Membrane-bound vs. soluble |
| subcellular_localization | multiclass | 10-class subcellular compartment |
| bindingdb | regression | Protein-ligand binding affinity (pKd) |
| pdbbind | regression | Protein-ligand binding affinity (-log Kd/Ki) |
| ppiaffinity | regression | Protein-protein binding affinity |
| humanppi | binary | Human protein-protein interaction |
| yeastppi | binary | Yeast protein-protein interaction |

### CALM

Protein property and annotation tasks:

| Task | Type | Description |
|------|------|-------------|
| meltome | regression | Melting temperature |
| solubility | regression | Solubility value |
| protein_abundance | regression | Protein abundance (log scale) |
| transcript_abundance | regression | Transcript abundance (log scale) |
| localization | multilabel | Subcellular localization (10 locations) |
| function_bp | multilabel | GO biological process |
| function_cc | multilabel | GO cellular component |
| function_mf | multilabel | GO molecular function |

### MoleculeACE

Small molecule activity prediction across 30 ChEMBL targets (CHEMBL2034, CHEMBL2047, ..., CHEMBL5608). Each task predicts binding activity (pKi or pEC50) from SMILES strings.

## Task Framings

Every regression task is automatically available in three framings:

- **regression** — predict the numeric value directly
- **binary** — predict "high" or "low" (median split on training data)
- **binned** — predict a quartile bin: "very_low", "low", "high", "very_high"

This lets you compare how well a model handles the same underlying data under different question formats.

## Installation

```bash
pip install -e .
```

For local inference with vLLM:

```bash
pip install -e ".[vllm]"
```

## Usage

### Evaluate via API

Any OpenAI-compatible endpoint works (OpenAI, Anthropic via proxy, vLLM serve, Ollama, etc.):

```bash
# OpenAI
litmus eval --model gpt-4o --backend api

# Custom endpoint
litmus eval --model my-model --backend api --base-url http://localhost:8000/v1

# Filter tasks and framings
litmus eval --model gpt-4o --backend api \
    --tasks peer:fluorescence,peer:stability \
    --framing regression \
    --max-examples 50
```

### Evaluate with vLLM (local)

Loads the model locally and runs batched inference:

```bash
litmus eval --model meta-llama/Llama-3.1-8B-Instruct --backend vllm

# Multi-GPU
litmus eval --model meta-llama/Llama-3.1-70B-Instruct \
    --backend vllm \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9
```

### List available tasks

```bash
litmus list
litmus list --tasks peer --framing binary
```

### Output formats

```bash
litmus eval --model gpt-4o --format markdown   # default, table to stdout
litmus eval --model gpt-4o --format csv
litmus eval --model gpt-4o --format json
litmus eval --model gpt-4o --output results.json  # save detailed results with per-example predictions
```

## CLI Reference

```
litmus eval [OPTIONS]

  --model TEXT                Model name or HuggingFace ID (required)
  --backend [api|vllm]       Inference backend (auto-detected if omitted)
  --base-url TEXT             API base URL
  --api-key TEXT              API key (default: OPENAI_API_KEY env var)
  --tasks TEXT                Comma-separated task prefixes to filter
  --framing TEXT              Filter by framing: regression, binary, binned
  --max-examples INT          Cap examples per task
  --max-concurrent INT        Max concurrent API requests (default: 10)
  --temperature FLOAT         Sampling temperature (default: 0.0)
  --max-tokens INT            Max response tokens (default: 1024)
  --output TEXT               Save JSON results to file
  --format [markdown|json|csv]  Output format (default: markdown)
  --verbose                   Enable debug logging
  --tensor-parallel-size INT  GPUs for tensor parallelism (vllm, default: 1)
  --gpu-memory-utilization FLOAT  GPU memory fraction (vllm, default: 0.9)
  --dtype TEXT                Model dtype (vllm, default: auto)
  --max-model-len INT         Max context length (vllm)
```

## Scoring

| Framing | Metrics |
|---------|---------|
| regression | Spearman, Pearson, R², RMSE, MAE |
| binary | Accuracy, F1 (macro/weighted), AUROC |
| multiclass / binned | Accuracy, F1 (macro/weighted) |
| multilabel | Exact match, F1 macro |

## How It Works

1. **Task loading** — datasets are fetched from HuggingFace (`taylor-joren/peer`, `taylor-joren/calm-property`, `joren/MoleculeACE`)
2. **Prompting** — each example is formatted into a system + user message asking the model to reason and provide an answer in `<answer>` tags
3. **Parsing** — responses are parsed with tag extraction, float parsing, and fuzzy label matching
4. **Scoring** — predictions are compared against ground truth using task-appropriate metrics

## Adding New Tasks

Litmus is designed to be extended with new tasks and benchmarks. Below is a step-by-step guide.

### Key concepts

Every task is a `TaskConfig` (defined in `src/litmus/tasks/_base.py`):

```python
@dataclass(frozen=True)
class TaskConfig:
    name: str                        # unique ID, e.g. "mybench:mytask"
    benchmark: str                   # parent benchmark name
    task_type: str                   # "regression", "binary", "multiclass", or "multilabel"
    framing: str                     # how the LLM sees it: "regression", "binary", "binned", etc.
    system_prompt: str               # system message sent to the LLM
    user_prompt_template: str        # prompt template with placeholders like {sequence}
    load_fn: Callable[[], list[dict]]  # returns [{"input": ..., "target": ...}, ...]
    metric: str                      # primary metric: "spearman", "accuracy", "f1_macro", etc.
    choices: list[str] | None        # valid labels for classification tasks
    target_formatter: Callable | None  # converts raw target to the label the LLM should predict
```

The `load_fn` is called at evaluation time. It must return a list of dicts, each with:
- `"input"` — a string (e.g. protein sequence, SMILES) or a dict of strings (e.g. `{"protein": ..., "ligand": ...}` for paired tasks)
- `"target"` — the ground truth value (float for regression, int/str for classification, list for multilabel)

The `user_prompt_template` uses Python format placeholders. For single-input tasks, use `{sequence}`. For paired tasks, use named keys matching the dict keys in `"input"` (e.g. `{protein}`, `{ligand}`).

### Step 1: Add a data loader

Create or extend a loader in `src/litmus/tasks/_loader.py`. The loader fetches data (typically from HuggingFace) and returns the standardized `list[dict]` format.

```python
# src/litmus/tasks/_loader.py

def load_mybench_task(task: str, split: str = "test") -> list[dict]:
    """Load a task from the MyBench dataset."""
    ds = load_dataset(
        "username/mybench",
        data_files=f"{task}/{split}.parquet",
        split="train",  # load_dataset uses "train" when loading from data_files
    )

    examples = []
    for row in ds:
        examples.append({"input": row["sequence"], "target": row["target"]})
    return examples
```

For paired inputs (e.g. protein + ligand), return a dict as the input:

```python
examples.append({
    "input": {"protein": row["protein"], "ligand": row["ligand"]},
    "target": row["target"],
})
```

### Step 2: Define the tasks

Create a new file `src/litmus/tasks/mybench.py`. Each task needs a `TaskConfig` and a registration function.

**Regression task:**

```python
# src/litmus/tasks/mybench.py

from functools import partial

from litmus.tasks._base import SYSTEM_PROMPT, TaskConfig
from litmus.tasks._framing import make_binary_task, make_binned_task
from litmus.tasks._loader import load_mybench_task


def register_mybench_tasks() -> list[TaskConfig]:
    tasks: list[TaskConfig] = []

    # A regression task
    base = TaskConfig(
        name="mybench:binding_affinity",
        benchmark="mybench",
        task_type="regression",
        framing="regression",
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=(
            "What is the binding affinity of this protein?\n\n"
            "Protein sequence:\n{sequence}"
        ),
        load_fn=partial(load_mybench_task, "binding_affinity", "test"),
        metric="spearman",
    )
    tasks.append(base)

    # Auto-generate binary and binned variants from training data
    train_examples = load_mybench_task("binding_affinity", "train")
    train_targets = [ex["target"] for ex in train_examples]
    tasks.append(make_binary_task(base, train_targets))
    tasks.append(make_binned_task(base, train_targets))

    return tasks
```

`make_binary_task` splits at the training median into "high"/"low". `make_binned_task` splits at training quartiles into "very_low", "low", "high", "very_high". Both automatically generate appropriate prompts and target formatters from the base regression task.

**Binary classification task:**

```python
tasks.append(
    TaskConfig(
        name="mybench:is_toxic",
        benchmark="mybench",
        task_type="binary",
        framing="binary",
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=(
            "Is this molecule toxic?\n\n"
            "Choices: yes, no\n\n"
            "SMILES:\n{sequence}"
        ),
        load_fn=partial(load_mybench_task, "toxicity", "test"),
        metric="accuracy",
        choices=["yes", "no"],
        # target_formatter maps the raw target (e.g. 1/0) to a choice label
        target_formatter=lambda t: "yes" if t == 1 else "no",
    )
)
```

**Multiclass task:**

```python
tasks.append(
    TaskConfig(
        name="mybench:enzyme_class",
        benchmark="mybench",
        task_type="multiclass",
        framing="multiclass",
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=(
            "Which enzyme class does this protein belong to?\n\n"
            "Choices: oxidoreductase, transferase, hydrolase, lyase, isomerase, ligase\n\n"
            "Protein sequence:\n{sequence}"
        ),
        load_fn=partial(load_mybench_task, "enzyme_class", "test"),
        metric="accuracy",
        choices=["oxidoreductase", "transferase", "hydrolase", "lyase", "isomerase", "ligase"],
    )
)
```

**Multilabel task:**

```python
tasks.append(
    TaskConfig(
        name="mybench:go_terms",
        benchmark="mybench",
        task_type="multilabel",
        framing="multilabel",
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=(
            "Which functions does this protein have?\n\n"
            "Choices: binding, catalysis, transport, signaling, structural\n\n"
            "Respond with a comma-separated list.\n\n"
            "Protein sequence:\n{sequence}"
        ),
        load_fn=partial(load_mybench_task, "go_terms", "test"),
        metric="f1_macro",
        choices=["binding", "catalysis", "transport", "signaling", "structural"],
    )
)
```

### Step 3: Register with the task registry

Add your registration function to `src/litmus/tasks/__init__.py`:

```python
from litmus.tasks.mybench import register_mybench_tasks

def build_registry() -> dict[str, TaskConfig]:
    if not TASK_REGISTRY:
        _register(register_peer_tasks())
        _register(register_calm_tasks())
        _register(register_moleculeace_tasks())
        _register(register_mybench_tasks())      # <-- add this line
    return TASK_REGISTRY
```

### Step 4: Verify

```bash
# List your new tasks
litmus list --tasks mybench

# Run a quick smoke test with a few examples
litmus eval --model gpt-4o --backend api --tasks mybench --max-examples 5
```

### Summary of task types and their requirements

| `task_type` | `framing` | `metric` | `choices` | `target_formatter` | Parsing |
|-------------|-----------|----------|-----------|---------------------|---------|
| `regression` | `regression` | `spearman` | `None` | `None` | `parse_float` — extracts a number |
| `binary` (auto) | `binary` | `accuracy` | `["high", "low"]` | median split (auto) | `parse_label` — fuzzy label match |
| `multiclass` (auto) | `binned` | `accuracy` | `["very_low", ..., "very_high"]` | quartile bins (auto) | `parse_label` |
| `binary` | `binary` | `accuracy` | e.g. `["yes", "no"]` | maps raw target to label | `parse_label` |
| `multiclass` | `multiclass` | `accuracy` | list of class labels | optional | `parse_label` |
| `multilabel` | `multilabel` | `f1_macro` | list of all labels | optional | `parse_multilabel` — comma/semicolon split |

Rows marked "(auto)" are generated by `make_binary_task` / `make_binned_task` and don't need to be defined manually.

### Tips

- **Prompt placeholders must match your input format.** If `load_fn` returns `{"input": "ACGT..."}` (a string), the template should use `{sequence}`. If it returns `{"input": {"protein": ..., "ligand": ...}}` (a dict), the template should use `{protein}` and `{ligand}`.
- **Always list choices in the prompt.** The LLM needs to know the valid options. The `choices` field in `TaskConfig` is used for parsing, but the prompt text is what the model actually sees.
- **Use `partial` for `load_fn`.** The registry is built eagerly, but `load_fn` is only called at evaluation time. Using `functools.partial` defers the actual data loading.
- **Binary/binned variants need training data.** `make_binary_task` and `make_binned_task` compute thresholds from training targets, so they call the loader for the train split at registration time.
- **The system prompt is shared.** `SYSTEM_PROMPT` tells the model to provide reasoning and wrap the final answer in `<answer></answer>` tags. Use it unless you have a specific reason to override.
- **`target_formatter` is required when raw targets don't match choice labels.** For example, if the dataset stores `1`/`0` but your choices are `"yes"`/`"no"`, provide a formatter: `lambda t: "yes" if t == 1 else "no"`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Project Structure

```
src/litmus/
    __init__.py
    cli.py              # Click CLI entry point
    model.py            # APIModel (OpenAI-compatible) and VLLMModel backends
    runner.py           # Evaluation pipeline: prompt -> parse -> score
    parsing.py          # Response parsing: extract_answer, parse_float, parse_label
    scoring.py          # Metrics: regression, classification, multilabel
    report.py           # Output formatting: markdown, JSON, CSV
    tasks/
        __init__.py     # Task registry and filtering
        _base.py        # TaskConfig dataclass and system prompt
        _framing.py     # Binary/binned framing generators
        _loader.py      # HuggingFace dataset loaders
        peer.py         # PEER task definitions
        calm.py         # CALM task definitions
        moleculeace.py  # MoleculeACE task definitions
```
