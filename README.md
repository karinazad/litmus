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
