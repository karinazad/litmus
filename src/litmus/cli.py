"""CLI entry point for Litmus benchmark."""

import asyncio
import logging
import os
import sys

import click

from litmus.report import format_csv, format_json, format_markdown
from litmus.runner import run_eval
from litmus.tasks import get_tasks


@click.group()
def main() -> None:
    """Litmus: LLM Property Prediction Benchmark."""
    pass


@main.command()
@click.option("--model", required=True, help="Model name: HF model ID, local path, or API model name")
@click.option(
    "--backend",
    type=click.Choice(["api", "azure", "vllm"]),
    default=None,
    help="Backend to use. If omitted, auto-detects: 'azure' when AZURE_OPENAI_ENDPOINT is set, 'api' when --base-url or --api-key is set, otherwise 'vllm'.",
)
@click.option("--base-url", default=None, help="API base URL (implies --backend api)")
@click.option("--api-key", default=None, help="API key (implies --backend api, default: OPENAI_API_KEY env var)")
@click.option("--tasks", "task_filter", default=None, help="Comma-separated task prefixes")
@click.option("--framing", default=None, type=click.Choice(["regression", "binary", "binned", "multiclass", "multilabel"]), help="Filter by framing")
@click.option("--max-examples", default=None, type=int, help="Max examples per task")
@click.option("--max-concurrent", default=10, type=int, help="Max concurrent API requests (api backend)")
@click.option("--temperature", default=0.0, type=float, help="Sampling temperature")
@click.option("--max-tokens", default=1024, type=int, help="Max tokens in model response")
@click.option("--output", default=None, help="Output file path (JSON)")
@click.option("--format", "fmt", default="markdown", type=click.Choice(["markdown", "json", "csv"]))
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
# vLLM-specific options
@click.option("--tensor-parallel-size", default=1, type=int, help="Number of GPUs for tensor parallelism (vllm backend)")
@click.option("--gpu-memory-utilization", default=0.9, type=float, help="GPU memory fraction (vllm backend)")
@click.option("--dtype", default="auto", help="Model dtype: auto, float16, bfloat16 (vllm backend)")
@click.option("--max-model-len", default=None, type=int, help="Max context length (vllm backend)")
def eval(
    model: str,
    backend: str | None,
    base_url: str | None,
    api_key: str | None,
    task_filter: str | None,
    framing: str | None,
    max_examples: int | None,
    max_concurrent: int,
    temperature: float,
    max_tokens: int,
    output: str | None,
    fmt: str,
    verbose: bool,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    max_model_len: int | None,
) -> None:
    """Evaluate an LLM on property prediction tasks."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    tasks = get_tasks(filter_str=task_filter, framing=framing)
    if not tasks:
        click.echo("No tasks matched the filter. Use 'litmus list' to see available tasks.")
        sys.exit(1)

    # Auto-detect backend
    if backend is None:
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):
            backend = "azure"
        elif base_url or api_key:
            backend = "api"
        else:
            backend = "vllm"

    click.echo(f"Running {len(tasks)} task(s) with model {model} (backend: {backend})")

    if backend == "azure":
        from litmus.model import AzureModel

        llm = AzureModel(
            model=model,
            max_concurrent=max_concurrent,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif backend == "api":
        from litmus.model import APIModel

        llm = APIModel(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_concurrent=max_concurrent,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        from litmus.model import VLLMModel

        llm = VLLMModel(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_model_len=max_model_len,
        )

    results = asyncio.run(run_eval(llm, tasks, max_examples=max_examples))

    if fmt == "markdown":
        text = format_markdown(results)
    elif fmt == "json":
        text = format_json(results, include_predictions=True)
    else:
        text = format_csv(results)

    click.echo(text)

    if output:
        with open(output, "w") as f:
            f.write(format_json(results, include_predictions=True))
        click.echo(f"\nDetailed results saved to {output}")


@main.command(name="list")
@click.option("--tasks", "task_filter", default=None, help="Filter by task prefix")
@click.option("--framing", default=None, help="Filter by framing")
def list_tasks(task_filter: str | None, framing: str | None) -> None:
    """List available evaluation tasks."""
    tasks = get_tasks(filter_str=task_filter, framing=framing)

    if not tasks:
        click.echo("No tasks found.")
        return

    click.echo(f"{'Name':<40} {'Benchmark':<15} {'Type':<12} {'Framing':<10} {'Metric':<10}")
    click.echo("-" * 87)
    for t in sorted(tasks, key=lambda x: x.name):
        click.echo(f"{t.name:<40} {t.benchmark:<15} {t.task_type:<12} {t.framing:<10} {t.metric:<10}")


if __name__ == "__main__":
    main()
