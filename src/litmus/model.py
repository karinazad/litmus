"""Model interface for Litmus benchmark."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Model(Protocol):
    """Protocol for LLM model backends."""

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Generate a completion for the given messages.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dicts with "role" and "content" keys.

        Returns
        -------
        str
            The model's response text.
        """
        ...


@runtime_checkable
class BatchModel(Protocol):
    """Protocol for model backends that support batched inference."""

    async def batch_complete(
        self, messages_batch: list[list[dict[str, str]]]
    ) -> list[str]:
        """Generate completions for a batch of message lists.

        Parameters
        ----------
        messages_batch : list[list[dict[str, str]]]
            List of conversation message lists.

        Returns
        -------
        list[str]
            List of response texts, one per conversation.
        """
        ...


class APIModel:
    """OpenAI-compatible async model client with retry and rate limiting.

    Parameters
    ----------
    model : str
        Model identifier, e.g. "gpt-4o".
    base_url : str | None
        Custom API base URL (for vLLM, Ollama, etc.).
    api_key : str | None
        API key. If None, uses OPENAI_API_KEY env var.
    max_concurrent : int
        Maximum number of concurrent API requests.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens in the response.
    max_retries : int
        Maximum number of retries on transient errors.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_concurrent: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 5,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Generate a completion with retry and rate limiting.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dicts with "role" and "content" keys.

        Returns
        -------
        str
            The model's response text.

        Raises
        ------
        openai.APIError
            If all retries are exhausted.
        """
        from openai import APIError, RateLimitError

        async with self._semaphore:
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    response = await self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    return response.choices[0].message.content or ""
                except RateLimitError as e:
                    last_error = e
                    wait = 2 ** attempt
                    logger.warning(
                        "Rate limited (attempt %d/%d), waiting %ds",
                        attempt + 1,
                        self.max_retries,
                        wait,
                    )
                    await asyncio.sleep(wait)
                except APIError as e:
                    if e.status_code and e.status_code >= 500:
                        last_error = e
                        wait = 2 ** attempt
                        logger.warning(
                            "Server error %d (attempt %d/%d), waiting %ds",
                            e.status_code,
                            attempt + 1,
                            self.max_retries,
                            wait,
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise
            raise last_error  # type: ignore[misc]


class VLLMModel:
    """vLLM offline inference model with batched generation.

    Loads a HuggingFace model locally via vLLM and runs inference
    using vLLM's efficient batched engine. Supports both single
    and batch completion.

    Parameters
    ----------
    model : str
        HuggingFace model ID or local path, e.g. "meta-llama/Llama-3.1-8B-Instruct".
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens in the response.
    tensor_parallel_size : int
        Number of GPUs for tensor parallelism.
    gpu_memory_utilization : float
        Fraction of GPU memory to use (0.0 to 1.0).
    dtype : str
        Data type for model weights, e.g. "auto", "float16", "bfloat16".
    max_model_len : int | None
        Maximum model context length. If None, uses the model's default.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        max_model_len: int | None = None,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vllm is required for local model inference. "
                "Install it with: pip install vllm"
            ) from e

        self.model_name = model
        self._sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        logger.info("Loading model %s with vLLM...", model)
        llm_kwargs: dict = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self._llm = LLM(**llm_kwargs)
        logger.info("Model loaded successfully.")

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Generate a completion for a single conversation.

        Wraps the synchronous vLLM call in an executor for async compatibility.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dicts with "role" and "content" keys.

        Returns
        -------
        str
            The model's response text.
        """
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._generate_chat, [messages]
        )
        return results[0]

    async def batch_complete(
        self, messages_batch: list[list[dict[str, str]]]
    ) -> list[str]:
        """Generate completions for a batch of conversations.

        Uses vLLM's native batching for efficient throughput.

        Parameters
        ----------
        messages_batch : list[list[dict[str, str]]]
            List of conversation message lists.

        Returns
        -------
        list[str]
            List of response texts, one per conversation.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._generate_chat, messages_batch
        )

    def _generate_chat(
        self, messages_batch: list[list[dict[str, str]]]
    ) -> list[str]:
        """Run batched chat generation synchronously via vLLM.

        Parameters
        ----------
        messages_batch : list[list[dict[str, str]]]
            List of conversation message lists.

        Returns
        -------
        list[str]
            Generated response texts.
        """
        outputs = self._llm.chat(
            messages=messages_batch,
            sampling_params=self._sampling_params,
        )
        return [output.outputs[0].text for output in outputs]
