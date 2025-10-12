"""Utilities for batching inference requests to remote API providers.

This module centralises the logic for talking to hosted inference providers
(eg. OpenAI, Lambda Labs) and exposes a single helper function,
``batch_inference_api``, that takes a list of prompts and returns generated
completions. Providers and models can be chosen explicitly or resolved via a
``provider:model`` shorthand string.

Example usages
--------------

>>> batch_inference_api(
...     "openai:gpt-4o-mini",
...     ["Write a haiku about llamas."],
...     temperature=0.7,
... )

>>> batch_inference_api(
...     "mistral-7b-instruct",
...     ["Explain GANs in two sentences."],
...     provider="lambda",
...     provider_options={"api_key": "...", "base_url": "https://api.lambdalabs.com/v1"},
... )

The public helper aims to be defensive: it normalises prompts, chunks large
batches, performs basic error handling, and raises informative exceptions when
optional dependencies are missing.

For on-device inference (eg. Hugging Face ``nn.Module`` or vLLM), use
``llm_gan.utils.infernece_local`` instead.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = {"openai", "lambda"}
DEFAULT_BATCH_SIZE = 256


class InferenceProvider:
    """Small protocol for inference providers."""

    def generate(self, prompts: Sequence[str], **request_options: Any) -> List[str]:  # pragma: no cover - interface
        raise NotImplementedError


def batch_inference_api(
    model: str | Sequence[str],
    prompts: Optional[Sequence[str]] = None,
    *,
    provider: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    provider_options: Optional[Dict[str, Any]] = None,
    request_options: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[str]:
    """Send a batch of prompts to the requested inference backend.

    Parameters
    ----------
    model:
        Either the model name (optionally prefixed with ``provider:``) or, if
        ``prompts`` is omitted, the iterable of prompts. When prompts are
        supplied as the first positional argument, the model is resolved from
        ``kwargs['model']`` or ``LLM_MODEL``.
    prompts:
        Sequence of prompt strings. If omitted, the first positional argument is
        treated as the prompt sequence.
    provider:
        Explicit provider name (``"openai"`` or ``"lambda"``).
    batch_size:
        Number of prompts to send in a single request chunk. Providers that do
        not support batching will handle this internally and emit one request
        per prompt.
    provider_options:
        Keyword arguments forwarded to the provider constructor (eg. API keys).
    request_options:
        Keyword arguments forwarded to each generation call (eg. temperature).
    kwargs:
        Convenience shim – any remaining keyword arguments are merged into
        ``request_options`` so callers can pass ``temperature=0.7`` etc without
        wrapping them in a dictionary.
    """

    provider_options = dict(provider_options or {})
    request_options = dict(request_options or {})

    # Allow direct keyword usage for common generation settings.
    for key, value in kwargs.items():
        request_options.setdefault(key, value)

    prompt_values, model_name = _resolve_prompts_and_model(model, prompts, request_options)
    provider_name, clean_model_name = _resolve_provider(model_name, provider)

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    inference_client = _get_provider(provider_name, clean_model_name, provider_options)

    responses: List[str] = []
    for chunk in _chunked(prompt_values, batch_size):
        responses.extend(inference_client.generate(chunk, **request_options))

    return responses


def _resolve_prompts_and_model(
    model_or_prompts: str | Sequence[str],
    explicit_prompts: Optional[Sequence[str]],
    request_options: Dict[str, Any],
) -> Tuple[List[str], str]:
    if explicit_prompts is None:
        prompts = _normalise_prompts(model_or_prompts)
        model_name = request_options.pop("model", None) or os.environ.get("LLM_MODEL")
        if not model_name:
            raise ValueError(
                "Model name is required when prompts are passed as the first argument."
            )
    else:
        if not isinstance(model_or_prompts, str):
            raise TypeError("Model name must be a string when prompts are provided separately.")
        model_name = model_or_prompts
        prompts = _normalise_prompts(explicit_prompts)

    return prompts, model_name


def _resolve_provider(model_name: str, provider: Optional[str]) -> Tuple[str, str]:
    if provider:
        provider_key = provider.lower()
        if provider_key in {"local", "local_vllm", "vllm"}:
            raise ValueError(
                "Local providers have moved to llm_gan.utils.infernece_local. "
                "Use batch_local_inference instead of batch_inference_api for on-device models."
            )
        if provider_key not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider '{provider}'. Supported providers: {sorted(SUPPORTED_PROVIDERS)}")
        return provider_key, model_name

    if ":" in model_name:
        prefix, remainder = model_name.split(":", 1)
        prefix_key = prefix.lower()
        if prefix_key in SUPPORTED_PROVIDERS:
            return prefix_key, remainder
        if prefix_key in {"local", "local_vllm", "vllm"}:
            raise ValueError(
                "Local providers have moved to llm_gan.utils.infernece_local. "
                "Use batch_local_inference instead of batch_inference_api for on-device models."
            )

    env_provider = os.environ.get("LLM_PROVIDER")
    if env_provider:
        provider_key = env_provider.lower()
        if provider_key in {"local", "local_vllm", "vllm"}:
            raise ValueError(
                "Local providers have moved to llm_gan.utils.infernece_local. "
                "Use batch_local_inference instead of batch_inference_api for on-device models."
            )
        if provider_key not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Provider '{env_provider}' from environment variable LLM_PROVIDER is not supported."
            )
        return provider_key, model_name

    # Default to OpenAI when nothing is specified.
    return "openai", model_name


def _normalise_prompts(prompts: str | Sequence[str]) -> List[str]:
    if isinstance(prompts, str):
        return [prompts]

    if not isinstance(prompts, Iterable):
        raise TypeError("Prompts must be a string or an iterable of strings.")

    normalised: List[str] = []
    for prompt in prompts:
        if not isinstance(prompt, str):
            raise TypeError("All prompts in the iterable must be strings.")
        normalised.append(prompt)
    if not normalised:
        raise ValueError("At least one prompt must be provided.")
    return normalised


def _chunked(values: Sequence[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _get_provider(provider_name: str, model_name: str, provider_options: Dict[str, Any]) -> InferenceProvider:
    provider_key = provider_name.lower()
    if provider_key in {"openai"}:
        return _OpenAIProvider(model_name, **provider_options)
    if provider_key == "lambda":
        return _LambdaProvider(model_name, **provider_options)
    raise ValueError(f"No provider implementation available for '{provider_name}'.")


@dataclass
class _OpenAIProvider(InferenceProvider):
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    timeout: float = float(os.environ.get("OPENAI_TIMEOUT", "60"))

    def __post_init__(self) -> None:
        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime error path
            raise RuntimeError(
                "The 'openai' package is required for the OpenAI provider. Install it via 'pip install openai'."
            ) from exc

        self._is_client_api = hasattr(openai, "OpenAI")
        self._openai_module = openai

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key in provider_options.")

        self._client = None
        if self._is_client_api:
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=self.base_url or os.environ.get("OPENAI_BASE_URL"),
                organization=self.organization or os.environ.get("OPENAI_ORG"),
                timeout=self.timeout,
            )
        else:
            openai.api_key = api_key  # type: ignore[attr-defined]
            if self.base_url or os.environ.get("OPENAI_BASE_URL"):
                openai.api_base = (self.base_url or os.environ.get("OPENAI_BASE_URL"))  # type: ignore[attr-defined]
            if self.organization or os.environ.get("OPENAI_ORG"):
                openai.organization = (self.organization or os.environ.get("OPENAI_ORG"))  # type: ignore[attr-defined]

    def generate(self, prompts: Sequence[str], **request_options: Any) -> List[str]:
        temperature = request_options.get("temperature", 0.7)
        top_p = request_options.get("top_p", 0.95)
        max_tokens = request_options.get("max_tokens", 512)
        system_prompt = request_options.get("system_prompt")
        stop = request_options.get("stop")
        presence_penalty = request_options.get("presence_penalty")
        frequency_penalty = request_options.get("frequency_penalty")

        responses: List[str] = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            if stop is not None:
                payload["stop"] = stop
            if presence_penalty is not None:
                payload["presence_penalty"] = presence_penalty
            if frequency_penalty is not None:
                payload["frequency_penalty"] = frequency_penalty

            try:
                if self._is_client_api and self._client is not None:
                    response = self._client.chat.completions.create(**payload)
                    content = response.choices[0].message.content or ""
                else:
                    response = self._openai_module.ChatCompletion.create(**payload)
                    content = response["choices"][0]["message"]["content"]
            except Exception:  # pragma: no cover - network path
                logger.exception("OpenAI request failed")
                raise

            responses.append(content.strip())

        return responses


@dataclass
class _LambdaProvider(InferenceProvider):
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    route: str = "chat/completions"
    timeout: float = float(os.environ.get("LAMBDA_TIMEOUT", "60"))
    extra_headers: Optional[Dict[str, str]] = None
    extra_payload: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime error path
            raise RuntimeError(
                "The 'requests' package is required for the Lambda provider. Install it via 'pip install requests'."
            ) from exc

        self._requests = requests
        self._session = requests.Session()

        api_key = self.api_key or os.environ.get("LAMBDA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Lambda Labs API key not provided. Set LAMBDA_API_KEY or pass api_key in provider_options."
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.extra_headers:
            headers.update(self.extra_headers)
        self._session.headers.update(headers)

        self._base_url = (self.base_url or os.environ.get("LAMBDA_API_BASE") or "https://api.lambdalabs.com/v1").rstrip("/")
        self._route = f"/{self.route.lstrip('/')}"
        self._endpoint = f"{self._base_url}{self._route}"

    def generate(self, prompts: Sequence[str], **request_options: Any) -> List[str]:
        temperature = request_options.get("temperature", 0.7)
        top_p = request_options.get("top_p", 0.95)
        max_tokens = request_options.get("max_tokens", 512)
        system_prompt = request_options.get("system_prompt")
        stop = request_options.get("stop")

        payload_defaults = self.extra_payload or {}
        responses: List[str] = []

        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            body: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            if stop is not None:
                body["stop"] = stop
            body.update(payload_defaults)

            try:
                response = self._session.post(self._endpoint, json=body, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
            except Exception:  # pragma: no cover - network path
                logger.exception("Lambda Labs request failed")
                raise

            responses.append(_extract_text_from_payload(data))

        return responses


def _extract_text_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content.strip()
                text = first.get("text")
                if isinstance(text, str):
                    return text.strip()
        for key in ("output", "output_text", "generated_text"):
            value = payload.get(key)
            if isinstance(value, str):
                return value.strip()

    raise RuntimeError(f"Unable to extract text from Lambda response: {payload}")
