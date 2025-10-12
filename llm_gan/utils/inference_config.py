"""Utilities for describing and executing model inference backends."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

from .infernece_api import batch_inference_api
from .infernece_local import DEFAULT_LOCAL_BATCH_SIZE, batch_local_inference

BackendType = Literal["api", "local"]


@dataclass
class InferenceConfig:
    """Lightweight description of how to execute inference for a model.

    Parameters
    ----------
    backend:
        ``"api"`` to route through ``batch_inference_api`` or ``"local"`` to use
        ``batch_local_inference``.
    model:
        For API backends this should usually be a model identifier string. For
        local backends it can be a ``torch.nn.Module`` (Hugging Face) or a model
        name understood by the local provider (eg. vLLM).
    tokenizer:
        Optional tokenizer used by local backends that require one (eg. Hugging
        Face ``AutoTokenizer``).
    provider:
        Provider hint (eg. ``"openai"`` or ``"huggingface"``). For API backends
        it maps directly to ``batch_inference_api``. For local backends it maps
        to ``batch_local_inference`` (defaults to ``"huggingface"`` when
        omitted).
    provider_options:
        Keyword arguments forwarded to the underlying provider constructor.
    request_options:
        Keyword arguments forwarded to each generation request.
    batch_size:
        Number of prompts to process per batch.
    """

    backend: BackendType
    model: Any
    tokenizer: Any = None
    provider: Optional[str] = None
    provider_options: Dict[str, Any] = None
    request_options: Dict[str, Any] = None
    batch_size: int = DEFAULT_LOCAL_BATCH_SIZE

    def __post_init__(self) -> None:
        self.provider_options = dict(self.provider_options or {})
        self.request_options = dict(self.request_options or {})
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

    def run(self, prompts: Union[str, Iterable[str]]) -> List[str]:
        """Execute inference using the configured backend."""

        if self.backend == "api":
            return batch_inference_api(
                self.model,
                prompts,
                provider=self.provider,
                batch_size=self.batch_size,
                provider_options=self.provider_options,
                request_options=self.request_options,
            )

        if self.backend == "local":
            return batch_local_inference(
                prompts,
                model=self.model,
                tokenizer=self.tokenizer,
                provider=self.provider,
                batch_size=self.batch_size,
                provider_options=self.provider_options,
                request_options=self.request_options,
            )

        raise ValueError(f"Unsupported backend '{self.backend}'.")

    def override(self, **updates: Any) -> "InferenceConfig":
        """Return a copy with updated fields without mutating self."""

        return replace(self, **updates)

    @classmethod
    def coerce(
        cls,
        spec: "InferenceConfig" | Dict[str, Any] | str | Any,
        *,
        default_backend: Optional[BackendType] = None,
        default_provider: Optional[str] = None,
        default_batch_size: Optional[int] = None,
    ) -> "InferenceConfig":
        """Convert a variety of specifications into an ``InferenceConfig``.

        Accepted specs:
        - ``InferenceConfig``: returned as-is (optionally overridden by defaults)
        - ``str``: interpreted as API model identifier
        - ``dict``: keys map directly to the dataclass fields (``backend`` is
          optional; inferred when missing)
        - Other objects: treated as local models (requires ``tokenizer`` either
          in defaults or dict spec)
        """

        if isinstance(spec, cls):
            return cls._apply_defaults(spec, default_provider, default_batch_size)

        data: Dict[str, Any]
        if isinstance(spec, dict):
            data = dict(spec)
        elif isinstance(spec, str):
            data = {"model": spec, "backend": default_backend or "api"}
        else:
            data = {"model": spec, "backend": default_backend or "local"}

        backend = data.get("backend")
        if backend is None:
            backend = default_backend
        if backend is None:
            backend = "api" if isinstance(data.get("model"), str) else "local"
        if backend not in ("api", "local"):
            raise ValueError(f"Unsupported backend '{backend}'.")
        data["backend"] = backend

        if "provider" not in data and default_provider is not None:
            data["provider"] = default_provider
        if default_batch_size is not None and "batch_size" not in data:
            data["batch_size"] = default_batch_size

        provider_options = data.get("provider_options")
        if provider_options is None:
            data["provider_options"] = {}
        request_options = data.get("request_options")
        if request_options is None:
            data["request_options"] = {}

        return cls(**data)

    @staticmethod
    def _apply_defaults(
        config: "InferenceConfig",
        default_provider: Optional[str],
        default_batch_size: Optional[int],
    ) -> "InferenceConfig":
        updates: Dict[str, Any] = {}
        if default_provider is not None and config.provider is None:
            updates["provider"] = default_provider
        if default_batch_size is not None and config.batch_size == DEFAULT_LOCAL_BATCH_SIZE:
            updates["batch_size"] = default_batch_size
        if updates:
            return config.override(**updates)
        return config


__all__ = ["InferenceConfig", "BackendType"]
