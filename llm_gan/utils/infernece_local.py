"""Local inference helpers for on-device models (Hugging Face, vLLM, etc.)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

SUPPORTED_LOCAL_PROVIDERS = {"huggingface", "hf", "module", "vllm", "local_vllm"}
DEFAULT_LOCAL_BATCH_SIZE = 8


class LocalInferenceProvider:
    def generate(self, prompts: Sequence[str], **request_options: Any) -> List[str]:  # pragma: no cover - interface hook
        raise NotImplementedError


def batch_local_inference(
    prompts: Sequence[str] | str,
    *,
    model: Any = None,
    tokenizer: Any = None,
    provider: str = "huggingface",
    batch_size: int = DEFAULT_LOCAL_BATCH_SIZE,
    provider_options: Optional[Dict[str, Any]] = None,
    request_options: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[str]:
    """Run batched inference against a local model.

    Parameters
    ----------
    prompts:
        Either a string prompt or a sequence of prompt strings.
    model:
        Backing model instance. For the ``huggingface`` provider this should be a
        ``torch.nn.Module`` that exposes ``generate`` (eg. ``AutoModelForCausalLM``).
        When using VLLM, ``model`` can be a string model identifier.
    tokenizer:
        Tokenizer compatible with the supplied model (required for Hugging Face).
    provider:
        Local provider identifier – ``"huggingface"`` (default) or ``"vllm"``.
    batch_size:
        Number of prompts processed per local forward/generation pass.
    provider_options:
        Options forwarded to the provider constructor (eg. ``device``,
        ``generation_defaults`` for Hugging Face or vLLM initialisation kwargs).
    request_options:
        Per-call generation overrides (merged with ``kwargs``).
    kwargs:
        Convenience alias for ``request_options`` so callers can pass
        ``temperature=0.7`` etc directly.
    """

    provider_options = dict(provider_options or {})
    request_options = dict(request_options or {})
    for key, value in kwargs.items():
        request_options.setdefault(key, value)

    prompt_values = _normalise_prompts(prompts)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    backend = _get_local_provider(provider, model=model, tokenizer=tokenizer, provider_options=provider_options)

    responses: List[str] = []
    for chunk in _chunked(prompt_values, batch_size):
        responses.extend(backend.generate(chunk, **request_options))
    return responses


def _normalise_prompts(prompts: Sequence[str] | str) -> List[str]:
    if isinstance(prompts, str):
        return [prompts]
    if not isinstance(prompts, Iterable):
        raise TypeError("prompts must be a string or an iterable of strings")
    normalised: List[str] = []
    for prompt in prompts:
        if not isinstance(prompt, str):
            raise TypeError("all prompts must be strings")
        normalised.append(prompt)
    if not normalised:
        raise ValueError("at least one prompt must be provided")
    return normalised


def _chunked(values: Sequence[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _get_local_provider(
    provider: Optional[str],
    *,
    model: Any,
    tokenizer: Any,
    provider_options: Dict[str, Any],
) -> LocalInferenceProvider:
    provider_key = (provider or "huggingface").lower()
    if provider_key in {"huggingface", "hf", "module"}:
        if model is None or tokenizer is None:
            raise ValueError("Hugging Face local inference requires both 'model' and 'tokenizer'.")
        return _HuggingFaceModuleProvider(model, tokenizer, **provider_options)
    if provider_key in {"vllm", "local_vllm"}:
        return _VLLMLocalProvider.get_instance(model, **provider_options)
    raise ValueError(f"Unsupported local provider '{provider}'. Supported providers: {sorted(SUPPORTED_LOCAL_PROVIDERS)}")


class _HuggingFaceModuleProvider(LocalInferenceProvider):
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: Optional[str] = None,
        generation_defaults: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        move_to_device: bool = True,
    ) -> None:
        torch = _lazy_import_torch()
        self._torch = torch
        self._tokenizer = tokenizer
        self._generation_defaults = dict(generation_defaults or {})
        self._tokenizer_kwargs = dict(tokenizer_kwargs or {})

        resolved_device = device or _infer_model_device(model, torch)
        self._device = torch.device(resolved_device)
        if move_to_device and hasattr(model, "to"):
            model.to(self._device)
        if hasattr(model, "eval"):
            model.eval()
        self._model = model

    def generate(self, prompts: Sequence[str], **request_options: Any) -> List[str]:
        torch = self._torch
        generation_args = dict(self._generation_defaults)
        generation_args.update(request_options)

        if "max_tokens" in generation_args and "max_new_tokens" not in generation_args:
            generation_args["max_new_tokens"] = generation_args.pop("max_tokens")
        else:
            generation_args.pop("max_tokens", None)
        generation_args.pop("stop", None)

        encode_kwargs = dict(self._tokenizer_kwargs)
        encode_kwargs.setdefault("return_tensors", "pt")
        encode_kwargs.setdefault("padding", True)
        encoded = self._tokenizer(list(prompts), **encode_kwargs)

        if hasattr(encoded, "to"):
            encoded = encoded.to(self._device)
        else:
            encoded = {
                key: (value.to(self._device) if hasattr(value, "to") else torch.tensor(value, device=self._device))
                for key, value in encoded.items()
            }

        with torch.inference_mode():
            sequences = self._model.generate(**encoded, **generation_args)

        if hasattr(sequences, "to"):
            sequences = sequences.to("cpu")

        decoded = self._tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return [text.strip() for text in decoded]


@dataclass
class _VLLMLocalProvider(LocalInferenceProvider):
    model_name: str
    init_options: Dict[str, Any]

    _instances: Dict[str, "_VLLMLocalProvider"] = field(default_factory=dict) #WHAT??? this is incorrect code

    def __post_init__(self) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime error path
            raise RuntimeError(
                "The 'vllm' package is required for local vLLM inference. Install it via 'pip install vllm'."
            ) from exc

        sampling_params = self.init_options.pop("sampling_params", {}) or {}
        temperature = self.init_options.pop("temperature", sampling_params.get("temperature", 0.7))
        top_p = self.init_options.pop("top_p", sampling_params.get("top_p", 0.95))
        max_tokens = self.init_options.pop("max_tokens", sampling_params.get("max_tokens", 512))
        stop = self.init_options.pop("stop", sampling_params.get("stop"))

        self._SamplingParams = SamplingParams
        self._sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )

        init_kwargs = self.init_options.pop("init_kwargs", {}) or {}
        init_kwargs.update({k: v for k, v in self.init_options.items() if v is not None})
        self._llm = LLM(model=self.model_name, **init_kwargs)

    @classmethod
    def get_instance(cls, model: Any, **init_options: Any) -> "_VLLMLocalProvider":
        if not isinstance(model, str):
            raise ValueError("vLLM local provider expects 'model' to be a string identifier.")
        instance = cls._instances.get(model)
        if instance is None:
            instance = cls(model, init_options)
            cls._instances[model] = instance
        return instance

    def generate(self, prompts: Sequence[str], **request_options: Any) -> List[str]:
        sampling_params = self._sampling_params
        if request_options:
            sampling_params = self._SamplingParams(
                temperature=request_options.get("temperature", sampling_params.temperature),
                top_p=request_options.get("top_p", sampling_params.top_p),
                max_tokens=request_options.get("max_tokens", sampling_params.max_tokens),
                stop=request_options.get("stop", sampling_params.stop),
            )

        outputs = self._llm.generate(list(prompts), sampling_params)
        decoded: List[str] = []
        for output in outputs:
            if not output.outputs:
                decoded.append("")
                continue
            decoded.append(output.outputs[0].text.strip())
        return decoded


def _lazy_import_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime error path
        raise RuntimeError(
            "PyTorch is required for Hugging Face local inference. Install it via 'pip install torch'."
        ) from exc
    return torch


def _infer_model_device(model: Any, torch_module: Any) -> str:
    if hasattr(model, "device") and model.device is not None:
        return str(model.device)
    if hasattr(model, "parameters"):
        try:
            first_param = next(model.parameters())
            return str(first_param.device)
        except StopIteration:
            pass
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


__all__ = [
    "batch_local_inference",
    "SUPPORTED_LOCAL_PROVIDERS",
    "DEFAULT_LOCAL_BATCH_SIZE",
]
