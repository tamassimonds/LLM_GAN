"""Evaluation utilities for the generator model."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from llm_gan.eval.common import (
    extract_tagged_text,
    load_eval_dataframe,
    ensure_column,
    run_pairwise_judge,
)
from llm_gan.prompts import llm_generator_prompt
from llm_gan.utils.inference_config import InferenceConfig


def eval_generator(
    generator: Optional[Any],
    judge: Any,
    *,
    dataset_path: str = "data/combined_eval_stories.csv",
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
    return_details: bool = False,
    generator_backend: Optional[str] = None,
    judge_backend: Optional[str] = None,
    generator_provider: Optional[str] = None,
    judge_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate how often the generator can fool the judge into misclassification.

    ``generator`` and ``judge`` can be:
    - an ``InferenceConfig``
    - a dict specification compatible with :class:`InferenceConfig`
    - a string model name (assumed API backend)
    - for the generator, a local ``torch.nn.Module`` or similar
    """

    df = load_eval_dataframe(dataset_path, num_samples=num_samples, seed=seed)

    titles = ensure_column(df, ["title"], "title")
    genres = ensure_column(df, ["genre"], "genre")
    human_stories = ensure_column(df, ["human_story", "story"], "human story")

    titles_list = titles.tolist()
    genres_list = genres.tolist()
    human_list = human_stories.tolist()

    generator_config = None if generator is None else InferenceConfig.coerce(
        generator,
        default_backend=generator_backend,
        default_provider=generator_provider,
    )

    judge_config = InferenceConfig.coerce(
        judge,
        default_backend=judge_backend,
        default_provider=judge_provider,
    )

    if generator_config is not None:
        prompts = [
            llm_generator_prompt(title, genre)
            for title, genre in zip(titles_list, genres_list)
        ]
        raw_generations = generator_config.run(prompts)
        ai_stories = [extract_tagged_text(output, "story") for output in raw_generations]
        df = df.assign(ai_story=ai_stories, generator_raw=raw_generations)
    else:
        ai_stories = ensure_column(df, ["ai_story", "generated_story"], "AI story").tolist()

    result = run_pairwise_judge(
        human_list,
        ai_stories,
        titles_list,
        genres_list,
        judge_config,
        seed=seed,
        return_details=return_details,
    )

    summary: Dict[str, Any] = {
        "num_samples": result.num_samples,
        "judge_accuracy": result.accuracy,
        "generator_fooling_rate": result.fooling_rate,
        "invalid_responses": result.invalid_responses,
    }
    if return_details:
        summary["details"] = result.details
    if generator_config is not None:
        summary["ai_stories"] = ai_stories
    return summary


__all__ = ["eval_generator"]
