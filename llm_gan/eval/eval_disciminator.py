"""Evaluation utilities for the discriminator/judge model."""

from __future__ import annotations

from typing import Any, Dict, Optional

from llm_gan.eval.common import ensure_column, load_eval_dataframe, run_pairwise_judge
from llm_gan.utils.inference_config import InferenceConfig


def eval_discriminator(
    judge: Any,
    *,
    dataset_path: str = "data/combined_eval_stories.csv",
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
    return_details: bool = False,
    judge_backend: Optional[str] = None,
    judge_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a discriminator on human vs AI story classification."""

    df = load_eval_dataframe(dataset_path, num_samples=num_samples, seed=seed)

    titles = ensure_column(df, ["title"], "title").tolist()
    genres = ensure_column(df, ["genre"], "genre").tolist()
    human_stories = ensure_column(df, ["human_story", "story"], "human story").tolist()
    ai_stories = ensure_column(df, ["ai_story", "generated_story"], "AI story").tolist()

    judge_config = InferenceConfig.coerce(
        judge,
        default_backend=judge_backend,
        default_provider=judge_provider,
    )

    result = run_pairwise_judge(
        human_stories,
        ai_stories,
        titles,
        genres,
        judge_config,
        seed=seed,
        return_details=return_details,
    )

    summary: Dict[str, Any] = {
        "num_samples": result.num_samples,
        "accuracy": result.accuracy,
        "invalid_responses": result.invalid_responses,
    }
    if return_details:
        summary["details"] = result.details
    return summary


__all__ = ["eval_discriminator"]
