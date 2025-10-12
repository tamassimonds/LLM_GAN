"""Shared helpers for evaluation scripts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from llm_gan.prompts import llm_generator_discriminator_prompt
from llm_gan.utils.inference_config import InferenceConfig
from llm_gan.utils.parse import parse_tags


@dataclass
class PairwiseJudgeResult:
    """Container holding results from a pairwise human vs AI evaluation."""

    accuracy: float
    fooling_rate: float
    invalid_responses: int
    num_samples: int
    answers: List[Optional[int]]
    labels: List[int]
    responses: List[str]
    details: List[Dict[str, Any]]


def load_eval_dataframe(
    path: str,
    *,
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Load the evaluation CSV and optionally subsample rows."""

    df = pd.read_csv(path)
    if num_samples is not None and num_samples < len(df):
        df = df.sample(n=num_samples, random_state=seed)
    return df.reset_index(drop=True)


def ensure_column(df: pd.DataFrame, candidates: Sequence[str], friendly_name: str) -> pd.Series:
    """Return the first matching column from ``candidates`` as strings."""

    for name in candidates:
        if name in df.columns:
            return df[name].fillna("").astype(str)
    raise KeyError(f"Expected a column for {friendly_name} (any of {candidates}) in dataset: {list(df.columns)}")


def extract_tagged_text(raw: str, tag: str) -> str:
    """Extract content inside ``<tag>...</tag>`` or return stripped fallback."""

    value = parse_tags(raw, tag)
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, str):
        return value.strip()
    return raw.strip()


def parse_answer_tag(raw: str) -> Optional[int]:
    """Parse the first ``<answer>`` tag (or fallback digits) into ``1`` or ``2``."""

    value = parse_tags(raw, "answer")
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, str):
        value = value.strip()
    if isinstance(value, str) and value:
        for char in value:
            if char in {"1", "2"}:
                return int(char)
    if isinstance(raw, str):
        for char in raw:
            if char in {"1", "2"}:
                return int(char)
    return None


def run_pairwise_judge(
    human_stories: Iterable[str],
    ai_stories: Iterable[str],
    titles: Iterable[str],
    genres: Iterable[str],
    judge: InferenceConfig,
    *,
    seed: Optional[int] = None,
    return_details: bool = False,
) -> PairwiseJudgeResult:
    """Evaluate a judge model on (human, AI) story pairs."""

    prompts: List[str] = []
    labels: List[int] = []
    metadata: List[Dict[str, Any]] = []

    rng = random.Random(seed)

    for human, artificial, title, genre in zip(human_stories, ai_stories, titles, genres):
        human = (human or "").strip()
        artificial = (artificial or "").strip()
        if not human or not artificial:
            continue
        order = rng.randint(0, 1)
        if order == 0:
            story1, story2 = human, artificial
            label = 1  # story 1 is human
        else:
            story1, story2 = artificial, human
            label = 2  # story 2 is human
        prompt = llm_generator_discriminator_prompt(title, genre, story1, story2)
        prompts.append(prompt)
        labels.append(label)
        metadata.append(
            {
                "title": title,
                "genre": genre,
                "human_first": order == 0,
                "story1": story1,
                "story2": story2,
            }
        )

    if not prompts:
        return PairwiseJudgeResult(
            accuracy=0.0,
            fooling_rate=0.0,
            invalid_responses=0,
            num_samples=0,
            answers=[],
            labels=[],
            responses=[],
            details=[],
        )

    responses = judge.run(prompts)
    answers: List[Optional[int]] = [parse_answer_tag(response) for response in responses]

    correct = 0
    invalid = 0
    details: List[Dict[str, Any]] = []

    for meta, expected, answer, response in zip(metadata, labels, answers, responses):
        is_correct = answer == expected
        if answer not in (1, 2):
            invalid += 1
            is_correct = False
        if is_correct:
            correct += 1
        if return_details:
            details.append(
                {
                    **meta,
                    "expected_answer": expected,
                    "model_answer": answer,
                    "raw_response": response,
                    "is_correct": is_correct,
                }
            )

    total = len(labels)
    accuracy = correct / total if total else 0.0
    fooling_rate = 1.0 - accuracy

    return PairwiseJudgeResult(
        accuracy=accuracy,
        fooling_rate=fooling_rate,
        invalid_responses=invalid,
        num_samples=total,
        answers=answers,
        labels=labels,
        responses=responses,
        details=details,
    )


__all__ = [
    "PairwiseJudgeResult",
    "load_eval_dataframe",
    "ensure_column",
    "extract_tagged_text",
    "parse_answer_tag",
    "run_pairwise_judge",
]
