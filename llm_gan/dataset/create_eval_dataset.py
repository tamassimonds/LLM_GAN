"""Build evaluation datasets by prompting a generator model."""

from __future__ import annotations

import argparse
import sys

from datasets import load_dataset
import pandas as pd

from llm_gan.eval.common import extract_tagged_text
from llm_gan.prompts import llm_generator_prompt
from llm_gan.utils.inference_config import InferenceConfig

DEFAULT_DATASET = "FareedKhan/1k_stories_100_genre"
DEFAULT_OUTPUT = "data/ai_eval_stories.csv"
DEFAULT_COMBINED_OUTPUT = "data/combined_eval_stories.csv"


def generate_ai_stories(
    generator_spec: InferenceConfig | dict | str | object,
    human_stories: pd.DataFrame,
    *,
    generator_backend: str | None = None,
    generator_provider: str | None = None,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Prompt the generator for each (title, genre) pair and return responses."""

    generator_config = InferenceConfig.coerce(
        generator_spec,
        default_backend=generator_backend,
        default_provider=generator_provider,
        default_batch_size=batch_size,
    )

    stories = human_stories.copy()
    stories["title"] = stories["title"].astype(str)
    stories["genre"] = stories["genre"].astype(str)

    prompts = [
        llm_generator_prompt(title, genre)
        for title, genre in zip(stories["title"], stories["genre"])
    ]
    total = len(prompts)
    responses: list[str] = []

    if total == 0:
        raise ValueError("No prompts available. Check the human stories dataframe.")

    chunk_size = generator_config.batch_size
    try:
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            chunk = prompts[start:stop]
            chunk_responses = generator_config.run(chunk)
            responses.extend(chunk_responses)
            if show_progress:
                pct = (stop / total) * 100.0
                print(
                    f"Generated {stop}/{total} prompts ({pct:.1f}%)",
                    file=sys.stderr,
                )
    except KeyboardInterrupt:
        print("Generation interrupted; returning responses gathered so far.", file=sys.stderr)

    if len(responses) != total:
        responses.extend([""] * (total - len(responses)))

    parsed_stories = [extract_tagged_text(text, "story") for text in responses]
    ai_df = pd.DataFrame(
        {
            "ai_story_raw": responses,
            "ai_story": parsed_stories,
        }
    )
    return ai_df


def build_combined_dataset(
    generator_spec: InferenceConfig | dict | str | object,
    *,
    dataset_name: str = DEFAULT_DATASET,
    split: str = "train",
    output_path: str = DEFAULT_OUTPUT,
    combined_output_path: str = DEFAULT_COMBINED_OUTPUT,
    limit: int | None = None,
    generator_backend: str | None = None,
    generator_provider: str | None = None,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> dict[str, str]:
    """Create AI and combined evaluation CSVs."""

    dataset = load_dataset(dataset_name)
    human_df = pd.DataFrame(dataset[split])
    if limit is not None:
        human_df = human_df.head(limit)

    if "id" not in human_df:
        human_df = human_df.reset_index().rename(columns={"index": "id"})

    if "human_story" not in human_df:
        human_df["human_story"] = human_df["story"].astype(str).str.split("\n").str[0]

    generator_df = generate_ai_stories(
        generator_spec,
        human_df,
        generator_backend=generator_backend,
        generator_provider=generator_provider,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    ai_df = pd.concat(
        [
            human_df[["id", "title", "genre", "human_story"]].reset_index(drop=True),
            generator_df.reset_index(drop=True),
        ],
        axis=1,
    )
    ai_df.to_csv(output_path, index=False)

    combined_df = ai_df.merge(human_df, on="id", suffixes=("_gen", ""))
    combined_df.to_csv(combined_output_path, index=False)

    return {
        "ai_output": output_path,
        "combined_output": combined_output_path,
        "num_rows": str(len(ai_df)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AI evaluation datasets.")
    parser.add_argument(
        "generator",
        help="Generator specification (e.g. 'openai:gpt-4.1-mini' or path to local model).",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset identifier")
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="AI stories CSV output path")
    parser.add_argument(
        "--combined-output",
        default=DEFAULT_COMBINED_OUTPUT,
        help="Combined CSV output path",
    )
    parser.add_argument("--limit", type=int, help="Limit the number of rows processed")
    parser.add_argument("--backend", choices=["api", "local"], help="Force backend choice")
    parser.add_argument("--provider", help="Provider hint (e.g. openai, lambda, huggingface)")
    parser.add_argument("--batch-size", type=int, help="Override batch size for generation")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress updates during generation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    generator = args.generator

    summary = build_combined_dataset(
        generator,
        dataset_name=args.dataset,
        split=args.split,
        output_path=args.output,
        combined_output_path=args.combined_output,
        limit=args.limit,
        generator_backend=args.backend,
        generator_provider=args.provider,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
    )

    print("Generation complete:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
