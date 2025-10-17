#!/usr/bin/env python3
"""Benchmark generated stories against human stories with GPT-5."""

import json
import sys
import os
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_gan.eval.log_parser import GeneratedStory
from llm_gan.eval.story_sampler import sample_story_pairs, StoryPair
from llm_gan.eval.gpt_evaluator import GPTEvaluator, EvaluationCriteria

def load_human_stories(csv_path, num_stories=None):
    """Load human stories from the training CSV file."""
    import pandas as pd
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   Loaded CSV with {len(df)} total stories")
        
        # Sample random stories if limit specified
        if num_stories and num_stories < len(df):
            df = df.sample(n=num_stories, random_state=42)
            print(f"   Sampled {num_stories} random stories")
        
        human_stories = []
        for idx, row in df.iterrows():
            # Create GeneratedStory objects for compatibility
            story_obj = GeneratedStory(
                story=row['human_story'],
                title=row['title'],
                genre=row['genre'],
                epoch=-1,  # Mark as human
                batch_idx=-1,
                step=-1,
                story_idx=idx,
                generator_reward=1.0,  # Human stories get perfect reward
                judge_accuracy=1.0,
                story_length=len(row['human_story'])
            )
            human_stories.append(story_obj)
        
        return human_stories
        
    except Exception as e:
        print(f"Error loading human stories: {e}")
        return []

def parse_batch_file(filepath):
    """Parse a batch file and extract stories."""
    with open(filepath, 'r') as f:
        batch_data = json.load(f)
    
    stories = []
    epoch = batch_data.get('epoch', 0)
    batch_idx = batch_data.get('batch_idx', 0)
    step = batch_data.get('step', 0)
    
    titles = batch_data.get('titles', [])
    genres = batch_data.get('genres', [])
    generated_stories = batch_data.get('generated_stories', [])
    generator_rewards = batch_data.get('generator_rewards', [])
    
    for i, (title, genre, story, reward) in enumerate(zip(titles, genres, generated_stories, generator_rewards)):
        story_obj = GeneratedStory(
            story=story,
            title=title,
            genre=genre,
            epoch=epoch,
            batch_idx=batch_idx,
            step=step,
            story_idx=i,
            generator_reward=reward,
            judge_accuracy=0.0,  # Not needed for evaluation
            story_length=len(story)
        )
        stories.append(story_obj)
    
    return stories

def normalize_story_length(story_a, story_b, target_length=None):
    """Normalize two stories to the same length."""
    if target_length is None:
        # Use the shorter of the two stories
        target_length = min(len(story_a.story), len(story_b.story))
    
    # Truncate both stories to target length
    normalized_a = GeneratedStory(
        story=story_a.story[:target_length],
        title=story_a.title,
        genre=story_a.genre,
        epoch=story_a.epoch,
        batch_idx=story_a.batch_idx,
        step=story_a.step,
        story_idx=story_a.story_idx,
        generator_reward=story_a.generator_reward,
        judge_accuracy=story_a.judge_accuracy,
        story_length=target_length
    )
    
    normalized_b = GeneratedStory(
        story=story_b.story[:target_length],
        title=story_b.title,
        genre=story_b.genre,
        epoch=story_b.epoch,
        batch_idx=story_b.batch_idx,
        step=story_b.step,
        story_idx=story_b.story_idx,
        generator_reward=story_b.generator_reward,
        judge_accuracy=story_b.judge_accuracy,
        story_length=target_length
    )
    
    return normalized_a, normalized_b

def main():
    # Parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: python benchmark_vs_human.py <batch.json> <human_stories.csv> <num_pairs>")
        print("Example: python benchmark_vs_human.py batch_0_0.json data/stories.csv 20")
        sys.exit(1)
    
    batch_path = sys.argv[1]
    human_csv_path = sys.argv[2]
    num_pairs = int(sys.argv[3])
    
    # Extract batch name for display
    batch_name = batch_path.replace('.json', '').replace('batch_', '')
    
    print("🚀 Benchmarking generated stories vs human stories with GPT-5")
    print("============================================================")
    print(f"🆚 Comparing: {batch_name} vs Human Stories")
    
    # Load generated stories
    print(f"📁 Loading generated stories from {batch_path}...")
    generated_stories = parse_batch_file(batch_path)
    print(f"   Found {len(generated_stories)} generated stories")
    
    # Load human stories
    print(f"📁 Loading human stories from {human_csv_path}...")
    human_stories = load_human_stories(human_csv_path, num_stories=num_pairs*2)  # Load extra for better sampling
    print(f"   Found {len(human_stories)} human stories")
    
    if len(human_stories) == 0:
        print("❌ No human stories loaded. Check your CSV file path.")
        sys.exit(1)
    
    # Sample pairs for evaluation
    actual_pairs = min(len(generated_stories), len(human_stories), num_pairs)
    print(f"📊 Creating {actual_pairs} normalized story pairs...")
    
    # Create pairs manually to ensure length normalization
    random.seed(42)
    pairs = []
    used_generated = set()
    used_human = set()
    
    for i in range(actual_pairs):
        # Sample unused stories
        available_generated = [s for j, s in enumerate(generated_stories) if j not in used_generated]
        available_human = [s for j, s in enumerate(human_stories) if j not in used_human]
        
        if not available_generated or not available_human:
            break
            
        gen_story = random.choice(available_generated)
        human_story = random.choice(available_human)
        
        # Normalize lengths
        norm_gen, norm_human = normalize_story_length(gen_story, human_story)
        
        # Track usage
        used_generated.add(generated_stories.index(gen_story))
        used_human.add(human_stories.index(human_story))
        
        # Create pair
        pair = StoryPair(
            story_a=norm_gen,
            story_b=norm_human,
            comparison_id=f"gen_vs_human_{i}"
        )
        pairs.append(pair)
    
    print(f"   Created {len(pairs)} normalized pairs")
    
    # Shuffle A/B assignment to remove positional bias
    random.seed(42)
    shuffled_pairs = []
    swap_mapping = {}
    
    for i, pair in enumerate(pairs):
        if random.random() < 0.5:
            # Keep original order (A=generated, B=human)
            shuffled_pairs.append(pair)
            swap_mapping[pair.comparison_id] = False
        else:
            # Swap order (A=human, B=generated)
            swapped_pair = StoryPair(
                story_a=pair.story_b,
                story_b=pair.story_a,
                comparison_id=f"swapped_{pair.comparison_id}"
            )
            shuffled_pairs.append(swapped_pair)
            swap_mapping[swapped_pair.comparison_id] = True
    
    pairs = shuffled_pairs
    swapped_count = sum(swap_mapping.values())
    print(f"   Swapped {swapped_count}/{len(pairs)} pairs to remove positional bias")
    
    # Calculate average story length
    avg_length = sum(len(pair.story_a.story) for pair in pairs) / len(pairs)
    print(f"   Average normalized story length: {avg_length:.0f} characters")
    
    # Run GPT-5 evaluation
    print(f"🤖 Running batch evaluation with GPT-5...")
    evaluator = GPTEvaluator(model="gpt-5")
    
    results = evaluator.evaluate_pairs_batch(
        pairs,
        criteria=EvaluationCriteria.OVERALL_QUALITY
    )
    
    print(f"✅ Completed {len(results)} evaluations")
    
    # Analyze results accounting for swapped pairs
    generated_wins = 0
    human_wins = 0
    ties = 0
    errors = 0
    
    for result in results:
        pair_id = result.pair_id
        was_swapped = swap_mapping.get(pair_id, False)
        
        if result.winner == 'error':
            errors += 1
        elif result.winner == 'TIE':
            ties += 1
        elif result.winner == 'A':
            if not was_swapped:
                generated_wins += 1  # A = generated
            else:
                human_wins += 1  # A = human (swapped)
        elif result.winner == 'B':
            if not was_swapped:
                human_wins += 1  # B = human
            else:
                generated_wins += 1  # B = generated (swapped)
    
    print(f"\n📈 Benchmark Results (accounting for swapping):")
    print(f"   {batch_name} (generated) wins: {generated_wins}")
    print(f"   Human stories wins: {human_wins}")
    print(f"   Ties: {ties}")
    print(f"   Errors: {errors}")
    
    if len(results) > errors:
        valid_results = len(results) - errors
        print(f"   {batch_name} win rate: {generated_wins/valid_results:.1%}")
        print(f"   Human win rate: {human_wins/valid_results:.1%}")
        
        avg_confidence = sum(r.confidence for r in results if r.winner != 'error') / valid_results
        print(f"   Average confidence: {avg_confidence:.3f}")
        
        # Determine winner
        if generated_wins > human_wins:
            margin = (generated_wins - human_wins) / valid_results
            print(f"\n🤖 Generated stories outperform humans by {margin:.1%}")
        elif human_wins > generated_wins:
            margin = (human_wins - generated_wins) / valid_results
            print(f"\n👥 Human stories outperform generated by {margin:.1%}")
        else:
            print(f"\n🤝 Generated and human stories perform equally")
    
    # Save detailed results
    os.makedirs("logs/evals", exist_ok=True)
    output_file = f"logs/evals/benchmark_{batch_name}_vs_human.json"
    with open(output_file, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    
    # Save summary
    if len(results) > errors:
        valid_results = len(results) - errors
        summary = {
            "evaluation_type": "human_benchmark",
            "batch_name": batch_name,
            "comparison": f"{batch_name} vs Human Stories",
            "total_pairs": len(results),
            "valid_pairs": valid_results,
            "errors": errors,
            "generated_wins": generated_wins,
            "human_wins": human_wins,
            "ties": ties,
            "generated_win_rate": generated_wins / valid_results if valid_results > 0 else 0,
            "human_win_rate": human_wins / valid_results if valid_results > 0 else 0,
            "average_confidence": sum(r.confidence for r in results if r.winner != 'error') / valid_results if valid_results > 0 else 0,
            "average_story_length": avg_length,
            "swapped_pairs": swapped_count,
            "model_used": "gpt-5"
        }
        
        summary_file = f"logs/evals/summary_{batch_name}_vs_human.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Detailed results saved to: {output_file}")
        print(f"📊 Summary saved to: {summary_file}")
    else:
        print(f"💾 Detailed results saved to: {output_file}")
    
    # Show sample results
    print(f"\n🔍 Sample evaluations:")
    for i, result in enumerate(results[:3]):
        pair_id = result.pair_id
        was_swapped = swap_mapping.get(pair_id, False)
        
        if was_swapped:
            a_type, b_type = "Human", "Generated"
        else:
            a_type, b_type = "Generated", "Human"
            
        print(f"   Pair {i+1}: {a_type} vs {b_type}")
        print(f"           Winner: {result.winner}, Confidence: {result.confidence:.3f}")
        print(f"           Reasoning: {result.reasoning[:100]}...")

if __name__ == "__main__":
    main()