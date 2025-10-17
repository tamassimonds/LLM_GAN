#!/usr/bin/env python3
"""Test script for batch evaluation of story files."""

import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_gan.eval.log_parser import GeneratedStory
from llm_gan.eval.story_sampler import sample_story_pairs
from llm_gan.eval.gpt_evaluator import GPTEvaluator, EvaluationCriteria

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

def main():
    import sys
    
    # Parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: python test_batch_eval.py <batch_a.json> <batch_b.json> <num_pairs>")
        print("Example: python test_batch_eval.py batch_0_0.json batch_1_44.json 15")
        sys.exit(1)
    
    batch_a_path = sys.argv[1]
    batch_b_path = sys.argv[2]
    num_pairs = int(sys.argv[3])
    
    # Extract batch names for display
    batch_a_name = batch_a_path.replace('.json', '').replace('batch_', '')
    batch_b_name = batch_b_path.replace('.json', '').replace('batch_', '')
    
    print("🚀 Testing batch evaluation with GPT-5")
    print("======================================")
    print(f"🆚 Comparing: {batch_a_name} vs {batch_b_name}")
    
    print(f"📁 Loading {batch_a_path}...")
    stories_a = parse_batch_file(batch_a_path)
    print(f"   Found {len(stories_a)} stories")
    
    print(f"📁 Loading {batch_b_path}...")
    stories_b = parse_batch_file(batch_b_path)
    print(f"   Found {len(stories_b)} stories")
    
    # Sample pairs for evaluation
    num_pairs = min(len(stories_a), len(stories_b), num_pairs)
    print(f"📊 Sampling {num_pairs} story pairs...")
    
    pairs, stats = sample_story_pairs(
        stories_a, stories_b, 
        num_pairs=num_pairs, 
        strategy="random",
        random_seed=42
    )
    
    # Shuffle A/B assignment to remove positional bias
    import random
    random.seed(42)  # For reproducibility
    shuffled_pairs = []
    swap_mapping = {}  # Track which pairs were swapped
    
    for i, pair in enumerate(pairs):
        if random.random() < 0.5:
            # Keep original order (A=batch_a, B=batch_b)
            shuffled_pairs.append(pair)
            swap_mapping[pair.comparison_id] = False
        else:
            # Swap order (A=batch_b, B=batch_a)
            from llm_gan.eval.story_sampler import StoryPair
            swapped_pair = StoryPair(
                story_a=pair.story_b,
                story_b=pair.story_a,
                comparison_id=f"swapped_{pair.comparison_id}"
            )
            shuffled_pairs.append(swapped_pair)
            swap_mapping[swapped_pair.comparison_id] = True
    
    pairs = shuffled_pairs
    
    # Count how many were swapped
    swapped_count = sum(swap_mapping.values())
    print(f"   Swapped {swapped_count}/{len(pairs)} pairs to remove positional bias")
    
    print(f"   Sampled {len(pairs)} pairs")
    print(f"   Genre match rate: {stats.get('genre_match_rate', 0):.2%}")
    
    # Run batch evaluation with GPT-5
    print(f"🤖 Running batch evaluation with GPT-5...")
    evaluator = GPTEvaluator(model="gpt-5")
    
    results = evaluator.evaluate_pairs_batch(
        pairs,
        criteria=EvaluationCriteria.OVERALL_QUALITY
    )
    
    print(f"✅ Completed {len(results)} evaluations")
    
    # Analyze results accounting for swapped pairs
    batch_a_wins = 0
    batch_b_wins = 0
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
                batch_a_wins += 1  # A = batch_a
            else:
                batch_b_wins += 1  # A = batch_b (swapped)
        elif result.winner == 'B':
            if not was_swapped:
                batch_b_wins += 1  # B = batch_b
            else:
                batch_a_wins += 1  # B = batch_a (swapped)
    
    print(f"\n📈 Results Summary (accounting for swapping):")
    print(f"   {batch_a_name} wins: {batch_a_wins}")
    print(f"   {batch_b_name} wins: {batch_b_wins}")
    print(f"   Ties: {ties}")
    print(f"   Errors: {errors}")
    
    if len(results) > errors:
        valid_results = len(results) - errors
        print(f"   {batch_a_name} win rate: {batch_a_wins/valid_results:.1%}")
        print(f"   {batch_b_name} win rate: {batch_b_wins/valid_results:.1%}")
        
        avg_confidence = sum(r.confidence for r in results if r.winner != 'error') / valid_results
        print(f"   Average confidence: {avg_confidence:.3f}")
    
    # Save detailed results
    os.makedirs("logs/evals", exist_ok=True)
    output_file = f"logs/evals/eval_{batch_a_name}_vs_{batch_b_name}.json"
    with open(output_file, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    
    # Save summary
    if len(results) > errors:
        valid_results = len(results) - errors
        summary = {
            "evaluation_type": "batch_comparison",
            "batch_a_name": batch_a_name,
            "batch_b_name": batch_b_name,
            "comparison": f"{batch_a_name} vs {batch_b_name}",
            "total_pairs": len(results),
            "valid_pairs": valid_results,
            "errors": errors,
            "batch_a_wins": batch_a_wins,
            "batch_b_wins": batch_b_wins,
            "ties": ties,
            "batch_a_win_rate": batch_a_wins / valid_results if valid_results > 0 else 0,
            "batch_b_win_rate": batch_b_wins / valid_results if valid_results > 0 else 0,
            "average_confidence": sum(r.confidence for r in results if r.winner != 'error') / valid_results if valid_results > 0 else 0,
            "swapped_pairs": swapped_count,
            "model_used": "gpt-5"
        }
        
        summary_file = f"logs/evals/summary_{batch_a_name}_vs_{batch_b_name}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Detailed results saved to: {output_file}")
        print(f"📊 Summary saved to: {summary_file}")
    else:
        print(f"💾 Detailed results saved to: {output_file}")
    
    # Show first few results for verification
    print(f"\n🔍 Sample results:")
    for i, result in enumerate(results[:3]):
        print(f"   Pair {i+1}: Winner={result.winner}, Confidence={result.confidence:.3f}")
        print(f"            Reasoning: {result.reasoning[:100]}...")

if __name__ == "__main__":
    main()