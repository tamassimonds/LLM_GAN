"""Benchmark evaluation for judge model performance tracking."""

import json
import random
from typing import Dict, Any, List, Tuple
import numpy as np

from llm_gan.train.evaluation import assess_judge_with_outputs


def load_benchmark_data(benchmark_path: str = "data/benchmark_eval.json") -> Dict[str, List[str]]:
    """Load the static benchmark evaluation dataset."""
    with open(benchmark_path, 'r') as f:
        return json.load(f)


def run_benchmark_evaluation(judge_model, tokenizer, benchmark_path: str = "data/benchmark_eval.json", 
                           max_judge_tokens: int = 512, max_story_length: int = 500) -> Dict[str, Any]:
    """Run benchmark evaluation on the judge model.
    
    Returns:
        Dict containing benchmark results including accuracy, detailed outputs, etc.
    """
    # Load benchmark data
    benchmark_data = load_benchmark_data(benchmark_path)
    
    titles = benchmark_data["titles"]
    genres = benchmark_data["genres"] 
    ai_stories = benchmark_data["ai_stories"]
    human_stories = benchmark_data["human_stories"]
    
    # Clip stories to equal length for fair comparison
    clipped_human_stories = [story[:max_story_length] for story in human_stories]
    clipped_ai_stories = [story[:max_story_length] for story in ai_stories]
    
    print(f"Running benchmark evaluation on {len(titles)} story pairs...")
    
    # Run judge evaluation
    judge_correct, judge_outputs = assess_judge_with_outputs(
        titles, genres, clipped_human_stories, clipped_ai_stories, 
        judge_model, tokenizer, max_judge_tokens
    )
    
    # Calculate metrics
    accuracy = sum(judge_correct) / len(judge_correct) if judge_correct else 0.0
    total_pairs = len(judge_correct)
    correct_count = sum(judge_correct)
    failed_count = sum(1 for parsed in judge_outputs['parsed_outputs'] if parsed == -1)
    
    # Calculate accuracy by genre
    genre_accuracy = {}
    for i, genre in enumerate(genres):
        if genre not in genre_accuracy:
            genre_accuracy[genre] = {'correct': 0, 'total': 0}
        genre_accuracy[genre]['total'] += 1
        if judge_correct[i]:
            genre_accuracy[genre]['correct'] += 1
    
    # Convert to percentages
    for genre in genre_accuracy:
        total = genre_accuracy[genre]['total']
        correct = genre_accuracy[genre]['correct'] 
        genre_accuracy[genre]['accuracy'] = correct / total if total > 0 else 0.0
    
    benchmark_results = {
        'overall_accuracy': accuracy,
        'total_pairs': total_pairs,
        'correct_count': correct_count,
        'failed_count': failed_count,  # Number of unparseable responses
        'failed_rate': failed_count / total_pairs if total_pairs > 0 else 0.0,
        'genre_accuracy': genre_accuracy,
        'detailed_results': {
            'titles': titles,
            'genres': genres,
            'judge_correct': judge_correct,
            'targets': judge_outputs['targets'],
            'parsed_outputs': judge_outputs['parsed_outputs'],
            'judge_responses': judge_outputs['judge_outputs']
        }
    }
    
    print(f"Benchmark Results:")
    print(f"  Overall Accuracy: {accuracy:.4f}")
    print(f"  Correct: {correct_count}/{total_pairs}")
    print(f"  Failed to parse: {failed_count}/{total_pairs} ({failed_count/total_pairs*100:.1f}%)")
    
    return benchmark_results


def save_benchmark_results(results: Dict[str, Any], save_path: str):
    """Save benchmark results to file."""
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results saved to: {save_path}")


def compare_benchmark_results(current_results: Dict[str, Any], 
                            previous_results_path: str = None) -> Dict[str, float]:
    """Compare current benchmark results with previous results.
    
    Returns:
        Dict containing improvement metrics.
    """
    if not previous_results_path:
        return {"improvement": 0.0, "message": "No previous results to compare"}
    
    try:
        with open(previous_results_path, 'r') as f:
            previous_results = json.load(f)
        
        current_acc = current_results['overall_accuracy']
        previous_acc = previous_results['overall_accuracy']
        improvement = current_acc - previous_acc
        
        current_failed_rate = current_results['failed_rate']
        previous_failed_rate = previous_results['failed_rate']
        failed_improvement = previous_failed_rate - current_failed_rate  # Lower is better
        
        return {
            "accuracy_improvement": improvement,
            "failed_rate_improvement": failed_improvement,
            "current_accuracy": current_acc,
            "previous_accuracy": previous_acc,
            "message": f"Accuracy {'improved' if improvement > 0 else 'decreased'} by {abs(improvement):.4f}"
        }
    
    except FileNotFoundError:
        return {"improvement": 0.0, "message": f"Previous results file not found: {previous_results_path}"}
    except Exception as e:
        return {"improvement": 0.0, "message": f"Error comparing results: {e}"}