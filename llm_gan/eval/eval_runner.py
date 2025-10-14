"""Main evaluation runner for orchestrating story comparisons."""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .log_parser import parse_log_directory, GeneratedStory
from .story_sampler import sample_story_pairs, StoryPair
from .gpt_evaluator import evaluate_story_pairs, EvaluationResult, EvaluationCriteria


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""
    log_dir_a: str
    log_dir_b: str
    label_a: str = "Model A"
    label_b: str = "Model B"
    num_pairs: int = 100
    sampling_strategy: str = "balanced"
    evaluation_criteria: EvaluationCriteria = EvaluationCriteria.OVERALL_QUALITY
    gpt_model: str = "gpt-4o"
    min_story_length: int = 50
    filter_placeholders: bool = True
    delay_between_calls: float = 1.0
    random_seed: Optional[int] = None
    output_dir: Optional[str] = None


class EvaluationRunner:
    """Main orchestrator for story evaluation comparisons."""
    
    def __init__(self, config: EvaluationConfig):
        """Initialize with evaluation configuration."""
        self.config = config
        
        # Set up output directory
        if config.output_dir:
            self.output_dir = Path(config.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"eval_results/comparison_{timestamp}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(self._config_to_dict(), f, indent=2)
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation pipeline."""
        
        print(f"Starting evaluation: {self.config.label_a} vs {self.config.label_b}")
        print(f"Output directory: {self.output_dir}")
        
        # Step 1: Parse log directories
        print("\nStep 1: Parsing training logs...")
        stories_a, stats_a = parse_log_directory(
            self.config.log_dir_a,
            self.config.min_story_length,
            self.config.filter_placeholders
        )
        
        stories_b, stats_b = parse_log_directory(
            self.config.log_dir_b,
            self.config.min_story_length,
            self.config.filter_placeholders
        )
        
        print(f"  {self.config.label_a}: {len(stories_a)} stories")
        print(f"  {self.config.label_b}: {len(stories_b)} stories")
        
        if len(stories_a) == 0 or len(stories_b) == 0:
            raise ValueError("One or both log directories contain no valid stories")
        
        # Save parsing stats
        with open(self.output_dir / "parsing_stats.json", 'w') as f:
            json.dump({
                'stories_a_stats': stats_a,
                'stories_b_stats': stats_b
            }, f, indent=2)
        
        # Step 2: Sample story pairs
        print(f"\nStep 2: Sampling {self.config.num_pairs} story pairs...")
        pairs, sampling_stats = sample_story_pairs(
            stories_a, 
            stories_b,
            self.config.num_pairs,
            self.config.sampling_strategy,
            self.config.random_seed
        )
        
        print(f"  Sampled {len(pairs)} pairs")
        print(f"  Genre match rate: {sampling_stats.get('genre_match_rate', 0):.2%}")
        
        # Save pairs and sampling stats
        self._save_story_pairs(pairs)
        with open(self.output_dir / "sampling_stats.json", 'w') as f:
            json.dump(sampling_stats, f, indent=2)
        
        # Step 3: Run GPT evaluations
        print(f"\nStep 3: Running GPT evaluations with {self.config.gpt_model}...")
        print(f"  Criteria: {self.config.evaluation_criteria.value}")
        print(f"  Rate limit: {self.config.delay_between_calls}s between calls")
        
        try:
            evaluation_results = evaluate_story_pairs(
                pairs,
                model=self.config.gpt_model,
                criteria=self.config.evaluation_criteria,
                delay_between_calls=self.config.delay_between_calls
            )
        except Exception as e:
            print(f"Error during GPT evaluation: {e}")
            print("Make sure you have set your OPENAI_API_KEY environment variable")
            raise
        
        print(f"  Completed {len(evaluation_results)} evaluations")
        
        # Save evaluation results
        self._save_evaluation_results(evaluation_results)
        
        # Step 4: Analyze results
        print("\nStep 4: Analyzing results...")
        analysis = self._analyze_results(evaluation_results, pairs, stories_a, stories_b)
        
        # Save analysis
        with open(self.output_dir / "analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Print summary
        self._print_summary(analysis)
        
        return {
            'config': self._config_to_dict(),
            'parsing_stats': {'stories_a': stats_a, 'stories_b': stats_b},
            'sampling_stats': sampling_stats,
            'evaluation_results': [r.to_dict() for r in evaluation_results],
            'analysis': analysis,
            'output_dir': str(self.output_dir)
        }
    
    def _save_story_pairs(self, pairs: List[StoryPair]):
        """Save story pairs to JSON file."""
        pairs_data = []
        for pair in pairs:
            pairs_data.append({
                'comparison_id': pair.comparison_id,
                'story_a': {
                    'story': pair.story_a.story,
                    'title': pair.story_a.title,
                    'genre': pair.story_a.genre,
                    'epoch': pair.story_a.epoch,
                    'step': pair.story_a.step,
                    'reward': pair.story_a.generator_reward
                },
                'story_b': {
                    'story': pair.story_b.story,
                    'title': pair.story_b.title,
                    'genre': pair.story_b.genre,
                    'epoch': pair.story_b.epoch,
                    'step': pair.story_b.step,
                    'reward': pair.story_b.generator_reward
                }
            })
        
        with open(self.output_dir / "story_pairs.json", 'w') as f:
            json.dump(pairs_data, f, indent=2)
    
    def _save_evaluation_results(self, results: List[EvaluationResult]):
        """Save evaluation results to JSON file."""
        results_data = [r.to_dict() for r in results]
        
        with open(self.output_dir / "evaluation_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def _analyze_results(self, results: List[EvaluationResult], 
                        pairs: List[StoryPair],
                        stories_a: List[GeneratedStory], 
                        stories_b: List[GeneratedStory]) -> Dict[str, Any]:
        """Analyze evaluation results and compute statistics."""
        
        # Basic win/loss counts
        wins_a = sum(1 for r in results if r.winner == 'A')
        wins_b = sum(1 for r in results if r.winner == 'B') 
        ties = sum(1 for r in results if r.winner == 'TIE')
        errors = sum(1 for r in results if r.winner == 'error')
        
        total_valid = len(results) - errors
        
        # Win rates
        win_rate_a = wins_a / total_valid if total_valid > 0 else 0
        win_rate_b = wins_b / total_valid if total_valid > 0 else 0
        tie_rate = ties / total_valid if total_valid > 0 else 0
        
        # Confidence statistics
        valid_results = [r for r in results if r.winner != 'error']
        avg_confidence = sum(r.confidence for r in valid_results) / len(valid_results) if valid_results else 0
        
        # Genre-based analysis
        genre_analysis = self._analyze_by_genre(results, pairs)
        
        # Epoch-based analysis (if applicable)
        epoch_analysis = self._analyze_by_epoch(results, pairs)
        
        analysis = {
            'summary': {
                'total_evaluations': len(results),
                'valid_evaluations': total_valid,
                'errors': errors,
                'wins_a': wins_a,
                'wins_b': wins_b,
                'ties': ties,
                'win_rate_a': win_rate_a,
                'win_rate_b': win_rate_b,
                'tie_rate': tie_rate,
                'avg_confidence': avg_confidence
            },
            'genre_analysis': genre_analysis,
            'epoch_analysis': epoch_analysis,
            'confidence_distribution': {
                'high_confidence': sum(1 for r in valid_results if r.confidence >= 0.8),
                'medium_confidence': sum(1 for r in valid_results if 0.5 <= r.confidence < 0.8),
                'low_confidence': sum(1 for r in valid_results if r.confidence < 0.5)
            }
        }
        
        return analysis
    
    def _analyze_by_genre(self, results: List[EvaluationResult], 
                         pairs: List[StoryPair]) -> Dict[str, Any]:
        """Analyze results broken down by genre."""
        genre_stats = {}
        
        for result, pair in zip(results, pairs):
            if result.winner == 'error':
                continue
                
            genre = pair.story_a.genre  # Should match story_b.genre for genre-matched sampling
            
            if genre not in genre_stats:
                genre_stats[genre] = {'wins_a': 0, 'wins_b': 0, 'ties': 0, 'total': 0}
            
            genre_stats[genre]['total'] += 1
            if result.winner == 'A':
                genre_stats[genre]['wins_a'] += 1
            elif result.winner == 'B':
                genre_stats[genre]['wins_b'] += 1
            else:
                genre_stats[genre]['ties'] += 1
        
        # Calculate win rates by genre
        for genre in genre_stats:
            total = genre_stats[genre]['total']
            if total > 0:
                genre_stats[genre]['win_rate_a'] = genre_stats[genre]['wins_a'] / total
                genre_stats[genre]['win_rate_b'] = genre_stats[genre]['wins_b'] / total
                genre_stats[genre]['tie_rate'] = genre_stats[genre]['ties'] / total
        
        return genre_stats
    
    def _analyze_by_epoch(self, results: List[EvaluationResult], 
                         pairs: List[StoryPair]) -> Dict[str, Any]:
        """Analyze results broken down by training epoch."""
        epoch_stats = {}
        
        for result, pair in zip(results, pairs):
            if result.winner == 'error':
                continue
            
            # Use average epoch for the pair
            avg_epoch = (pair.story_a.epoch + pair.story_b.epoch) / 2
            epoch_bucket = f"epoch_{int(avg_epoch)}"
            
            if epoch_bucket not in epoch_stats:
                epoch_stats[epoch_bucket] = {'wins_a': 0, 'wins_b': 0, 'ties': 0, 'total': 0}
            
            epoch_stats[epoch_bucket]['total'] += 1
            if result.winner == 'A':
                epoch_stats[epoch_bucket]['wins_a'] += 1
            elif result.winner == 'B':
                epoch_stats[epoch_bucket]['wins_b'] += 1
            else:
                epoch_stats[epoch_bucket]['ties'] += 1
        
        # Calculate win rates by epoch
        for epoch in epoch_stats:
            total = epoch_stats[epoch]['total']
            if total > 0:
                epoch_stats[epoch]['win_rate_a'] = epoch_stats[epoch]['wins_a'] / total
                epoch_stats[epoch]['win_rate_b'] = epoch_stats[epoch]['wins_b'] / total
                epoch_stats[epoch]['tie_rate'] = epoch_stats[epoch]['ties'] / total
        
        return epoch_stats
    
    def _print_summary(self, analysis: Dict[str, Any]):
        """Print a summary of the evaluation results."""
        summary = analysis['summary']
        
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total evaluations: {summary['total_evaluations']}")
        print(f"Valid evaluations: {summary['valid_evaluations']}")
        print(f"Errors: {summary['errors']}")
        print()
        print(f"{self.config.label_a} wins: {summary['wins_a']} ({summary['win_rate_a']:.1%})")
        print(f"{self.config.label_b} wins: {summary['wins_b']} ({summary['win_rate_b']:.1%})")
        print(f"Ties: {summary['ties']} ({summary['tie_rate']:.1%})")
        print()
        print(f"Average confidence: {summary['avg_confidence']:.3f}")
        
        # Determine winner
        if summary['win_rate_a'] > summary['win_rate_b']:
            margin = summary['win_rate_a'] - summary['win_rate_b']
            print(f"\n🏆 {self.config.label_a} wins by {margin:.1%} margin")
        elif summary['win_rate_b'] > summary['win_rate_a']:
            margin = summary['win_rate_b'] - summary['win_rate_a']
            print(f"\n🏆 {self.config.label_b} wins by {margin:.1%} margin")
        else:
            print(f"\n🤝 Tie between {self.config.label_a} and {self.config.label_b}")
        
        print(f"\nDetailed results saved to: {self.output_dir}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return {
            'log_dir_a': self.config.log_dir_a,
            'log_dir_b': self.config.log_dir_b,
            'label_a': self.config.label_a,
            'label_b': self.config.label_b,
            'num_pairs': self.config.num_pairs,
            'sampling_strategy': self.config.sampling_strategy,
            'evaluation_criteria': self.config.evaluation_criteria.value,
            'gpt_model': self.config.gpt_model,
            'min_story_length': self.config.min_story_length,
            'filter_placeholders': self.config.filter_placeholders,
            'delay_between_calls': self.config.delay_between_calls,
            'random_seed': self.config.random_seed,
            'output_dir': str(self.output_dir)
        }


def run_evaluation(log_dir_a: str, log_dir_b: str, 
                  num_pairs: int = 100,
                  label_a: str = "Model A",
                  label_b: str = "Model B",
                  **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run a complete evaluation.
    
    Args:
        log_dir_a: Path to first training log directory
        log_dir_b: Path to second training log directory  
        num_pairs: Number of story pairs to evaluate
        label_a: Label for first model
        label_b: Label for second model
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary with complete evaluation results
    """
    config = EvaluationConfig(
        log_dir_a=log_dir_a,
        log_dir_b=log_dir_b,
        label_a=label_a,
        label_b=label_b,
        num_pairs=num_pairs,
        **kwargs
    )
    
    runner = EvaluationRunner(config)
    return runner.run_evaluation()