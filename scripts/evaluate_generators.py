#!/usr/bin/env python3
"""
Command-line interface for evaluating LLM generator performance using GPT.

This script compares generated stories from two different training runs by:
1. Parsing training log directories to extract generated stories
2. Sampling balanced pairs of stories for comparison
3. Using GPT-4/5 to evaluate which stories are better
4. Providing statistical analysis of the results

Usage examples:
    # Basic comparison with 50 story pairs
    python evaluate_generators.py logs/run_A logs/run_B --num_pairs 50
    
    # Detailed comparison with custom labels and output directory
    python evaluate_generators.py logs/baseline logs/improved \\
        --label_a "Baseline Model" --label_b "Improved Model" \\
        --num_pairs 100 --output_dir results/baseline_vs_improved \\
        --gpt_model gpt-4o
        
    # Genre-matched sampling with creativity evaluation
    python evaluate_generators.py logs/run_1 logs/run_2 \\
        --strategy genre_matched --criteria creativity \\
        --delay 2.0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_gan.eval.eval_runner import run_evaluation, EvaluationConfig
from llm_gan.eval.gpt_evaluator import EvaluationCriteria
from llm_gan.eval.results_analyzer import analyze_evaluation_results


def setup_api_key():
    """Check and setup OpenAI API key."""
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set.")
        print("   Set it with: export OPENAI_API_KEY='your-api-key-here'")
        print("   Or create a .env file with: OPENAI_API_KEY=your-api-key-here")
        
        # Check for .env file
        env_file = Path('.env')
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv()
                if os.getenv('OPENAI_API_KEY'):
                    print("✅ Loaded API key from .env file")
                    return True
            except ImportError:
                print("   Install python-dotenv to use .env files: pip install python-dotenv")
        
        return False
    return True


def validate_log_directory(log_dir: str) -> bool:
    """Validate that a log directory exists and contains batch files."""
    path = Path(log_dir)
    if not path.exists():
        print(f"❌ Error: Log directory '{log_dir}' does not exist")
        return False
    
    batch_files = list(path.glob("batch_*.json"))
    if not batch_files:
        print(f"❌ Error: No batch log files found in '{log_dir}'")
        print("   Expected files like: batch_0_0.json, batch_0_1.json, etc.")
        return False
    
    print(f"✅ Found {len(batch_files)} batch files in '{log_dir}'")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM generator performance using GPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s logs/run_A logs/run_B --num_pairs 50
  %(prog)s logs/baseline logs/improved --label_a "Baseline" --label_b "Improved" --output_dir results/
  %(prog)s logs/run_1 logs/run_2 --strategy genre_matched --criteria creativity
        """
    )
    
    # Required arguments
    parser.add_argument('log_dir_a', help='Path to first training log directory')
    parser.add_argument('log_dir_b', help='Path to second training log directory')
    
    # Model labels
    parser.add_argument('--label_a', default='Model A', 
                       help='Label for first model (default: Model A)')
    parser.add_argument('--label_b', default='Model B',
                       help='Label for second model (default: Model B)')
    
    # Sampling options
    parser.add_argument('--num_pairs', type=int, default=100,
                       help='Number of story pairs to evaluate (default: 100)')
    parser.add_argument('--strategy', choices=['balanced', 'random', 'genre_matched', 'epoch_matched'],
                       default='balanced', help='Sampling strategy (default: balanced)')
    parser.add_argument('--min_length', type=int, default=50,
                       help='Minimum story length in characters (default: 50)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    # Evaluation options  
    parser.add_argument('--criteria', choices=[c.value for c in EvaluationCriteria],
                       default=EvaluationCriteria.OVERALL_QUALITY.value,
                       help='Evaluation criteria (default: overall_quality)')
    parser.add_argument('--gpt_model', default='gpt-4o',
                       help='GPT model to use (default: gpt-4o)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between API calls in seconds (default: 1.0)')
    
    # Output options
    parser.add_argument('--output_dir', help='Output directory for results')
    parser.add_argument('--no_filter', action='store_true',
                       help='Don\'t filter placeholder stories')
    
    # Analysis options
    parser.add_argument('--analyze_only', help='Skip evaluation and analyze existing results directory')
    parser.add_argument('--no_plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Handle analyze-only mode
    if args.analyze_only:
        print(f"Analyzing existing results in: {args.analyze_only}")
        try:
            report = analyze_evaluation_results(args.analyze_only)
            print("✅ Analysis complete!")
            print(f"📊 Report saved to: {Path(args.analyze_only) / 'comprehensive_report.json'}")
            return
        except Exception as e:
            print(f"❌ Error analyzing results: {e}")
            sys.exit(1)
    
    # Setup and validation
    print("🔍 Setting up evaluation...")
    
    # Check API key
    if not setup_api_key():
        print("❌ OpenAI API key required for evaluation")
        sys.exit(1)
    
    # Validate log directories
    if not validate_log_directory(args.log_dir_a):
        sys.exit(1)
    if not validate_log_directory(args.log_dir_b):
        sys.exit(1)
    
    # Create evaluation config
    criteria = EvaluationCriteria(args.criteria)
    
    config = EvaluationConfig(
        log_dir_a=args.log_dir_a,
        log_dir_b=args.log_dir_b,
        label_a=args.label_a,
        label_b=args.label_b,
        num_pairs=args.num_pairs,
        sampling_strategy=args.strategy,
        evaluation_criteria=criteria,
        gpt_model=args.gpt_model,
        min_story_length=args.min_length,
        filter_placeholders=not args.no_filter,
        delay_between_calls=args.delay,
        random_seed=args.seed,
        output_dir=args.output_dir
    )
    
    # Print configuration
    print(f"📋 Configuration:")
    print(f"   {args.label_a}: {args.log_dir_a}")
    print(f"   {args.label_b}: {args.log_dir_b}")
    print(f"   Story pairs: {args.num_pairs}")
    print(f"   Strategy: {args.strategy}")
    print(f"   Criteria: {args.criteria}")
    print(f"   GPT model: {args.gpt_model}")
    print(f"   API delay: {args.delay}s")
    
    try:
        # Run evaluation
        print(f"\n🚀 Starting evaluation...")
        results = run_evaluation(
            log_dir_a=args.log_dir_a,
            log_dir_b=args.log_dir_b,
            label_a=args.label_a,
            label_b=args.label_b,
            num_pairs=args.num_pairs,
            sampling_strategy=args.strategy,
            evaluation_criteria=criteria,
            gpt_model=args.gpt_model,
            min_story_length=args.min_length,
            filter_placeholders=not args.no_filter,
            delay_between_calls=args.delay,
            random_seed=args.seed,
            output_dir=args.output_dir
        )
        
        output_dir = results['output_dir']
        print(f"✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        
        # Run additional analysis
        print(f"\n📊 Running statistical analysis...")
        try:
            report = analyze_evaluation_results(output_dir)
            print("✅ Statistical analysis complete!")
            
            # Print key conclusions
            conclusions = report.get('conclusions', [])
            if conclusions:
                print(f"\n🎯 Key Conclusions:")
                for conclusion in conclusions:
                    print(f"   • {conclusion}")
            
        except Exception as e:
            print(f"⚠️  Warning: Statistical analysis failed: {e}")
            print("   Basic results are still available in the output directory")
        
        print(f"\n📈 View detailed results:")
        print(f"   Basic analysis: {output_dir}/analysis.json")
        print(f"   Full report: {output_dir}/comprehensive_report.json")
        print(f"   Story pairs: {output_dir}/story_pairs.json")
        print(f"   Raw evaluations: {output_dir}/evaluation_results.json")
        
    except KeyboardInterrupt:
        print(f"\n❌ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()