# LLM GAN Generator Evaluation System

This system allows you to compare generated stories from different LLM GAN training runs using GPT-4/5 as an evaluator for stylistic quality assessment.

## Overview

The evaluation system consists of several components:

1. **Log Parser** - Extracts generated stories from training log directories
2. **Story Sampler** - Creates balanced pairs of stories for comparison  
3. **GPT Evaluator** - Uses OpenAI's GPT models to evaluate story quality
4. **Results Analyzer** - Provides statistical analysis and visualizations
5. **CLI Interface** - Easy-to-use command-line tool

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_eval.txt
```

### 2. Set Up OpenAI API Key

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

### 3. Run Evaluation

```bash
# Basic comparison
python evaluate_generators.py logs/training_logs_A logs/training_logs_B --num_pairs 50

# Detailed comparison
python evaluate_generators.py logs/baseline logs/improved \
    --label_a "Baseline Model" --label_b "Improved Model" \
    --num_pairs 100 --output_dir results/baseline_vs_improved \
    --gpt_model gpt-4o
```

## Usage Examples

### Basic Evaluation
```bash
python evaluate_generators.py logs/run_1 logs/run_2 --num_pairs 50
```

### Custom Labels and Output Directory
```bash
python evaluate_generators.py logs/baseline logs/improved \
    --label_a "Baseline" --label_b "Improved" \
    --output_dir results/comparison_1 \
    --num_pairs 100
```

### Genre-Matched Sampling
```bash
python evaluate_generators.py logs/run_A logs/run_B \
    --strategy genre_matched \
    --num_pairs 75
```

### Creativity-Focused Evaluation
```bash
python evaluate_generators.py logs/run_1 logs/run_2 \
    --criteria creativity \
    --gpt_model gpt-4 \
    --num_pairs 100
```

### Rate-Limited for API Limits
```bash
python evaluate_generators.py logs/run_A logs/run_B \
    --delay 2.0 \
    --num_pairs 200
```

### Analyze Existing Results
```bash
python evaluate_generators.py --analyze_only results/comparison_1
```

## Command Line Options

### Required Arguments
- `log_dir_a` - Path to first training log directory
- `log_dir_b` - Path to second training log directory

### Model Labels
- `--label_a` - Label for first model (default: "Model A")
- `--label_b` - Label for second model (default: "Model B")

### Sampling Options
- `--num_pairs` - Number of story pairs to evaluate (default: 100)
- `--strategy` - Sampling strategy: `balanced`, `random`, `genre_matched`, `epoch_matched` (default: balanced)
- `--min_length` - Minimum story length in characters (default: 50)
- `--seed` - Random seed for reproducibility

### Evaluation Options
- `--criteria` - Evaluation criteria: `overall_quality`, `creativity`, `coherence`, `style`, `engagement` (default: overall_quality)
- `--gpt_model` - GPT model to use (default: gpt-4o)
- `--delay` - Delay between API calls in seconds (default: 1.0)

### Output Options
- `--output_dir` - Custom output directory for results
- `--no_filter` - Don't filter placeholder stories
- `--no_plots` - Skip generating visualization plots

### Analysis Options
- `--analyze_only` - Skip evaluation and analyze existing results
- `--no_plots` - Skip generating plots

## Sampling Strategies

### Balanced (Default)
- Ensures balanced representation across genres
- Fills remaining pairs randomly if needed
- Good general-purpose strategy

### Random
- Completely random sampling
- Fastest option
- May not be representative

### Genre Matched
- Ensures both stories in each pair are from the same genre
- Good for genre-specific analysis
- Requires common genres between training runs

### Epoch Matched
- Pairs stories from similar training epochs
- Good for analyzing training progression
- Useful when comparing different training stages

## Evaluation Criteria

### Overall Quality (Default)
Comprehensive assessment of writing excellence including narrative structure, prose quality, and overall effectiveness.

### Creativity
Focus on originality, imagination, and creative elements in the storytelling.

### Coherence
Emphasis on narrative logic, consistency, and structural coherence.

### Style
Assessment of prose quality, writing style, and linguistic sophistication.

### Engagement
Evaluation of reader engagement and entertainment value.

## Output Files

The evaluation system generates several output files:

### Core Results
- `analysis.json` - Basic statistical analysis
- `evaluation_results.json` - Raw GPT evaluation results
- `story_pairs.json` - All story pairs used in evaluation
- `config.json` - Evaluation configuration

### Advanced Analysis
- `comprehensive_report.json` - Detailed statistical analysis
- `parsing_stats.json` - Statistics about log parsing
- `sampling_stats.json` - Statistics about story sampling

### Visualizations
- `win_distribution.png` - Pie chart of win/loss/tie distribution
- `confidence_distribution.png` - Histogram of evaluation confidence scores
- `genre_analysis.png` - Win rates by genre (if applicable)

## Statistical Analysis

The system provides comprehensive statistical analysis including:

- **Binomial Tests** - Tests if win rates significantly differ from chance
- **Confidence Intervals** - For win rate differences
- **Effect Size Analysis** - Cohen's h for practical significance
- **Confidence Pattern Analysis** - Patterns in evaluator confidence
- **Genre/Epoch Breakdowns** - Performance across different categories

## API Costs

Approximate costs for OpenAI API usage:

- **GPT-4o**: ~$0.03-0.06 per comparison (depending on story length)
- **GPT-4**: ~$0.15-0.30 per comparison
- **100 comparisons with GPT-4o**: ~$3-6
- **100 comparisons with GPT-4**: ~$15-30

Rate limiting with `--delay` helps manage costs and avoid hitting API limits.

## Troubleshooting

### Common Issues

1. **"No batch log files found"**
   - Ensure log directory contains `batch_*.json` files
   - Check that training actually generated logs

2. **"OpenAI API key required"**
   - Set `OPENAI_API_KEY` environment variable
   - Or create `.env` file with the key

3. **"No valid stories found"**
   - Lower `--min_length` threshold
   - Use `--no_filter` to include placeholder stories
   - Check that training logs contain generated stories

4. **API rate limiting errors**
   - Increase `--delay` parameter
   - Use a less expensive model like `gpt-3.5-turbo`

5. **Memory issues with large datasets**
   - Reduce `--num_pairs`
   - Filter stories by length or quality

### Log Directory Structure

Expected structure for training log directories:
```
logs/training_logs_20241201_120000/
├── batch_0_0.json
├── batch_0_1.json
├── batch_1_0.json
├── ...
└── epoch_0_summary.json
```

Each `batch_*.json` file should contain:
- `generated_stories` - List of generated story texts
- `titles` - List of story titles
- `genres` - List of story genres
- `epoch`, `batch_idx`, `step` - Training metadata

## Python API Usage

You can also use the evaluation system programmatically:

```python
from llm_gan.eval.eval_runner import run_evaluation
from llm_gan.eval.gpt_evaluator import EvaluationCriteria

# Run evaluation
results = run_evaluation(
    log_dir_a="logs/run_A",
    log_dir_b="logs/run_B", 
    num_pairs=50,
    label_a="Baseline",
    label_b="Improved",
    evaluation_criteria=EvaluationCriteria.CREATIVITY,
    gpt_model="gpt-4o"
)

# Analyze results
from llm_gan.eval.results_analyzer import analyze_evaluation_results
report = analyze_evaluation_results(results['output_dir'])
```

## Contributing

To extend the evaluation system:

1. **Add new evaluation criteria** - Modify `EvaluationCriteria` enum and update prompts
2. **Add new sampling strategies** - Extend `StorySampler` class
3. **Add new statistical tests** - Extend `ResultsAnalyzer` class
4. **Add new visualizations** - Modify `generate_visualizations()` method

## License

This evaluation system is part of the LLM GAN project and follows the same license terms.