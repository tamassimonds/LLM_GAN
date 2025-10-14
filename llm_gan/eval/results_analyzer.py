"""Advanced statistical analysis and visualization of evaluation results."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from .gpt_evaluator import EvaluationResult


@dataclass
class StatisticalTest:
    """Results of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_level: float
    interpretation: str


class ResultsAnalyzer:
    """Advanced analysis of evaluation results with statistical testing."""
    
    def __init__(self, results_dir: str):
        """Initialize with path to evaluation results directory."""
        self.results_dir = Path(results_dir)
        
        # Load data
        self.config = self._load_json("config.json")
        self.evaluation_results = self._load_json("evaluation_results.json")
        self.analysis = self._load_json("analysis.json")
        self.sampling_stats = self._load_json("sampling_stats.json")
        
    def _load_json(self, filename: str) -> Dict[str, Any]:
        """Load JSON file from results directory."""
        file_path = self.results_dir / filename
        if not file_path.exists():
            return {}
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def run_statistical_tests(self, confidence_level: float = 0.95) -> List[StatisticalTest]:
        """Run statistical tests on the evaluation results."""
        tests = []
        
        # Extract wins for each model
        results = [r for r in self.evaluation_results if r.get('winner') != 'error']
        
        if not results:
            return tests
        
        wins_a = sum(1 for r in results if r['winner'] == 'A')
        wins_b = sum(1 for r in results if r['winner'] == 'B')
        total_decisive = wins_a + wins_b  # Exclude ties
        
        if total_decisive == 0:
            return tests
        
        # Binomial test (is there a significant preference?)
        if total_decisive >= 10:  # Minimum sample size for meaningful test
            binom_test = self._binomial_test(wins_a, total_decisive, confidence_level)
            tests.append(binom_test)
        
        # Confidence interval for win rate difference
        if total_decisive >= 20:
            ci_test = self._confidence_interval_test(wins_a, wins_b, confidence_level)
            tests.append(ci_test)
        
        # Effect size (Cohen's h for proportions)
        effect_size_test = self._effect_size_test(wins_a, wins_b, total_decisive)
        tests.append(effect_size_test)
        
        return tests
    
    def _binomial_test(self, wins_a: int, total: int, confidence_level: float) -> StatisticalTest:
        """Test if win rate significantly differs from 50%."""
        # Two-tailed binomial test
        p_value = stats.binom_test(wins_a, total, p=0.5, alternative='two-sided')
        alpha = 1 - confidence_level
        significant = p_value < alpha
        
        win_rate = wins_a / total
        
        if significant:
            if win_rate > 0.5:
                interpretation = f"Model A wins significantly more than expected by chance (p={p_value:.4f})"
            else:
                interpretation = f"Model B wins significantly more than expected by chance (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference from random chance (p={p_value:.4f})"
        
        return StatisticalTest(
            test_name="Binomial Test",
            statistic=win_rate,
            p_value=p_value,
            significant=significant,
            confidence_level=confidence_level,
            interpretation=interpretation
        )
    
    def _confidence_interval_test(self, wins_a: int, wins_b: int, confidence_level: float) -> StatisticalTest:
        """Calculate confidence interval for win rate difference."""
        total = wins_a + wins_b
        p_a = wins_a / total
        p_b = wins_b / total
        diff = p_a - p_b
        
        # Standard error for difference in proportions
        se_diff = np.sqrt((p_a * (1 - p_a) + p_b * (1 - p_b)) / total)
        
        # Z-score for confidence level
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        # Confidence interval
        margin_error = z_score * se_diff
        ci_lower = diff - margin_error
        ci_upper = diff + margin_error
        
        # Check if CI includes 0 (no difference)
        significant = not (ci_lower <= 0 <= ci_upper)
        
        interpretation = f"Win rate difference: {diff:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] at {confidence_level:.0%} confidence"
        if significant:
            interpretation += " (significant difference)"
        else:
            interpretation += " (no significant difference)"
        
        return StatisticalTest(
            test_name="Confidence Interval",
            statistic=diff,
            p_value=1 - confidence_level if significant else confidence_level,
            significant=significant,
            confidence_level=confidence_level,
            interpretation=interpretation
        )
    
    def _effect_size_test(self, wins_a: int, wins_b: int, total: int) -> StatisticalTest:
        """Calculate Cohen's h effect size for proportion difference."""
        p_a = wins_a / total
        p_b = wins_b / total
        
        # Cohen's h for comparing two proportions
        h = 2 * (np.arcsin(np.sqrt(p_a)) - np.arcsin(np.sqrt(p_b)))
        
        # Interpret effect size
        if abs(h) < 0.2:
            effect = "negligible"
        elif abs(h) < 0.5:
            effect = "small"
        elif abs(h) < 0.8:
            effect = "medium"
        else:
            effect = "large"
        
        interpretation = f"Cohen's h = {h:.3f} ({effect} effect size)"
        
        return StatisticalTest(
            test_name="Effect Size (Cohen's h)",
            statistic=h,
            p_value=0.0,  # Not applicable for effect size
            significant=abs(h) >= 0.2,  # Arbitrary threshold for meaningful effect
            confidence_level=0.0,
            interpretation=interpretation
        )
    
    def analyze_confidence_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in evaluation confidence scores."""
        results = [r for r in self.evaluation_results if r.get('winner') != 'error']
        
        if not results:
            return {}
        
        confidences = [r['confidence'] for r in results]
        
        # Basic statistics
        confidence_stats = {
            'mean': np.mean(confidences),
            'median': np.median(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'quartiles': np.percentile(confidences, [25, 50, 75]).tolist()
        }
        
        # Confidence by outcome
        conf_by_outcome = {}
        for outcome in ['A', 'B', 'TIE']:
            outcome_confs = [r['confidence'] for r in results if r['winner'] == outcome]
            if outcome_confs:
                conf_by_outcome[outcome] = {
                    'mean': np.mean(outcome_confs),
                    'count': len(outcome_confs),
                    'std': np.std(outcome_confs)
                }
        
        return {
            'overall_stats': confidence_stats,
            'by_outcome': conf_by_outcome,
            'high_confidence_rate': sum(1 for c in confidences if c >= 0.8) / len(confidences),
            'low_confidence_rate': sum(1 for c in confidences if c < 0.5) / len(confidences)
        }
    
    def generate_visualizations(self, save_plots: bool = True) -> Dict[str, str]:
        """Generate visualization plots for the results."""
        plot_files = {}
        
        try:
            # Set style
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (10, 6)
            
            # 1. Win rate comparison
            fig, ax = plt.subplots()
            summary = self.analysis.get('summary', {})
            
            labels = [self.config.get('label_a', 'Model A'), 
                     self.config.get('label_b', 'Model B'), 
                     'Ties']
            sizes = [summary.get('wins_a', 0), 
                    summary.get('wins_b', 0), 
                    summary.get('ties', 0)]
            
            if sum(sizes) > 0:
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                plt.title('Evaluation Results Distribution')
                
                if save_plots:
                    plot_path = self.results_dir / 'win_distribution.png'
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plot_files['win_distribution'] = str(plot_path)
                
                plt.close()
            
            # 2. Confidence distribution
            confidences = [r['confidence'] for r in self.evaluation_results 
                          if r.get('winner') != 'error']
            
            if confidences:
                fig, ax = plt.subplots()
                plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Confidence Score')
                plt.ylabel('Frequency')
                plt.title('Distribution of Evaluation Confidence Scores')
                plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(confidences):.3f}')
                plt.legend()
                
                if save_plots:
                    plot_path = self.results_dir / 'confidence_distribution.png'
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plot_files['confidence_distribution'] = str(plot_path)
                
                plt.close()
            
            # 3. Genre-based analysis (if available)
            genre_analysis = self.analysis.get('genre_analysis', {})
            if genre_analysis:
                genres = list(genre_analysis.keys())
                win_rates_a = [genre_analysis[g].get('win_rate_a', 0) for g in genres]
                win_rates_b = [genre_analysis[g].get('win_rate_b', 0) for g in genres]
                
                x = np.arange(len(genres))
                width = 0.35
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(x - width/2, win_rates_a, width, label=self.config.get('label_a', 'Model A'))
                ax.bar(x + width/2, win_rates_b, width, label=self.config.get('label_b', 'Model B'))
                
                ax.set_xlabel('Genre')
                ax.set_ylabel('Win Rate')
                ax.set_title('Win Rates by Genre')
                ax.set_xticks(x)
                ax.set_xticklabels(genres, rotation=45, ha='right')
                ax.legend()
                ax.set_ylim(0, 1)
                
                if save_plots:
                    plot_path = self.results_dir / 'genre_analysis.png'
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plot_files['genre_analysis'] = str(plot_path)
                
                plt.close()
                
        except Exception as e:
            print(f"Error generating visualizations: {e}")
        
        return plot_files
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        
        # Run statistical tests
        statistical_tests = self.run_statistical_tests()
        
        # Analyze confidence patterns
        confidence_analysis = self.analyze_confidence_patterns()
        
        # Generate visualizations
        plot_files = self.generate_visualizations()
        
        # Create comprehensive report
        report = {
            'evaluation_metadata': {
                'evaluation_date': self.config.get('timestamp'),
                'model_a': self.config.get('label_a', 'Model A'),
                'model_b': self.config.get('label_b', 'Model B'),
                'total_pairs': self.config.get('num_pairs', 0),
                'gpt_model': self.config.get('gpt_model', 'unknown'),
                'criteria': self.config.get('evaluation_criteria', 'unknown')
            },
            'basic_results': self.analysis.get('summary', {}),
            'statistical_tests': [
                {
                    'test_name': test.test_name,
                    'statistic': test.statistic,
                    'p_value': test.p_value,
                    'significant': test.significant,
                    'interpretation': test.interpretation
                }
                for test in statistical_tests
            ],
            'confidence_analysis': confidence_analysis,
            'genre_breakdown': self.analysis.get('genre_analysis', {}),
            'epoch_breakdown': self.analysis.get('epoch_analysis', {}),
            'visualizations': plot_files,
            'conclusions': self._generate_conclusions(statistical_tests)
        }
        
        # Save report
        with open(self.results_dir / 'comprehensive_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _generate_conclusions(self, statistical_tests: List[StatisticalTest]) -> List[str]:
        """Generate textual conclusions based on the analysis."""
        conclusions = []
        
        summary = self.analysis.get('summary', {})
        total_valid = summary.get('valid_evaluations', 0)
        
        if total_valid == 0:
            conclusions.append("No valid evaluations were completed.")
            return conclusions
        
        # Basic outcome
        win_rate_a = summary.get('win_rate_a', 0)
        win_rate_b = summary.get('win_rate_b', 0)
        
        model_a = self.config.get('label_a', 'Model A')
        model_b = self.config.get('label_b', 'Model B')
        
        if win_rate_a > win_rate_b:
            margin = win_rate_a - win_rate_b
            conclusions.append(f"{model_a} outperformed {model_b} with a {margin:.1%} margin ({win_rate_a:.1%} vs {win_rate_b:.1%})")
        elif win_rate_b > win_rate_a:
            margin = win_rate_b - win_rate_a
            conclusions.append(f"{model_b} outperformed {model_a} with a {margin:.1%} margin ({win_rate_b:.1%} vs {win_rate_a:.1%})")
        else:
            conclusions.append(f"{model_a} and {model_b} performed equally well")
        
        # Statistical significance
        significant_tests = [t for t in statistical_tests if t.significant]
        if significant_tests:
            conclusions.append("The performance difference is statistically significant")
            for test in significant_tests:
                conclusions.append(f"• {test.interpretation}")
        else:
            conclusions.append("The performance difference is not statistically significant")
        
        # Sample size assessment
        if total_valid < 30:
            conclusions.append("Note: Small sample size limits the reliability of statistical tests")
        elif total_valid >= 100:
            conclusions.append("Large sample size provides reliable statistical power")
        
        # Confidence assessment
        avg_confidence = summary.get('avg_confidence', 0)
        if avg_confidence >= 0.8:
            conclusions.append("Evaluations show high confidence in the judgments")
        elif avg_confidence < 0.6:
            conclusions.append("Evaluations show relatively low confidence, suggesting close competition")
        
        return conclusions


def analyze_evaluation_results(results_dir: str) -> Dict[str, Any]:
    """
    Convenience function to run comprehensive analysis on evaluation results.
    
    Args:
        results_dir: Path to evaluation results directory
        
    Returns:
        Comprehensive analysis report
    """
    analyzer = ResultsAnalyzer(results_dir)
    return analyzer.generate_report()