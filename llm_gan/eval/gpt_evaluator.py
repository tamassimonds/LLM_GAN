"""GPT-based evaluation of story pairs for stylistic quality."""

import json
import time
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai package not installed. Install with: pip install openai")

from .story_sampler import StoryPair


class EvaluationCriteria(Enum):
    """Different criteria for story evaluation."""
    OVERALL_QUALITY = "overall_quality"
    CREATIVITY = "creativity" 
    COHERENCE = "coherence"
    STYLE = "style"
    ENGAGEMENT = "engagement"


@dataclass
class EvaluationResult:
    """Result of a single story pair evaluation."""
    pair_id: str
    winner: str  # "A", "B", or "tie"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    criteria: EvaluationCriteria
    model_used: str
    evaluation_time: float
    raw_response: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'pair_id': self.pair_id,
            'winner': self.winner,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'criteria': self.criteria.value,
            'model_used': self.model_used,
            'evaluation_time': self.evaluation_time,
            'raw_response': self.raw_response
        }


class GPTEvaluator:
    """Evaluate story pairs using GPT models."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize GPT evaluator.
        
        Args:
            model: OpenAI model to use (gpt-4o, gpt-4, gpt-3.5-turbo, etc.)
            api_key: OpenAI API key (if None, uses environment variable)
        """
        if not HAS_OPENAI:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        self.model = model
        
        # Set up OpenAI client
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            # Will use OPENAI_API_KEY environment variable
            self.client = openai.OpenAI()
    
    def evaluate_pairs_batch(self, pairs: List[StoryPair], 
                           criteria: EvaluationCriteria = EvaluationCriteria.OVERALL_QUALITY,
                           max_retries: int = 3) -> List[EvaluationResult]:
        """
        Evaluate multiple story pairs in a single batch API call.
        
        Args:
            pairs: List of StoryPair objects to evaluate
            criteria: Evaluation criteria to use
            max_retries: Maximum number of API call retries
            
        Returns:
            List of EvaluationResult objects
        """
        if not pairs:
            return []
        
        print(f"  🚀 Running batch evaluation of {len(pairs)} pairs...")
        
        # Create batch prompt with all pairs
        batch_prompt = self._create_batch_evaluation_prompt(pairs, criteria)
        
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                # Prepare API parameters based on model
                api_params = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are an expert literary critic and writing evaluator. You provide thoughtful, detailed analysis of creative writing. You will evaluate multiple story pairs and return results in JSON format."
                        },
                        {
                            "role": "user",
                            "content": batch_prompt
                        }
                    ]
                }
                
                # Only add parameters if not GPT-5
                if not self.model.startswith('gpt-5'):
                    api_params["temperature"] = 0.3
                    api_params["max_tokens"] = 4000  # Larger for batch response
                
                response = self.client.chat.completions.create(**api_params)
                
                evaluation_time = time.time() - start_time
                raw_response = response.choices[0].message.content
                
                # Parse batch response
                results = self._parse_batch_response(raw_response, pairs, criteria, self.model, evaluation_time / len(pairs))
                
                print(f"  ✅ Batch evaluation completed in {evaluation_time:.2f}s")
                return results
                
            except Exception as e:
                print(f"  ⚠️  Batch API call attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    # Fallback to individual evaluations
                    print(f"  🔄 Falling back to individual evaluations...")
                    return [self.evaluate_pair(pair, criteria, max_retries) for pair in pairs]
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return []  # Should not reach here
    
    def evaluate_pair(self, pair: StoryPair, 
                     criteria: EvaluationCriteria = EvaluationCriteria.OVERALL_QUALITY,
                     max_retries: int = 3) -> EvaluationResult:
        """
        Evaluate a single story pair.
        
        Args:
            pair: StoryPair to evaluate
            criteria: Evaluation criteria to use
            max_retries: Maximum number of API call retries
            
        Returns:
            EvaluationResult with the evaluation outcome
        """
        prompt = self._create_evaluation_prompt(pair, criteria)
        
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                # Prepare API parameters based on model
                api_params = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are an expert literary critic and writing evaluator. You provide thoughtful, detailed analysis of creative writing."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
                
                # Only add parameters if not GPT-5
                if not self.model.startswith('gpt-5'):
                    api_params["temperature"] = 0.3  # Lower temperature for more consistent evaluations
                    api_params["max_tokens"] = 1000
                
                response = self.client.chat.completions.create(**api_params)
                
                evaluation_time = time.time() - start_time
                raw_response = response.choices[0].message.content
                
                # Parse the response
                result = self._parse_response(
                    raw_response, 
                    pair.comparison_id, 
                    criteria, 
                    evaluation_time
                )
                
                return result
                
            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    # Return a failed evaluation result
                    return EvaluationResult(
                        pair_id=pair.comparison_id,
                        winner="error",
                        confidence=0.0,
                        reasoning=f"API call failed: {str(e)}",
                        criteria=criteria,
                        model_used=self.model,
                        evaluation_time=time.time() - start_time,
                        raw_response=""
                    )
                
                # Wait before retrying
                time.sleep(2 ** attempt)
    
    def evaluate_pairs_batch(self, pairs: List[StoryPair], 
                           criteria: EvaluationCriteria = EvaluationCriteria.OVERALL_QUALITY,
                           delay_between_calls: float = 1.0) -> List[EvaluationResult]:
        """
        Evaluate multiple story pairs with rate limiting.
        
        Args:
            pairs: List of StoryPair objects to evaluate
            criteria: Evaluation criteria to use
            delay_between_calls: Delay in seconds between API calls
            
        Returns:
            List of EvaluationResult objects
        """
        results = []
        
        for i, pair in enumerate(pairs):
            print(f"Evaluating pair {i+1}/{len(pairs)}: {pair.comparison_id}")
            
            result = self.evaluate_pair(pair, criteria)
            results.append(result)
            
            # Rate limiting
            if i < len(pairs) - 1:  # Don't delay after the last call
                time.sleep(delay_between_calls)
        
        return results
    
    def _create_evaluation_prompt(self, pair: StoryPair, criteria: EvaluationCriteria) -> str:
        """Create the evaluation prompt for a story pair."""
        
        criteria_descriptions = {
            EvaluationCriteria.OVERALL_QUALITY: "overall quality and writing excellence",
            EvaluationCriteria.CREATIVITY: "creativity and originality", 
            EvaluationCriteria.COHERENCE: "narrative coherence and logical flow",
            EvaluationCriteria.STYLE: "writing style and prose quality",
            EvaluationCriteria.ENGAGEMENT: "reader engagement and entertainment value"
        }
        
        criteria_desc = criteria_descriptions.get(criteria, "overall quality")
        
        prompt = f"""I need you to compare two stories and determine which one is better in terms of {criteria_desc}.

**STORY A:**
Title: {pair.story_a.title}
Genre: {pair.story_a.genre}
Story: {pair.story_a.story}

**STORY B:**
Title: {pair.story_b.title}
Genre: {pair.story_b.genre}
Story: {pair.story_b.story}

Please evaluate these stories based on {criteria_desc} and provide your analysis in the following format:

**ANALYSIS:**
[Provide detailed reasoning for your evaluation, discussing the strengths and weaknesses of each story]

**WINNER:** [A, B, or TIE]
**CONFIDENCE:** [Your confidence level from 0.0 to 1.0]

Focus specifically on {criteria_desc} in your evaluation. Consider factors like narrative structure, character development, prose quality, creativity, and overall effectiveness of the writing."""
        
        return prompt
    
    def _parse_response(self, response: str, pair_id: str, 
                       criteria: EvaluationCriteria, evaluation_time: float) -> EvaluationResult:
        """Parse the GPT response into an EvaluationResult."""
        
        # Extract winner
        winner = "error"
        confidence = 0.0
        reasoning = response
        
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('**WINNER:**'):
                winner_text = line.replace('**WINNER:**', '').strip().upper()
                if 'A' in winner_text:
                    winner = 'A'
                elif 'B' in winner_text:
                    winner = 'B'
                elif 'TIE' in winner_text:
                    winner = 'TIE'
            
            elif line.startswith('**CONFIDENCE:**'):
                conf_text = line.replace('**CONFIDENCE:**', '').strip()
                try:
                    confidence = float(conf_text)
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                except ValueError:
                    confidence = 0.5  # Default if parsing fails
        
        # Extract reasoning (everything before WINNER line)
        analysis_start = response.find('**ANALYSIS:**')
        winner_start = response.find('**WINNER:**')
        
        if analysis_start != -1 and winner_start != -1:
            reasoning = response[analysis_start + len('**ANALYSIS:**'):winner_start].strip()
        elif analysis_start != -1:
            reasoning = response[analysis_start + len('**ANALYSIS:**'):].strip()
        
        return EvaluationResult(
            pair_id=pair_id,
            winner=winner,
            confidence=confidence,
            reasoning=reasoning,
            criteria=criteria,
            model_used=self.model,
            evaluation_time=evaluation_time,
            raw_response=response
        )
    
    def _create_batch_evaluation_prompt(self, pairs: List[StoryPair], criteria: EvaluationCriteria) -> str:
        """Create a prompt for batch evaluation of multiple story pairs."""
        criteria_descriptions = {
            EvaluationCriteria.OVERALL_QUALITY: "overall quality and writing excellence including narrative structure, prose quality, and overall effectiveness",
            EvaluationCriteria.CREATIVITY: "creativity, originality, and imaginative elements in the storytelling",
            EvaluationCriteria.COHERENCE: "narrative logic, consistency, and structural coherence", 
            EvaluationCriteria.STYLE: "prose quality, writing style, and linguistic sophistication",
            EvaluationCriteria.ENGAGEMENT: "reader engagement and entertainment value"
        }
        
        criteria_desc = criteria_descriptions.get(criteria, criteria_descriptions[EvaluationCriteria.OVERALL_QUALITY])
        
        prompt = f"""I will present you with {len(pairs)} pairs of stories. For each pair, evaluate which story is better in terms of {criteria_desc}.

For each pair, analyze both stories and determine which is superior, providing your confidence level (0.1-1.0).

Return your evaluations in this JSON format:
{{
  "evaluations": [
    {{
      "pair_id": "pair_1", 
      "winner": "A",
      "confidence": 0.8,
      "reasoning": "Brief explanation..."
    }},
    ...
  ]
}}

Here are the story pairs to evaluate:

"""

        for i, pair in enumerate(pairs):
            prompt += f"""
=== PAIR {i+1} (ID: {pair.pair_id}) ===

**STORY A:**
{pair.story_a[:2000]}{"..." if len(pair.story_a) > 2000 else ""}

**STORY B:** 
{pair.story_b[:2000]}{"..." if len(pair.story_b) > 2000 else ""}

"""

        prompt += """
Please evaluate all pairs and return the JSON response with your evaluations."""
        
        return prompt
    
    def _parse_batch_response(self, response: str, pairs: List[StoryPair], 
                            criteria: EvaluationCriteria, model: str, avg_time: float) -> List[EvaluationResult]:
        """Parse batch evaluation response into individual results."""
        results = []
        
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                evaluations = data.get('evaluations', [])
                
                # Create results for each evaluation
                for i, evaluation in enumerate(evaluations):
                    if i < len(pairs):
                        pair = pairs[i]
                        result = EvaluationResult(
                            pair_id=evaluation.get('pair_id', pair.pair_id),
                            winner=evaluation.get('winner', 'A'),
                            confidence=float(evaluation.get('confidence', 0.5)),
                            reasoning=evaluation.get('reasoning', 'Batch evaluation'),
                            criteria=criteria,
                            model_used=model,
                            evaluation_time=avg_time,
                            raw_response=response
                        )
                        results.append(result)
                
                # Fill in missing results if needed
                while len(results) < len(pairs):
                    pair = pairs[len(results)]
                    result = EvaluationResult(
                        pair_id=pair.pair_id,
                        winner='A',
                        confidence=0.5,
                        reasoning='Batch parsing incomplete',
                        criteria=criteria,
                        model_used=model,
                        evaluation_time=avg_time,
                        raw_response=response
                    )
                    results.append(result)
                    
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"  ⚠️  Error parsing batch response: {e}")
            # Fallback: create default results
            for pair in pairs:
                result = EvaluationResult(
                    pair_id=pair.pair_id,
                    winner='A',
                    confidence=0.5,
                    reasoning=f'Batch parsing failed: {str(e)}',
                    criteria=criteria,
                    model_used=model,
                    evaluation_time=avg_time,
                    raw_response=response
                )
                results.append(result)
        
        return results


def evaluate_story_pairs(pairs: List[StoryPair],
                        model: str = "gpt-4o",
                        criteria: EvaluationCriteria = EvaluationCriteria.OVERALL_QUALITY,
                        api_key: Optional[str] = None,
                        delay_between_calls: float = 1.0) -> List[EvaluationResult]:
    """
    Convenience function to evaluate story pairs with GPT.
    
    Args:
        pairs: List of StoryPair objects to evaluate
        model: OpenAI model to use
        criteria: Evaluation criteria
        api_key: OpenAI API key (optional)
        delay_between_calls: Delay between API calls for rate limiting
        
    Returns:
        List of EvaluationResult objects
    """
    evaluator = GPTEvaluator(model=model, api_key=api_key)
    return evaluator.evaluate_pairs_batch(pairs, criteria, delay_between_calls)