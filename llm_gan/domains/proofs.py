"""Mathematical proofs domain implementation."""

from typing import Dict, List, Any
from .base import BaseDomain


class ProofsDomain(BaseDomain):
    """Domain implementation for mathematical proofs."""
    
    def get_data_columns(self) -> List[str]:
        """Return required CSV columns for proofs."""
        return ["problem", "human_solution", "source"]
    
    def load_data(self, csv_path: str):
        """Load and validate proofs data."""
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = self.get_data_columns()
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for {self.__class__.__name__}: {missing_cols}")
        
        # Clean data
        df = df.dropna(subset=['problem', 'human_solution'])
        df['source'] = df['source'].fillna('Unknown')
        
        return df
    
    def get_generator_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate proof prompt."""
        problem = sample.get("problem", "")
        source = sample.get("source", "")
        
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Solve the following problem{f' from {source}' if source else ''}:

{problem}

Provide a clear, rigorous mathematical proof with all steps explained.

Put your solution inside <output> tags like this:
<output>Your complete solution here</output><|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def get_judge_prompt(self, output1: str, output2: str, sample: Dict[str, Any]) -> str:
        """Generate judge prompt for comparing two proofs."""
        problem = sample.get("problem", "")
        
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

You need to determine which mathematical solution is better.

Problem:
{problem}

SOLUTION 1:
{output1}

SOLUTION 2:
{output2}

Evaluate based on correctness, clarity, rigor, and completeness.

Give your reasoning briefly, then put your final answer in \\boxed{{1}} or \\boxed{{2}} to indicate which solution is better.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def extract_output(self, generated_text: str) -> str:
        """Extract proof from generated text."""
        # First try the standard extraction
        output = super().extract_output(generated_text)
        
        # For proofs, also look for content in \boxed{} if present
        if "\\boxed{" in output:
            # Extract the main proof, not just the boxed answer
            # The proof should be everything before or around the boxed answer
            parts = output.split("\\boxed{")
            if parts[0].strip():
                # Use the part before the boxed answer as the main proof
                output = parts[0].strip()
        
        # Check for placeholder text specific to proofs
        placeholder_patterns = [
            "your solution here",
            "your complete solution",
            "solution here",
            "[solution]",
            "insert solution"
        ]
        
        output_lower = output.lower().strip()
        for pattern in placeholder_patterns:
            if pattern in output_lower and len(output) < 100:
                return "Solution: [Mathematical proof]"  # Fallback
        
        return output
    
    def get_evaluation_criteria(self) -> List[str]:
        """Return proof evaluation criteria."""
        return [
            "mathematical correctness",
            "logical rigor",
            "clarity of explanation",
            "completeness of proof",
            "proper use of notation",
            "step-by-step reasoning"
        ]
    
    def get_batch_metadata_keys(self) -> List[str]:
        """Return metadata keys for batch files."""
        return ["problems", "sources"]