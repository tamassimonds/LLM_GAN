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

You are comparing two mathematical solutions. DO NOT solve the problem yourself. Just evaluate which of the two given solutions is better.

Problem:
{problem}

SOLUTION 1:
{output1}

SOLUTION 2:
{output2}

Compare the solutions based on:
- Mathematical correctness
- Clarity of explanation  
- Logical rigor
- Completeness of proof

DO NOT solve the problem from scratch. Only evaluate the quality of the two given solutions.

Give your reasoning briefly, then put your final answer in \\boxed{{1}} or \\boxed{{2}} to indicate which solution is better.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def extract_output(self, generated_text: str) -> str:
        """Extract proof from generated text."""
        # First try the standard extraction
        output = super().extract_output(generated_text)
        
        # Debug: print what we extracted
        print(f"DEBUG: Extracted output length: {len(output)}, preview: {output[:200]}...")
        
        # Only check for very obvious placeholder patterns (be less aggressive)
        obvious_placeholders = [
            "your solution here",
            "your complete solution here", 
            "[solution]",
            "solution: [mathematical proof]"
        ]
        
        output_lower = output.lower().strip()
        for pattern in obvious_placeholders:
            if output_lower == pattern or (pattern in output_lower and len(output) < 50):
                print(f"DEBUG: Detected placeholder pattern: {pattern}")
                return "I need to solve this mathematical problem step by step."  # Better fallback
        
        # For proofs, also look for content in \boxed{} if present
        if "\\boxed{" in output:
            # Keep the whole proof including the boxed answer
            print("DEBUG: Found boxed answer in proof")
        
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