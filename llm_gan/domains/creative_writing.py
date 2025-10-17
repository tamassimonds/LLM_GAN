"""Creative writing domain implementation."""

from typing import Dict, List, Any
from .base import BaseDomain


class CreativeWritingDomain(BaseDomain):
    """Domain implementation for creative writing tasks."""
    
    def get_data_columns(self) -> List[str]:
        """Return required CSV columns for creative writing."""
        return ["human_story", "title", "genre"]
    
    def load_data(self, csv_path: str):
        """Load and validate creative writing data."""
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = self.get_data_columns()
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for {self.__class__.__name__}: {missing_cols}")
        
        # Clean data
        df = df.dropna(subset=['title', 'human_story'])
        df['genre'] = df['genre'].fillna('General Fiction')
        
        return df
    
    def get_generator_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate creative writing prompt."""
        title = sample.get("title", "Untitled")
        genre = sample.get("genre", "general fiction")
        
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Write a creative story in the {genre} genre with the title "{title}".

Make the story feel human-written and engaging. Write approximately 200-300 words.

Put your story inside <output> tags like this:
<output>Your story here</output><|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def get_judge_prompt(self, output1: str, output2: str, sample: Dict[str, Any]) -> str:
        """Generate judge prompt for comparing two stories."""
        title = sample.get("title", "Untitled")
        genre = sample.get("genre", "general fiction")
        
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

You need to determine which story is better

Title: {title}
Genre: {genre}

STORY 1:
{output1}

STORY 2:
{output2}


Give your reasoning briefly, then put your final answer in \\boxed{{1}} or \\boxed{{2}} to indicate which story is better.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def extract_output(self, generated_text: str) -> str:
        """Extract story from generated text."""
        # Use the base implementation but with story-specific fallbacks
        output = super().extract_output(generated_text)
        
        # Check for placeholder text specific to stories
        placeholder_patterns = [
            "your story here",
            "write your story",
            "story here",
            "[story]",
            "insert story here"
        ]
        
        output_lower = output.lower().strip()
        for pattern in placeholder_patterns:
            if pattern in output_lower and len(output) < 100:
                return "A creative story."  # Fallback
        
        return output
    
    def get_evaluation_criteria(self) -> List[str]:
        """Return creative writing evaluation criteria."""
        return [
            "narrative coherence",
            "prose quality", 
            "character development",
            "creativity and originality",
            "engagement and readability",
            "genre appropriateness"
        ]
    
    def get_batch_metadata_keys(self) -> List[str]:
        """Return metadata keys for batch files."""
        return ["titles", "genres"]