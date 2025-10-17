"""Base domain class for multi-domain training and evaluation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import pandas as pd


class BaseDomain(ABC):
    """Abstract base class for all domains."""
    
    @abstractmethod
    def get_data_columns(self) -> List[str]:
        """Return required CSV column names for this domain."""
        pass
    
    @abstractmethod
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and validate domain-specific data."""
        df = pd.read_csv(csv_path)
        required_cols = self.get_data_columns()
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for {self.__class__.__name__}: {missing_cols}")
        return df
    
    @abstractmethod
    def get_generator_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate prompt for the generator model."""
        pass
    
    @abstractmethod
    def get_judge_prompt(self, output1: str, output2: str, sample: Dict[str, Any]) -> str:
        """Generate prompt for the judge model to compare two outputs."""
        pass
    
    @abstractmethod
    def extract_output(self, generated_text: str) -> str:
        """Extract output from generated text (from <output> tags)."""
        # Default implementation that works for most domains
        from llm_gan.utils.parsing import parse_tags
        
        # Try to get output from <output> tags
        outputs = parse_tags(generated_text, "output")
        if outputs:
            if isinstance(outputs, list):
                return outputs[0] if outputs else ""
            return outputs
        
        # Try uppercase
        outputs = parse_tags(generated_text, "OUTPUT")
        if outputs:
            if isinstance(outputs, list):
                return outputs[0] if outputs else ""
            return outputs
            
        # Fallback: return everything after assistant header
        if "assistant" in generated_text:
            parts = generated_text.split("assistant")
            if len(parts) > 1:
                return parts[-1].strip()[:512]
        
        return generated_text[:512]
    
    @abstractmethod
    def get_evaluation_criteria(self) -> List[str]:
        """Return domain-specific evaluation criteria."""
        pass
    
    def normalize_outputs(self, output1: str, output2: str, target_length: int = None) -> Tuple[str, str]:
        """Normalize two outputs for fair comparison."""
        if target_length is None:
            target_length = min(len(output1), len(output2))
        
        return output1[:target_length], output2[:target_length]
    
    @abstractmethod
    def get_batch_metadata_keys(self) -> List[str]:
        """Return keys for batch file metadata specific to this domain."""
        pass