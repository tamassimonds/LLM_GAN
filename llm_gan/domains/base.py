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
        pass
    
    @abstractmethod
    def get_generator_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate prompt for the generator model."""
        pass
    
    @abstractmethod
    def get_judge_prompt(self, output1: str, output2: str, sample: Dict[str, Any]) -> str:
        """Generate prompt for the judge model to compare two outputs."""
        pass
    
    def extract_output(self, generated_text: str, use_tags: bool = True) -> str:
        """Extract output from generated text.
        
        Args:
            generated_text: The raw generated text
            use_tags: If True, look for <output> tags. If False, take whole output after assistant.
        """
        if use_tags:
            return self._extract_with_tags(generated_text)
        else:
            return self._extract_without_tags(generated_text)
    
    def _extract_with_tags(self, generated_text: str) -> str:
        """Extract output using <output> tags."""
        from llm_gan.utils.parse import parse_tags
        
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
        return self._extract_without_tags(generated_text)
    
    def _extract_without_tags(self, generated_text: str) -> str:
        """Extract output without using tags - take everything after assistant."""
        if "assistant" in generated_text:
            parts = generated_text.split("assistant")
            if len(parts) > 1:
                content = parts[-1].strip()
                # Clean up common chat formatting
                content = content.replace("<|eot_id|>", "").strip()
                return content
        
        return generated_text.strip()
    
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