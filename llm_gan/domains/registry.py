"""Domain registry for auto-detection and factory creation."""

from typing import Dict, Type, Optional, List
import pandas as pd
from .base import BaseDomain
from .creative_writing import CreativeWritingDomain
from .proofs import ProofsDomain


class DomainRegistry:
    """Registry for managing domain implementations."""
    
    # Map domain names to their implementations
    _domains: Dict[str, Type[BaseDomain]] = {
        "creative_writing": CreativeWritingDomain,
        "proofs": ProofsDomain,
    }
    
    # Map column signatures to domains for auto-detection
    _column_signatures = {
        frozenset(["human_story", "title", "genre"]): "creative_writing",
        frozenset(["problem", "human_solution", "source"]): "proofs",
    }
    
    @classmethod
    def register_domain(cls, name: str, domain_class: Type[BaseDomain]):
        """Register a new domain implementation."""
        cls._domains[name] = domain_class
    
    @classmethod
    def get_domain(cls, name: str) -> BaseDomain:
        """Get domain instance by name."""
        if name not in cls._domains:
            raise ValueError(f"Unknown domain: {name}. Available: {list(cls._domains.keys())}")
        return cls._domains[name]()
    
    @classmethod
    def auto_detect_domain(cls, csv_path: str) -> Optional[str]:
        """Auto-detect domain from CSV structure."""
        try:
            # Read just the header to check columns
            df = pd.read_csv(csv_path, nrows=0)
            columns = set(df.columns)
            
            # Check each signature
            for col_set, domain_name in cls._column_signatures.items():
                if col_set.issubset(columns):
                    print(f"Auto-detected domain: {domain_name}")
                    return domain_name
            
            print(f"Could not auto-detect domain from columns: {list(columns)}")
            return None
            
        except Exception as e:
            print(f"Error auto-detecting domain: {e}")
            return None
    
    @classmethod
    def create_domain(cls, csv_path: str = None, domain_name: str = None) -> BaseDomain:
        """Create domain instance with auto-detection fallback."""
        if domain_name:
            return cls.get_domain(domain_name)
        
        if csv_path:
            detected = cls.auto_detect_domain(csv_path)
            if detected:
                return cls.get_domain(detected)
        
        # Default to creative writing for backward compatibility
        print("Using default domain: creative_writing")
        return cls.get_domain("creative_writing")
    
    @classmethod
    def list_domains(cls) -> List[str]:
        """List all available domains."""
        return list(cls._domains.keys())