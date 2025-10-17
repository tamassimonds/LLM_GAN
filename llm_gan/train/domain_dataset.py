"""Domain-aware dataset for multi-domain training."""

import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Dict, Any


class DomainDataset(TorchDataset):
    """Dataset that works with any domain."""
    
    def __init__(self, domain, csv_path: str, min_output_length: int = 100):
        """Initialize domain-aware dataset.
        
        Args:
            domain: Domain object that defines data structure
            csv_path: Path to CSV file
            min_output_length: Minimum length for human outputs
        """
        self.domain = domain
        self.data = domain.load_data(csv_path)
        
        # Domain-specific filtering
        if hasattr(domain, 'filter_data'):
            self.data = domain.filter_data(self.data, min_output_length)
        else:
            # Default filtering for creative writing
            if 'human_story' in self.data.columns:
                self.data = self.data[self.data['human_story'].str.len() >= min_output_length]
                self.data['human_story'] = self.data['human_story'].str.strip()
            elif 'human_solution' in self.data.columns:
                self.data = self.data[self.data['human_solution'].str.len() >= min_output_length]
                self.data['human_solution'] = self.data['human_solution'].str.strip()
        
        print(f"Loaded {len(self.data)} samples for {domain.__class__.__name__}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """Return a sample dict that the domain can use."""
        row = self.data.iloc[idx]
        
        # Convert to dict for domain to use
        sample = row.to_dict()
        
        # Ensure backward compatibility for creative writing
        if 'human_story' in sample:
            sample['human_output'] = sample['human_story']
        elif 'human_solution' in sample:
            sample['human_output'] = sample['human_solution']
            
        return sample