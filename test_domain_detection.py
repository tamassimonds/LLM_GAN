#!/usr/bin/env python3
"""Test domain detection and loading."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_gan.domains import DomainRegistry

def test_domain_detection():
    print("Testing domain detection...")
    
    # Test creative writing
    print("\n1. Testing creative writing detection:")
    try:
        domain = DomainRegistry.create_domain(csv_path="data/stories.csv")
        print(f"   ✅ Detected: {domain.__class__.__name__}")
        print(f"   Required columns: {domain.get_data_columns()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test proofs
    print("\n2. Testing proofs detection:")
    try:
        domain = DomainRegistry.create_domain(csv_path="data/proofs.csv")
        print(f"   ✅ Detected: {domain.__class__.__name__}")
        print(f"   Required columns: {domain.get_data_columns()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test explicit domain
    print("\n3. Testing explicit domain specification:")
    try:
        domain = DomainRegistry.create_domain(domain_name="proofs")
        print(f"   ✅ Created: {domain.__class__.__name__}")
        print(f"   Required columns: {domain.get_data_columns()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_domain_detection()