#!/usr/bin/env python3
"""Fix all training issues for multi-domain support."""

import os
import re

def fix_clean_train():
    """Fix all issues in clean_train.py"""
    file_path = "llm_gan/train/clean_train.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    print("Fixing imports and variable names...")
    
    # Fix imports
    content = content.replace(
        "from .dataset import StoryDataset", 
        "# Domain dataset imported inline where needed"
    )
    
    # Fix dataset usage
    content = content.replace(
        "dataset = StoryDataset(data_path, min_story_length=min_story_length)",
        """# Use domain-aware dataset
        from llm_gan.train.domain_dataset import DomainDataset
        dataset = DomainDataset(domain_obj, data_path, min_output_length=min_story_length)"""
    )
    
    # Fix variable name references
    content = content.replace(
        "generator_model, tokenizer, generator_prompts, generated_stories, generator_ref_model",
        "generator_model, tokenizer, generator_prompts, generated_outputs, generator_ref_model"
    )
    
    content = content.replace(
        "generator_log_probs = calculate_log_probs(generator_model, tokenizer, generator_prompts, generated_stories)",
        "generator_log_probs = calculate_log_probs(generator_model, tokenizer, generator_prompts, generated_outputs)"
    )
    
    # Make sure domain import is added
    if "from llm_gan.domains import DomainRegistry" not in content:
        # Find the right place to add the import
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('def train_llm_gan('):
                # Add domain imports right at the start of the function
                lines.insert(i + 1, "    # Import domain system")
                lines.insert(i + 2, "    from llm_gan.domains import DomainRegistry")
                lines.insert(i + 3, "")
                break
        content = '\n'.join(lines)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed: {file_path}")

def check_domains_exist():
    """Check if domain files exist and create them if not"""
    domain_files = [
        "llm_gan/domains/__init__.py",
        "llm_gan/domains/base.py",
        "llm_gan/domains/creative_writing.py", 
        "llm_gan/domains/proofs.py",
        "llm_gan/domains/registry.py",
        "llm_gan/train/domain_dataset.py"
    ]
    
    missing = [f for f in domain_files if not os.path.exists(f)]
    
    if missing:
        print(f"❌ Missing domain files: {missing}")
        print("You need to copy the domain system files from the development machine.")
        return False
    else:
        print("✅ All domain files exist")
        return True

if __name__ == "__main__":
    print("Fixing all training issues for multi-domain support...")
    
    if check_domains_exist():
        fix_clean_train()
        print("\n✅ All fixes applied!")
        print("\nYou can now train with:")
        print("  torchrun --nproc_per_node=8 train_ddp.py --data_path data/proofs.csv --batch_size 4 --epochs 10")
    else:
        print("\n❌ Domain files missing. Please copy them first.")