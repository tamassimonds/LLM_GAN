#!/usr/bin/env python3
"""Quick fix for training imports to enable proofs training."""

import os

def fix_clean_train():
    """Fix imports in clean_train.py"""
    file_path = "llm_gan/train/clean_train.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove old import
    content = content.replace("from .dataset import StoryDataset", "# Domain dataset imported inline where needed")
    
    # Check if domain import already exists
    if "from llm_gan.domains import DomainRegistry" not in content:
        # Add domain import after other imports
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('from .benchmark import'):
                lines.insert(i+1, "    from llm_gan.domains import DomainRegistry")
                break
        content = '\n'.join(lines)
    
    # Fix dataset usage
    if "dataset = StoryDataset" in content:
        content = content.replace(
            "dataset = StoryDataset(data_path, min_story_length=min_story_length)",
            """# Use domain-aware dataset
        from llm_gan.train.domain_dataset import DomainDataset
        dataset = DomainDataset(domain_obj, data_path, min_output_length=min_story_length)"""
        )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed: {file_path}")

if __name__ == "__main__":
    print("Fixing training imports for multi-domain support...")
    fix_clean_train()
    print("✅ Complete! You can now train with: python train_ddp.py --data_path data/proofs.csv")