#!/usr/bin/env python3
"""Fix proof extraction and judge prompts."""

import os

def fix_proofs_domain():
    """Fix extraction and judge prompts in proofs domain."""
    file_path = "llm_gan/domains/proofs.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix extraction method - make it less aggressive
    old_extract = '''        # Check for placeholder text specific to proofs
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
                return "Solution: [Mathematical proof]"  # Fallback'''
                
    new_extract = '''        # Debug: print what we extracted
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
                return "I need to solve this mathematical problem step by step."  # Better fallback'''
    
    content = content.replace(old_extract, new_extract)
    
    # Fix judge prompt to be clearer
    old_judge = '''You need to determine which mathematical solution is better.

Problem:
{problem}

SOLUTION 1:
{output1}

SOLUTION 2:
{output2}

Evaluate based on correctness, clarity, rigor, and completeness.

Give your reasoning briefly, then put your final answer in \\boxed{{1}} or \\boxed{{2}} to indicate which solution is better.'''

    new_judge = '''You are comparing two mathematical solutions. DO NOT solve the problem yourself. Just evaluate which of the two given solutions is better.

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

Give your reasoning briefly, then put your final answer in \\boxed{{1}} or \\boxed{{2}} to indicate which solution is better.'''

    content = content.replace(old_judge, new_judge)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed: {file_path}")

if __name__ == "__main__":
    print("Fixing proof extraction and judge prompts...")
    fix_proofs_domain()
    print("✅ Complete! The extraction should work better now and judges won't solve from scratch.")