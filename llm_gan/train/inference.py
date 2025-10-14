"""Simple inference utilities for LLM generation."""

import torch
from typing import List


def simple_generate(model, tokenizer, prompts: List[str], max_new_tokens=512, temperature=0.8, batch_size=32) -> List[str]:
    """Simple batch generation with proper model state management."""
    # Ensure model is in eval mode and gradients are cleared
    model.eval()
    
    # Check model parameters for corruption before generation
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            print(f"ERROR: Model parameter {name} contains inf/nan values!")
            raise RuntimeError(f"Model corrupted: {name} has non-finite values")
    
    responses = []
    
    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        with torch.no_grad():
            # Tokenize batch
            encoded = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            
            # Move to model device
            device = next(model.parameters()).device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Generate
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode
            batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            responses.extend(batch_responses)
    
    return responses