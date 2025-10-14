"""REINFORCE training utilities for policy gradient updates."""

import torch
import torch.nn.functional as F
from typing import List


def calculate_log_probs(model, tokenizer, prompts: List[str], responses: List[str]) -> torch.Tensor:
    """Calculate log probabilities for responses given prompts."""
    log_probs = []
    
    for prompt, response in zip(prompts, responses):
        # Tokenize prompt and full sequence
        prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=False)
        full_text = prompt + response
        full_tokens = tokenizer(full_text, return_tensors="pt", padding=False)
        
        # Move to same device as model
        device = next(model.parameters()).device
        prompt_tokens = {k: v.to(device) for k, v in prompt_tokens.items()}
        full_tokens = {k: v.to(device) for k, v in full_tokens.items()}
        
        # Get model outputs (with gradients for training)
        outputs = model(**full_tokens)
        logits = outputs.logits
        
        # Calculate log probabilities for the response part only
        prompt_length = prompt_tokens['input_ids'].shape[1]
        response_tokens = full_tokens['input_ids'][0, prompt_length:]
        response_logits = logits[0, prompt_length-1:-1]  # Shift by 1 for next token prediction
        
        # Calculate log probabilities with numerical stability
        response_log_probs = F.log_softmax(response_logits, dim=-1)
        selected_log_probs = response_log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)
        
        # Use mean log probability (normalized by length)
        total_log_prob = selected_log_probs.mean()
        
        log_probs.append(total_log_prob)
    
    return torch.stack(log_probs)


def reinforce_update(model, optimizer, log_probs: torch.Tensor, rewards: List[float]) -> float:
    """Perform REINFORCE update using policy gradient."""
    # Convert rewards to tensor and normalize them to reduce variance
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=log_probs.device)
    
    # Normalize rewards to have zero mean and unit variance for stability
    if len(rewards_tensor) > 1:
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
    
    # Clamp log probabilities to prevent extreme values
    log_probs_clamped = torch.clamp(log_probs, min=-500, max=0)
    
    # Calculate policy gradient: -log_prob * reward (negative because we want to maximize)
    policy_loss = -(log_probs_clamped * rewards_tensor).mean()
    
    # Check for nan/inf values
    if not torch.isfinite(policy_loss):
        print(f"Warning: Non-finite loss detected: {policy_loss}")
        return 0.0
    
    # Backpropagate with gradient clipping
    optimizer.zero_grad()
    policy_loss.backward()
    
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Validate model parameters after update
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            print(f"ERROR: Model parameter {name} became corrupted after REINFORCE update!")
            return 0.0  # Skip this update
    
    return policy_loss.item()