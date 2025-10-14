"""REINFORCE training utilities for policy gradient updates."""

import torch
import torch.nn.functional as F
from typing import List


def calculate_log_probs(model, tokenizer, prompts: List[str], responses: List[str], reference_model=None):
    """Calculate log probabilities for responses given prompts.
    
    Args:
        model: Current policy model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        responses: List of response strings
        reference_model: Reference model for PPO (optional)
        
    Returns:
        If reference_model is None: torch.Tensor of log probabilities
        If reference_model is provided: Tuple of (current_log_probs, reference_log_probs)
    """
    current_log_probs = []
    reference_log_probs = [] if reference_model is not None else None
    
    for prompt, response in zip(prompts, responses):
        # Tokenize prompt and full sequence
        prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=False)
        full_text = prompt + response
        full_tokens = tokenizer(full_text, return_tensors="pt", padding=False)
        
        # Move to same device as model
        device = next(model.parameters()).device
        prompt_tokens = {k: v.to(device) for k, v in prompt_tokens.items()}
        full_tokens = {k: v.to(device) for k, v in full_tokens.items()}
        
        # Calculate log probabilities for the response part only
        prompt_length = prompt_tokens['input_ids'].shape[1]
        response_tokens = full_tokens['input_ids'][0, prompt_length:]
        
        # Current model log probabilities (with gradients for training)
        outputs = model(**full_tokens)
        logits = outputs.logits
        response_logits = logits[0, prompt_length-1:-1]  # Shift by 1 for next token prediction
        response_log_probs = F.log_softmax(response_logits, dim=-1)
        selected_log_probs = response_log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)
        total_log_prob = selected_log_probs.mean()
        current_log_probs.append(total_log_prob)
        
        # Reference model log probabilities (no gradients)
        if reference_model is not None:
            with torch.no_grad():
                ref_outputs = reference_model(**full_tokens)
                ref_logits = ref_outputs.logits
                ref_response_logits = ref_logits[0, prompt_length-1:-1]
                ref_response_log_probs = F.log_softmax(ref_response_logits, dim=-1)
                ref_selected_log_probs = ref_response_log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)
                ref_total_log_prob = ref_selected_log_probs.mean()
                reference_log_probs.append(ref_total_log_prob)
    
    current_log_probs = torch.stack(current_log_probs)
    
    if reference_model is not None:
        reference_log_probs = torch.stack(reference_log_probs)
        return current_log_probs, reference_log_probs
    else:
        return current_log_probs


def reinforce_update(model, optimizer, log_probs: torch.Tensor, rewards: List[float], max_len: int= None) -> float:
    """Perform REINFORCE update using policy gradient."""
    # Convert rewards to tensor and normalize them to reduce variance
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=log_probs.device)
    
    # Normalize rewards to have zero mean and unit variance for stability
    if len(rewards_tensor) > 1:
        #skeptical of this cause it cooks the gradient when we get low accuracy? Idk guess if we are having 0 fooling it's ok
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8) 
    
    # Clamp log probabilities to prevent extreme values
    log_probs_clamped = torch.clamp(log_probs, min=-500, max=0)
    
    # Calculate policy gradient: -log_prob * reward (negative because we want to maximize)
    if not max_len:
        policy_loss = -(log_probs_clamped * rewards_tensor).mean() #This is the part I'm skeptical of
    else:
        policy_loss = -(log_probs_clamped * rewards_tensor).sum() / max_len
    
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


def ppo_update(model, optimizer, current_log_probs: torch.Tensor, reference_log_probs: torch.Tensor, 
               rewards: List[float], clip_eps: float = 0.2, entropy_coef: float = 0.01) -> float:
    """Perform PPO update using clipped policy gradient."""
    # Convert rewards to tensor and normalize them (advantages)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=current_log_probs.device)
    
    # Normalize rewards to have zero mean and unit variance for stability (advantages)
    if len(rewards_tensor) > 1:
        advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
    else:
        advantages = rewards_tensor
    
    # Calculate probability ratio: exp(current_log_probs - reference_log_probs)
    log_ratio = current_log_probs - reference_log_probs
    ratio = torch.exp(log_ratio)
    
    # Clamp ratio to prevent extreme values
    ratio = torch.clamp(ratio, min=1e-8, max=10.0)
    
    # PPO clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Entropy bonus for exploration (approximation using log probabilities)
    entropy = -current_log_probs.mean()  # Simple entropy approximation
    entropy_loss = -entropy_coef * entropy
    
    # Total loss
    total_loss = policy_loss + entropy_loss
    
    # Check for nan/inf values
    if not torch.isfinite(total_loss):
        print(f"Warning: Non-finite loss detected: {total_loss}")
        return 0.0
    
    # Backpropagate with gradient clipping
    optimizer.zero_grad()
    total_loss.backward()
    
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Validate model parameters after update
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            print(f"ERROR: Model parameter {name} became corrupted after PPO update!")
            return 0.0  # Skip this update
    
    return total_loss.item()