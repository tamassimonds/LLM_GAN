import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
import random
import numpy as np
import json
import os
from datetime import datetime

from llm_gan.prompts import llm_generator_prompt, llm_generator_discriminator_prompt
from llm_gan.utils.parse import parse_tags, parse_boxed
from .dataset import StoryDataset




def simple_generate(model, tokenizer, prompts, max_new_tokens=512, temperature=0.8, batch_size=32):
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


def assess_judge(titles, genres, stories_human, stories_ai, judge_model, tokenizer):
    targets = [random.randint(0,1) for _ in range(len(titles))]
    
    prompts = []
    for title, genre, human_story, ai_story, target in zip(titles, genres, stories_human, stories_ai, targets):
        story_order = (human_story, ai_story) if target == 0 else (ai_story, human_story)
        prompt = llm_generator_discriminator_prompt(title, genre, *story_order)
        prompts.append(prompt)
    
    judge_outputs = simple_generate(
        model=judge_model, 
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=MAX_AGENT_TOKENS,
        batch_size=len(prompts)  # Process all prompts at once
    )
    
    parsed_outputs = []
    for output in judge_outputs:
        # Extract content after assistant header to get just the response
        if "assistant<|end_header_id|>" in output:
            response = output.split("assistant<|end_header_id|>")[-1].strip()
        elif "<|start_header_id|>assistant<|end_header_id|>" in output:
            response = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            response = output.strip()
        
        # Parse boxed answer from the response only
        answer = parse_boxed(response)
        
        if answer == "1":
            parsed_outputs.append(0)
        elif answer == "2":
            parsed_outputs.append(1)
        else:
            parsed_outputs.append(-1)
    
    correct = [parsed == target for parsed, target in zip(parsed_outputs, targets)]
    return correct

def assess_judge_with_outputs(titles, genres, stories_human, stories_ai, judge_model, tokenizer):
    targets = [random.randint(0,1) for _ in range(len(titles))]
    
    prompts = []
    for title, genre, human_story, ai_story, target in zip(titles, genres, stories_human, stories_ai, targets):
        story_order = (human_story, ai_story) if target == 0 else (ai_story, human_story)
        prompt = llm_generator_discriminator_prompt(title, genre, *story_order)
        prompts.append(prompt)
    
    judge_outputs = simple_generate(
        model=judge_model, 
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=MAX_JUDGE_TOKENS,
        batch_size=len(prompts)  # Process all prompts at once
    )
    
    parsed_outputs = []
    for output in judge_outputs:
        # Extract content after assistant header to get just the response
        if "assistant<|end_header_id|>" in output:
            response = output.split("assistant<|end_header_id|>")[-1].strip()
        elif "<|start_header_id|>assistant<|end_header_id|>" in output:
            response = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            response = output.strip()
        
        # Parse boxed answer from the response only
        answer = parse_boxed(response)
        
        if answer == "1":
            parsed_outputs.append(0)
        elif answer == "2":
            parsed_outputs.append(1)
        else:
            parsed_outputs.append(-1)
    
    # Extract just the judge responses (not the full prompt+response)
    judge_responses = []
    for output in judge_outputs:
        if "assistant<|end_header_id|>" in output:
            response = output.split("assistant<|end_header_id|>")[-1].strip()
        elif "<|start_header_id|>assistant<|end_header_id|>" in output:
            response = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            response = output.strip()
        judge_responses.append(response)
    
    correct = [parsed == target for parsed, target in zip(parsed_outputs, targets)]
    return correct, {
        'targets': targets,
        'prompts': prompts,
        'judge_outputs': judge_responses,  # Now contains only the judge responses
        'parsed_outputs': parsed_outputs,
        'correct': correct
    }


def calculate_rewards(judge_correct, generator_fooled_judge):
    judge_rewards = [1.0 if correct else -0.5 for correct in judge_correct]
    generator_rewards = [1.0 if fooled else -1.0 for fooled in generator_fooled_judge]
    return judge_rewards, generator_rewards


def calculate_log_probs(model, tokenizer, prompts, responses):
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
        
        # Sum log probabilities for the response with clamping for stability
        total_log_prob = selected_log_probs.mean()
        # total_log_prob = torch.clamp(total_log_prob, min=-100, max=0)  # Prevent extreme values
        
        log_probs.append(total_log_prob)
        # breakpoint()
    
    return torch.stack(log_probs)


def reinforce_update(model, optimizer, log_probs, rewards):
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

    # breakpoint()
    
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


# Training parameters
bs = 32  # Small batch for large model
epochs = 100  # Full training

MAX_AGENT_TOKENS = 512
MAX_JUDGE_TOKENS = 512


def collate_fn(batch):
    return batch

dataset = StoryDataset("data/ai_eval_stories.csv")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn)

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'  # Fix the padding warning
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

generator_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
    device_map="auto"
)
judge_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
    device_map="auto"
)
torch.backends.cuda.matmul.allow_tf32 = True

generator_optimizer = AdamW(generator_model.parameters(), lr=1e-5)  # Lower learning rate for stability
judge_optimizer = AdamW(judge_model.parameters(), lr=1e-5)  # Lower learning rate for stability

# Setup logging
log_dir = f"logs/training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)

print(f"Starting training with {len(dataset)} samples, {len(dataloader)} batches...")
print(f"Logging to: {log_dir}")

# Test if models are corrupted from the start
print("Testing model integrity...")
for name, param in generator_model.named_parameters():
    if not torch.isfinite(param).all():
        print(f"ERROR: Generator model parameter {name} contains inf/nan values at startup!")
        
for name, param in judge_model.named_parameters():
    if not torch.isfinite(param).all():
        print(f"ERROR: Judge model parameter {name} contains inf/nan values at startup!")

print("Models loaded successfully, starting training...")
for epoch in range(epochs):
    epoch_judge_accuracy = 0
    epoch_generator_reward = 0
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"Processing batch {batch_idx+1}/{len(dataloader)}...")
        
        titles = [item['title'] for item in batch]
        genres = [item['genre'] for item in batch] 
        human_stories = [item['human_story'] for item in batch]
        
        # Generate stories with generator
        generator_prompts = [llm_generator_prompt(title, genre) for title, genre in zip(titles, genres)]
        
        print("  Generating stories...")
        generated_stories_raw = simple_generate(
            model=generator_model,
            tokenizer=tokenizer,
            prompts=generator_prompts,
            max_new_tokens=512,
            temperature=0.8,
            batch_size=32  # Process all 32 at once
        )
        
        generated_stories = []
        for i, story_text in enumerate(generated_stories_raw):
            # First try to get story from <story> tags, but exclude the example
            all_stories = parse_tags(story_text, "story")
            story = None
            
            if isinstance(all_stories, list):
                # Multiple story tags found - filter out the example
                for s in all_stories:
                    if s and "Your story here" not in s and len(s) > 50:
                        story = s
                        break
            elif all_stories and "Your story here" not in all_stories:
                story = all_stories
            
            if story is None:
                # Try to get content from <STORY> tags (uppercase)
                story = parse_tags(story_text, "STORY")
            
            if story is None:
                # Look for content after assistant header - this is where the actual story is
                if "assistant" in story_text:
                    # Split on assistant and get the last part
                    parts = story_text.split("assistant")
                    if len(parts) > 1:
                        content = parts[-1].strip()
                        # Take the first substantial paragraph
                        lines = content.split('\n')
                        story_lines = []
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('<') and len(line) > 10:
                                story_lines.append(line)
                        if story_lines:
                            story = ' '.join(story_lines)
                
                if not story:
                    story = f"A {genres[i].lower()} story about {titles[i]}."  # Fallback
                
                # Clean up and limit length
                if story and len(story) > 10:
                    story = ' '.join(story.split())[:512]
                else:
                    story = f"A {genres[i].lower()} story titled '{titles[i]}'."
                    
            generated_stories.append(story)
        
        print("  Judging stories...")
        # Clip stories to equal length for fair comparison
        max_story_length = 500  # Max characters per story
        clipped_human_stories = [story[:max_story_length] for story in human_stories]
        clipped_generated_stories = [story[:max_story_length] for story in generated_stories]
        
        # Judge the stories (modified to return judge outputs too)
        judge_correct, judge_outputs = assess_judge_with_outputs(
            titles, genres, clipped_human_stories, clipped_generated_stories, judge_model, tokenizer
        )
        
        generator_fooled_judge = [not correct for correct in judge_correct]
        judge_rewards, generator_rewards = calculate_rewards(
            judge_correct, generator_fooled_judge
        )
        
        accuracy = sum(judge_correct) / len(judge_correct) if judge_correct else 0
        epoch_judge_accuracy += accuracy
        epoch_generator_reward += np.mean(generator_rewards)
        
        print(f"  Results: Judge Acc: {accuracy:.4f}, Avg Gen Reward: {np.mean(generator_rewards):.4f}")
        
        # REINFORCE training updates
        print("  Training generator with REINFORCE...")
        generator_model.train()  # Set to training mode
        generator_log_probs = calculate_log_probs(generator_model, tokenizer, generator_prompts, generated_stories)
        generator_loss = reinforce_update(generator_model, generator_optimizer, generator_log_probs, generator_rewards)
        
        print("  Training judge with REINFORCE...")
        judge_model.train()  # Set to training mode
        judge_log_probs = calculate_log_probs(judge_model, tokenizer, judge_outputs['prompts'], judge_outputs['judge_outputs'])
        judge_loss = reinforce_update(judge_model, judge_optimizer, judge_log_probs, judge_rewards)
        
        print(f"  Training losses - Generator: {generator_loss:.4f}, Judge: {judge_loss:.4f}")
        
        # Clear gradients after training
        generator_optimizer.zero_grad()
        judge_optimizer.zero_grad()
        
        # Save detailed logs
        batch_log = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'titles': titles,
            'genres': genres,
            'human_stories': human_stories,
            'generator_prompts': generator_prompts,
            'generated_stories_raw': generated_stories_raw,
            'generated_stories': generated_stories,
            'judge_data': judge_outputs,
            'judge_accuracy': accuracy,
            'generator_rewards': generator_rewards,
            'judge_rewards': judge_rewards,
            'generator_loss': generator_loss,
            'judge_loss': judge_loss
        }
        
        with open(f"{log_dir}/batch_{epoch}_{batch_idx}.json", 'w') as f:
            json.dump(batch_log, f, indent=2)
        
        # Run full training (removed break)
    
    epoch_summary = {
        'epoch': epoch,
        'judge_accuracy': epoch_judge_accuracy/len(dataloader),
        'generator_reward': epoch_generator_reward/len(dataloader),
        'total_batches': len(dataloader)
    }
    
    print(f"Epoch {epoch} Summary - Judge Accuracy: {epoch_summary['judge_accuracy']:.4f}, Generator Reward: {epoch_summary['generator_reward']:.4f}")
    
    # Save epoch summary
    with open(f"{log_dir}/epoch_{epoch}_summary.json", 'w') as f:
        json.dump(epoch_summary, f, indent=2)
    
    torch.save(generator_model.state_dict(), f"working_generator_epoch_{epoch}.pt")
    torch.save(judge_model.state_dict(), f"working_judge_epoch_{epoch}.pt")

print("Training completed successfully!")