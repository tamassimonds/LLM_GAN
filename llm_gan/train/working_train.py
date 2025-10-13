import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
import random
import numpy as np
import json
import os
from datetime import datetime

from llm_gan.prompts import llm_generator_prompt, llm_generator_discriminator_prompt
from llm_gan.utils import batch_local_inference
from llm_gan.utils.parse import parse_tags
from .dataset import StoryDataset


def assess_judge(titles, genres, stories_human, stories_ai, judge_model, tokenizer):
    targets = [random.randint(0,1) for _ in range(len(titles))]
    
    prompts = []
    for title, genre, human_story, ai_story, target in zip(titles, genres, stories_human, stories_ai, targets):
        story_order = (human_story, ai_story) if target == 0 else (ai_story, human_story)
        prompt = llm_generator_discriminator_prompt(title, genre, *story_order)
        prompts.append(prompt)
    
    judge_outputs = batch_local_inference(
        prompts, 
        model=judge_model, 
        tokenizer=tokenizer,
        max_new_tokens=256
    )
    
    parsed_outputs = []
    for output in judge_outputs:
        answer = parse_tags(output, "answer")
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
    
    judge_outputs = batch_local_inference(
        prompts, 
        model=judge_model, 
        tokenizer=tokenizer,
        max_new_tokens=256
    )
    
    parsed_outputs = []
    for output in judge_outputs:
        answer = parse_tags(output, "answer")
        if answer == "1":
            parsed_outputs.append(0)
        elif answer == "2":
            parsed_outputs.append(1)
        else:
            parsed_outputs.append(-1)
    
    correct = [parsed == target for parsed, target in zip(parsed_outputs, targets)]
    return correct, {
        'targets': targets,
        'prompts': prompts,
        'judge_outputs': judge_outputs,
        'parsed_outputs': parsed_outputs,
        'correct': correct
    }


def calculate_rewards(judge_correct, generator_fooled_judge):
    judge_rewards = [1.0 if correct else -0.5 for correct in judge_correct]
    generator_rewards = [1.0 if fooled else -1.0 for fooled in generator_fooled_judge]
    return judge_rewards, generator_rewards


# Training parameters
bs = 1  # Small batch for large model
epochs = 10  # Full training

def collate_fn(batch):
    return batch

dataset = StoryDataset("data/combined_eval_stories.csv")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn)

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'  # Fix the padding warning
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

generator_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
judge_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

generator_optimizer = AdamW(generator_model.parameters(), lr=1e-5)
judge_optimizer = AdamW(judge_model.parameters(), lr=1e-5)

# Setup logging
log_dir = f"training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)

print(f"Starting training with {len(dataset)} samples, {len(dataloader)} batches...")
print(f"Logging to: {log_dir}")
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
        generated_stories_raw = batch_local_inference(
            generator_prompts,
            model=generator_model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.8
        )
        
        generated_stories = []
        for story_text in generated_stories_raw:
            story = parse_tags(story_text, "story")
            if story is None:
                # Better fallback parsing
                if "ASSISTANT>" in story_text:
                    story = story_text.split("ASSISTANT>")[-1].strip()
                else:
                    story = story_text.strip()
                # Clean up any remaining XML/structure
                story = story.replace('</', '').replace('<', '').replace('>', '')
                story = ' '.join(story.split())[:512]  # Clean whitespace and limit length
            generated_stories.append(story)
        
        print("  Judging stories...")
        # Judge the stories (modified to return judge outputs too)
        judge_correct, judge_outputs = assess_judge_with_outputs(
            titles, genres, human_stories, generated_stories, judge_model, tokenizer
        )
        
        generator_fooled_judge = [not correct for correct in judge_correct]
        judge_rewards, generator_rewards = calculate_rewards(
            judge_correct, generator_fooled_judge
        )
        
        accuracy = sum(judge_correct) / len(judge_correct) if judge_correct else 0
        epoch_judge_accuracy += accuracy
        epoch_generator_reward += np.mean(generator_rewards)
        
        print(f"  Results: Judge Acc: {accuracy:.4f}, Avg Gen Reward: {np.mean(generator_rewards):.4f}")
        
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
            'judge_rewards': judge_rewards
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