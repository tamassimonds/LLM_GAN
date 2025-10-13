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
        max_new_tokens=256,
        batch_size=len(prompts)  # Process all prompts at once
    )
    
    parsed_outputs = []
    for output in judge_outputs:
        # Get all answer tags and take the last valid one
        all_answers = parse_tags(output, "answer")
        
        # If multiple answers, take the last one
        if isinstance(all_answers, list) and len(all_answers) > 0:
            answer = all_answers[-1]  # Take last answer
        else:
            answer = all_answers  # Single answer or None
        
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
        max_new_tokens=256,
        batch_size=len(prompts)  # Process all prompts at once
    )
    
    parsed_outputs = []
    for output in judge_outputs:
        # Get all answer tags and take the last valid one
        all_answers = parse_tags(output, "answer")
        
        # If multiple answers, take the last one
        if isinstance(all_answers, list) and len(all_answers) > 0:
            answer = all_answers[-1]  # Take last answer
        else:
            answer = all_answers  # Single answer or None
        
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


# Training parameters
bs = 1  # Small batch for large model
epochs = 10  # Full training

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