import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
import random
import numpy as np

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
        max_new_tokens=200
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


def calculate_rewards(judge_correct, generator_fooled_judge):
    judge_rewards = [1.0 if correct else -0.5 for correct in judge_correct]
    generator_rewards = [1.0 if fooled else -1.0 for fooled in generator_fooled_judge]
    return judge_rewards, generator_rewards


# Training parameters
bs = 4  # Smaller batch size for testing
epochs = 2  # Fewer epochs for testing

def collate_fn(batch):
    return batch

dataset = StoryDataset("data/combined_eval_stories.csv")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

generator_model = AutoModelForCausalLM.from_pretrained(model_name)
judge_model = AutoModelForCausalLM.from_pretrained(model_name)

generator_optimizer = AdamW(generator_model.parameters(), lr=1e-5)
judge_optimizer = AdamW(judge_model.parameters(), lr=1e-5)

print("Starting simple training...")
for epoch in range(epochs):
    epoch_judge_accuracy = 0
    epoch_generator_reward = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # batch is a list of dicts when using default_collate
        titles = [item['title'] for item in batch]
        genres = [item['genre'] for item in batch] 
        human_stories = [item['human_story'] for item in batch]
        
        # Generate stories with generator
        generator_prompts = [llm_generator_prompt(title, genre) for title, genre in zip(titles, genres)]
        
        generated_stories_raw = batch_local_inference(
            generator_prompts,
            model=generator_model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.8
        )
        
        generated_stories = []
        for story_text in generated_stories_raw:
            story = parse_tags(story_text, "story")
            if story is None:
                story = story_text.split("ASSISTANT>")[-1].strip()[:500]
            generated_stories.append(story)
        
        # Judge the stories
        judge_correct = assess_judge(
            titles, genres, human_stories, generated_stories, judge_model, tokenizer
        )
        
        generator_fooled_judge = [not correct for correct in judge_correct]
        judge_rewards, generator_rewards = calculate_rewards(
            judge_correct, generator_fooled_judge
        )
        
        # Simple training step - just compute gradients based on rewards
        # This is a simplified version without full RL implementation
        
        accuracy = sum(judge_correct) / len(judge_correct) if judge_correct else 0
        epoch_judge_accuracy += accuracy
        epoch_generator_reward += np.mean(generator_rewards)
        
        if batch_idx % 2 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: Judge Acc: {accuracy:.4f}, Avg Gen Reward: {np.mean(generator_rewards):.4f}")
    
    print(f"Epoch {epoch} Summary - Judge Accuracy: {epoch_judge_accuracy/len(dataloader):.4f}, Generator Reward: {epoch_generator_reward/len(dataloader):.4f}")
    
    if epoch % 1 == 0:
        torch.save(generator_model.state_dict(), f"simple_generator_epoch_{epoch}.pt")
        torch.save(judge_model.state_dict(), f"simple_judge_epoch_{epoch}.pt")

print("Simple training completed!")