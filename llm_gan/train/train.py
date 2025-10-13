
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model

from llm_gan.prompts import llm_generator_prompt, llm_generator_discriminator_prompt
from llm_gan.utils import batch_local_inference
from llm_gan.utils.parse import parse_tags
from .dataset import StoryDataset


"""
Guess this doesn't super make sense seeing that we aern't doing GRPO?
"""
# def genereate_rollout(model, prompt: list[str], n_rollouts) -> list[str]:

#     pas


def assess_judge(titles, genres, stories_human, stories_ai, judge_model, tokenizer) -> tuple[list[bool], list[float]]:
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
    
    



bs = 64
epochs = 100

dataset = StoryDataset("data/combined_eval_stories.csv")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

generator_model = AutoModelForCausalLM.from_pretrained(model_name)
judge_model = AutoModelForCausalLM.from_pretrained(model_name)

# Create PPO models without PEFT for now (simpler setup)
generator_ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(generator_model)
judge_ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(judge_model)

# Fix generation config
from transformers import GenerationConfig
gen_config = GenerationConfig.from_pretrained(model_name)
generator_ppo_model.generation_config = gen_config
judge_ppo_model.generation_config = gen_config





ppo_config = PPOConfig(
    output_dir="./ppo_output",
    learning_rate=1.41e-5,
    per_device_train_batch_size=bs//4,
    num_mini_batches=4,
    num_ppo_epochs=4,
    gradient_accumulation_steps=1,
    total_episodes=epochs * len(dataloader),
    response_length=200,
)

generator_ppo_trainer = PPOTrainer(
    args=ppo_config,
    processing_class=tokenizer,
    model=generator_ppo_model,
    ref_model=None,
    reward_model=judge_ppo_model,
    train_dataset=dataset,
    value_model=generator_ppo_model,
)

judge_ppo_config = PPOConfig(
    output_dir="./judge_ppo_output",
    learning_rate=1e-5,
    per_device_train_batch_size=bs//4,
    num_mini_batches=4,
    num_ppo_epochs=4,
    gradient_accumulation_steps=1,
    total_episodes=epochs * len(dataloader),
    response_length=200,
)

judge_ppo_trainer = PPOTrainer(
    args=judge_ppo_config,
    processing_class=tokenizer,
    model=judge_ppo_model,
    ref_model=None,
    reward_model=generator_ppo_model,
    train_dataset=dataset,
    value_model=judge_ppo_model,
)


print("Starting training...")
for epoch in range(epochs):
    epoch_judge_loss = 0
    epoch_generator_reward = 0
    
    for batch_idx, batch in enumerate(dataloader):
        titles = [item['title'] for item in batch]
        genres = [item['genre'] for item in batch]
        human_stories = [item['human_story'] for item in batch]
        
        generator_prompts = [llm_generator_prompt(title, genre) for title, genre in zip(titles, genres)]
        
        query_tensors = []
        for prompt in generator_prompts:
            tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            query_tensors.append(tokens['input_ids'].squeeze())
        
        response_tensors = generator_ppo_trainer.generate(
            query_tensors,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id
        )
        
        generated_stories = []
        for response in response_tensors:
            story_text = tokenizer.decode(response, skip_special_tokens=True)
            story = parse_tags(story_text, "story")
            if story is None:
                story = story_text.split("ASSISTANT>")[-1].strip()[:500]
            generated_stories.append(story)
        
        judge_correct = assess_judge(
            titles, genres, human_stories, generated_stories, judge_ppo_model, tokenizer
        )
        
        generator_fooled_judge = [not correct for correct in judge_correct]
        judge_rewards, generator_rewards = calculate_rewards(
            judge_correct, generator_fooled_judge
        )
        
        rewards_tensor = [torch.tensor([reward], dtype=torch.float) for reward in generator_rewards]
        
        gen_stats = generator_ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)
        
        # Train judge with PPO - rewards based on accuracy
        judge_query_tensors = []
        for prompt in [llm_generator_discriminator_prompt(t, g, h, gs) for t, g, h, gs in zip(titles, genres, human_stories, generated_stories)]:
            tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            judge_query_tensors.append(tokens['input_ids'].squeeze())
        
        judge_response_tensors = judge_ppo_trainer.generate(
            judge_query_tensors,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.pad_token_id
        )
        
        judge_rewards_tensor = [torch.tensor([reward], dtype=torch.float) for reward in judge_rewards]
        judge_stats = judge_ppo_trainer.step(judge_query_tensors, judge_response_tensors, judge_rewards_tensor)
        
        epoch_judge_loss += np.mean(judge_rewards)
        epoch_generator_reward += np.mean(generator_rewards)
        
        if batch_idx % 10 == 0:
            accuracy = sum(judge_correct) / len(judge_correct) if judge_correct else 0
            print(f"Epoch {epoch}, Batch {batch_idx}: Judge Acc: {accuracy:.4f}, Avg Judge Reward: {np.mean(judge_rewards):.4f}, Avg Gen Reward: {np.mean(generator_rewards):.4f}")
    
    print(f"Epoch {epoch} Summary - Judge Reward: {epoch_judge_loss/len(dataloader):.4f}, Generator Reward: {epoch_generator_reward/len(dataloader):.4f}")
    
    if epoch % 10 == 0:
        generator_ppo_model.save_pretrained(f"generator_epoch_{epoch}")
        judge_ppo_model.save_pretrained(f"judge_epoch_{epoch}")

print("Training completed!")







