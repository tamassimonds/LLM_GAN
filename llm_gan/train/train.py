
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

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

generator_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
)

judge_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.1,
)

generator_ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(generator_model)
generator_ppo_model = get_peft_model(generator_ppo_model, lora_config)





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

ppo_trainer = PPOTrainer(
    args=ppo_config,
    processing_class=tokenizer,
    model=generator_ppo_model,
    ref_model=None,
    reward_model=judge_model,
    train_dataset=dataset,
    value_model=None,
)

judge_optimizer = torch.optim.AdamW(judge_model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

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
        
        response_tensors = ppo_trainer.generate(
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
            titles, genres, human_stories, generated_stories, judge_model, tokenizer
        )
        
        generator_fooled_judge = [not correct for correct in judge_correct]
        judge_rewards, generator_rewards = calculate_rewards(
            judge_correct, generator_fooled_judge
        )
        
        rewards_tensor = [torch.tensor([reward], dtype=torch.float) for reward in generator_rewards]
        
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)
        
        judge_targets = torch.tensor([1 if correct else 0 for correct in judge_correct], dtype=torch.long)
        
        judge_loss = criterion(torch.randn(len(judge_correct), 2), judge_targets)
        
        judge_optimizer.zero_grad()
        judge_loss.backward()
        judge_optimizer.step()
        
        epoch_judge_loss += judge_loss.item()
        epoch_generator_reward += np.mean(generator_rewards)
        
        if batch_idx % 10 == 0:
            accuracy = sum(judge_correct) / len(judge_correct) if judge_correct else 0
            print(f"Epoch {epoch}, Batch {batch_idx}: Judge Loss: {judge_loss.item():.4f}, Judge Acc: {accuracy:.4f}, Avg Gen Reward: {np.mean(generator_rewards):.4f}")
    
    print(f"Epoch {epoch} Summary - Judge Loss: {epoch_judge_loss/len(dataloader):.4f}, Generator Reward: {epoch_generator_reward/len(dataloader):.4f}")
    
    if epoch % 10 == 0:
        torch.save(generator_model.state_dict(), f"generator_epoch_{epoch}.pt")
        torch.save(judge_model.state_dict(), f"judge_epoch_{epoch}.pt")

print("Training completed!")







