"""Clean LLM GAN training script with wandb logging."""

import torch
import wandb
import numpy as np
import json
import os
from datetime import datetime
from torch.optim import AdamW

from llm_gan.prompts import llm_generator_prompt
from llm_gan.utils.parse import parse_tags
from .dataset import StoryDataset
from .inference import simple_generate
from .evaluation import assess_judge_with_outputs, calculate_rewards
from .training import calculate_log_probs, reinforce_update


def extract_story_from_generation(story_text: str, title: str, genre: str) -> str:
    """Extract story content from generated text."""
    
    def is_placeholder_text(text: str) -> bool:
        """Check if text is placeholder content."""
        if not text:
            return True
        
        text_lower = text.lower().strip()
        
        # Common placeholder patterns
        placeholder_patterns = [
            "your story here",
            "story here", 
            "write your story",
            "**your story here**",
            "*your story here*",
            "[story]",
            "[placeholder",
            "insert story",
            "add story",
            "story content"
        ]
        
        # Check if text matches any placeholder pattern
        for pattern in placeholder_patterns:
            if pattern in text_lower:
                print(f"PLACEHOLDER DETECTED: '{text}' matches pattern '{pattern}'")
                return True
        
        # Check if text is too short to be meaningful
        if len(text.strip()) < 20:
            print(f"PLACEHOLDER DETECTED: Text too short: '{text}'")
            return True
            
        return False
    
    # First try to get story from <story> tags, but exclude placeholders
    all_stories = parse_tags(story_text, "story")
    story = None
    
    if isinstance(all_stories, list):
        # Multiple story tags found - filter out placeholders
        for s in all_stories:
            if s and not is_placeholder_text(s) and len(s) > 50:
                story = s
                break
    elif all_stories and not is_placeholder_text(all_stories):
        story = all_stories
    
    if story is None:
        # Try to get content from <STORY> tags (uppercase)
        uppercase_story = parse_tags(story_text, "STORY")
        if uppercase_story and not is_placeholder_text(uppercase_story):
            story = uppercase_story
    
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
                    potential_story = ' '.join(story_lines)
                    if not is_placeholder_text(potential_story):
                        story = potential_story
    
    # Final fallback - but still check for placeholders
    if not story or is_placeholder_text(story):
        fallback_story = f"A {genre.lower()} story about {title}."
        print(f"FALLBACK USED: Generated placeholder content, using fallback: '{fallback_story}'")
        story = fallback_story
    
    # Clean up and limit length
    if story and len(story) > 10:
        story = ' '.join(story.split())[:512]
    else:
        story = f"A {genre.lower()} story titled '{title}'."
        
    return story


def train_llm_gan(
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    data_path: str = "data/ai_eval_stories.csv",
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-5,
    max_story_length: int = 500,
    min_story_length: int = 100,
    max_agent_tokens: int = 512,
    max_judge_tokens: int = 1024,
    project_name: str = "llm-gan",
    run_name: str = None
):
    """Main training function for LLM GAN."""
    
    # Initialize wandb
    if run_name is None:
        run_name = f"llm_gan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "max_story_length": max_story_length,
            "min_story_length": min_story_length,
            "max_agent_tokens": max_agent_tokens,
            "max_judge_tokens": max_judge_tokens
        }
    )
    
    # Setup logging
    log_dir = f"logs/training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Load dataset
    def collate_fn(batch):
        return batch
    
    dataset = StoryDataset(data_path, min_story_length=min_story_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Load models and tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    generator_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    judge_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Setup optimizers
    generator_optimizer = AdamW(generator_model.parameters(), lr=learning_rate)
    judge_optimizer = AdamW(judge_model.parameters(), lr=learning_rate)
    
    # Test model integrity
    print("Testing model integrity...")
    for name, param in generator_model.named_parameters():
        if not torch.isfinite(param).all():
            print(f"ERROR: Generator model parameter {name} contains inf/nan values at startup!")
            
    for name, param in judge_model.named_parameters():
        if not torch.isfinite(param).all():
            print(f"ERROR: Judge model parameter {name} contains inf/nan values at startup!")

    print(f"Starting training with {len(dataset)} samples, {len(dataloader)} batches...")
    print(f"Logging to: {log_dir}")
    
    # Training loop
    step = 0
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
                max_new_tokens=max_agent_tokens,
                temperature=0.8,
                batch_size=batch_size
            )
            
            # Extract stories from generated text
            generated_stories = [
                extract_story_from_generation(story_text, titles[i], genres[i]) 
                for i, story_text in enumerate(generated_stories_raw)
            ]
            
            print("  Judging stories...")
            # Clip stories to equal length for fair comparison
            clipped_human_stories = [story[:max_story_length] for story in human_stories]
            clipped_generated_stories = [story[:max_story_length] for story in generated_stories]
            
            # Judge the stories
            judge_correct, judge_outputs = assess_judge_with_outputs(
                titles, genres, clipped_human_stories, clipped_generated_stories, 
                judge_model, tokenizer, max_judge_tokens
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
            generator_model.train()
            generator_log_probs = calculate_log_probs(generator_model, tokenizer, generator_prompts, generated_stories)
            generator_loss = reinforce_update(generator_model, generator_optimizer, generator_log_probs, generator_rewards)
            
            print("  Training judge with REINFORCE...")
            judge_model.train()
            judge_log_probs = calculate_log_probs(judge_model, tokenizer, judge_outputs['prompts'], judge_outputs['judge_outputs'])
            judge_loss = reinforce_update(judge_model, judge_optimizer, judge_log_probs, judge_rewards)
            
            print(f"  Training losses - Generator: {generator_loss:.4f}, Judge: {judge_loss:.4f}")
            
            # Clear gradients after training
            generator_optimizer.zero_grad()
            judge_optimizer.zero_grad()
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "batch": batch_idx,
                "step": step,
                "judge_accuracy": accuracy,
                "generator_reward_mean": np.mean(generator_rewards),
                "generator_loss": generator_loss,
                "judge_loss": judge_loss,
                "judge_reward_mean": np.mean(judge_rewards)
            })
            
            # Save detailed logs
            batch_log = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'step': step,
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
            
            step += 1
        
        # Epoch summary
        epoch_summary = {
            'epoch': epoch,
            'judge_accuracy': epoch_judge_accuracy/len(dataloader),
            'generator_reward': epoch_generator_reward/len(dataloader),
            'total_batches': len(dataloader)
        }
        
        print(f"Epoch {epoch} Summary - Judge Accuracy: {epoch_summary['judge_accuracy']:.4f}, Generator Reward: {epoch_summary['generator_reward']:.4f}")
        
        # Log epoch summary to wandb
        wandb.log({
            "epoch_judge_accuracy": epoch_summary['judge_accuracy'],
            "epoch_generator_reward": epoch_summary['generator_reward']
        })
        
        # Save epoch summary
        with open(f"{log_dir}/epoch_{epoch}_summary.json", 'w') as f:
            json.dump(epoch_summary, f, indent=2)
        
        # Save model checkpoints
        torch.save(generator_model.state_dict(), f"generator_epoch_{epoch}.pt")
        torch.save(judge_model.state_dict(), f"judge_epoch_{epoch}.pt")

    print("Training completed successfully!")
    wandb.finish()


if __name__ == "__main__":
    train_llm_gan()