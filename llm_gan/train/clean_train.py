"""Clean LLM GAN training script with wandb logging and DDP support."""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
from .training import calculate_log_probs, reinforce_update, ppo_update
from .benchmark import run_benchmark_evaluation, save_benchmark_results, compare_benchmark_results

import signal
import sys
import time
from contextlib import contextmanager

@contextmanager
def timeout(duration):
    """Context manager for timing out operations."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)

def setup_distributed():
    """Setup distributed training environment with robust error handling."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Running with torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        print(f"Attempting to setup distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        if world_size > 1:
            try:
                # Check CUDA availability first
                if not torch.cuda.is_available():
                    print("ERROR: CUDA not available but distributed training requested")
                    print("Falling back to CPU/single GPU mode")
                    return 0, 1, 0
                
                # Check if the specified GPU exists
                if local_rank >= torch.cuda.device_count():
                    print(f"ERROR: local_rank {local_rank} >= available GPUs {torch.cuda.device_count()}")
                    print("Falling back to single GPU mode")
                    return 0, 1, 0
                
                # Set device before initializing process group
                torch.cuda.set_device(local_rank)
                device = torch.device(f"cuda:{local_rank}")
                
                # Test basic CUDA operations on this device
                try:
                    test_tensor = torch.randn(2, 2, device=device)
                    _ = test_tensor + 1  # Simple operation to verify GPU works
                    print(f"GPU {local_rank} basic functionality verified")
                except Exception as e:
                    print(f"ERROR: GPU {local_rank} failed basic test: {e}")
                    print("Falling back to single GPU mode")
                    return 0, 1, 0
                
                # Initialize process group with timeout and error handling
                print(f"Initializing NCCL process group (rank {rank})...")
                
                with timeout(60):  # 60 second timeout for NCCL init
                    dist.init_process_group(
                        backend='nccl',
                        timeout=torch.distributed.default_pg_timeout
                    )
                
                # Verify the process group is working
                print(f"Testing NCCL communication (rank {rank})...")
                test_tensor = torch.ones(1, device=device) * rank
                dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
                expected_sum = sum(range(world_size))
                
                if abs(test_tensor.item() - expected_sum) > 1e-6:
                    raise RuntimeError(f"NCCL communication test failed: got {test_tensor.item()}, expected {expected_sum}")
                
                print(f"NCCL setup successful (rank {rank})")
                return rank, world_size, local_rank
                
            except TimeoutError as e:
                print(f"ERROR: NCCL initialization timed out: {e}")
                print("This usually indicates network/driver issues")
                print("Falling back to single GPU mode")
                cleanup_distributed_safe()
                return 0, 1, 0
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                if 'nccl' in error_msg:
                    print(f"ERROR: NCCL error during setup: {e}")
                    print("Common causes:")
                    print("- Incompatible CUDA/driver versions")
                    print("- Network connectivity issues between GPUs")
                    print("- Insufficient GPU memory")
                    print("- Mixed GPU architectures")
                elif 'cuda' in error_msg:
                    print(f"ERROR: CUDA error during setup: {e}")
                    print("Check GPU availability and memory")
                else:
                    print(f"ERROR: Distributed setup failed: {e}")
                
                print("Falling back to single GPU mode")
                cleanup_distributed_safe()
                return 0, 1, 0
                
            except Exception as e:
                print(f"ERROR: Unexpected error during distributed setup: {e}")
                print(f"Error type: {type(e).__name__}")
                print("Falling back to single GPU mode")
                cleanup_distributed_safe()
                return 0, 1, 0
        else:
            print("Single GPU mode requested")
            return rank, world_size, local_rank
    else:
        # Single GPU fallback
        print("No distributed environment detected, using single GPU mode")
        return 0, 1, 0

def cleanup_distributed_safe():
    """Safely cleanup distributed training with error handling."""
    try:
        if dist.is_initialized():
            print("Cleaning up distributed process group...")
            dist.destroy_process_group()
            print("Distributed cleanup completed")
    except Exception as e:
        print(f"Warning: Error during distributed cleanup: {e}")
        # Don't raise - this is cleanup, we want to continue


def extract_output_from_generation(generated_text: str, title: str = "", genre: str = "") -> str:
    """Extract output content from generated text."""
    
    def is_placeholder_text(text: str) -> bool:
        """Check if text is placeholder content."""
        if not text:
            return True
        
        text_lower = text.lower().strip()
        
        # Check if text is too short to be meaningful
        if len(text.strip()) < 20:
            return True
        
        # Only check for exact placeholder matches, not substrings that might be in examples
        exact_placeholder_patterns = [
            "your story here",
            "**your story here**",
            "*your story here*",
            "write your story",
            "story here",
            "[story]",
            "insert story here",
            "add your story",
            "put your story here"
        ]
        
        # Check if text IS a placeholder pattern (not just contains it)
        for pattern in exact_placeholder_patterns:
            if text_lower == pattern or text_lower == pattern.strip('*'):
                return True
        
        # Check for standalone placeholder phrases (surrounded by whitespace/punctuation)
        import re
        for pattern in ["your story here", "story here"]:
            if re.search(r'\b' + re.escape(pattern) + r'\b', text_lower) and len(text.strip()) < 50:
                return True
            
        return False
    
    # First try to get output from <output> tags, but exclude placeholders
    all_outputs = parse_tags(generated_text, "output")
    output = None
    
    if isinstance(all_outputs, list):
        # Multiple output tags found - filter out placeholders
        for o in all_outputs:
            if o and not is_placeholder_text(o) and len(o) > 50:
                output = o
                break
    elif all_outputs and not is_placeholder_text(all_outputs):
        output = all_outputs
    
    if output is None:
        # Try to get content from <OUTPUT> tags (uppercase)
        uppercase_output = parse_tags(generated_text, "OUTPUT")
        if uppercase_output and not is_placeholder_text(uppercase_output):
            output = uppercase_output
    
    if output is None:
        # Look for content after assistant header - this is where the actual output is
        if "assistant" in generated_text:
            # Split on assistant and get the last part
            parts = generated_text.split("assistant")
            if len(parts) > 1:
                content = parts[-1].strip()
                # Take the first substantial paragraph
                lines = content.split('\n')
                output_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('<') and len(line) > 10:
                        output_lines.append(line)
                if output_lines:
                    potential_output = ' '.join(output_lines)
                    if not is_placeholder_text(potential_output):
                        output = potential_output
    
    # Final fallback - but still check for placeholders
    if not output or is_placeholder_text(output):
        fallback_output = f"A {genre.lower() if genre else 'generated'} output about {title if title else 'this topic'}."
        output = fallback_output
    
    # Clean up and limit length
    if output and len(output) > 10:
        output = ' '.join(output.split())[:512]
    else:
        output = f"A {genre.lower() if genre else 'generated'} output titled '{title if title else 'Untitled'}'."
        
    return output


def train_llm_gan(
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    data_path: str = "data/stories.csv",
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-5,
    max_story_length: int = 500,
    min_story_length: int = 100,
    max_agent_tokens: int = 512,
    max_judge_tokens: int = 1024,
    project_name: str = "llm-gan",
    run_name: str = None,
    save_checkpoints: bool = False,
    checkpoint_freq: int = 20,
    use_ppo: bool = False,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    domain: str = None
):
    """Main training function for LLM GAN with DDP support."""
    
    # Import domain system
    from llm_gan.domains import DomainRegistry
    
    # Create domain instance
    domain_obj = DomainRegistry.create_domain(csv_path=data_path, domain_name=domain)
    if hasattr(domain_obj, '__class__'):
        print(f"Using domain: {domain_obj.__class__.__name__}")
    
    try:
        # Setup distributed training with error handling
        rank, world_size, local_rank = setup_distributed()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        
        print(f"Training setup: rank={rank}, world_size={world_size}, device={device}")
        
        # Only initialize wandb on rank 0
        if rank == 0:
            import wandb
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
                    "max_judge_tokens": max_judge_tokens,
                    "save_checkpoints": save_checkpoints,
                    "checkpoint_freq": checkpoint_freq,
                    "use_ppo": use_ppo,
                    "clip_eps": clip_eps,
                    "entropy_coef": entropy_coef,
                    "world_size": world_size
                }
            )
            
            # Setup logging (only on rank 0)
            log_dir = f"logs/training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_dir = None
        
        # Load dataset
        def collate_fn(batch):
            return batch
        
        # Use domain-aware dataset
        from llm_gan.train.domain_dataset import DomainDataset
        dataset = DomainDataset(domain_obj, data_path, min_output_length=min_story_length)
        
        # Setup distributed sampler
        if world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
        else:
            sampler = None
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        # Load models and tokenizer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load models without device_map for DDP
        generator_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16
        )
        judge_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16
        )
        
        # Move models to device
        generator_model = generator_model.to(device)
        judge_model = judge_model.to(device)
        
        # Create reference models for PPO (if enabled)
        if use_ppo:
            # Reference models are copies that stay fixed during updates
            generator_ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16
            ).to(device)
            judge_ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16
            ).to(device)
            
            # Set reference models to eval mode (no training)
            generator_ref_model.eval()
            judge_ref_model.eval()
            
            # Wrap reference models with DDP if needed (for inference only)
            if world_size > 1:
                generator_ref_model = DDP(generator_ref_model, device_ids=[local_rank])
                judge_ref_model = DDP(judge_ref_model, device_ids=[local_rank])
        else:
            generator_ref_model = None
            judge_ref_model = None
        
        # Wrap models with DDP if using multiple GPUs
        if world_size > 1:
            generator_model = DDP(generator_model, device_ids=[local_rank])
            judge_model = DDP(judge_model, device_ids=[local_rank])
        
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

        if rank == 0:
            print(f"Starting training with {len(dataset)} samples, {len(dataloader)} batches...")
            print(f"World size: {world_size}")
            print(f"Logging to: {log_dir}")
        
        # Training loop
        step = 0
        for epoch in range(epochs):
            # Set epoch for distributed sampler
            if world_size > 1:
                sampler.set_epoch(epoch)
            
            # Update reference models for PPO (copy current state)
            if use_ppo:
                # Get the actual model (unwrap DDP if needed)
                gen_model_for_copy = generator_model.module if world_size > 1 else generator_model
                judge_model_for_copy = judge_model.module if world_size > 1 else judge_model
                ref_gen_model = generator_ref_model.module if world_size > 1 else generator_ref_model
                ref_judge_model = judge_ref_model.module if world_size > 1 else judge_ref_model
                
                # Copy current model state to reference models
                ref_gen_model.load_state_dict(gen_model_for_copy.state_dict())
                ref_judge_model.load_state_dict(judge_model_for_copy.state_dict())
                
                if rank == 0:
                    print(f"Updated reference models for epoch {epoch}")
            
            epoch_judge_accuracy = 0
            epoch_generator_reward = 0
            
            for batch_idx, batch in enumerate(dataloader):
                print(f"Processing batch {batch_idx+1}/{len(dataloader)}...")
                
                # Extract data based on domain
                human_outputs = []
                for item in batch:
                    if 'human_output' in item:
                        human_outputs.append(item['human_output'])
                    elif 'human_story' in item:
                        human_outputs.append(item['human_story'])
                    elif 'human_solution' in item:
                        human_outputs.append(item['human_solution'])
                
                # Generate prompts using domain
                generator_prompts = [domain_obj.get_generator_prompt(item) for item in batch]
                
                print("  Generating outputs...")
                generated_outputs_raw = simple_generate(
                    model=generator_model,
                    tokenizer=tokenizer,
                    prompts=generator_prompts,
                    max_new_tokens=max_agent_tokens,
                    temperature=0.8,
                    batch_size=batch_size
                )
                
                # Extract outputs from generated text using domain
                generated_outputs = [
                    domain_obj.extract_output(output_text) 
                    for output_text in generated_outputs_raw
                ]
                
                print("  Judging outputs...")
                # Clip outputs to equal length for fair comparison, but enforce minimum length
                min_words = 128  # Minimum word count to prevent gaming
                
                clipped_human_outputs = []
                clipped_generated_outputs = []
                
                for human_output, generated_output in zip(human_outputs, generated_outputs):
                    # Clip to max length first
                    human_clipped = human_output[:max_story_length]
                    generated_clipped = generated_output[:max_story_length]
                    
                    # Count words (rough approximation: chars/5)
                    human_word_count = len(human_clipped.split())
                    generated_word_count = len(generated_clipped.split())
                    
                    # If either story is too short, penalize by padding or truncating to min viable comparison
                    if human_word_count < min_words or generated_word_count < min_words:
                        # Use max_story_length for both to ensure fair comparison of longer content
                        clipped_human_outputs.append(human_clipped)
                        clipped_generated_outputs.append(generated_clipped)
                    else:
                        # Both stories meet minimum length, clip to same actual length
                        min_length = min(len(human_clipped), len(generated_clipped))
                        clipped_human_outputs.append(human_clipped[:min_length])
                        clipped_generated_outputs.append(generated_clipped[:min_length])
                
                # Judge the outputs using domain prompts
                # For now, we'll pass batch items to reconstruct prompts
                judge_correct, judge_outputs = assess_judge_with_outputs(
                    batch, domain_obj, clipped_human_outputs, clipped_generated_outputs, 
                    judge_model, tokenizer, max_judge_tokens
                )
                
                generator_fooled_judge = [not correct for correct in judge_correct]
                judge_rewards, generator_rewards = calculate_rewards(
                    judge_correct, generator_fooled_judge, judge_outputs['parsed_outputs']
                )
                
                accuracy = sum(judge_correct) / len(judge_correct) if judge_correct else 0
                epoch_judge_accuracy += accuracy
                epoch_generator_reward += np.mean(generator_rewards)
                
                print(f"  Results: Judge Acc: {accuracy:.4f}, Avg Gen Reward: {np.mean(generator_rewards):.4f}")
                
                # Training updates (PPO or REINFORCE)
                if use_ppo:
                    print("  Training generator with PPO...")
                    generator_model.train()
                    generator_log_probs, generator_ref_log_probs = calculate_log_probs(
                        generator_model, tokenizer, generator_prompts, generated_stories, generator_ref_model
                    )
                    generator_loss = ppo_update(
                        generator_model, generator_optimizer, generator_log_probs, generator_ref_log_probs, 
                        generator_rewards, clip_eps, entropy_coef
                    )
                    
                    print("  Training judge with PPO...")
                    judge_model.train()
                    judge_log_probs, judge_ref_log_probs = calculate_log_probs(
                        judge_model, tokenizer, judge_outputs['prompts'], judge_outputs['judge_outputs'], judge_ref_model
                    )
                    judge_loss = ppo_update(
                        judge_model, judge_optimizer, judge_log_probs, judge_ref_log_probs, 
                        judge_rewards, clip_eps, entropy_coef
                    )
                else:
                    print("  Training generator with REINFORCE...")
                    generator_model.train()
                    generator_log_probs = calculate_log_probs(generator_model, tokenizer, generator_prompts, generated_stories)
                    generator_loss = reinforce_update(generator_model, generator_optimizer, generator_log_probs, generator_rewards)
                    
                    print("  Training judge with REINFORCE...")
                    judge_model.train()
                    judge_log_probs = calculate_log_probs(judge_model, tokenizer, judge_outputs['prompts'], judge_outputs['judge_outputs'])
                    judge_loss = reinforce_update(judge_model, judge_optimizer, judge_log_probs, judge_rewards, max_judge_tokens)
                
                print(f"  Training losses - Generator: {generator_loss:.4f}, Judge: {judge_loss:.4f}")
                
                # Clear gradients after training
                generator_optimizer.zero_grad()
                judge_optimizer.zero_grad()
                
                # Log to wandb (only on rank 0)
                if rank == 0:
                    import wandb
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
                    
                    # Save detailed logs (domain-agnostic format)
                    batch_log = {
                        'domain': domain_obj.__class__.__name__.replace('Domain', '').lower(),
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'step': step,
                        'batch_items': batch,
                        'human_outputs': human_outputs,
                        'generator_prompts': generator_prompts,
                        'generated_outputs_raw': generated_outputs_raw,
                        'generated_outputs': generated_outputs,
                        'judge_data': judge_outputs,
                        'judge_accuracy': accuracy,
                        'generator_rewards': generator_rewards,
                        'judge_rewards': judge_rewards,
                        'generator_loss': generator_loss,
                        'judge_loss': judge_loss
                    }
                    
                    with open(f"{log_dir}/batch_{epoch}_{batch_idx}.json", 'w') as f:
                        json.dump(batch_log, f, indent=2)
                
                # Run benchmark evaluation every 10 steps (only on rank 0)
                if step % 10 == 0 and rank == 0:
                    print(f"\nRunning benchmark evaluation at step {step}...")
                    # Get underlying model for DDP
                    eval_judge_model = judge_model.module if world_size > 1 else judge_model
                    benchmark_results = run_benchmark_evaluation(
                        eval_judge_model, tokenizer, 
                        max_judge_tokens=max_judge_tokens,
                        max_story_length=max_story_length
                    )
                    
                    # Log benchmark to wandb
                    import wandb
                    wandb.log({
                        "step": step,
                        "benchmark_accuracy": benchmark_results['overall_accuracy'],
                        "benchmark_failed_rate": benchmark_results['failed_rate']
                    })
                    
                    print(f"Benchmark results: Accuracy={benchmark_results['overall_accuracy']:.4f}, Failed Rate={benchmark_results['failed_rate']:.4f}")
                
                step += 1
                
                # Save step-based checkpoints (only on rank 0)
                if rank == 0 and save_checkpoints and step % checkpoint_freq == 0:
                    # Get underlying model state dict for DDP
                    gen_state_dict = generator_model.module.state_dict() if world_size > 1 else generator_model.state_dict()
                    judge_state_dict = judge_model.module.state_dict() if world_size > 1 else judge_model.state_dict()
                    
                    torch.save(gen_state_dict, f"generator_step_{step}.pt")
                    torch.save(judge_state_dict, f"judge_step_{step}.pt")
                    print(f"Saved model checkpoints at step {step}")
            
            # Epoch summary
            epoch_summary = {
                'epoch': epoch,
                'judge_accuracy': epoch_judge_accuracy/len(dataloader),
                'generator_reward': epoch_generator_reward/len(dataloader),
                'total_batches': len(dataloader)
            }
            
            if rank == 0:
                print(f"Epoch {epoch} Summary - Judge Accuracy: {epoch_summary['judge_accuracy']:.4f}, Generator Reward: {epoch_summary['generator_reward']:.4f}")
                
                # Log epoch summary to wandb
                import wandb
                wandb.log({
                    "epoch_judge_accuracy": epoch_summary['judge_accuracy'],
                    "epoch_generator_reward": epoch_summary['generator_reward']
                })
                
                # Save epoch summary
                with open(f"{log_dir}/epoch_{epoch}_summary.json", 'w') as f:
                    json.dump(epoch_summary, f, indent=2)
                
                # Save model checkpoints (if enabled)
                if save_checkpoints:
                    # Get underlying model state dict for DDP
                    gen_state_dict = generator_model.module.state_dict() if world_size > 1 else generator_model.state_dict()
                    judge_state_dict = judge_model.module.state_dict() if world_size > 1 else judge_model.state_dict()
                    
                    torch.save(gen_state_dict, f"generator_epoch_{epoch}.pt")
                    torch.save(judge_state_dict, f"judge_epoch_{epoch}.pt")
                    print(f"Saved model checkpoints for epoch {epoch}")
                else:
                    print(f"Skipping checkpoint save for epoch {epoch} (save_checkpoints=False)")

        if rank == 0:
            print("Training completed successfully!")
            import wandb
            wandb.finish()
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        raise
    except RuntimeError as e:
        error_msg = str(e).lower()
        if 'nccl' in error_msg or 'cuda' in error_msg:
            print(f"CUDA/NCCL error during training: {e}")
            print("This may be due to:")
            print("- GPU memory exhaustion")
            print("- Network issues between GPUs")
            print("- Hardware problems")
            raise
        else:
            print(f"Runtime error during training: {e}")
            raise
    except Exception as e:
        print(f"Unexpected error during training: {e}")
        print(f"Error type: {type(e).__name__}")
        raise
        
    finally:
        # Always cleanup, even if there was an error
        cleanup_distributed_safe()
        
        if rank == 0:
            try:
                import wandb
                wandb.finish()
            except:
                pass  # Don't fail on wandb cleanup


if __name__ == "__main__":
    train_llm_gan(epochs=10, batch_size=32)  # Quick test