#!/usr/bin/env python3
"""
Launch script for distributed LLM GAN training with DDP.

Usage:
    # Test with 2 GPUs:
    torchrun --nproc_per_node=2 train_ddp.py --batch_size 16 --epochs 5
    
    # Full 8 GPU training:
    torchrun --nproc_per_node=8 train_ddp.py --batch_size 8 --epochs 100
"""

import argparse
from llm_gan.train.clean_train import train_llm_gan

def main():
    parser = argparse.ArgumentParser(description='Distributed LLM GAN Training')
    
    # Model and data arguments
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B-Instruct',
                        help='Model name from HuggingFace')
    parser.add_argument('--data_path', type=str, default='data/stories.csv',
                        help='Path to training data CSV')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    
    # Story generation arguments
    parser.add_argument('--max_story_length', type=int, default=500,
                        help='Maximum story length in characters')
    parser.add_argument('--min_story_length', type=int, default=100,
                        help='Minimum story length in characters')
    parser.add_argument('--max_agent_tokens', type=int, default=512,
                        help='Max tokens for generator')
    parser.add_argument('--max_judge_tokens', type=int, default=1024,
                        help='Max tokens for judge')
    
    # Logging and checkpointing
    parser.add_argument('--project_name', type=str, default='llm-gan-ddp',
                        help='Wandb project name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Wandb run name (auto-generated if None)')
    parser.add_argument('--save_checkpoints', action='store_true',
                        help='Save model checkpoints')
    
    args = parser.parse_args()
    
    # Launch training
    train_llm_gan(
        model_name=args.model_name,
        data_path=args.data_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_story_length=args.max_story_length,
        min_story_length=args.min_story_length,
        max_agent_tokens=args.max_agent_tokens,
        max_judge_tokens=args.max_judge_tokens,
        project_name=args.project_name,
        run_name=args.run_name,
        save_checkpoints=args.save_checkpoints
    )

if __name__ == '__main__':
    main()