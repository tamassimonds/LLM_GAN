# LLM GAN Project

A Generative Adversarial Network implementation where two LLMs (generator and judge) train adversarially to generate high-quality content across multiple domains.

## Project Structure

```
├── llm_gan/                 # Main package
│   ├── domains/            # Multi-domain support (creative writing, proofs, etc.)
│   ├── train/              # Training modules
│   ├── eval/               # Evaluation system
│   └── utils/              # Utilities
├── scripts/                # User-facing scripts
│   ├── train_ddp.py        # Main training entry point
│   ├── evaluate_generators.py  # Evaluation tool
│   ├── run_ddp_training.sh # Training convenience script
│   ├── set_api_keys.sh     # API key setup
│   └── utils/              # Development utilities
├── tests/                  # Test suite
├── docs/                   # Documentation
├── dev/                    # Development notebooks
├── data/                   # Training datasets
└── logs/                   # Training outputs
```

## Quick Start

### 1. Training

#### Creative Writing (Stories)
```bash
# Single GPU training
uv run python scripts/train_ddp.py --data_path data/stories.csv --epochs 10

# Multi-GPU distributed training (2 GPUs)
uv run torchrun --nproc_per_node=2 scripts/train_ddp.py --data_path data/stories.csv --batch_size 16

# Full multi-GPU training (8 GPUs)
uv run torchrun --nproc_per_node=8 scripts/train_ddp.py --data_path data/stories.csv --batch_size 8 --epochs 100
```

#### Mathematical Proofs
```bash
# Single GPU training
uv run python scripts/train_ddp.py --data_path data/proofs.csv --epochs 10

# Multi-GPU training
uv run torchrun --nproc_per_node=2 scripts/train_ddp.py --data_path data/proofs.csv --batch_size 16
```

#### Advanced Training Options
```bash
# PPO training instead of REINFORCE
uv run python scripts/train_ddp.py --data_path data/stories.csv --use_ppo --clip_eps 0.2

# Custom model and parameters
uv run python scripts/train_ddp.py \
  --model_name "meta-llama/Llama-3.2-3B-Instruct" \
  --data_path data/stories.csv \
  --batch_size 16 \
  --learning_rate 5e-6 \
  --epochs 50
```

### 2. Evaluation
```bash
# Compare two training runs
uv run python scripts/evaluate_generators.py logs/run_A logs/run_B --num_pairs 50

# Human vs AI benchmark
uv run python scripts/utils/benchmark_vs_human.py batch_1_44.json --save_dir logs/evals/
```

## Features

- **Multi-Domain Support**: Train on creative writing, mathematical proofs, or custom domains
- **Adversarial Training**: Generator and judge models train against each other
- **Multi-GPU Support**: Distributed training with DDP across multiple GPUs
- **PPO & REINFORCE**: Advanced policy gradient optimization methods
- **GPT Evaluation**: Batch evaluation system using GPT models for quality assessment
- **Domain Auto-Detection**: Automatically detects domain from CSV column structure
- **Comprehensive Logging**: Detailed training logs and wandb integration

## Supported Domains

### Creative Writing
- **Dataset**: `data/stories.csv`
- **Columns**: `title`, `genre`, `human_story`
- **Features**: Story generation with genre and title conditioning

### Mathematical Proofs  
- **Dataset**: `data/proofs.csv`
- **Columns**: `problem`, `human_solution`, `source`
- **Features**: Mathematical proof generation and evaluation

### Custom Domains
Extend the system by creating new domain classes in `llm_gan/domains/`. See existing implementations for reference.

## Training Commands Reference

```bash
# Basic training commands
uv run python scripts/train_ddp.py --data_path <dataset> [options]

# Common options:
--epochs 10              # Number of training epochs
--batch_size 32          # Batch size per GPU
--learning_rate 1e-5     # Learning rate
--use_ppo               # Use PPO instead of REINFORCE
--save_checkpoints      # Save model checkpoints
--project_name "my-run" # Wandb project name
```

## Documentation

- **[Distributed Training Guide](docs/DDP_README.md)** - Multi-GPU training setup
- **[Evaluation System Guide](docs/EVALUATION_README.md)** - Story comparison evaluation

## Development

```bash
# Setup development environment
uv sync

# Run tests
uv run python -m pytest tests/

# Development utilities
ls scripts/utils/
```