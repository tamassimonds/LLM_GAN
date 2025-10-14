# LLM GAN Project

A Generative Adversarial Network implementation where two LLMs (generator and judge) train adversarially to generate high-quality stories.

## Project Structure

```
├── llm_gan/                 # Main package
│   ├── train/              # Training modules
│   ├── eval/               # Evaluation system
│   └── utils/              # Utilities
├── scripts/                # Standalone scripts
│   ├── train_ddp.py        # Distributed training launcher
│   ├── evaluate_generators.py  # Story evaluation tool
│   └── run_ddp_training.sh # Training convenience script
├── tests/                  # Test suite
├── docs/                   # Documentation
│   ├── DDP_README.md       # Distributed training guide
│   └── EVALUATION_README.md # Evaluation system guide
├── dev/                    # Development files
│   ├── *.ipynb            # Jupyter notebooks
│   └── notes.txt          # Development notes
├── data/                   # Training data
└── logs/                   # Training logs
```

## Quick Start

### 1. Training
```bash
# Single GPU training
python -m llm_gan.train.clean_train

# Multi-GPU distributed training
scripts/run_ddp_training.sh test
```

### 2. Evaluation
```bash
# Compare two training runs
python scripts/evaluate_generators.py logs/run_A logs/run_B --num_pairs 50
```

## Features

- **Adversarial Training**: Generator and judge models train against each other
- **Multi-GPU Support**: Distributed training with DDP across multiple GPUs
- **Evaluation System**: GPT-based evaluation of story quality with statistical analysis
- **REINFORCE Training**: Policy gradient optimization for both models
- **Comprehensive Logging**: Detailed training logs and wandb integration

## Documentation

- **[Distributed Training Guide](docs/DDP_README.md)** - Multi-GPU training setup
- **[Evaluation System Guide](docs/EVALUATION_README.md)** - Story comparison evaluation

## Development

Development files and notebooks are in the `dev/` directory. Tests are in `tests/`.

```bash
# Run tests
python -m pytest tests/

# Install development dependencies
pip install -r requirements_eval.txt
```