# Distributed Data Parallel (DDP) Training Setup

This branch implements multi-GPU distributed training for the LLM GAN project using PyTorch's DistributedDataParallel.

## Quick Start

### 1. Test DDP Setup
```bash
# Test basic DDP functionality with 2 GPUs
torchrun --nproc_per_node=2 tests/test_ddp_setup.py
```

### 2. Launch Training

#### Option A: Using the launch script (recommended)
```bash
# Quick test (2 GPUs, 5 epochs)
scripts/run_ddp_training.sh test

# Medium training (4 GPUs, 20 epochs)
scripts/run_ddp_training.sh small

# Full training (all GPUs, 100 epochs, save checkpoints)
scripts/run_ddp_training.sh full
```

#### Option B: Direct torchrun commands
```bash
# Test with 2 GPUs
torchrun --nproc_per_node=2 train_ddp.py --batch_size 16 --epochs 5

# Full training with 8 GPUs
torchrun --nproc_per_node=8 train_ddp.py --batch_size 8 --epochs 100 --save_checkpoints

# Full training with frequent checkpointing every 10 steps
torchrun --nproc_per_node=8 train_ddp.py --batch_size 8 --epochs 100 --save_checkpoints --checkpoint_freq 10
```

## Key Changes for DDP

### 1. Model Loading
- Removed `device_map="auto"` 
- Load models to specific GPU based on local rank
- Wrap with `DistributedDataParallel`

### 2. Data Loading
- Added `DistributedSampler` for proper data distribution
- Each GPU processes different batches
- Call `sampler.set_epoch(epoch)` for proper shuffling

### 3. Inference Updates
- Updated `simple_generate()` to handle DDP wrapped models
- Use `model.module` to access underlying model

### 4. Logging and Checkpointing
- Only rank 0 performs logging and saves files
- Benchmark evaluation only runs on rank 0
- Model checkpoints save underlying model state dict
- **Step-based checkpointing**: Save every N steps with `--checkpoint_freq` (default: 20)
- **Epoch-based checkpointing**: Save at end of each epoch (existing behavior)
- Checkpoint files: `generator_step_X.pt`, `judge_step_X.pt`, `generator_epoch_X.pt`, `judge_epoch_X.pt`

## Performance Benefits

With 8 A100 GPUs, you should see:
- ~8x speedup in training time
- Better GPU utilization (distributed across all GPUs)
- Larger effective batch sizes

## Monitoring

- Only rank 0 logs to wandb
- All processes participate in training
- Check `nvidia-smi` to verify all GPUs are being used

## Troubleshooting

### Common Issues
1. **NCCL errors**: Check CUDA/driver compatibility
2. **Memory errors**: Reduce `--batch_size` per GPU
3. **Hanging**: Ensure all processes can communicate

### Debug Commands
```bash
# Check GPU availability
nvidia-smi

# Test with single GPU (fallback)
python llm_gan/train/clean_train.py

# Verbose torchrun output
torchrun --nproc_per_node=2 --log_level=INFO train_ddp.py
```

## File Structure
- `train_ddp.py` - Main training launcher with argument parsing
- `run_ddp_training.sh` - Convenient bash script for common scenarios  
- `test_ddp_setup.py` - Basic DDP functionality test
- `llm_gan/train/clean_train.py` - Updated training code with DDP support
- `llm_gan/train/inference.py` - Updated inference for DDP models