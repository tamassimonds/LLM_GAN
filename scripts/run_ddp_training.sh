#!/bin/bash

# DDP Training Launch Scripts for LLM GAN

echo "LLM GAN Distributed Training Launcher"
echo "======================================"

# Check for CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA may not be available."
    exit 1
fi

# Display available GPUs
echo "Available GPUs:"
nvidia-smi --list-gpus

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Function to run training with specified number of GPUs
run_training() {
    local ngpus=$1
    local batch_size=$2
    local epochs=$3
    local extra_args=$4
    
    echo ""
    echo "Starting training with $ngpus GPUs, batch_size=$batch_size, epochs=$epochs"
    echo "Command: torchrun --nproc_per_node=$ngpus scripts/train_ddp.py --batch_size $batch_size --epochs $epochs $extra_args"
    echo ""
    
    torchrun --nproc_per_node=$ngpus scripts/train_ddp.py \
        --batch_size $batch_size \
        --epochs $epochs \
        $extra_args
}

# Parse command line arguments
case "$1" in
    "test")
        echo "Running test with 2 GPUs..."
        run_training 2 16 5 "--project_name llm-gan-ddp-test"
        ;;
    "small")
        echo "Running small training with 4 GPUs..."
        run_training 4 8 20 "--project_name llm-gan-ddp-small"
        ;;
    "full")
        echo "Running full training with all available GPUs..."
        run_training $NUM_GPUS 4 100 "--save_checkpoints --project_name llm-gan-ddp-full"
        ;;
    *)
        echo "Usage: $0 {test|small|full}"
        echo ""
        echo "  test  - Quick test with 2 GPUs, small batch, 5 epochs"
        echo "  small - Medium training with 4 GPUs, 20 epochs"  
        echo "  full  - Full training with all GPUs, 100 epochs, save checkpoints"
        echo ""
        echo "Custom usage:"
        echo "  torchrun --nproc_per_node=N scripts/train_ddp.py [options]"
        exit 1
        ;;
esac