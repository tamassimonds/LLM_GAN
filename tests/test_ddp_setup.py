#!/usr/bin/env python3
"""
Simple test script to verify DDP setup works correctly.

Usage:
    # Test with 2 GPUs:
    torchrun --nproc_per_node=2 test_ddp_setup.py
"""

import torch
import torch.distributed as dist
import os

def test_ddp_setup():
    """Test basic DDP functionality."""
    
    # Get distributed environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        print(f"Process {rank}/{world_size} on GPU {local_rank}")
        
        # Initialize process group
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        # Test basic tensor operations
        device = torch.device(f"cuda:{local_rank}")
        tensor = torch.randn(4, 4, device=device) * (rank + 1)
        
        print(f"Rank {rank}: Created tensor with shape {tensor.shape} on {device}")
        
        # Test all-reduce
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"Rank {rank}: After all-reduce, tensor sum: {tensor.sum().item():.2f}")
        
        # Test model creation
        model = torch.nn.Linear(10, 1).to(device)
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[local_rank])
        
        print(f"Rank {rank}: Created model on {device}")
        
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
            
        print(f"Rank {rank}: DDP test completed successfully!")
        
    else:
        print("Single GPU mode - no distributed setup needed")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tensor = torch.randn(4, 4, device=device)
        print(f"Created tensor with shape {tensor.shape} on {device}")
        print("Single GPU test completed successfully!")

if __name__ == '__main__':
    test_ddp_setup()