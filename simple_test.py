#!/usr/bin/env python3
"""
Simple test for batching functionality
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm.src.batcher import RequestBatcher

def test_simple_batching():
    """Test batching with a simple model"""
    print("Testing simple batching...")
    
    # Create a simple model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create batcher
    batcher = RequestBatcher(model)
    batcher.tokenizer = tokenizer
    
    # Add requests
    batcher.add_request("The capital of France is")
    batcher.add_request("Quantum computing is")
    
    # Process batch
    outputs = batcher.finalize_batch()
    
    if outputs is not None:
        print("✓ Simple batching successful")
        print(f"Output type: {type(outputs)}")
        return True
    else:
        print("✗ Simple batching failed")
        return False

if __name__ == "__main__":
    test_simple_batching()