#!/usr/bin/env python3
"""
Direct test of the batcher component
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm.src.batcher import RequestBatcher

def test_batcher_directly():
    """Test the batcher directly with a real model"""
    print("Testing batcher directly...")
    
    # Load model and tokenizer directly
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create batcher with the model
    batcher = RequestBatcher(model)
    batcher.tokenizer = tokenizer
    
    # Add requests
    batcher.add_request("The capital of France is")
    batcher.add_request("Quantum computing is")
    
    # Process batch
    outputs = batcher.finalize_batch()
    
    if outputs is not None:
        print("✓ Batcher test successful")
        print(f"Output type: {type(outputs)}")
        return True
    else:
        print("✗ Batcher test failed")
        return False

if __name__ == "__main__":
    test_batcher_directly()