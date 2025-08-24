#!/usr/bin/env python3
"""
Debug script for batching dtype issues
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm.src.batcher import RequestBatcher

def debug_batching():
    """Debug batching dtype issues"""
    print("Debugging batching...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Test tokenization
    text = "The capital of France is"
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    print(f"Tokenized input shape: {tokens.input_ids.shape}")
    print(f"Tokenized input dtype: {tokens.input_ids.dtype}")
    print(f"Tokenized input device: {tokens.input_ids.device}")
    
    # Move to same device as model
    input_ids = tokens.input_ids.to(next(model.parameters()).device)
    print(f"Moved input dtype: {input_ids.dtype}")
    
    # Try direct inference
    try:
        with torch.no_grad():
            outputs = model(input_ids)
        print(f"Direct inference successful, output type: {type(outputs)}")
    except Exception as e:
        print(f"Direct inference failed: {e}")

if __name__ == "__main__":
    debug_batching()