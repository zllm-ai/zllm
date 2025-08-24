#!/usr/bin/env python3
"""
Debug batcher implementation
"""

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def debug_batcher_implementation():
    """Debug the batcher implementation step by step"""
    print("Debugging batcher implementation...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Simulate what happens in the batcher
    inputs = []
    
    # Add requests
    texts = ["The capital of France is", "Quantum computing is"]
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids[0]
        tokens = tokens.to(next(model.parameters()).device).long()
        inputs.append(tokens)
        print(f"Tokens for '{text}': shape={tokens.shape}, dtype={tokens.dtype}")
    
    # Pad inputs
    max_len = max(len(tensor) for tensor in inputs)
    print(f"Max length: {max_len}")
    
    # Pad all sequences
    padded_inputs = []
    for tensor in inputs:
        if len(tensor) < max_len:
            padded_tensor = F.pad(tensor, (0, max_len - len(tensor)), value=tokenizer.pad_token_id or 0)
        else:
            padded_tensor = tensor[:max_len]
        print(f"Padded tensor: shape={padded_tensor.shape}, dtype={padded_tensor.dtype}")
        padded_inputs.append(padded_tensor)
    
    # Stack tensors
    inputs_tensor = torch.stack(padded_inputs).long()
    print(f"Stacked tensor: shape={inputs_tensor.shape}, dtype={inputs_tensor.dtype}")
    
    # Move to device
    inputs_tensor = inputs_tensor.to(next(model.parameters()).device)
    print(f"Final tensor: shape={inputs_tensor.shape}, dtype={inputs_tensor.dtype}, device={inputs_tensor.device}")
    
    # Try inference
    try:
        with torch.no_grad():
            outputs = model(inputs_tensor)
        print(f"Inference successful, output type: {type(outputs)}")
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    debug_batcher_implementation()