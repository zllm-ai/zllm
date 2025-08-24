#!/usr/bin/env python3
"""
Debug test for batching with detailed output
"""

from vllm.src.model_loader import HuggingFaceModelLoader
from vllm.src.batcher import RequestBatcher

def debug_test():
    """Debug test with detailed output"""
    print("Running debug test...")
    
    # Load model and tokenizer
    model_loader = HuggingFaceModelLoader("gpt2")
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    
    # Create batcher
    batcher = RequestBatcher(model)
    batcher.tokenizer = tokenizer
    
    # Add requests
    batcher.add_request("The capital of France is")
    batcher.add_request("Quantum computing is")
    
    # Process batch
    outputs = batcher.finalize_batch()
    
    if outputs is not None:
        print("✓ Debug test successful")
    else:
        print("✗ Debug test failed")

if __name__ == "__main__":
    debug_test()