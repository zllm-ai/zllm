#!/usr/bin/env python3
"""
Debug model loader dtype
"""

from vllm.src.model_loader import HuggingFaceModelLoader

def debug_model_loader():
    """Debug the model loader dtype"""
    print("Debugging model loader...")
    
    # Load model through our loader
    model_loader = HuggingFaceModelLoader("gpt2")
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Check if CUDA is available
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")

if __name__ == "__main__":
    debug_model_loader()