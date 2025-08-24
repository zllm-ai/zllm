#!/usr/bin/env python3
"""
Test script to verify our fixes for the custom vLLM implementation
"""

import torch
from vllm.src.model_loader import HuggingFaceModelLoader
from vllm.src.batcher import RequestBatcher

def test_model_loading():
    """Test model loading with CPU device"""
    print("Testing model loading with CPU...")
    
    try:
        # Test with a small model that works on CPU
        model_loader = HuggingFaceModelLoader(
            model_name="gpt2", 
            quantization=None, 
            device="cpu"
        )
        
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()
        
        print("✓ Model loaded successfully on CPU")
        
        # Test batcher
        batcher = RequestBatcher(model, max_seq_len=512, batch_size=4)
        batcher.tokenizer = tokenizer
        
        # Add a test request
        batcher.add_request("Hello, how are you?")
        
        # Process the batch
        outputs = batcher.finalize_batch()
        
        if outputs is not None:
            print(f"✓ Batch processing successful, output shape: {outputs.shape}")
            return True
        else:
            print("✗ Batch processing failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n✓ All tests passed! The fixes are working correctly.")
    else:
        print("\n✗ Some tests failed. Please check the implementation.")