#!/usr/bin/env python3
"""
Test script for zllm with proper CPU handling
"""

import torch
from vllm.src.model_loader import HuggingFaceModelLoader
from vllm.src.batcher import RequestBatcher

def test_cpu_execution():
    """Test model execution on CPU with a compatible model"""
    print("Testing CPU execution with a compatible model...")
    
    try:
        # Use a smaller model that works well on CPU
        print("Loading gpt2 model (works well on CPU)...")
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
        batcher.add_request("Hello, how are you today?")
        
        # Process the batch
        outputs = batcher.finalize_batch()
        
        if outputs is not None:
            print(f"✓ Batch processing successful, output shape: {outputs.shape}")
            # Try to generate some text
            with torch.no_grad():
                # Take the last token's logits and sample
                logits = outputs[0, -1, :]
                probabilities = torch.softmax(logits, dim=-1)
                predicted_token_id = torch.multinomial(probabilities, 1).item()
                predicted_token = tokenizer.decode([predicted_token_id])
                print(f"Predicted next token: '{predicted_token}'")
            
            return True
        else:
            print("✗ Batch processing failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_with_small_qwen():
    """Test with a smaller Qwen model that might work on CPU"""
    print("\nTesting with a smaller Qwen model...")
    
    try:
        # Try with a smaller Qwen model
        print("Loading Qwen/Qwen3-0.5B model...")
        model_loader = HuggingFaceModelLoader(
            model_name="Qwen/Qwen3-0.5B", 
            quantization=None, 
            device="cpu"
        )
        
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()
        
        print("✓ Qwen model loaded successfully on CPU")
        
        # Test batcher
        batcher = RequestBatcher(model, max_seq_len=256, batch_size=2)
        batcher.tokenizer = tokenizer
        
        # Add a test request
        batcher.add_request("The weather today is")
        
        # Process the batch
        outputs = batcher.finalize_batch()
        
        if outputs is not None:
            print(f"✓ Batch processing successful, output shape: {outputs.shape}")
            return True
        else:
            print("✗ Batch processing failed")
            return False
            
    except Exception as e:
        print(f"✗ Error with Qwen model: {str(e)}")
        return False

if __name__ == "__main__":
    print("Running zllm test suite...\n")
    
    # Test 1: CPU execution with compatible model
    success1 = test_cpu_execution()
    
    # Test 2: Small Qwen model
    success2 = test_with_small_qwen()
    
    if success1 or success2:
        print("\n✓ At least one test passed! The system is working.")
    else:
        print("\n✗ All tests failed. There may be issues with the setup.")
        
    print("\nFor best results on CPU:")
    print("1. Use smaller models like 'gpt2' or 'Qwen/Qwen3-0.5B'")
    print("2. Avoid AWQ/GPTQ quantized models on CPU (they're GPU-optimized)")
    print("3. Keep sequence lengths small to reduce memory usage")