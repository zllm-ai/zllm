#!/usr/bin/env python3
"""
Test script for custom vLLM implementation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm.src.model_loader import HuggingFaceModelLoader
from vllm.src.quantization import Quantizer
from vllm.src.batcher import RequestBatcher
from vllm.src.inference import InferenceEngine

def test_model_loading():
    """Test model loading functionality"""
    print("Testing model loading...")
    try:
        # Using a small model for testing
        model_loader = HuggingFaceModelLoader("gpt2")
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()
        print("✓ Model loading successful")
        return model, tokenizer
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None, None

def test_quantization(model):
    """Test quantization functionality"""
    print("Testing quantization...")
    try:
        if model is not None:
            quantizer = Quantizer(model, quantization="gptq")
            quantized_model = quantizer.quantize_model()
            print("✓ Quantization successful")
            return quantized_model
        else:
            print("✗ Quantization skipped (no model)")
            return None
    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        return None

def test_batching(model, tokenizer):
    """Test batching functionality"""
    print("Testing batching...")
    try:
        if model is not None and tokenizer is not None:
            # Debug information
            print(f"Model device: {next(model.parameters()).device}")
            print(f"Model dtype: {next(model.parameters()).dtype}")
            
            # Create a fresh batcher instance with the model
            batcher = RequestBatcher(model)
            batcher.tokenizer = tokenizer
            
            # Debug tokenizer information
            print(f"Tokenizer pad token: {tokenizer.pad_token}")
            print(f"Tokenizer pad token ID: {tokenizer.pad_token_id}")
            
            # Add some test requests
            batcher.add_request("The capital of France is")
            batcher.add_request("Quantum computing is")
            
            # Process batch
            outputs = batcher.finalize_batch()
            if outputs is not None:
                print("✓ Batching successful")
                return True
            else:
                print("✗ Batching failed (no outputs)")
                return False
        else:
            print("✗ Batching skipped (no model/tokenizer)")
            return False
    except Exception as e:
        print(f"✗ Batching failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference_engine():
    """Test inference engine functionality"""
    print("Testing inference engine...")
    try:
        engine = InferenceEngine()
        dummy_input = torch.randn(1, 10)
        samples = engine.parallel_sampling(dummy_input, num_samples=2)
        print("✓ Inference engine successful")
        return True
    except Exception as e:
        print(f"✗ Inference engine failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running custom vLLM tests...\n")
    
    # Test components
    model, tokenizer = test_model_loading()
    quantized_model = test_quantization(model)
    batching_success = test_batching(model, tokenizer)
    inference_success = test_inference_engine()
    
    # Summary
    print("\nTest Summary:")
    print(f"  Model Loading: {'✓' if model is not None else '✗'}")
    print(f"  Quantization:  {'✓' if quantized_model is not None else '✗'}")
    print(f"  Batching:      {'✓' if batching_success else '✗'}")
    print(f"  Inference:     {'✓' if inference_success else '✗'}")
    
    if all([model is not None, batching_success, inference_success]):
        print("\n🎉 All core components working!")
    else:
        print("\n⚠️  Some components need attention.")

if __name__ == "__main__":
    main()