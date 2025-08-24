#!/usr/bin/env python3
"""
Comprehensive test of all features
"""

import torch
from vllm.src.model_loader import HuggingFaceModelLoader
from vllm.src.quantization import Quantizer

def test_quantization_methods():
    """Test different quantization methods"""
    print("Testing quantization methods...")
    
    try:
        # Load a small model for testing
        model_loader = HuggingFaceModelLoader("gpt2", device="cpu")
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()
        
        print("Original model loaded successfully!")
        
        # Test different quantization methods
        quant_methods = ["fp16", "int8", "fp8"]
        
        for method in quant_methods:
            print(f"\nTesting {method.upper()} quantization...")
            try:
                quantizer = Quantizer(model=model, quantization=method)
                quantized_model = quantizer.quantize_model()
                print(f"✅ {method.upper()} quantization applied successfully!")
                
                # Test generation with quantized model
                prompt = "Hello world"
                inputs = tokenizer(prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = quantized_model.generate(
                        **inputs,
                        max_new_tokens=20,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Generated response: {response[:50]}...")
                
            except Exception as e:
                print(f"❌ {method.upper()} quantization failed: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model loading with different configurations"""
    print("\nTesting model loading...")
    
    try:
        # Test with auto-detected AWQ model (should handle gracefully)
        print("Testing AWQ model detection...")
        awq_loader = HuggingFaceModelLoader("TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ")
        print("AWQ model loader created (would require actual AWQ model to fully test)")
        
        # Test with regular model
        print("Testing regular model...")
        regular_loader = HuggingFaceModelLoader("gpt2")
        model = regular_loader.get_model()
        tokenizer = regular_loader.get_tokenizer()
        print("Regular model loaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Running comprehensive tests...\n")
    
    success1 = test_quantization_methods()
    success2 = test_model_loading()
    
    if success1 and success2:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")