#!/usr/bin/env python3
"""
Test script for Qwen model loading
"""

import torch
from vllm.src.model_loader import HuggingFaceModelLoader

def test_qwen_loading():
    """Test loading Qwen model"""
    print("Testing Qwen model loading...")
    
    try:
        # Load Qwen model (smaller version for testing)
        print("Loading Qwen/Qwen3-0.5B model...")
        model_loader = HuggingFaceModelLoader("Qwen/Qwen3-0.5B", device="cpu")
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()
        
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Tokenizer type: {type(tokenizer)}")
        
        # Test generation
        print("\nTesting text generation...")
        prompt = "Hello, how are you today?"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_qwen_loading()
    if success:
        print("\n✅ Qwen test passed!")
    else:
        print("\n❌ Qwen test failed!")