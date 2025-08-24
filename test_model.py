#!/usr/bin/env python3
"""
Simple test script to verify model loading and generation
"""

import torch
from vllm.src.model_loader import HuggingFaceModelLoader

def test_model_loading():
    """Test loading a small model and generating text"""
    print("Testing model loading and generation...")
    
    try:
        # Load a small model
        print("Loading gpt2 model...")
        model_loader = HuggingFaceModelLoader("gpt2", device="cpu")
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()
        
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Tokenizer type: {type(tokenizer)}")
        
        # Test generation
        print("\nTesting text generation...")
        prompt = "The future of artificial intelligence"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
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
    success = test_model_loading()
    if success:
        print("\n✅ Test passed!")
    else:
        print("\n❌ Test failed!")
