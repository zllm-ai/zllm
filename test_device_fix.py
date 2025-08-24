#!/usr/bin/env python3
"""
Simple test to verify device placement fixes
"""

import torch
from vllm.src.model_loader import HuggingFaceModelLoader

def test_device_placement():
    """Test proper device placement"""
    print("Testing device placement fixes...")
    
    try:
        # Load a small model
        print("Loading gpt2 model...")
        model_loader = HuggingFaceModelLoader("gpt2", device="cpu")
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()
        
        print("Model loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Test generation with proper device placement
        prompt = "The future of AI"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        
        # Move inputs to same device as model
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated response: {response}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_device_placement()
    if success:
        print("\n✅ Device placement test passed!")
    else:
        print("\n❌ Device placement test failed!")