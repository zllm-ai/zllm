#!/usr/bin/env python3
"""
Simple test with tiny model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_tiny_model():
    """Test with a tiny model"""
    print("Testing with tiny model (gpt2)...")
    
    try:
        # Load tiny model
        model_name = "gpt2"
        print(f"Loading {model_name}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded successfully!")
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Model loaded successfully!")
        
        # Test generation
        prompt = "The future of artificial intelligence"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated response:\n{response}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tiny_model()
    if success:
        print("\n✅ Tiny model test passed!")
    else:
        print("\n❌ Tiny model test failed!")