#!/usr/bin/env python3
"""
Simple test for proper model loading and generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_proper_loading():
    """Test proper model loading and generation"""
    print("Testing proper model loading and generation...")
    
    try:
        # Load a small, publicly available model
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        print(f"Loading {model_name}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("Tokenizer loaded successfully!")
        
        # Load model with proper settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        print("Model loaded successfully!")
        
        # Test generation
        prompt = "Write a short poem about technology."
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Move inputs to device if needed
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
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
    success = test_proper_loading()
    if success:
        print("\n✅ Test passed!")
    else:
        print("\n❌ Test failed!")
