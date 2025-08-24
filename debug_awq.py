#!/usr/bin/env python3
"""
Debug script for AWQ model loading
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_awq_loading():
    """Test loading AWQ model directly"""
    print("Testing AWQ model loading...")
    
    try:
        # Test with a public AWQ model
        model_name = "TheBloke/Llama-2-7B-AWQ"  # This is a public AWQ model
        print(f"Loading tokenizer for {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("Tokenizer loaded successfully!")
        
        print(f"Loading model for {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("Model loaded successfully!")
        
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
    success = test_awq_loading()
    if success:
        print("\n✅ AWQ test passed!")
    else:
        print("\n❌ AWQ test failed!")