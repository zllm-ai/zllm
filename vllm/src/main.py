"""
Main entry point for the custom vLLM/llama.cpp CLI application
"""

import torch
import warnings
from torch.nn import functional as F
from typing import List, Tuple, Dict
from vllm.src.model_loader import HuggingFaceModelLoader
from vllm.src.quantization import Quantizer
from vllm.src.batcher import RequestBatcher
from vllm.src.inference import InferenceEngine


def main():
    print("Custom vLLM Implementation")
    print("=" * 30)
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Prompt for model name
    model_name = input("Enter Hugging Face model name (e.g., gpt2, Qwen/Qwen3-0.5B): ").strip()
    if not model_name:
        model_name = "gpt2"  # Default to a small model that works on CPU
    
    # Prompt for quantization type
    quantization_type = input("Choose quantization type (gptq/awq/none) [default: none]: ").strip()
    if not quantization_type:
        quantization_type = None
    
    # Prompt for device selection
    device_choice = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        device_choice = input("Select device (cuda/cpu) [default: cuda]: ").strip()
        if not device_choice:
            device_choice = "cuda"
    else:
        print("CUDA not available, using CPU")
        device_choice = "cpu"
    
    print(f"Loading model: {model_name}")
    print(f"Quantization: {quantization_type}")
    print(f"Device: {device_choice}")
    
    # Special handling for AWQ models on CPU
    if quantization_type in ["awq", "AWQ"] and device_choice == "cpu":
        print("Warning: AWQ models are optimized for GPU. Performance on CPU may be poor.")
        print("Consider using a non-quantized model for better CPU performance.")
        choice = input("Continue anyway? (y/n) [default: n]: ").strip().lower()
        if choice != 'y':
            print("Exiting...")
            return
    
    try:
        # Initialize model loader with warnings suppressed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_loader = HuggingFaceModelLoader(model_name=model_name, quantization=quantization_type, device=device_choice)
        
        # Get model and tokenizer
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()
        
        # Apply quantization if specified
        if quantization_type:
            print(f"Applying {quantization_type.upper()} quantization...")
            quantizer = Quantizer(model=model, quantization=quantization_type)
            model = quantizer.model  # Get quantized model
        
        # Initialize inference engine
        inference_engine = InferenceEngine(device=device_choice)
        
        # Initialize request batcher
        batcher = RequestBatcher(model=model, max_seq_len=512, batch_size=4)  # Smaller defaults for CPU
        batcher.tokenizer = tokenizer
        
        print("Model loaded successfully!")
        
        # Example usage - process a sample request
        while True:
            user_input = input("\nEnter your prompt (or 'quit' to exit): ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if user_input:
                try:
                    # For better text generation, we'll use the model's generate method directly
                    # Tokenize the input
                    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    # Generate response
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=100,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                        )
                    
                    # Decode and print the response
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"Generated response: {response}")
                        
                except Exception as e:
                    print(f"Error processing request: {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()