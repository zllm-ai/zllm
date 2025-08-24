"""
Main entry point for the custom vLLM/llama.cpp CLI application
"""

import torch
from torch.nn import functional as F
from typing import List, Tuple, Dict
from vllm.src.model_loader import HuggingFaceModelLoader
from vllm.src.quantization import Quantizer
from vllm.src.batcher import RequestBatcher
from vllm.src.inference import InferenceEngine


def main():
    print("Custom vLLM Implementation")
    print("=" * 30)
    
    # Prompt for model name
    model_name = input("Enter Hugging Face model name (e.g., meta-llama/Llama-3-8b): ").strip()
    if not model_name:
        model_name = "meta-llama/Llama-3-8b"  # Default model
    
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
    
    try:
        # Initialize model loader with device parameter
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
        batcher = RequestBatcher(model=model, max_seq_len=2048, batch_size=8)
        batcher.tokenizer = tokenizer
        
        print("Model loaded successfully!")
        
        # Example usage - process a sample request
        while True:
            user_input = input("\nEnter your prompt (or 'quit' to exit): ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if user_input:
                try:
                    # Add request to batcher
                    batcher.add_request(user_input)
                    
                    # Process the batch
                    outputs = batcher.finalize_batch()
                    
                    # Decode output
                    if outputs is not None:
                        # For simplicity, we're just showing tensor shapes
                        print(f"Output shape: {outputs.shape}")
                        print("Inference completed successfully!")
                    else:
                        print("No output generated")
                        
                    # Clear the batch for next request
                    batcher.inputs = []
                except Exception as e:
                    print(f"Error processing request: {str(e)}")
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return


if __name__ == "__main__":
    main()