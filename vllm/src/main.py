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
    # Prompt for model name
    model_name = input("Enter Hugging Face model name (e.g., meta-llama/Llama-3-8b): ").strip()
    
    # Prompt for quantization type
    quantization_type = input("Choose quantization type (gptq/awq) [default: gptq]: ").strip()
    if not quantization_type:
        quantization_type = "gptq"
    
    # Prompt for device selection
    device_choice = input("Select device (cuda/hip) [default: cuda]: ").strip()
    if not device_choice:
        device_choice = "cuda"
    
    # Initialize model loader
    model_loader = HuggingFaceModelLoader(model_name=model_name)
    
    # Initialize quantizer
    quantizer = Quantizer(quantization_type=quantization_type)
    
    # Initialize request batcher
    batcher = RequestBatcher()
    
    # Initialize inference engine
    inference_engine = InferenceEngine()
    
    # Configure device
    device = torch.device(device_choice)
    model_loader.device = device
    quantizer.device = device
    inference_engine.device = device
    
    # Load and quantize model
    model = model_loader.load_model()
    quantized_model = quantizer.quantize_model(model)
    
    # Start request batching
    batcher.start_batching(quantized_model, inference_engine)


if __name__ == "__main__":
    main()