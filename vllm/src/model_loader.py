"""
Hugging Face Model Loader for Custom vLLM Implementation

This module provides functionality to load models from Hugging Face using the transformers library.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
from accelerate import Accelerator

class HuggingFaceModelLoader:
    def __init__(self, model_name: str, quantization: str = None):
        self.model_name = model_name
        self.quantization = quantization
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.accelerator = Accelerator()
        self.init_model()
    
    def init_model(self):
        """Initialize the model and tokenizer with quantization support"""
        try:
            print(f"Loading tokenizer for {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Loading model for {self.model_name}...")
            # Load model with appropriate settings
            load_kwargs = {}
            
            # Only use float16 if CUDA is available
            if torch.cuda.is_available():
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            print("Model loaded successfully!")
            
            # Apply quantization if specified
            if self.quantization:
                self.apply_quantization()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # Try with minimal settings
            try:
                print("Trying fallback loading method...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=True
                )
            except Exception as e2:
                print(f"Fallback method also failed: {str(e2)}")
                raise e
    
    def apply_quantization(self):
        """Apply quantization techniques to the model"""
        print(f"Applying {self.quantization.upper()} quantization...")
        # Note: In a full implementation, this would contain actual quantization code
        # For now, we're just printing a message
        if self.quantization == "gptq":
            print("GPTQ quantization would be applied here")
        elif self.quantization == "awq":
            print("AWQ quantization would be applied here")
        else:
            print(f"Unknown quantization method: {self.quantization}")
    
    def save_model(self, save_path: str):
        """Save the loaded model to a local directory"""
        os.makedirs(save_path, exist_ok=True)
        self.tokenizer.save_pretrained(save_path)
        self.model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def load_model(self):
        """Load and return the model (compatibility method)"""
        return self.get_model()