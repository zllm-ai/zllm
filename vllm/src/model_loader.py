"""
Hugging Face Model Loader for Custom vLLM Implementation

This module provides functionality to load models from Hugging Face using the transformers library.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
import warnings
from accelerate import Accelerator

class HuggingFaceModelLoader:
    def __init__(self, model_name: str, quantization: str = None, device: str = "cuda"):
        self.model_name = model_name
        self.quantization = quantization
        self.tokenizer = None
        self.model = None
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator()
        self.init_model()
    
    def init_model(self):
        """Initialize the model and tokenizer with quantization support"""
        try:
            print(f"Loading tokenizer for {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Loading model for {self.model_name}...")
            
            # Suppress kernel version warning and other warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Handle different quantization methods
                quant_type = self.quantization.lower() if self.quantization else "none"
                
                if "awq" in self.model_name.lower() or quant_type == "awq":
                    # AWQ quantized model
                    print("Loading AWQ quantized model...")
                    try:
                        # Try to load with autoawq if available
                        from awq import AutoAWQForCausalLM
                        self.model = AutoAWQForCausalLM.from_quantized(
                            self.model_name,
                            fuse_layers=True,
                            trust_remote_code=True,
                            safetensors=True,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                        print("Loaded AWQ model with AutoAWQ")
                    except ImportError:
                        # Fallback to regular loading for AWQ models
                        print("AutoAWQ not available, loading with standard method...")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                    except Exception as e:
                        print(f"AWQ loading failed: {e}")
                        # Last resort fallback
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                elif "gptq" in self.model_name.lower() or quant_type == "gptq":
                    # GPTQ quantized model
                    print("Loading GPTQ quantized model...")
                    try:
                        from auto_gptq import AutoGPTQForCausalLM
                        self.model = AutoGPTQForCausalLM.from_quantized(
                            self.model_name,
                            trust_remote_code=True,
                            use_safetensors=True,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                        print("Loaded GPTQ model with AutoGPTQ")
                    except ImportError:
                        # Fallback to regular loading
                        print("AutoGPTQ not available, loading with standard method...")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                    except Exception as e:
                        print(f"GPTQ loading failed: {e}")
                        # Last resort fallback
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                elif self.device == "cpu":
                    # Standard CPU loading
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                    # Move model to CPU explicitly and ensure all tensors are on CPU
                    self.model = self.model.to(torch.device("cpu"))
                else:
                    # Standard GPU loading with proper device mapping
                    load_kwargs = {}
                    if torch.cuda.is_available():
                        load_kwargs["torch_dtype"] = torch.float16
                        load_kwargs["device_map"] = "auto"
                    else:
                        load_kwargs["low_cpu_mem_usage"] = True
                        load_kwargs["torch_dtype"] = torch.float32
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        **load_kwargs
                    )
            
            print("Model loaded successfully!")
            
            # Ensure consistent device placement
            if hasattr(self.model, 'device') and self.model.device != torch.device(self.device):
                try:
                    self.model = self.model.to(torch.device(self.device))
                except Exception:
                    pass  # Model might already be on the correct device
                    
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            # Try with minimal settings
            try:
                print("Trying fallback loading method...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                if self.device == "cpu":
                    self.model = self.model.to(torch.device("cpu"))
            except Exception as e2:
                print(f"Fallback method also failed: {str(e2)}")
                raise e
    
    def apply_quantization(self):
        """Apply quantization techniques to the model"""
        quant_type = self.quantization.lower() if self.quantization else "none"
        print(f"Applying {quant_type.upper()} quantization...")
        # Note: In a full implementation, this would contain actual quantization code
        # For now, we're just printing a message
        if quant_type == "gptq":
            print("GPTQ quantization would be applied here")
        elif quant_type == "awq":
            print("AWQ quantization would be applied here")
        else:
            print(f"Unknown quantization method: {quant_type}")
    
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