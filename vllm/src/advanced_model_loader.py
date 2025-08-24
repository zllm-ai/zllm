"""
Advanced Model Loader with Comprehensive Device Management

This module provides robust model loading with support for all quantization methods,
proper device placement, error handling, and enterprise features.
"""

import torch
import warnings
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, pipeline
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import gc
import os
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedModelLoader:
    """
    Advanced Model Loader with comprehensive device management, 
    quantization support, and enterprise features.
    """
    
    def __init__(self, 
                 model_name: str = "gpt2",
                 quantization_method: str = "none",
                 device: str = "cuda",
                 trust_remote_code: bool = True,
                 max_workers: int = 4):
        """
        Initialize the Advanced Model Loader.
        
        Args:
            model_name: Hugging Face model identifier
            quantization_method: Quantization method to use
            device: Target device ('cuda', 'cpu', 'auto')
            trust_remote_code: Whether to trust remote code
            max_workers: Maximum number of worker threads
        """
        self.model_name = model_name
        self.quantization_method = quantization_method.lower()
        self.requested_device = device
        self.trust_remote_code = trust_remote_code
        self.max_workers = max_workers
        
        # Initialize state
        self.model = None
        self.tokenizer = None
        self.model_config = None
        self.device_map = None
        self.quantization_config = None
        self.loading_lock = threading.Lock()
        self.cache_dir = Path.home() / ".cache" / "advanced_vllm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine actual device
        self.actual_device = self._determine_actual_device()
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def _determine_actual_device(self) -> torch.device:
        """Determine the actual device to use based on availability and request."""
        if self.requested_device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")
        elif self.requested_device == "cpu":
            return torch.device("cpu")
        else:
            # Auto mode
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
    
    def _detect_quantization_from_model_name(self) -> str:
        """Auto-detect quantization type from model name."""
        model_lower = self.model_name.lower()
        
        if "awq" in model_lower:
            return "awq"
        elif "gptq" in model_lower:
            return "gptq"
        elif "int8" in model_lower:
            return "int8"
        elif "int4" in model_lower:
            return "int4"
        elif "fp8" in model_lower:
            return "fp8"
        elif "fp4" in model_lower:
            return "fp4"
        elif "nf4" in model_lower:
            return "nf4"
        elif "fp16" in model_lower:
            return "fp16"
        else:
            return "none"
    
    def _validate_quantization_support(self) -> bool:
        """Check if required libraries are available for quantization method."""
        quant_method = self.quantization_method
        
        if quant_method in ["none", "int8", "int4", "fp8", "fp4", "nf4", "fp16"]:
            # These are handled internally or don't require external libraries
            return True
        elif quant_method == "awq":
            try:
                import awq
                return True
            except ImportError:
                self.logger.warning("AWQ quantization requires 'autoawq' library (deprecated)")
                return False
        elif quant_method == "gptq":
            try:
                import auto_gptq
                return True
            except ImportError:
                self.logger.warning("GPTQ quantization requires 'auto-gptq' library")
                return False
        else:
            self.logger.warning(f"Unknown quantization method: {quant_method}")
            return False
    
    def _create_quantization_config(self) -> Optional[Dict[str, Any]]:
        """Create quantization configuration based on method."""
        quant_method = self.quantization_method
        
        if quant_method == "none":
            return None
        elif quant_method == "int8":
            return {
                "load_in_8bit": True,
                "llm_int8_threshold": 6.0,
                "llm_int8_has_fp16_weight": False
            }
        elif quant_method == "int4":
            return {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.float16 if self.actual_device.type == "cuda" else torch.float32,
                "bnb_4bit_use_double_quant": True
            }
        elif quant_method == "fp8":
            # FP8 is experimental
            return {
                "torch_dtype": torch.float16 if self.actual_device.type == "cuda" else torch.float32
            }
        elif quant_method in ["fp4", "nf4"]:
            return {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": quant_method,
                "bnb_4bit_compute_dtype": torch.float16 if self.actual_device.type == "cuda" else torch.float32,
                "bnb_4bit_use_double_quant": False
            }
        elif quant_method == "fp16":
            return {
                "torch_dtype": torch.float16
            }
        else:
            # For AWQ/GPTQ, return None as they're handled separately
            return None
    
    def _load_with_awq(self):
        """Load model with AWQ quantization."""
        try:
            from awq import AutoAWQForCausalLM
            
            self.logger.info("Loading AWQ quantized model...")
            
            # Load AWQ model
            self.model = AutoAWQForCausalLM.from_quantized(
                self.model_name,
                fuse_layers=True,
                trust_remote_code=self.trust_remote_code,
                safetensors=True,
                device_map="auto" if self.actual_device.type == "cuda" else None
            )
            
            self.logger.info("AWQ model loaded successfully")
            return True
            
        except ImportError:
            self.logger.error("AWQ library not available")
            return False
        except Exception as e:
            self.logger.error(f"Error loading AWQ model: {str(e)}")
            return False
    
    def _load_with_gptq(self):
        """Load model with GPTQ quantization."""
        try:
            from auto_gptq import AutoGPTQForCausalLM
            
            self.logger.info("Loading GPTQ quantized model...")
            
            # Load GPTQ model
            self.model = AutoGPTQForCausalLM.from_quantized(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                use_safetensors=True,
                device_map="auto" if self.actual_device.type == "cuda" else None
            )
            
            self.logger.info("GPTQ model loaded successfully")
            return True
            
        except ImportError:
            self.logger.error("GPTQ library not available")
            return False
        except Exception as e:
            self.logger.error(f"Error loading GPTQ model: {str(e)}")
            return False
    
    def _load_standard_model(self):
        """Load standard model with or without quantization."""
        try:
            self.logger.info(f"Loading standard model: {self.model_name}")
            
            # Create model loading configuration
            load_config = {
                "trust_remote_code": self.trust_remote_code,
                "low_cpu_mem_usage": True
            }
            
            # Add device map for CUDA
            if self.actual_device.type == "cuda":
                load_config["device_map"] = "auto"
            
            # Add quantization config if available
            if self.quantization_config:
                load_config.update(self.quantization_config)
            
            # Add dtype for CPU
            if self.actual_device.type == "cpu":
                load_config["torch_dtype"] = torch.float32
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_config
            )
            
            self.logger.info("Standard model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading standard model: {str(e)}")
            return False
    
    def _ensure_proper_device_placement(self):
        """Ensure all model tensors are on the correct device."""
        if self.model is None:
            return
            
        try:
            # Move model to target device if needed
            model_device = next(self.model.parameters()).device
            if model_device != self.actual_device:
                self.logger.info(f"Moving model from {model_device} to {self.actual_device}")
                self.model = self.model.to(self.actual_device)
                
            # Ensure all parameters are on the correct device
            for name, param in self.model.named_parameters():
                if param.device != self.actual_device:
                    self.logger.debug(f"Moving parameter {name} to {self.actual_device}")
                    param.data = param.data.to(self.actual_device)
                    
        except Exception as e:
            self.logger.warning(f"Error ensuring device placement: {str(e)}")
    
    def _load_tokenizer(self):
        """Load tokenizer with proper configuration."""
        try:
            self.logger.info(f"Loading tokenizer for {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.logger.info("Tokenizer loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {str(e)}")
            return False
    
    def _extract_model_config(self):
        """Extract model configuration information."""
        try:
            if hasattr(self.model, 'config'):
                self.model_config = self.model.config
                
                # Log some key configuration information
                config_info = {
                    "model_type": getattr(self.model_config, "model_type", "unknown"),
                    "hidden_size": getattr(self.model_config, "hidden_size", "unknown"),
                    "num_hidden_layers": getattr(self.model_config, "num_hidden_layers", "unknown"),
                    "max_position_embeddings": getattr(self.model_config, "max_position_embeddings", "unknown")
                }
                
                self.logger.info(f"Model config: {config_info}")
                
        except Exception as e:
            self.logger.warning(f"Could not extract model config: {str(e)}")
    
    def load_model(self) -> bool:
        """
        Load the model with comprehensive error handling and device management.
        
        Returns:
            bool: True if successful, False otherwise
        """
        with self.loading_lock:
            try:
                self.logger.info(f"Loading model {self.model_name} with {self.quantization_method} quantization on {self.actual_device}")
                
                # Auto-detect quantization if not explicitly set
                if self.quantization_method == "none":
                    detected_quant = self._detect_quantization_from_model_name()
                    if detected_quant != "none":
                        self.logger.info(f"Auto-detected {detected_quant} quantization")
                        self.quantization_method = detected_quant
                
                # Validate quantization support
                if not self._validate_quantization_support():
                    self.logger.warning(f"Quantization method {self.quantization_method} not supported, falling back to none")
                    self.quantization_method = "none"
                
                # Create quantization config
                self.quantization_config = self._create_quantization_config()
                
                # Load tokenizer first
                if not self._load_tokenizer():
                    return False
                
                # Load model based on quantization method
                success = False
                if self.quantization_method == "awq":
                    success = self._load_with_awq()
                    if not success:
                        self.logger.info("Falling back to standard loading for AWQ model")
                        self.quantization_method = "none"
                        success = self._load_standard_model()
                elif self.quantization_method == "gptq":
                    success = self._load_with_gptq()
                    if not success:
                        self.logger.info("Falling back to standard loading for GPTQ model")
                        self.quantization_method = "none"
                        success = self._load_standard_model()
                else:
                    success = self._load_standard_model()
                
                if not success:
                    self.logger.error("Failed to load model with all methods")
                    return False
                
                # Ensure proper device placement
                self._ensure_proper_device_placement()
                
                # Extract model configuration
                self._extract_model_config()
                
                self.logger.info("Model loading completed successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Critical error during model loading: {str(e)}")
                return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the loaded model."""
        info = {
            "model_name": self.model_name,
            "quantization_method": self.quantization_method,
            "device": str(self.actual_device),
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
        }
        
        if self.model_config:
            info.update({
                "model_type": getattr(self.model_config, "model_type", "unknown"),
                "hidden_size": getattr(self.model_config, "hidden_size", "unknown"),
                "num_hidden_layers": getattr(self.model_config, "num_hidden_layers", "unknown"),
                "max_position_embeddings": getattr(self.model_config, "max_position_embeddings", "unknown")
            })
        
        return info
    
    def get_model(self):
        """Get the loaded model."""
        return self.model
    
    def get_tokenizer(self):
        """Get the loaded tokenizer."""
        return self.tokenizer
    
    def unload_model(self):
        """Unload the model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("Model unloaded and memory freed")


# Example usage and testing
if __name__ == "__main__":
    # Test the advanced model loader
    loader = AdvancedModelLoader(
        model_name="gpt2",
        quantization_method="none",
        device="cpu"
    )
    
    print("Testing Advanced Model Loader...")
    
    # Load model
    if loader.load_model():
        print("✅ Model loaded successfully!")
        
        # Get model info
        info = loader.get_model_info()
        print(f"Model info: {info}")
        
        # Test generation
        tokenizer = loader.get_tokenizer()
        model = loader.get_model()
        
        if tokenizer and model:
            prompt = "The future of artificial intelligence"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            
            # Move inputs to model device
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
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
        
        # Unload model
        loader.unload_model()
        print("✅ Model unloaded successfully!")
    else:
        print("❌ Failed to load model")