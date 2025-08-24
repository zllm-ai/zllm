"""
Ultimate Custom vLLM Implementation - Professional Edition

This is a comprehensive implementation that combines the best features of 
both vLLM and llama.cpp with additional enterprise-grade enhancements.
"""

import torch
import warnings
import sys
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import json
import time
import threading
from contextlib import contextmanager

# Import our advanced modules
from vllm.src.advanced_model_loader import AdvancedModelLoader
from vllm.src.advanced_inference import (
    AdvancedInferenceEngine, 
    InferenceConfig, 
    DeviceStrategy,
    AttentionMode
)
from vllm.src.advanced_quantization import (
    AdvancedQuantizer, 
    QuantizationMethod, 
    QuantizationConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UltimateVLLM:
    """
    Ultimate Custom vLLM Implementation - Professional Edition
    
    Combines the best features of vLLM and llama.cpp with enterprise enhancements.
    """
    
    def __init__(self):
        self.model_loader = None
        self.inference_engine = None
        self.quantizer = None
        self.model = None
        self.tokenizer = None
        self.config = None
        self.is_initialized = False
        self.save_mode_enabled = False
        
        # Performance monitoring
        self.stats = {
            "models_loaded": 0,
            "inferences_performed": 0,
            "total_tokens_generated": 0,
            "average_response_time_ms": 0.0
        }
        
        # Threading support
        self.lock = threading.RLock()
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Ultimate vLLM initialized")
    
    def _detect_quantization_from_model(self, model_name: str) -> str:
        """
        Auto-detect quantization type from model name.
        
        Args:
            model_name: Model name or path
            
        Returns:
            str: Detected quantization method
        """
        model_lower = model_name.lower()
        
        # Check for quantization indicators in model name
        quant_indicators = {
            "awq": ["awq", "autoawq"],
            "gptq": ["gptq", "autogptq"],
            "int8": ["int8", "8bit"],
            "int4": ["int4", "4bit"],
            "fp8": ["fp8", "float8"],
            "fp4": ["fp4", "float4"],
            "nf4": ["nf4", "normalfloat4"],
            "fp16": ["fp16", "float16"]
        }
        
        for quant_method, indicators in quant_indicators.items():
            if any(indicator in model_lower for indicator in indicators):
                self.logger.info(f"Auto-detected {quant_method.upper()} quantization from model name")
                return quant_method
        
        # Default to none
        return "none"
    
    def _validate_hardware_requirements(self, quantization_method: str, device: str) -> bool:
        """
        Validate hardware requirements for quantization method.
        
        Args:
            quantization_method: Quantization method to validate
            device: Target device
            
        Returns:
            bool: True if requirements met, False otherwise
        """
        quant_method = quantization_method.lower()
        
        # Check CUDA requirements
        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available")
            return False
            
        # Check quantization library requirements
        if quant_method == "awq":
            try:
                import awq
                self.logger.info("AutoAWQ library available")
                return True
            except ImportError:
                self.logger.warning("AutoAWQ library not available (deprecated)")
                return True  # Still allow, will fall back to standard loading
                
        elif quant_method == "gptq":
            try:
                import auto_gptq
                self.logger.info("AutoGPTQ library available")
                return True
            except ImportError:
                self.logger.warning("AutoGPTQ library not available")
                choice = input("Continue anyway? (y/n) [default: n]: ").strip().lower()
                return choice == 'y'
        
        # Other methods don't have special requirements
        return True
    
    def load_model(self,
                   model_name: str = "gpt2",
                   quantization_method: str = "none",
                   device: str = "cuda",
                   trust_remote_code: bool = True,
                   max_context_length: Optional[int] = None) -> bool:
        """
        Load model with comprehensive configuration.
        
        Args:
            model_name: Model name or path
            quantization_method: Quantization method
            device: Target device ('cuda', 'cpu', 'auto')
            trust_remote_code: Whether to trust remote code
            max_context_length: Maximum context length override
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            try:
                self.logger.info(f"Loading model: {model_name}")
                self.logger.info(f"Quantization: {quantization_method}")
                self.logger.info(f"Device: {device}")
                
                # Auto-detect quantization if not specified
                if quantization_method == "none":
                    detected_quant = self._detect_quantization_from_model(model_name)
                    if detected_quant != "none":
                        quantization_method = detected_quant
                
                # Validate hardware requirements
                if not self._validate_hardware_requirements(quantization_method, device):
                    self.logger.error("Hardware requirements not met")
                    return False
                
                # Initialize advanced model loader
                self.model_loader = AdvancedModelLoader(
                    model_name=model_name,
                    quantization_method=quantization_method,
                    device=device,
                    trust_remote_code=trust_remote_code
                )
                
                # Load model
                if not self.model_loader.load_model():
                    self.logger.error("Failed to load model")
                    return False
                
                # Get model and tokenizer
                self.model = self.model_loader.get_model()
                self.tokenizer = self.model_loader.get_tokenizer()
                
                if self.model is None or self.tokenizer is None:
                    self.logger.error("Model or tokenizer not loaded properly")
                    return False
                
                # Apply quantization if needed (for non-prequantized models)
                if quantization_method not in ["none", "awq", "gptq"]:
                    self.logger.info(f"Applying {quantization_method.upper()} quantization...")
                    try:
                        quant_config = QuantizationConfig(
                            method=QuantizationMethod(quantization_method.lower()),
                            bits=8 if quantization_method in ["int8", "fp8"] else 4
                        )
                        self.quantizer = AdvancedQuantizer(
                            self.model,
                            quantization_method,
                            quant_config
                        )
                        self.model = self.quantizer.quantize_model()
                        self.logger.info("Quantization completed successfully")
                    except Exception as e:
                        self.logger.warning(f"Quantization failed: {str(e)}. Continuing with original model.")
                
                # Determine actual device
                actual_device = self.model_loader.actual_device
                
                # Initialize inference engine with appropriate device strategy
                inference_config = InferenceConfig(
                    max_batch_size=1,
                    max_sequence_length=max_context_length or 2048,
                    attention_mode=AttentionMode.STANDARD,
                    device_strategy=DeviceStrategy.SAVE_MODE if self.save_mode_enabled else DeviceStrategy.ALL_GPU,
                    use_cuda_graph=torch.cuda.is_available() and device == "cuda",
                    enable_streaming=True
                )
                
                # Set up device strategy for save mode
                cache_device = "cpu" if self.save_mode_enabled else device
                
                self.inference_engine = AdvancedInferenceEngine(
                    config=inference_config,
                    model_device=actual_device,
                    cache_device=cache_device
                )
                
                # Enable save mode if requested
                if self.save_mode_enabled:
                    self.inference_engine.enable_save_mode()
                
                self.is_initialized = True
                self.stats["models_loaded"] += 1
                
                self.logger.info("Model loaded successfully!")
                self.logger.info(f"Model info: {self.model_loader.get_model_info()}")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error loading model: {str(e)}")
                self.logger.exception("Detailed error trace:")
                return False
    
    def enable_save_mode(self) -> bool:
        """
        Enable save mode - model weights on GPU, KV cache on CPU.
        
        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            if not torch.cuda.is_available():
                self.logger.warning("Save mode requires CUDA. CUDA not available.")
                return False
            
            try:
                self.logger.info("Enabling Save Mode...")
                self.logger.info("Moving model weights to GPU, keeping KV cache operations on CPU...")
                
                # Enable save mode in inference engine
                if self.inference_engine:
                    success = self.inference_engine.enable_save_mode()
                    if success:
                        self.save_mode_enabled = True
                        self.logger.info("✅ Save Mode enabled successfully!")
                        return True
                    else:
                        self.logger.error("Failed to enable save mode in inference engine")
                        return False
                else:
                    self.logger.warning("Inference engine not initialized")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error enabling save mode: {str(e)}")
                return False
    
    def disable_save_mode(self):
        """Disable save mode."""
        with self.lock:
            if self.inference_engine:
                self.inference_engine.disable_save_mode()
            self.save_mode_enabled = False
            self.logger.info("Save mode disabled")
    
    def _prepare_inputs(self, 
                        prompt: Union[str, List[str]], 
                        max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for model generation.
        
        Args:
            prompt: Input prompt(s)
            max_length: Maximum sequence length
            
        Returns:
            Dict[str, torch.Tensor]: Prepared inputs
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        # Handle single prompt
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length or 2048
        )
        
        # Move to model device
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        return inputs
    
    def generate(self,
                 prompt: Union[str, List[str]],
                 max_new_tokens: int = 100,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 repetition_penalty: float = 1.1,
                 do_sample: bool = True,
                 **kwargs) -> Union[str, List[str]]:
        """
        Generate text from prompt(s).
        
        Args:
            prompt: Input prompt(s)
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to sample or use greedy decoding
            **kwargs: Additional generation parameters
            
        Returns:
            Union[str, List[str]]: Generated text
        """
        with self.lock:
            if not self.is_initialized:
                raise RuntimeError("Model not initialized. Call load_model() first.")
            
            start_time = time.time()
            
            try:
                # Prepare inputs
                inputs = self._prepare_inputs(prompt)
                
                # Move inputs to model device
                model_device = next(self.model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                
                # Prepare generation parameters
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "do_sample": do_sample,
                    "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                
                # Add top_k if specified
                if top_k > 0:
                    gen_kwargs["top_k"] = top_k
                
                # Add any additional parameters
                gen_kwargs.update(kwargs)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_kwargs)
                
                # Decode outputs
                if isinstance(prompt, str):
                    # Single prompt
                    full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if full_response.startswith(prompt):
                        response = full_response[len(prompt):].strip()
                    else:
                        response = full_response
                    
                    # Update stats
                    self.stats["inferences_performed"] += 1
                    self.stats["total_tokens_generated"] += len(response.split())
                    
                else:
                    # Multiple prompts
                    responses = []
                    for i, p in enumerate(prompt):
                        full_response = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                        if full_response.startswith(p):
                            response = full_response[len(p):].strip()
                        else:
                            response = full_response
                        responses.append(response)
                    
                    # Update stats
                    self.stats["inferences_performed"] += len(prompt)
                    for resp in responses:
                        self.stats["total_tokens_generated"] += len(resp.split())
                    
                    response = responses
                
                # Update timing stats
                elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                self.stats["average_response_time_ms"] = (
                    (self.stats["average_response_time_ms"] * (self.stats["inferences_performed"] - 1) + elapsed_time) 
                    / self.stats["inferences_performed"]
                )
                
                return response
                
            except Exception as e:
                self.logger.error(f"Error during generation: {str(e)}")
                raise
    
    def stream_generate(self,
                       prompt: str,
                       max_new_tokens: int = 100,
                       temperature: float = 0.7,
                       top_p: float = 0.9,
                       top_k: int = 50,
                       repetition_penalty: float = 1.1,
                       do_sample: bool = True,
                       **kwargs):
        """
        Stream generate text token by token.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to sample or use greedy decoding
            **kwargs: Additional generation parameters
            
        Yields:
            str: Generated tokens
        """
        with self.lock:
            if not self.is_initialized:
                raise RuntimeError("Model not initialized. Call load_model() first.")
            
            try:
                # Prepare inputs
                inputs = self._prepare_inputs(prompt)
                
                # Move inputs to model device
                model_device = next(self.model.parameters()).device
                current_inputs = {k: v.clone().to(model_device) for k, v in inputs.items()}
                
                # Prepare generation parameters
                gen_kwargs = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "do_sample": do_sample,
                    "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                
                # Add top_k if specified
                if top_k > 0:
                    gen_kwargs["top_k"] = top_k
                
                # Add any additional parameters
                gen_kwargs.update(kwargs)
                
                # Stream generation token by token
                for i in range(max_new_tokens):
                    with torch.no_grad():
                        # Generate next token with only 1 new token
                        outputs = self.model.generate(
                            **current_inputs,
                            max_new_tokens=1,
                            **{k: v for k, v in gen_kwargs.items() if k not in ['max_new_tokens']}
                        )
                    
                    # Get the new token ID
                    new_token_id = outputs[0, -1].unsqueeze(0).unsqueeze(0)
                    
                    # Check for EOS token
                    if new_token_id.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Decode the new token
                    new_token = self.tokenizer.decode([new_token_id.item()])
                    
                    # Yield the new token
                    yield new_token
                    
                    # Update inputs for next iteration
                    current_inputs = {
                        k: torch.cat([v, new_token_id.to(v.device)], dim=1) 
                        for k, v in current_inputs.items()
                    }
                    
                    # Small delay for streaming effect (optional)
                    yield ""  # Empty yield to allow for streaming
                    
            except Exception as e:
                self.logger.error(f"Error during streaming generation: {str(e)}")
                raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        with self.lock:
            info = {
                "initialized": self.is_initialized,
                "save_mode_enabled": self.save_mode_enabled,
                "stats": self.stats.copy()
            }
            
            if self.model_loader:
                info.update(self.model_loader.get_model_info())
            
            if self.inference_engine:
                info["inference_stats"] = self.inference_engine.get_stats()
            
            return info
    
    def unload_model(self):
        """Unload the model and free all resources."""
        with self.lock:
            try:
                # Unload model from model loader
                if self.model_loader:
                    self.model_loader.unload_model()
                
                # Clear inference engine
                if self.inference_engine:
                    self.inference_engine.clear_cache()
                
                # Reset state
                self.model = None
                self.tokenizer = None
                self.model_loader = None
                self.inference_engine = None
                self.quantizer = None
                self.is_initialized = False
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.logger.info("Model unloaded and resources freed")
                
            except Exception as e:
                self.logger.error(f"Error unloading model: {str(e)}")
    
    @contextmanager
    def streaming_context(self):
        """Context manager for streaming operations."""
        with self.lock:
            try:
                if self.inference_engine:
                    with self.inference_engine.streaming_context():
                        yield self
                else:
                    yield self
            finally:
                pass


# Interactive CLI Interface
class UltimateVLLMCLI:
    """Interactive CLI interface for Ultimate vLLM."""
    
    def __init__(self):
        self.engine = UltimateVLLM()
        self.streaming_enabled = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _show_welcome_message(self):
        """Show welcome message and instructions."""
        print("=" * 60)
        print("🚀 Ultimate Custom vLLM Implementation - Professional Edition")
        print("=" * 60)
        print("Combining the best features of vLLM and llama.cpp")
        print("With enterprise-grade enhancements and optimizations")
        print()
        print("💡 Tips for best results:")
        print("   • Be specific with your prompts")
        print("   • For lists, specify the exact number of items")
        print("   • For explanations, ask for step-by-step breakdowns")
        print("   • Use 'config' to reconfigure parameters")
        print("   • Use 'stream' to toggle streaming mode")
        print("   • Use 'save' to enable save mode (weights on GPU, KV cache on CPU)")
        print("   • Use 'quit' or 'exit' to stop")
        print("=" * 60)
    
    def _interactive_model_setup(self) -> bool:
        """Interactive model setup."""
        print("\n🔧 Model Configuration")
        print("-" * 25)
        
        # Model name
        model_name = input("Enter Hugging Face model name [default: gpt2]: ").strip()
        if not model_name:
            model_name = "gpt2"
        
        # Auto-detect quantization
        auto_quant = self.engine._detect_quantization_from_model(model_name)
        if auto_quant != "none":
            print(f"🔍 Auto-detected {auto_quant.upper()} quantization from model name")
            quantization_method = auto_quant
        else:
            print("\nAvailable quantization methods:")
            print("  none  - No quantization (default)")
            print("  int8  - 8-bit integer quantization")
            print("  int4  - 4-bit integer quantization")
            print("  fp8   - 8-bit floating point")
            print("  fp4   - 4-bit floating point")
            print("  nf4   - NormalFloat 4-bit")
            print("  fp16  - 16-bit floating point")
            print("  awq   - Activation-aware Weight Quantization (deprecated)")
            print("  gptq  - Post-training Quantization")
            
            quantization_method = input("Choose quantization method [default: none]: ").strip()
            if not quantization_method:
                quantization_method = "none"
        
        # Device selection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            device = input("Select device (cuda/cpu) [default: cuda]: ").strip()
            if not device:
                device = "cuda"
        else:
            print("CUDA not available, using CPU")
            device = "cpu"
        
        print(f"\n📥 Loading model...")
        print(f"   Model: {model_name}")
        print(f"   Quantization: {quantization_method}")
        print(f"   Device: {device}")
        
        # Load model
        success = self.engine.load_model(
            model_name=model_name,
            quantization_method=quantization_method,
            device=device
        )
        
        if success:
            print("✅ Model loaded successfully!")
            model_info = self.engine.get_model_info()
            print(f"📊 Model info: {model_info}")
        else:
            print("❌ Failed to load model")
        
        return success
    
    def _show_current_config(self):
        """Show current configuration."""
        print("\n⚙️  Current Configuration")
        print("-" * 25)
        model_info = self.engine.get_model_info()
        for key, value in model_info.items():
            print(f"   {key}: {value}")
    
    def run_interactive_session(self):
        """Run interactive session."""
        self._show_welcome_message()
        
        # Setup model
        if not self._interactive_model_setup():
            print("❌ Failed to setup model. Exiting...")
            return
        
        # Show configuration
        self._show_current_config()
        
        # Interactive loop
        while True:
            try:
                prompt = input("\n📝 Enter your prompt (or commands): ").strip()
                
                if prompt.lower() in ['quit', 'exit']:
                    break
                elif prompt.lower() == 'config':
                    # Reconfigure model
                    print("\n🔧 Reconfiguration")
                    print("Note: For major changes, restart the application")
                    # For now, just show config
                    self._show_current_config()
                    continue
                elif prompt.lower() == 'stream':
                    self.streaming_enabled = not self.streaming_enabled
                    print(f"🌊 Streaming mode {'enabled' if self.streaming_enabled else 'disabled'}")
                    continue
                elif prompt.lower() == 'save':
                    if not self.engine.save_mode_enabled:
                        success = self.engine.enable_save_mode()
                        if success:
                            print("💾 Save mode enabled successfully!")
                        else:
                            print("❌ Failed to enable save mode")
                    else:
                        self.engine.disable_save_mode()
                        print("💾 Save mode disabled")
                    continue
                elif prompt.lower() == 'info':
                    model_info = self.engine.get_model_info()
                    print("\nℹ️  Model Information:")
                    for key, value in model_info.items():
                        print(f"   {key}: {value}")
                    continue
                elif prompt.lower() == 'clear':
                    # Clear cache
                    if self.engine.inference_engine:
                        self.engine.inference_engine.clear_cache()
                    print("🧹 Cache cleared")
                    continue
                
                if prompt:
                    try:
                        print("\n🔄 Generating response...")
                        
                        if self.streaming_enabled:
                            # Stream response
                            print("📡 Streaming response: ", end="", flush=True)
                            response_tokens = []
                            
                            for token in self.engine.stream_generate(
                                prompt=prompt,
                                max_new_tokens=200,
                                temperature=0.7,
                                top_p=0.9,
                                top_k=50
                            ):
                                if token:  # Skip empty tokens
                                    print(token, end="", flush=True)
                                    response_tokens.append(token)
                            
                            print()  # New line after streaming
                            print(f"✅ Streaming completed ({len(response_tokens)} tokens)")
                            
                        else:
                            # Generate full response
                            start_time = time.time()
                            response = self.engine.generate(
                                prompt=prompt,
                                max_new_tokens=200,
                                temperature=0.7,
                                top_p=0.9,
                                top_k=50
                            )
                            elapsed_time = time.time() - start_time
                            
                            print(f"🤖 Response: {response}")
                            print(f"⏱️  Generated in {elapsed_time:.2f}s")
                            
                    except Exception as e:
                        print(f"❌ Error generating response: {str(e)}")
                        print("💡 Try a shorter prompt or different model")
                        
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted by user. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 End of input. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")
                self.logger.exception("Detailed error trace:")
    
    def run_with_defaults(self):
        """Run with default configuration."""
        print("🚀 Starting Ultimate vLLM with default settings...")
        
        # Load default model
        success = self.engine.load_model(
            model_name="gpt2",
            quantization_method="none",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        if not success:
            print("❌ Failed to load default model")
            return
        
        print("✅ Default model loaded successfully!")
        self._show_current_config()
        
        # Simple interactive loop
        while True:
            try:
                prompt = input("\n📝 Enter your prompt (or 'quit' to exit): ").strip()
                
                if prompt.lower() in ['quit', 'exit']:
                    break
                
                if prompt:
                    response = self.engine.generate(
                        prompt=prompt,
                        max_new_tokens=100,
                        temperature=0.7
                    )
                    print(f"🤖 Response: {response}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")


def main():
    """Main entry point."""
    try:
        print("🚀 Ultimate Custom vLLM Implementation")
        print("=" * 50)
        print("1. Interactive Setup (Full customization)")
        print("2. Quick Start (Default settings)")
        
        choice = input("Select mode (1/2) [default: 1]: ").strip()
        
        cli = UltimateVLLMCLI()
        
        if choice == "2":
            cli.run_with_defaults()
        else:
            cli.run_interactive_session()
            
    except KeyboardInterrupt:
        print("\n\n👋 Application interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        logging.exception("Fatal error details:")


if __name__ == "__main__":
    main()