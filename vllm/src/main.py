"""
Main entry point for the custom vLLM/llama.cpp CLI application
"""

import torch
import warnings
import sys
from typing import List, Tuple, Dict, Optional
from vllm.src.model_loader import HuggingFaceModelLoader
from vllm.src.quantization import Quantizer
from vllm.src.inference import InferenceEngine


class CustomVLLMCLI:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.inference_engine = None
        self.model_config = None
        self.default_generation_config = {}
        self.save_mode_enabled = False
    
    def get_model_max_tokens(self):
        """Extract maximum context length from the loaded model"""
        max_context_length = 2048  # Default fallback
        
        try:
            if hasattr(self.model, 'config'):
                config = self.model.config
                
                # Try different attributes for max context length
                if hasattr(config, 'max_position_embeddings'):
                    max_context_length = config.max_position_embeddings
                elif hasattr(config, 'n_ctx'):
                    max_context_length = config.n_ctx
                elif hasattr(config, 'max_sequence_length'):
                    max_context_length = config.max_sequence_length
                elif hasattr(config, 'seq_length'):
                    max_context_length = config.seq_length
                    
        except Exception as e:
            print(f"Could not extract max context length: {e}")
            
        return max_context_length
    
    def get_model_defaults(self):
        """Extract default parameters from the loaded model"""
        # Get model's maximum context length
        max_context_length = self.get_model_max_tokens()
        
        defaults = {
            "max_new_tokens": min(300, max_context_length // 4),  # Use 1/4 of context or 300, whichever is smaller
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "early_stopping": True,
            "context_length": max_context_length,
        }
        
        try:
            # Try to get model-specific defaults from config
            if hasattr(self.model, 'config'):
                config = self.model.config
                
                # Get generation defaults from model config
                if hasattr(config, 'temperature') and config.temperature is not None:
                    defaults['temperature'] = config.temperature
                if hasattr(config, 'top_p') and config.top_p is not None:
                    defaults['top_p'] = config.top_p
                if hasattr(config, 'repetition_penalty') and config.repetition_penalty is not None:
                    defaults['repetition_penalty'] = config.repetition_penalty
                    
        except Exception as e:
            print(f"Could not extract model defaults: {e}")
            
        return defaults
    
    def enable_save_mode(self):
        """Enable save mode - model weights on GPU, KV cache on CPU"""
        if not torch.cuda.is_available():
            print("Save mode requires CUDA. CUDA not available.")
            return False
            
        try:
            print("Enabling Save Mode...")
            print("Moving model weights to GPU, keeping KV cache operations on CPU...")
            
            # This is a simplified implementation - in a full implementation,
            # we would need to modify the model's attention mechanisms to
            # handle KV cache on CPU while keeping weights on GPU
            
            # For demonstration, we'll just set a flag and show what would be done
            self.save_mode_enabled = True
            print("✅ Save Mode enabled successfully!")
            print("Note: This is a conceptual implementation. Full save mode requires")
            print("      custom attention kernels that are not implemented in this demo.")
            return True
            
        except Exception as e:
            print(f"Error enabling save mode: {str(e)}")
            return False
    
    def interactive_setup(self):
        """Interactive setup for model and generation parameters"""
        print("Custom vLLM Implementation - Smart Setup")
        print("=" * 45)
        
        # Model configuration
        model_name = input("Enter Hugging Face model name (e.g., gpt2, microsoft/Phi-3-mini-4k-instruct) [default: gpt2]: ").strip()
        if not model_name:
            model_name = "gpt2"
        
        # Auto-detect quantization type from model name
        auto_quantization = None
        if "awq" in model_name.lower():
            auto_quantization = "awq"
        elif "gptq" in model_name.lower():
            auto_quantization = "gptq"
        
        if auto_quantization:
            print(f"Detected {auto_quantization.upper()} quantized model")
            quantization_type = auto_quantization
        else:
            print("Available quantization methods:")
            print("  none  - No quantization (default)")
            print("  awq   - Activation-aware Weight Quantization")
            print("  gptq  - Post-training Quantization")
            print("  int8  - 8-bit integer quantization")
            print("  int4  - 4-bit integer quantization")
            print("  fp8   - 8-bit floating point")
            print("  fp4   - 4-bit floating point")
            print("  nf4   - NormalFloat 4-bit")
            print("  fp16  - 16-bit floating point")
            
            quantization_type = input("Choose quantization type [default: none]: ").strip()
            if not quantization_type:
                quantization_type = None
        
        device_choice = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            device_choice = input("Select device (cuda/cpu) [default: cuda]: ").strip()
            if not device_choice:
                device_choice = "cuda"
        else:
            print("CUDA not available, using CPU")
            device_choice = "cpu"
        
        print(f"\nLoading model: {model_name}")
        print(f"Quantization: {quantization_type}")
        print(f"Device: {device_choice}")
        
        # Special handling for quantized models
        if quantization_type in ["awq", "AWQ", "gptq", "GPTQ"]:
            if device_choice == "cpu":
                print("Warning: Quantized models are optimized for GPU. Performance on CPU may be poor.")
                choice = input("Continue anyway? (y/n) [default: n]: ").strip().lower()
                if choice != 'y':
                    print("Exiting...")
                    return False
            # Check if required libraries are installed
            if quantization_type.lower() == "awq":
                try:
                    import awq
                except ImportError:
                    print("Warning: AWQ quantization requires 'autoawq' library.")
                    print("Note: AutoAWQ is deprecated. Consider using 'none' or 'int8' quantization instead.")
                    choice = input("Continue anyway? (y/n) [default: n]: ").strip().lower()
                    if choice != 'y':
                        print("Exiting...")
                        return False
        
        try:
            # Initialize model loader with warnings suppressed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_loader = HuggingFaceModelLoader(model_name=model_name, quantization=quantization_type, device=device_choice)
            
            # Get model and tokenizer
            self.model = model_loader.get_model()
            self.tokenizer = model_loader.get_tokenizer()
            self.model_config = getattr(self.model, 'config', None)
            
            # Apply quantization if specified and needed
            if quantization_type and quantization_type.lower() not in ["none", "awq", "gptq"]:
                # For non-prequantized quantization methods, apply after loading
                if not (auto_quantization and auto_quantization.lower() in ["awq", "gptq"]):
                    print(f"Applying {quantization_type.upper()} quantization...")
                    from vllm.src.quantization import Quantizer
                    quantizer = Quantizer(model=self.model, quantization=quantization_type)
                    self.model = quantizer.quantize_model()
            
            # Initialize inference engine
            self.inference_engine = InferenceEngine(device=device_choice)
            
            # Get model defaults
            self.default_generation_config = self.get_model_defaults()
            
            print("Model loaded successfully!")
            print(f"Detected model context length: {self.default_generation_config['context_length']} tokens")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("\nTroubleshooting tips:")
            print("- For AWQ models, ensure you have 'autoawq' installed (note: it's deprecated)")
            print("- For GPTQ models, ensure you have 'auto-gptq' installed")
            print("- Some models require Hugging Face authentication")
            print("- Large models may require significant GPU memory")
            return False
    
    def auto_configure_generation_params(self):
        """Automatically configure generation parameters based on model defaults"""
        print("\nUsing automatic configuration based on model defaults...")
        return self.default_generation_config.copy()
    
    def manual_configure_generation_params(self):
        """Configure generation parameters interactively with model defaults as starting point"""
        print("\nGeneration Parameters Configuration (Press Enter to use model defaults)")
        print("-" * 65)
        
        # Get current defaults
        config = self.default_generation_config.copy()
        
        # Show detected model limits
        print(f"Model maximum context length: {config.get('context_length', 2048)} tokens")
        context_length_input = input(f"Context length [default: {config.get('context_length', 2048)}]: ").strip()
        if context_length_input:
            context_length = int(context_length_input)
            # Validate context_length
            max_allowed = config.get('context_length', 8192)
            if context_length > max_allowed:
                print(f"Warning: Setting context length to model maximum: {max_allowed}")
                context_length = max_allowed
            elif context_length < 128:
                print("Warning: Context length should be at least 128. Using 128.")
                context_length = 128
            config['context_length'] = context_length
        
        max_suggested_new_tokens = min(500, config.get('context_length', 2048) // 4)
        print(f"Model suggested max new tokens: {max_suggested_new_tokens}")
        max_new_tokens_input = input(f"Max new tokens [default: {max_suggested_new_tokens}]: ").strip()
        if max_new_tokens_input:
            max_new_tokens = int(max_new_tokens_input)
            # Validate max_new_tokens
            max_allowed = config.get('context_length', 2048) // 2
            if max_new_tokens > max_allowed:
                print(f"Warning: Setting max_new_tokens to half of context length: {max_allowed}")
                max_new_tokens = max_allowed
            elif max_new_tokens < 1:
                print(f"Warning: max_new_tokens must be positive. Using suggested value: {max_suggested_new_tokens}")
                max_new_tokens = max_suggested_new_tokens
            config['max_new_tokens'] = max_new_tokens
        
        print(f"Model default temperature: {config.get('temperature', 0.7)}")
        temperature_input = input(f"Temperature (0.0-2.0) [default: {config.get('temperature', 0.7)}]: ").strip()
        if temperature_input:
            temperature = float(temperature_input)
            # Validate temperature
            if temperature < 0.0 or temperature > 2.0:
                print("Warning: Temperature should be between 0.0 and 2.0. Using default value.")
                temperature = config['temperature']
            config['temperature'] = temperature
        
        print(f"Model default top-p: {config.get('top_p', 0.9)}")
        top_p_input = input(f"Top-p (nucleus sampling) [default: {config.get('top_p', 0.9)}]: ").strip()
        if top_p_input:
            top_p = float(top_p_input)
            # Validate top_p
            if top_p <= 0.0 or top_p > 1.0:
                print("Warning: Top-p should be between 0.0 and 1.0. Using default value.")
                top_p = config['top_p']
            config['top_p'] = top_p
        
        print(f"Model default top-k: {config.get('top_k', 50)}")
        top_k_input = input(f"Top-k (0 to disable) [default: {config.get('top_k', 50)}]: ").strip()
        if top_k_input:
            top_k = int(top_k_input)
            # Validate top_k
            if top_k < 0:
                print("Warning: Top-k should be non-negative. Using default value.")
                top_k = config['top_k']
            config['top_k'] = top_k
        
        print(f"Model default repetition penalty: {config.get('repetition_penalty', 1.1)}")
        repetition_penalty_input = input(f"Repetition penalty [default: {config.get('repetition_penalty', 1.1)}]: ").strip()
        if repetition_penalty_input:
            repetition_penalty = float(repetition_penalty_input)
            # Validate repetition_penalty
            if repetition_penalty < 1.0 or repetition_penalty > 2.0:
                print("Warning: Repetition penalty should be between 1.0 and 2.0. Using default value.")
                repetition_penalty = config['repetition_penalty']
            config['repetition_penalty'] = repetition_penalty
        
        return config
    
    def show_current_config(self, config):
        """Display current configuration"""
        print("\nCurrent Configuration:")
        print("-" * 25)
        for key, value in config.items():
            print(f"{key}: {value}")
        if self.save_mode_enabled:
            print("SAVE MODE: Enabled")
        print()
    
    def stream_response(self, prompt: str, config: Dict):
        """Stream response token by token"""
        try:
            # Tokenize the input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.get('context_length', 2048)
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            print("Generating response: ", end="", flush=True)
            
            # Suppress transformer warnings
            import transformers
            transformers.logging.set_verbosity_error()
            
            # For proper streaming, we need to generate token by token
            # Create initial input
            current_inputs = {k: v.clone() for k, v in inputs.items()}
            generated_text = ""
            
            # Prepare generation parameters
            gen_kwargs = {
                "temperature": config['temperature'],
                "top_p": config['top_p'],
                "repetition_penalty": config['repetition_penalty'],
                "do_sample": config['do_sample'],
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "early_stopping": True,
            }
            
            # Add top_k if specified and > 0
            if config.get('top_k', 0) > 0:
                gen_kwargs["top_k"] = config['top_k']
            
            # Generate tokens one by one for true streaming
            max_tokens = config['max_new_tokens']
            
            with torch.no_grad():
                for i in range(max_tokens):
                    # Generate next token with only 1 new token
                    outputs = self.model.generate(
                        **current_inputs,
                        max_new_tokens=1,
                        **{k: v for k, v in gen_kwargs.items() if k not in ['max_new_tokens']}
                    )
                    
                    # Get the new token ID
                    new_token_id = outputs[0, -1].unsqueeze(0).unsqueeze(0)
                    
                    # Decode the new token
                    new_token = self.tokenizer.decode([new_token_id.item()])
                    
                    # Check for end of sequence
                    if new_token_id.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Print the new token immediately
                    print(new_token, end="", flush=True)
                    generated_text += new_token
                    
                    # Update inputs for next iteration
                    current_inputs = {
                        k: torch.cat([v, new_token_id.to(v.device)], dim=1) 
                        for k, v in current_inputs.items()
                    }
                    
                    # Small delay for better streaming experience
                    import time
                    time.sleep(0.02)
            
            print()  # New line after streaming
            return generated_text
            
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def generate_response(self, prompt: str, config: Dict):
        """Generate response with given configuration"""
        try:
            # Tokenize the input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.get('context_length', 2048)
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            print("Generating response...")
            
            # Suppress transformer warnings
            import transformers
            transformers.logging.set_verbosity_error()
            
            # Prepare generation parameters
            gen_kwargs = {
                "max_new_tokens": config['max_new_tokens'],
                "temperature": config['temperature'],
                "top_p": config['top_p'],
                "repetition_penalty": config['repetition_penalty'],
                "do_sample": config['do_sample'],
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "early_stopping": True,
            }
            
            # Add top_k if specified and > 0
            if config.get('top_k', 0) > 0:
                gen_kwargs["top_k"] = config['top_k']
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode and clean the response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the response if it's included
            if full_response.startswith(prompt):
                response = full_response[len(prompt):].strip()
            else:
                response = full_response
            
            # Handle empty or very short responses
            if not response or len(response.strip()) < 5:
                response = "(No meaningful response generated. Try a more specific prompt.)"
            
            return response
            
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def run_interactive_session(self):
        """Run interactive session with smart defaults"""
        if not self.interactive_setup():
            return
        
        # Ask user if they want to customize parameters
        print("\nConfiguration Options:")
        print("1. Automatic (Use model defaults)")
        print("2. Manual (Customize parameters)")
        
        config_choice = input("Select configuration mode (1/2) [default: 1]: ").strip()
        
        if config_choice == "2":
            config = self.manual_configure_generation_params()
        else:
            config = self.auto_configure_generation_params()
        
        self.show_current_config(config)
        
        print("\nTips for better responses:")
        print("- Be specific with your prompts")
        print("- For lists, specify the number of items (e.g., 'List 5 facts about...')")
        print("- For explanations, ask for step-by-step breakdowns")
        print("- Use 'config' to reconfigure parameters")
        print("- Use 'stream' to enable streaming mode")
        print("- Use 'save' to enable save mode (weights on GPU, KV cache on CPU)")
        print("- Use 'quit' or 'exit' to stop")
        
        # Interactive session
        streaming_enabled = False
        while True:
            try:
                prompt = input("\nEnter your prompt (or 'quit' to exit, 'config' to reconfigure, 'stream' to toggle streaming, 'save' to toggle save mode): ").strip()
                if prompt.lower() in ['quit', 'exit']:
                    break
                elif prompt.lower() == 'config':
                    print("\nConfiguration Options:")
                    print("1. Automatic (Use model defaults)")
                    print("2. Manual (Customize parameters)")
                    
                    config_choice = input("Select configuration mode (1/2) [default: 1]: ").strip()
                    
                    if config_choice == "2":
                        config = self.manual_configure_generation_params()
                    else:
                        config = self.auto_configure_generation_params()
                    self.show_current_config(config)
                    continue
                elif prompt.lower() == 'stream':
                    streaming_enabled = not streaming_enabled
                    print(f"Streaming mode {'enabled' if streaming_enabled else 'disabled'}")
                    continue
                elif prompt.lower() == 'save':
                    if torch.cuda.is_available():
                        self.save_mode_enabled = not self.save_mode_enabled
                        if self.save_mode_enabled:
                            self.enable_save_mode()
                        else:
                            print("Save mode disabled")
                    else:
                        print("Save mode requires CUDA. CUDA not available.")
                    continue
                    
                if prompt:
                    try:
                        if streaming_enabled:
                            response = self.stream_response(prompt, config)
                            print(f"\nStreamed response completed.")
                        else:
                            response = self.generate_response(prompt, config)
                            print(f"Generated response: {response}")
                            
                    except Exception as e:
                        print(f"Error processing request: {str(e)}")
                        print("Try a shorter prompt, reduce max_new_tokens, or use a different model.")
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Exiting...")
                break
            except EOFError:
                print("\n\nEnd of input. Exiting...")
                break
    
    def run_with_defaults(self):
        """Run with default configuration for quick start"""
        print("Custom vLLM Implementation - Quick Start")
        print("=" * 40)
        
        # Quick setup with defaults
        model_name = "gpt2"
        device_choice = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading default model: {model_name} on {device_choice}...")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_loader = HuggingFaceModelLoader(model_name=model_name, device=device_choice)
            
            self.model = model_loader.get_model()
            self.tokenizer = model_loader.get_tokenizer()
            self.model_config = getattr(self.model, 'config', None)
            self.inference_engine = InferenceEngine(device=device_choice)
            
            # Get model defaults
            self.default_generation_config = self.get_model_defaults()
            
            print("Model loaded successfully!")
            print(f"Detected model context length: {self.default_generation_config['context_length']} tokens")
            print("\nUsing automatic configuration based on model defaults:")
            self.show_current_config(self.default_generation_config)
            
            # Interactive session with defaults
            streaming_enabled = False
            while True:
                try:
                    prompt = input("\nEnter your prompt (or 'quit' to exit, 'stream' to toggle streaming, 'save' to toggle save mode): ").strip()
                    if prompt.lower() in ['quit', 'exit']:
                        break
                    elif prompt.lower() == 'stream':
                        streaming_enabled = not streaming_enabled
                        print(f"Streaming mode {'enabled' if streaming_enabled else 'disabled'}")
                        continue
                    elif prompt.lower() == 'save':
                        if torch.cuda.is_available():
                            self.save_mode_enabled = not self.save_mode_enabled
                            if self.save_mode_enabled:
                                self.enable_save_mode()
                            else:
                                print("Save mode disabled")
                        else:
                            print("Save mode requires CUDA. CUDA not available.")
                        continue
                        
                    if prompt:
                        try:
                            if streaming_enabled:
                                response = self.stream_response(prompt, self.default_generation_config)
                                print(f"\nStreamed response completed.")
                            else:
                                response = self.generate_response(prompt, self.default_generation_config)
                                print(f"Generated response: {response}")
                                
                        except Exception as e:
                            print(f"Error processing request: {str(e)}")
                except KeyboardInterrupt:
                    print("\n\nInterrupted by user. Exiting...")
                    break
                except EOFError:
                    print("\n\nEnd of input. Exiting...")
                    break
        
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    cli = CustomVLLMCLI()
    
    # Ask user which mode they want
    print("Custom vLLM Implementation")
    print("=" * 30)
    print("1. Smart Setup (Automatic configuration with customization option)")
    print("2. Quick Start (Default settings)")
    
    try:
        choice = input("Select mode (1/2) [default: 1]: ").strip()
        
        if choice == "2":
            cli.run_with_defaults()
        else:
            cli.run_interactive_session()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except EOFError:
        print("\n\nEnd of input. Exiting...")


if __name__ == "__main__":
    main()