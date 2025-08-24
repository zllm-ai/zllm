"""
Main entry point for the custom vLLM/llama.cpp CLI application
"""

import torch
import warnings
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
    
    def get_model_defaults(self):
        """Extract default parameters from the loaded model"""
        defaults = {
            "max_new_tokens": 200,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "early_stopping": False,
        }
        
        try:
            # Try to get model-specific defaults
            if hasattr(self.model, 'config'):
                config = self.model.config
                
                # Get max context length
                if hasattr(config, 'max_position_embeddings'):
                    defaults['context_length'] = config.max_position_embeddings
                elif hasattr(config, 'n_ctx'):
                    defaults['context_length'] = config.n_ctx
                else:
                    defaults['context_length'] = 2048  # Default fallback
                
                # Get generation defaults from model config
                if hasattr(config, 'temperature'):
                    defaults['temperature'] = config.temperature
                if hasattr(config, 'top_p'):
                    defaults['top_p'] = config.top_p
                if hasattr(config, 'repetition_penalty'):
                    defaults['repetition_penalty'] = config.repetition_penalty
                    
        except Exception as e:
            print(f"Could not extract model defaults: {e}")
            defaults['context_length'] = 2048
            
        return defaults
    
    def interactive_setup(self):
        """Interactive setup for model and generation parameters"""
        print("Custom vLLM Implementation - Smart Setup")
        print("=" * 45)
        
        # Model configuration
        model_name = input("Enter Hugging Face model name (e.g., gpt2, Qwen/Qwen3-0.5B) [default: gpt2]: ").strip()
        if not model_name:
            model_name = "gpt2"
        
        quantization_type = input("Choose quantization type (gptq/awq/none) [default: none]: ").strip()
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
        
        # Special handling for AWQ models on CPU
        if quantization_type in ["awq", "AWQ"] and device_choice == "cpu":
            print("Warning: AWQ models are optimized for GPU. Performance on CPU may be poor.")
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
            
            # Apply quantization if specified
            if quantization_type and quantization_type.lower() != "none":
                print(f"Applying {quantization_type.upper()} quantization...")
                quantizer = Quantizer(model=self.model, quantization=quantization_type)
                self.model = quantizer.model  # Get quantized model
            
            # Initialize inference engine
            self.inference_engine = InferenceEngine(device=device_choice)
            
            # Get model defaults
            self.default_generation_config = self.get_model_defaults()
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Make sure you have enough GPU memory for the selected model.")
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
        
        # Allow user to customize parameters
        print(f"Model default context length: {config.get('context_length', 2048)}")
        context_length_input = input(f"Context length [default: {config.get('context_length', 2048)}]: ").strip()
        if context_length_input:
            context_length = int(context_length_input)
            # Validate context_length
            max_allowed = min(config.get('context_length', 8192), 8192)
            if context_length > max_allowed:
                print(f"Warning: Setting context length to maximum allowed: {max_allowed}")
                context_length = max_allowed
            elif context_length < 128:
                print("Warning: Context length should be at least 128. Using 128.")
                context_length = 128
            config['context_length'] = context_length
        
        print(f"Model default max new tokens: {config.get('max_new_tokens', 200)}")
        max_new_tokens_input = input(f"Max new tokens [default: {config.get('max_new_tokens', 200)}]: ").strip()
        if max_new_tokens_input:
            max_new_tokens = int(max_new_tokens_input)
            # Validate max_new_tokens
            if max_new_tokens > 4096:
                print("Warning: Very large max_new_tokens may cause memory issues. Using 4096.")
                max_new_tokens = 4096
            elif max_new_tokens < 1:
                print("Warning: max_new_tokens must be positive. Using default value.")
                max_new_tokens = config['max_new_tokens']
            config['max_new_tokens'] = max_new_tokens
        
        print(f"Model default temperature: {config.get('temperature', 1.0)}")
        temperature_input = input(f"Temperature (0.0-2.0) [default: {config.get('temperature', 1.0)}]: ").strip()
        if temperature_input:
            temperature = float(temperature_input)
            # Validate temperature
            if temperature < 0.0 or temperature > 2.0:
                print("Warning: Temperature should be between 0.0 and 2.0. Using default value.")
                temperature = config['temperature']
            config['temperature'] = temperature
        
        print(f"Model default top-p: {config.get('top_p', 1.0)}")
        top_p_input = input(f"Top-p (nucleus sampling) [default: {config.get('top_p', 1.0)}]: ").strip()
        if top_p_input:
            top_p = float(top_p_input)
            # Validate top_p
            if top_p <= 0.0 or top_p > 1.0:
                print("Warning: Top-p should be between 0.0 and 1.0. Using default value.")
                top_p = config['top_p']
            config['top_p'] = top_p
        
        print(f"Model default top-k: {config.get('top_k', 0)}")
        top_k_input = input(f"Top-k (0 to disable) [default: {config.get('top_k', 0)}]: ").strip()
        if top_k_input:
            top_k = int(top_k_input)
            # Validate top_k
            if top_k < 0:
                print("Warning: Top-k should be non-negative. Using default value.")
                top_k = config['top_k']
            config['top_k'] = top_k
        
        print(f"Model default repetition penalty: {config.get('repetition_penalty', 1.0)}")
        repetition_penalty_input = input(f"Repetition penalty [default: {config.get('repetition_penalty', 1.0)}]: ").strip()
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
        print()
    
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
            }
            
            # Add top_k if specified and > 0
            if config.get('top_k', 0) > 0:
                gen_kwargs["top_k"] = config['top_k']
            
            # Only add early_stopping if the model supports it and it's enabled
            if config.get('early_stopping', False):
                try:
                    # Test if early_stopping is supported
                    dummy_inputs = {k: v[:1] for k, v in inputs.items()}  # Create minimal inputs
                    self.model.generate(**dummy_inputs, **gen_kwargs, early_stopping=True, max_new_tokens=1)
                    gen_kwargs["early_stopping"] = True
                except Exception:
                    # early_stopping not supported, continue without it
                    pass
            
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
        print("- Use 'quit' or 'exit' to stop")
        
        # Interactive session
        while True:
            user_input = input("\nEnter your prompt (or 'quit' to exit, 'config' to reconfigure): ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
            elif user_input.lower() == 'config':
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
                
            if user_input:
                try:
                    response = self.generate_response(user_input, config)
                    
                    # Handle empty responses
                    if not response:
                        print("Generated response: (No response generated)")
                    else:
                        print(f"Generated response: {response}")
                        
                except Exception as e:
                    print(f"Error processing request: {str(e)}")
                    print("Try a shorter prompt, reduce max_new_tokens, or use a different model.")
    
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
            print("\nUsing automatic configuration based on model defaults:")
            self.show_current_config(self.default_generation_config)
            
            # Interactive session with defaults
            while True:
                user_input = input("\nEnter your prompt (or 'quit' to exit): ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    break
                    
                if user_input:
                    try:
                        response = self.generate_response(user_input, self.default_generation_config)
                        
                        if not response:
                            print("Generated response: (No response generated)")
                        else:
                            print(f"Generated response: {response}")
                            
                    except Exception as e:
                        print(f"Error processing request: {str(e)}")
        
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    cli = CustomVLLMCLI()
    
    # Ask user which mode they want
    print("Custom vLLM Implementation")
    print("=" * 30)
    print("1. Smart Setup (Automatic configuration with customization option)")
    print("2. Quick Start (Default settings)")
    
    choice = input("Select mode (1/2) [default: 1]: ").strip()
    
    if choice == "2":
        cli.run_with_defaults()
    else:
        cli.run_interactive_session()


if __name__ == "__main__":
    main()