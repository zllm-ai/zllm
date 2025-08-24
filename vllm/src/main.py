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
        self.default_config = {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "early_stopping": True,
            "context_length": 2048,
            "kv_cache_size": None,  # Will be determined automatically
        }
    
    def interactive_setup(self):
        """Interactive setup for model and generation parameters"""
        print("Custom vLLM Implementation - Advanced Setup")
        print("=" * 50)
        
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
            
            # Apply quantization if specified
            if quantization_type and quantization_type.lower() != "none":
                print(f"Applying {quantization_type.upper()} quantization...")
                quantizer = Quantizer(model=self.model, quantization=quantization_type)
                self.model = quantizer.model  # Get quantized model
            
            # Initialize inference engine
            self.inference_engine = InferenceEngine(device=device_choice)
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Make sure you have enough GPU memory for the selected model.")
            return False
    
    def configure_generation_params(self):
        """Configure generation parameters interactively"""
        print("\nGeneration Parameters Configuration")
        print("-" * 35)
        
        # Get current defaults
        config = self.default_config.copy()
        
        # Allow user to customize parameters
        max_new_tokens = input(f"Max new tokens [default: {config['max_new_tokens']}]: ").strip()
        if max_new_tokens:
            config['max_new_tokens'] = int(max_new_tokens)
        
        temperature = input(f"Temperature (0.0-2.0) [default: {config['temperature']}]: ").strip()
        if temperature:
            config['temperature'] = float(temperature)
        
        top_p = input(f"Top-p (nucleus sampling) [default: {config['top_p']}]: ").strip()
        if top_p:
            config['top_p'] = float(top_p)
        
        top_k = input(f"Top-k [default: {config['top_k']}]: ").strip()
        if top_k:
            config['top_k'] = int(top_k)
        
        repetition_penalty = input(f"Repetition penalty [default: {config['repetition_penalty']}]: ").strip()
        if repetition_penalty:
            config['repetition_penalty'] = float(repetition_penalty)
        
        context_length = input(f"Context length [default: {config['context_length']}]: ").strip()
        if context_length:
            config['context_length'] = int(context_length)
        
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
                max_length=config['context_length']
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            print("Generating response...")
            
            # Prepare generation parameters
            gen_kwargs = {
                "max_new_tokens": config['max_new_tokens'],
                "temperature": config['temperature'],
                "top_p": config['top_p'],
                "repetition_penalty": config['repetition_penalty'],
                "do_sample": config['do_sample'],
                "early_stopping": config['early_stopping'],
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Add top_k if specified
            if config.get('top_k') and config['top_k'] > 0:
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
            
            return response
            
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def run_interactive_session(self):
        """Run interactive session with full customization"""
        if not self.interactive_setup():
            return
        
        # Configure generation parameters
        config = self.configure_generation_params()
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
                config = self.configure_generation_params()
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
            self.inference_engine = InferenceEngine(device=device_choice)
            
            print("Model loaded successfully!")
            print("\nUsing default generation parameters:")
            self.show_current_config(self.default_config)
            
            # Interactive session with defaults
            while True:
                user_input = input("\nEnter your prompt (or 'quit' to exit): ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    break
                    
                if user_input:
                    try:
                        response = self.generate_response(user_input, self.default_config)
                        
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
    print("1. Advanced Setup (Full customization)")
    print("2. Quick Start (Default settings)")
    
    choice = input("Select mode (1/2) [default: 2]: ").strip()
    
    if choice == "1":
        cli.run_interactive_session()
    else:
        cli.run_with_defaults()


if __name__ == "__main__":
    main()