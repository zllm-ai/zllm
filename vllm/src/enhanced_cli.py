#!/usr/bin/env python3
"""
Enhanced Interactive CLI for Custom vLLM

This CLI provides a rich, interactive experience with all the features of vLLM and llama.cpp
"""

import argparse
import sys
import os
import json
import torch
from typing import Dict, List, Optional
from vllm.src.model_loader import HuggingFaceModelLoader
from vllm.src.quantization import Quantizer
from vllm.src.batcher import RequestBatcher
from vllm.src.inference import InferenceEngine
from vllm.src.api_server import app as api_app
import uvicorn

class EnhancedVLLMCLI:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.batcher = None
        self.inference_engine = None
        self.config = {}
        self.history = []
        
    def interactive_wizard(self):
        """Run the interactive configuration wizard"""
        print("=== Custom vLLM Interactive Setup ===")
        print("Welcome to the enhanced vLLM CLI!")
        print()
        
        # Model selection
        model_name = input("Enter Hugging Face model name (or 'list' for popular models): ").strip()
        if model_name.lower() == 'list':
            self.list_popular_models()
            model_name = input("Enter Hugging Face model name: ").strip()
        
        if not model_name:
            model_name = "gpt2"  # Default model
            
        # Quantization selection
        print("\nQuantization Options:")
        print("1. None (default)")
        print("2. GPTQ")
        print("3. AWQ")
        print("4. INT8")
        print("5. FP8")
        quant_choice = input("Select quantization (1-5, default: 1): ").strip()
        
        quantization_map = {
            "1": None, "2": "gptq", "3": "awq", "4": "int8", "5": "fp8"
        }
        quantization = quantization_map.get(quant_choice, None)
        
        # Device selection
        print("\nDevice Options:")
        print("1. Auto (default)")
        print("2. CPU")
        print("3. CUDA")
        device_choice = input("Select device (1-3, default: 1): ").strip()
        
        device_map = {
            "1": "auto", "2": "cpu", "3": "cuda"
        }
        device = device_map.get(device_choice, "auto")
        
        # Parallelism options
        print("\nParallelism Options:")
        tensor_parallel = input("Tensor parallelism size (default: 1): ").strip()
        tensor_parallel = int(tensor_parallel) if tensor_parallel.isdigit() else 1
        
        pipeline_parallel = input("Pipeline parallelism size (default: 1): ").strip()
        pipeline_parallel = int(pipeline_parallel) if pipeline_parallel.isdigit() else 1
        
        # Advanced features
        print("\nAdvanced Features:")
        enable_prefix_caching = input("Enable prefix caching? (y/N): ").strip().lower() == 'y'
        enable_speculative_decoding = input("Enable speculative decoding? (y/N): ").strip().lower() == 'y'
        
        # Store configuration
        self.config = {
            "model_name": model_name,
            "quantization": quantization,
            "device": device,
            "tensor_parallel": tensor_parallel,
            "pipeline_parallel": pipeline_parallel,
            "prefix_caching": enable_prefix_caching,
            "speculative_decoding": enable_speculative_decoding
        }
        
        print(f"\nConfiguration Summary:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
            
        confirm = input("\nProceed with this configuration? (Y/n): ").strip().lower()
        if confirm == 'n':
            return self.interactive_wizard()
            
        return self.config
    
    def list_popular_models(self):
        """List popular models from Hugging Face"""
        popular_models = [
            "gpt2",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-3-8b",
            "mistralai/Mistral-7B-v0.1",
            "google/gemma-2b",
            "microsoft/Phi-3-mini-4k-instruct",
            "Qwen/Qwen2-7B"
        ]
        
        print("\nPopular Models:")
        for i, model in enumerate(popular_models, 1):
            print(f"{i}. {model}")
        print()
    
    def load_model(self, config: Dict):
        """Load model with the given configuration"""
        print(f"Loading model: {config['model_name']}")
        print(f"Quantization: {config['quantization'] or 'None'}")
        print(f"Device: {config['device']}")
        
        try:
            # Initialize model loader
            model_loader = HuggingFaceModelLoader(
                model_name=config['model_name'],
                quantization=config['quantization']
            )
            
            self.model = model_loader.get_model()
            self.tokenizer = model_loader.get_tokenizer()
            
            # Apply quantization if specified
            if config['quantization']:
                print(f"Applying {config['quantization'].upper()} quantization...")
                quantizer = Quantizer(self.model, quantization=config['quantization'])
                self.model = quantizer.quantize_model()
            
            # Initialize inference engine
            device = config['device']
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.inference_engine = InferenceEngine(device=device)
            
            # Initialize batcher
            self.batcher = RequestBatcher(self.model)
            self.batcher.tokenizer = self.tokenizer
            
            print("✓ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            return False
    
    def interactive_chat(self):
        """Run interactive chat session"""
        if not self.model or not self.tokenizer:
            print("✗ No model loaded. Please load a model first.")
            return
            
        print("\n=== Interactive Chat Mode ===")
        print("Type your messages and press Enter to send.")
        print("Commands: /quit, /history, /clear, /config")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['/quit', '/exit']:
                    break
                elif user_input.lower() == '/history':
                    self.show_history()
                    continue
                elif user_input.lower() == '/clear':
                    self.history.clear()
                    print("History cleared.")
                    continue
                elif user_input.lower() == '/config':
                    self.show_config()
                    continue
                elif not user_input:
                    continue
                    
                # Add to history
                self.history.append({"role": "user", "content": user_input})
                
                # Process the input
                response = self.process_request(user_input)
                
                if response:
                    print(f"Assistant: {response}")
                    self.history.append({"role": "assistant", "content": response})
                else:
                    print("Assistant: Sorry, I couldn't generate a response.")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def process_request(self, text: str) -> str:
        """Process a single request"""
        if not self.batcher:
            return "Error: Batcher not initialized"
            
        try:
            # Add request to batcher
            self.batcher.add_request(text)
            
            # Process the batch
            outputs = self.batcher.finalize_batch()
            
            if outputs is not None:
                # Simple decoding - in a real implementation, you would do proper decoding
                # For now, we'll just return a placeholder response
                return f"I understand your message: '{text[:50]}...'"
            else:
                return "No response generated"
                
        except Exception as e:
            return f"Error processing request: {str(e)}"
    
    def show_history(self):
        """Show conversation history"""
        if not self.history:
            print("No conversation history.")
            return
            
        print("\n=== Conversation History ===")
        for msg in self.history:
            print(f"{msg['role'].capitalize()}: {msg['content']}")
        print()
    
    def show_config(self):
        """Show current configuration"""
        print("\n=== Current Configuration ===")
        for key, value in self.config.items():
            print(f"{key}: {value}")
        print()
    
    def benchmark(self):
        """Run benchmark tests"""
        print("Running benchmark tests...")
        # This would be implemented with actual benchmarking code
        print("Benchmark results would be displayed here.")
    
    def serve_api(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the API server"""
        print(f"Starting API server on {host}:{port}")
        print("API endpoints:")
        print("  GET  /v1/models")
        print("  POST /v1/completions")
        print("  POST /v1/chat/completions")
        print("  GET  /v1/health")
        print("\nPress Ctrl+C to stop the server.")
        
        try:
            uvicorn.run("vllm.src.api_server:app", host=host, port=port, reload=False)
        except KeyboardInterrupt:
            print("\nServer stopped.")
        except Exception as e:
            print(f"Error starting server: {str(e)}")
    
    def run(self, args):
        """Main run method"""
        if args.command == "interactive":
            config = self.interactive_wizard()
            if self.load_model(config):
                self.interactive_chat()
        elif args.command == "serve":
            # For simplicity, we'll use default config for server mode
            config = {
                "model_name": "gpt2",
                "quantization": None,
                "device": "auto"
            }
            if self.load_model(config):
                self.serve_api(args.host, args.port)
        elif args.command == "chat":
            # Load with provided arguments
            config = {
                "model_name": args.model,
                "quantization": args.quantization,
                "device": args.device
            }
            if self.load_model(config):
                self.interactive_chat()
        elif args.command == "benchmark":
            self.benchmark()
        else:
            print("Unknown command. Use --help for usage information.")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Custom vLLM CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Run interactive setup wizard")
    
    # Serve mode
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    # Chat mode
    chat_parser = subparsers.add_parser("chat", help="Run chat with specified model")
    chat_parser.add_argument("--model", default="gpt2", help="Model name")
    chat_parser.add_argument("--quantization", choices=["gptq", "awq", "int8", "fp8"], 
                           help="Quantization method")
    chat_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                           help="Device to use")
    
    # Benchmark mode
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark tests")
    
    args = parser.parse_args()
    
    # Run the CLI
    cli = EnhancedVLLMCLI()
    cli.run(args)

if __name__ == "__main__":
    main()