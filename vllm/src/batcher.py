"""
Request Batching for Custom vLLM/llama.cpp

This module provides functionality for continuous request batching in the serving architecture.
"""

import torch
from torch.nn import functional as F
from typing import List, Tuple, Dict, Optional

class RequestBatcher:
    def __init__(self, model: torch.nn.Module, max_seq_len: int = 2048, batch_size: int = 8):
        self.model = model
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.inputs = []
        self.tokenizer = None
        self.device = next(model.parameters()).device if model.parameters() else torch.device("cpu")
    
    def add_request(self, text: str):
        """Add a new request to the batcher"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Please set the tokenizer before adding requests.")
        
        # Tokenize the input text
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, 
                               max_length=self.max_seq_len).input_ids[0]
        
        # Move to device (keep as long for embedding layers)
        tokens = tokens.to(self.device).long()
        
        # Check if the input is within the max sequence length
        if len(tokens) > self.max_seq_len:
            print(f"Warning: Input text is too long: {len(tokens)} > {self.max_seq_len}. Truncating...")
            tokens = tokens[:self.max_seq_len]
        
        self.inputs.append(tokens)
    
    def finalize_batch(self):
        """Finalize the current batch and process it"""
        if not self.inputs:
            return None
        
        try:
            # Debug information
            print(f"Number of inputs: {len(self.inputs)}")
            for i, tensor in enumerate(self.inputs):
                print(f"Input {i}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
            
            # Pad the inputs to the maximum sequence length in the batch
            max_len = max(len(tensor) for tensor in self.inputs)
            max_len = min(max_len, self.max_seq_len)  # Don't exceed max_seq_len
            print(f"Max length: {max_len}")
            
            # Pad all sequences to the same length (using long for embedding)
            padded_inputs = []
            for i, tensor in enumerate(self.inputs):
                if len(tensor) < max_len:
                    padded_tensor = F.pad(tensor, (0, max_len - len(tensor)), value=self.tokenizer.pad_token_id or 0)
                else:
                    padded_tensor = tensor[:max_len]
                print(f"Padded tensor {i}: shape={padded_tensor.shape}, dtype={padded_tensor.dtype}, device={padded_tensor.device}")
                padded_inputs.append(padded_tensor)
            
            # Convert to tensor (keep as long for embedding layers)
            inputs_tensor = torch.stack(padded_inputs).long()
            print(f"Stacked tensor: shape={inputs_tensor.shape}, dtype={inputs_tensor.dtype}, device={inputs_tensor.device}")
            
            # Move to device
            inputs_tensor = inputs_tensor.to(self.device)
            print(f"Final tensor: shape={inputs_tensor.shape}, dtype={inputs_tensor.dtype}, device={inputs_tensor.device}")
            
            # Debug model information
            print(f"Model device: {next(self.model.parameters()).device}")
            print(f"Model dtype: {next(self.model.parameters()).dtype}")
            
            # Process the batch
            with torch.no_grad():
                outputs = self.model(inputs_tensor)
            
            # Return the processed outputs
            return outputs
            
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Clear the inputs for the next batch
            self.inputs = []
    
    def start_batching(self, model: torch.nn.Module = None, inference_engine = None):
        """Start continuous batching process (simplified implementation)"""
        if model is not None:
            self.model = model
            
        print("Starting request batching...")
        print("Note: This is a simplified implementation. A full continuous batching system would be more complex.")
        
        # In a full implementation, this would run a background process
        # that continuously collects requests and processes them in batches

# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Create a simple model for demonstration
    model = torch.nn.Linear(10, 10)
    
    batcher = RequestBatcher(model, max_seq_len=2048, batch_size=8)
    
    # Note: In practice, you would set a real tokenizer here
    # batcher.tokenizer = tokenizer
    
    print("RequestBatcher initialized successfully")