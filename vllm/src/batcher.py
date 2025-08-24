"""
Request Batching for Custom vLLM/llama.cpp

This module provides functionality for continuous request batching in the serving architecture.
"""

import torch
from torch.nn import functional as F
from typing import List, Tuple, Dict

class RequestBatcher:
    def __init__(self, model: torch.nn.Module, max_seq_len: int = 2048, batch_size: int = 8):
        self.model = model
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.inputs = []
        self.targets = []
        self.tokenizer = None
    
    def add_request(self, text: str):
        """Add a new request to the batcher"""
        # Tokenize the input text
        tokens = self.tokenizer(text, return_tensors="pt").input_ids[0]
        
        # Check if the input is within the max sequence length
        if len(tokens) > self.max_seq_len:
            raise ValueError(f"Input text is too long: {len(tokens)} > {self.max_seq_len}")
        
        self.inputs.append(tokens)
    
    def finalize_batch(self):
        """Finalize the current batch and process it"""
        if not self.inputs:
            return None
        
        # Pad the inputs to the maximum sequence length
        self.inputs = [F.pad(tensor, (0, self.max_seq_len - len(tensor)), value=0) for tensor in self.inputs]
        
        # Convert to tensor
        inputs_tensor = torch.stack(self.inputs)
        
        # Process the batch
        outputs = self.model(inputs_tensor)
        
        # Return the processed outputs
        return outputs
    
    def get_tokenizer(self):
        return self.tokenizer


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")
    model = torch.nn.Linear(10, 10)
    
    batcher = RequestBatcher(model, max_seq_len=2048, batch_size=8)
    batcher.tokenizer = tokenizer
    
    # Add some requests
    batcher.add_request("What is the capital of France?")
    batcher.add_request("Who is the president of the United States?")
    batcher.add_request("Explain quantum computing in simple terms.")
    
    # Finalize and process the batch
    outputs = batcher.finalize_batch()
    print(outputs)