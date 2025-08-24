"""
OpenAI-compatible API Server for Custom vLLM

This module provides a FastAPI-based server that implements the OpenAI API specification
for text generation, making it easy to integrate with existing applications.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import uvicorn
import asyncio
import uuid
import time
from vllm.src.model_loader import HuggingFaceModelLoader
from vllm.src.quantization import Quantizer
from vllm.src.inference import InferenceEngine


app = FastAPI(title="Custom vLLM API", description="OpenAI-compatible API for custom vLLM implementation")

# Global model variables
model = None
tokenizer = None
inference_engine = None

# Default generation configuration
default_generation_config = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "early_stopping": True,
}

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class CompletionResponseChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionResponseChoice]
    usage: Optional[Dict[str, int]] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = 16
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[Dict[str, int]] = None

class ModelConfigRequest(BaseModel):
    model: str
    quantization: Optional[str] = None
    device: Optional[str] = None

class GenerationConfigRequest(BaseModel):
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = None
    early_stopping: Optional[bool] = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup with default settings"""
    global model, tokenizer, inference_engine
    
    # For demonstration, we'll use a small model
    # In practice, you would load the model specified in the request
    try:
        model_loader = HuggingFaceModelLoader("gpt2")
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()
        
        # Initialize inference engine
        inference_engine = InferenceEngine()
        
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

@app.post("/v1/models/load")
async def load_model(config: ModelConfigRequest):
    """Load a specific model with configuration"""
    global model, tokenizer, inference_engine
    
    try:
        model_loader = HuggingFaceModelLoader(
            model_name=config.model,
            quantization=config.quantization,
            device=config.device
        )
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()
        
        # Initialize inference engine
        inference_engine = InferenceEngine(device=config.device or "cuda")
        
        return {"status": "success", "message": f"Model {config.model} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/v1/config/generation")
async def configure_generation(config: GenerationConfigRequest):
    """Configure generation parameters"""
    global default_generation_config
    
    # Update only provided parameters
    for key, value in config.dict().items():
        if value is not None:
            default_generation_config[key] = value
    
    return {"status": "success", "config": default_generation_config}

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [{
            "id": "gpt2",
            "object": "model",
            "created": 1677610602,
            "owned_by": "custom-vllm"
        }]
    }

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """Create a completion for the provided prompt"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Tokenize the input
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Prepare generation parameters
        gen_config = default_generation_config.copy()
        gen_config.update({
            "max_new_tokens": request.max_tokens or gen_config["max_new_tokens"],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "repetition_penalty": 1.0 + request.presence_penalty,  # Approximation
        })
        
        # Add top_k if specified
        if request.top_k and request.top_k > 0:
            gen_config["top_k"] = request.top_k
        
        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": gen_config["max_new_tokens"],
            "temperature": gen_config["temperature"],
            "top_p": gen_config["top_p"],
            "repetition_penalty": gen_config["repetition_penalty"],
            "do_sample": gen_config["do_sample"],
            "early_stopping": gen_config["early_stopping"],
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # Add top_k if specified
        if gen_config.get("top_k") and gen_config["top_k"] > 0:
            gen_kwargs["top_k"] = gen_config["top_k"]
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # Decode and clean the response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the response if it's included and echo is False
        if request.echo:
            response_text = full_response
        else:
            if full_response.startswith(request.prompt):
                response_text = full_response[len(request.prompt):].strip()
            else:
                response_text = full_response
        
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:10]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionResponseChoice(
                    text=response_text,
                    index=0,
                    finish_reason="length"
                )
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion for the provided messages"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Format messages as a single prompt
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        prompt += "\nassistant:"
        
        # Tokenize the input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Prepare generation parameters
        gen_config = default_generation_config.copy()
        gen_config.update({
            "max_new_tokens": request.max_tokens or gen_config["max_new_tokens"],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "repetition_penalty": 1.0 + request.presence_penalty,  # Approximation
        })
        
        # Add top_k if specified
        if request.top_k and request.top_k > 0:
            gen_config["top_k"] = request.top_k
        
        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": gen_config["max_new_tokens"],
            "temperature": gen_config["temperature"],
            "top_p": gen_config["top_p"],
            "repetition_penalty": gen_config["repetition_penalty"],
            "do_sample": gen_config["do_sample"],
            "early_stopping": gen_config["early_stopping"],
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # Add top_k if specified
        if gen_config.get("top_k") and gen_config["top_k"] > 0:
            gen_kwargs["top_k"] = gen_config["top_k"]
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # Decode and clean the response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if prompt in full_response:
            response_text = full_response[len(prompt):].strip()
        else:
            response_text = full_response
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/v1/config")
async def get_config():
    """Get current generation configuration"""
    return {"generation_config": default_generation_config}

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)