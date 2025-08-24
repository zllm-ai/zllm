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
import warnings
from vllm.src.model_loader import HuggingFaceModelLoader
from vllm.src.quantization import Quantizer
from vllm.src.inference import InferenceEngine


app = FastAPI(title="Custom vLLM API", description="OpenAI-compatible API for custom vLLM implementation")

# Global model variables
model = None
tokenizer = None
inference_engine = None
model_config = None

# Default generation configuration (will be updated when model is loaded)
default_generation_config = {
    "max_new_tokens": 200,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "do_sample": True,
    "early_stopping": False,
}

def get_model_defaults(model_obj):
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
        if hasattr(model_obj, 'config'):
            config = model_obj.config
            
            # Get generation defaults from model config
            if hasattr(config, 'temperature'):
                defaults['temperature'] = config.temperature
            if hasattr(config, 'top_p'):
                defaults['top_p'] = config.top_p
            if hasattr(config, 'repetition_penalty'):
                defaults['repetition_penalty'] = config.repetition_penalty
                
    except Exception as e:
        print(f"Could not extract model defaults: {e}")
        
    return defaults

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: Optional[int] = None
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
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
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
    global model, tokenizer, inference_engine, model_config, default_generation_config
    
    # For demonstration, we'll use a small model
    # In practice, you would load the model specified in the request
    try:
        model_loader = HuggingFaceModelLoader("gpt2")
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()
        model_config = getattr(model, 'config', None)
        
        # Initialize inference engine
        inference_engine = InferenceEngine()
        
        # Get model defaults
        default_generation_config = get_model_defaults(model)
        
        print("Model loaded successfully with automatic configuration")
        print(f"Model defaults: {default_generation_config}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

@app.post("/v1/models/load")
async def load_model(config: ModelConfigRequest):
    """Load a specific model with configuration"""
    global model, tokenizer, inference_engine, model_config, default_generation_config
    
    try:
        model_loader = HuggingFaceModelLoader(
            model_name=config.model,
            quantization=config.quantization,
            device=config.device
        )
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()
        model_config = getattr(model, 'config', None)
        
        # Initialize inference engine
        inference_engine = InferenceEngine(device=config.device or "cuda")
        
        # Get model defaults
        default_generation_config = get_model_defaults(model)
        
        return {"status": "success", "message": f"Model {config.model} loaded successfully", "defaults": default_generation_config}
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
    global model, tokenizer, default_generation_config
    
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
        
        # Prepare generation parameters using model defaults, overridden by request parameters
        gen_config = default_generation_config.copy()
        
        # Override with request parameters if provided
        if request.max_tokens is not None:
            gen_config["max_new_tokens"] = request.max_tokens
        if request.temperature is not None:
            gen_config["temperature"] = request.temperature
        if request.top_p is not None:
            gen_config["top_p"] = request.top_p
        if request.top_k is not None:
            gen_config["top_k"] = request.top_k
        if request.presence_penalty is not None:
            gen_config["repetition_penalty"] = 1.0 + request.presence_penalty  # Approximation
        
        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": gen_config["max_new_tokens"],
            "temperature": gen_config["temperature"],
            "top_p": gen_config["top_p"],
            "repetition_penalty": gen_config["repetition_penalty"],
            "do_sample": gen_config["do_sample"],
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # Add top_k if specified and > 0
        if gen_config.get("top_k", 0) > 0:
            gen_kwargs["top_k"] = gen_config["top_k"]
        
        # Only add early_stopping if the model supports it and it's enabled
        if gen_config.get('early_stopping', False):
            try:
                # Test if early_stopping is supported
                dummy_inputs = {k: v[:1] for k, v in inputs.items()}  # Create minimal inputs
                model.generate(**dummy_inputs, **gen_kwargs, early_stopping=True, max_new_tokens=1)
                gen_kwargs["early_stopping"] = True
            except Exception:
                # early_stopping not supported, continue without it
                pass
        
        # Suppress transformer warnings
        warnings.filterwarnings("ignore")
        
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
    global model, tokenizer, default_generation_config
    
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
        
        # Prepare generation parameters using model defaults, overridden by request parameters
        gen_config = default_generation_config.copy()
        
        # Override with request parameters if provided
        if request.max_tokens is not None:
            gen_config["max_new_tokens"] = request.max_tokens
        if request.temperature is not None:
            gen_config["temperature"] = request.temperature
        if request.top_p is not None:
            gen_config["top_p"] = request.top_p
        if request.top_k is not None:
            gen_config["top_k"] = request.top_k
        if request.presence_penalty is not None:
            gen_config["repetition_penalty"] = 1.0 + request.presence_penalty  # Approximation
        
        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": gen_config["max_new_tokens"],
            "temperature": gen_config["temperature"],
            "top_p": gen_config["top_p"],
            "repetition_penalty": gen_config["repetition_penalty"],
            "do_sample": gen_config["do_sample"],
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # Add top_k if specified and > 0
        if gen_config.get("top_k", 0) > 0:
            gen_kwargs["top_k"] = gen_config["top_k"]
        
        # Only add early_stopping if the model supports it and it's enabled
        if gen_config.get('early_stopping', False):
            try:
                # Test if early_stopping is supported
                dummy_inputs = {k: v[:1] for k, v in inputs.items()}  # Create minimal inputs
                model.generate(**dummy_inputs, **gen_kwargs, early_stopping=True, max_new_tokens=1)
                gen_kwargs["early_stopping"] = True
            except Exception:
                # early_stopping not supported, continue without it
                pass
        
        # Suppress transformer warnings
        warnings.filterwarnings("ignore")
        
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
    return {"status": "healthy", "model_loaded": model is not None, "model_defaults": default_generation_config}

@app.get("/v1/config")
async def get_config():
    """Get current generation configuration"""
    return {"generation_config": default_generation_config}

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)