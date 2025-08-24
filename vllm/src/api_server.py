"""
OpenAI-compatible API Server for Custom vLLM

This module provides a FastAPI-based server that implements the OpenAI API specification
for text generation, making it easy to integrate with existing applications.
"""

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, AsyncGenerator
import torch
import uvicorn
import asyncio
import uuid
import time
import warnings
import json
from vllm.src.model_loader import HuggingFaceModelLoader
from vllm.src.quantization import Quantizer
from vllm.src.inference import InferenceEngine


app = FastAPI(title="Custom vLLM API", description="OpenAI-compatible API for custom vLLM implementation")

# Global model variables
model = None
tokenizer = None
inference_engine = None
model_config = None
save_mode_enabled = False

# Default generation configuration (will be updated when model is loaded)
default_generation_config = {
    "max_new_tokens": 300,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "early_stopping": True,
    "context_length": 2048,
}

def get_model_max_tokens(model_obj):
    """Extract maximum context length from the loaded model"""
    max_context_length = 2048  # Default fallback
    
    try:
        if hasattr(model_obj, 'config'):
            config = model_obj.config
            
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

def get_model_defaults(model_obj):
    """Extract default parameters from the loaded model"""
    # Get model's maximum context length
    max_context_length = get_model_max_tokens(model_obj)
    
    defaults = {
        "max_new_tokens": min(500, max_context_length // 4),  # Use 1/4 of context or 500, whichever is smaller
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
        if hasattr(model_obj, 'config'):
            config = model_obj.config
            
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
    save_mode: Optional[bool] = False  # New parameter for save mode

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
    save_mode: Optional[bool] = False  # New parameter for save mode

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
    save_mode: Optional[bool] = False  # New parameter for save mode

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
    global model, tokenizer, inference_engine, model_config, default_generation_config, save_mode_enabled
    
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
        
        # Enable save mode if requested
        if config.save_mode and torch.cuda.is_available():
            save_mode_enabled = inference_engine.enable_save_mode()
        else:
            save_mode_enabled = False
        
        # Get model defaults
        default_generation_config = get_model_defaults(model)
        
        return {
            "status": "success", 
            "message": f"Model {config.model} loaded successfully", 
            "defaults": default_generation_config,
            "save_mode": save_mode_enabled
        }
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

async def stream_completion_response(prompt: str, gen_config: dict, request_id: str, use_save_mode: bool = False) -> AsyncGenerator[str, None]:
    """Stream completion response token by token"""
    global model, tokenizer, inference_engine
    
    try:
        # Enable save mode if requested
        if use_save_mode and torch.cuda.is_available():
            inference_engine.enable_save_mode()
        
        # Tokenize the input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=gen_config.get('context_length', 2048)
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Prepare generation parameters
        gen_kwargs = {
            "temperature": gen_config["temperature"],
            "top_p": gen_config["top_p"],
            "repetition_penalty": gen_config["repetition_penalty"],
            "do_sample": gen_config["do_sample"],
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "early_stopping": True,
        }
        
        # Add top_k if specified and > 0
        if gen_config.get("top_k", 0) > 0:
            gen_kwargs["top_k"] = gen_config["top_k"]
        
        # Stream token by token
        current_inputs = {k: v.clone() for k, v in inputs.items()}
        generated_text = ""
        
        for i in range(gen_config["max_new_tokens"]):
            # Generate next token
            outputs = model.generate(
                **current_inputs,
                max_new_tokens=1,
                **{k: v for k, v in gen_kwargs.items() if k not in ['max_new_tokens']}
            )
            
            # Get the new token
            new_token_id = outputs[0, -1].unsqueeze(0).unsqueeze(0)
            new_token = tokenizer.decode([new_token_id.item()])
            
            # Check for EOS token
            if new_token_id.item() == tokenizer.eos_token_id:
                break
            
            # Append to generated text
            generated_text += new_token
            
            # Create streaming response
            chunk = {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": "custom-vllm",
                "choices": [{
                    "text": new_token,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None
                }]
            }
            
            yield f"data: {json.dumps(chunk)}\n\n"
            
            # Update inputs for next iteration
            current_inputs = {
                k: torch.cat([v, new_token_id.to(v.device)], dim=1) 
                for k, v in current_inputs.items()
            }
            
            # Small delay to allow for streaming
            await asyncio.sleep(0.01)
        
        # Send final chunk with finish reason
        final_chunk = {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": "custom-vllm",
            "choices": [{
                "text": "",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length"
            }]
        }
        
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "streaming_error",
                "param": None,
                "code": None
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """Create a completion for the provided prompt"""
    global model, tokenizer, default_generation_config, inference_engine, save_mode_enabled
    
    # Enable save mode if requested
    if request.save_mode and torch.cuda.is_available():
        save_mode_enabled = inference_engine.enable_save_mode()
    
    # Handle streaming requests
    if request.stream:
        request_id = f"cmpl-{uuid.uuid4().hex[:10]}"
        return StreamingResponse(
            stream_completion_response(request.prompt, default_generation_config, request_id, request.save_mode),
            media_type="text/event-stream"
        )
    
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
            "early_stopping": True,
        }
        
        # Add top_k if specified and > 0
        if gen_config.get("top_k", 0) > 0:
            gen_kwargs["top_k"] = gen_config["top_k"]
        
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
        
        # Handle empty or very short responses
        if not response_text or len(response_text.strip()) < 5:
            response_text = "(No meaningful response generated. Try a more specific prompt.)"
        
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

async def stream_chat_response(prompt: str, gen_config: dict, request_id: str, use_save_mode: bool = False) -> AsyncGenerator[str, None]:
    """Stream chat response token by token"""
    global model, tokenizer, inference_engine
    
    try:
        # Enable save mode if requested
        if use_save_mode and torch.cuda.is_available():
            inference_engine.enable_save_mode()
        
        # Tokenize the input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=gen_config.get('context_length', 2048)
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Prepare generation parameters
        gen_kwargs = {
            "temperature": gen_config["temperature"],
            "top_p": gen_config["top_p"],
            "repetition_penalty": gen_config["repetition_penalty"],
            "do_sample": gen_config["do_sample"],
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "early_stopping": True,
        }
        
        # Add top_k if specified and > 0
        if gen_config.get("top_k", 0) > 0:
            gen_kwargs["top_k"] = gen_config["top_k"]
        
        # Stream token by token
        current_inputs = {k: v.clone() for k, v in inputs.items()}
        generated_text = ""
        
        for i in range(gen_config["max_new_tokens"]):
            # Generate next token
            outputs = model.generate(
                **current_inputs,
                max_new_tokens=1,
                **{k: v for k, v in gen_kwargs.items() if k not in ['max_new_tokens']}
            )
            
            # Get the new token
            new_token_id = outputs[0, -1].unsqueeze(0).unsqueeze(0)
            new_token = tokenizer.decode([new_token_id.item()])
            
            # Check for EOS token
            if new_token_id.item() == tokenizer.eos_token_id:
                break
            
            # Append to generated text
            generated_text += new_token
            
            # Create streaming response
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "custom-vllm",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": new_token
                    },
                    "finish_reason": None
                }]
            }
            
            yield f"data: {json.dumps(chunk)}\n\n"
            
            # Update inputs for next iteration
            current_inputs = {
                k: torch.cat([v, new_token_id.to(v.device)], dim=1) 
                for k, v in current_inputs.items()
            }
            
            # Small delay to allow for streaming
            await asyncio.sleep(0.01)
        
        # Send final chunk with finish reason
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "custom-vllm",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "length"
            }]
        }
        
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "streaming_error",
                "param": None,
                "code": None
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion for the provided messages"""
    global model, tokenizer, default_generation_config, inference_engine, save_mode_enabled
    
    # Enable save mode if requested
    if request.save_mode and torch.cuda.is_available():
        save_mode_enabled = inference_engine.enable_save_mode()
    
    # Handle streaming requests
    if request.stream:
        # Format messages as a single prompt
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        prompt += "\nassistant:"
        request_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"
        return StreamingResponse(
            stream_chat_response(prompt, default_generation_config, request_id, request.save_mode),
            media_type="text/event-stream"
        )
    
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
            "early_stopping": True,
        }
        
        # Add top_k if specified and > 0
        if gen_config.get("top_k", 0) > 0:
            gen_kwargs["top_k"] = gen_config["top_k"]
        
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
        
        # Handle empty or very short responses
        if not response_text or len(response_text.strip()) < 5:
            response_text = "(No meaningful response generated. Try a more specific prompt.)"
        
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
    return {
        "status": "healthy", 
        "model_loaded": model is not None, 
        "model_defaults": default_generation_config,
        "save_mode_enabled": save_mode_enabled
    }

@app.get("/v1/config")
async def get_config():
    """Get current generation configuration"""
    return {"generation_config": default_generation_config}

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)