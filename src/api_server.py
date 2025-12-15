# -*- coding: utf-8 -*-
import json
import time
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import our custom modules
from model import Llama, ModelArgs
from load_weights import load_weights

# Try to import kernel to enable Triton if available
try:
    import kernel 
except ImportError:
    pass

# -----------------------------------------------------------------------------
# Global State (Singleton Pattern)
# -----------------------------------------------------------------------------
# We store the model in a global dictionary to ensure it's loaded only once.
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # [Startup]: Executed when the server starts
    print("[INFO] Loading model into GPU memory...")
    try:
        # Load weights using our custom loader
        model, args = load_weights()
        ml_models["model"] = model
        ml_models["args"] = args
        
        # Load Tokenizer (using HF implementation for simplicity)
        # 修改前
        # ml_models["tokenizer"] = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        # 修改后
        # 这一行要和 load_weights.py 里的 MODEL_PATH 保持一致
        local_model_path = "./weight" 

        from transformers import AutoTokenizer
        # local_files_only=True 保证如果本地没文件它会报错而不是悄悄去联网下载
        ml_models["tokenizer"] = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
        
        print("[INFO] Model loaded successfully! Ready to serve.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise e
        
    yield # Server is running...
    
    # [Shutdown]: Executed when server stops
    print("[INFO] Cleaning up GPU memory...")
    ml_models.clear()
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

# -----------------------------------------------------------------------------
# Inference Logic (Generator)
# -----------------------------------------------------------------------------
def stream_generation_logic(prompt, max_new_tokens, temperature):
    """
    A Python Generator that yields tokens one by one.
    Used for Server-Sent Events (SSE).
    """
    model = ml_models["model"]
    tokenizer = ml_models["tokenizer"]
    device = next(model.parameters()).device
    
    # 1. Prefill Phase
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    # Reset/Update KV Cache logic would go here in a production system.
    # For this demo, we assume the model handles cache reset internally or we rely on fresh inference.
    
    curr_pos = 0
    start_pos = 0 
    
    # First Forward Pass (Processing Prompt)
    with torch.no_grad():
        logits = model(input_ids, start_pos=start_pos)
    
    # Get the first token
    next_token_logits = logits[:, -1, :]
    if temperature > 0:
        probs = torch.softmax(next_token_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
    # Decode first word
    first_word = tokenizer.decode(next_token[0], skip_special_tokens=True)
    yield first_word
    
    # Update state
    current_token_tensor = next_token
    # start_pos becomes the length of the prompt
    start_pos = input_ids.shape[1] 
    
    # 2. Decoding Loop
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(current_token_tensor, start_pos=start_pos)
            
        next_token_logits = logits[:, -1, :]
        if temperature > 0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
        word = tokenizer.decode(next_token[0], skip_special_tokens=True)
        
        # Stop if EOS token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break
            
        yield word
        
        current_token_tensor = next_token
        start_pos += 1

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def generate_stream(req: GenerationRequest):
    """
    OpenAI-compatible streaming interface
    """
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    print(f"[Request] Prompt: {req.prompt[:50]}...")

    # Create generator
    generator = stream_generation_logic(req.prompt, req.max_tokens, req.temperature)

    # SSE Wrapper
    def sse_wrapper():
        start_time = time.time()
        token_count = 0
        for token in generator:
            token_count += 1
            # Construct JSON chunk
            chunk = {
                "choices": [{"delta": {"content": token}}],
                "usage": None
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        
        end_time = time.time()
        print(f"[Done] Speed: {token_count / (end_time - start_time):.2f} t/s")
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_wrapper(), media_type="text/event-stream")

if __name__ == "__main__":
    # Start uvicorn server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)