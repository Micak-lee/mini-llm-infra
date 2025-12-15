# -*- coding: utf-8 -*-
import time
import torch
from load_weights import load_weights  # 复用之前的加载函数

def generate_stream(model, prompt, max_new_tokens=100):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    device = next(model.parameters()).device
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    # ---------------------------------------------------------
    # 阶段 1: Prefill (处理 Prompt)
    # ---------------------------------------------------------
    start_time = time.time()
    
    # 第一次 forward，处理整个 Prompt，并填满 KV Cache
    with torch.no_grad():
        logits = model(input_ids, start_pos=0)
    
    # 获取第一个生成的 token
    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
    generated_ids = [next_token.item()]
    
    print(f"Prompt: {prompt}")
    print(f"Generated: ", end="", flush=True)
    
    # 记录当前 Cache 填到了哪里
    curr_pos = input_ids.shape[1] 
    current_token_tensor = next_token
    
    # ---------------------------------------------------------
    # 阶段 2: Decoding (逐个生成) - 这里的速度最关键
    # ---------------------------------------------------------
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # 关键点：只喂入最新的 1 个 token
            # start_pos 告诉模型去 Cache 的哪个位置取历史信息
            logits = model(current_token_tensor, start_pos=curr_pos)
            
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        
        print(tokenizer.decode(next_token[0]), end="", flush=True)
        generated_ids.append(next_token.item())
        
        # 更新状态
        current_token_tensor = next_token
        curr_pos += 1
        
        if next_token.item() == tokenizer.eos_token_id:
            break

    end_time = time.time()
    total_tokens = len(generated_ids)
    duration = end_time - start_time
    
    print(f"\n\n[Stats]")
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Total Time: {duration:.2f}s")
    print(f"Throughput: {total_tokens / duration:.2f} tokens/s")

if __name__ == "__main__":
    # 1. 加载模型 (复用之前的逻辑)
    model, args = load_weights()
    
    # 2. 运行 KV Cache 推理
    prompt = "The future of AI infrastructure is"
    generate_stream(model, prompt, max_new_tokens=100)