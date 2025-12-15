# -*- coding: utf-8 -*-
import os
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download
from model import Llama, ModelArgs

# === 配置路径 ===
# 如果你真的想要根目录下的 /weight，请确保你有权限 (sudo mkdir /weight && sudo chmod 777 /weight)
# 推荐使用项目内的相对路径 "./weight"
MODEL_PATH = "./weight" 
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_weights():
    print(f"1. Checking model weights in {MODEL_PATH}...")
    
    # === 1. 下载阶段 (如果本地没有，就去下载) ===
    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        print(f"   [Info] Weights not found. Downloading to {MODEL_PATH}...")
        try:
            snapshot_download(
                repo_id=MODEL_ID, 
                local_dir=MODEL_PATH, 
                local_dir_use_symlinks=False # 关键：不使用软链接，直接下载真实文件
            )
            print("   [Success] Download complete.")
        except Exception as e:
            print(f"   [Error] Download failed: {e}")
            raise e
    else:
        print("   [Info] Local weights found. Skipping download.")

    # === 2. 加载 HF 模型 (用于提取权重) ===
    print("2. Loading HF model from local storage...")
    # 注意：这里路径直接传 MODEL_PATH (本地文件夹路径)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="cpu",
        local_files_only=True # 强制只读本地，断网也能跑
    )
    hf_state_dict = hf_model.state_dict()
    print("   HF Model loaded. Extracting weights...")

    print("3. Initializing Local Llama Model (Low RAM Mode)...")
    
    # TinyLlama Config
    args = ModelArgs(
        dim=2048,
        n_layers=22,
        n_heads=32,
        n_kv_heads=4,
        vocab_size=32000,
        multiple_of=256,
        norm_eps=1e-5,
        max_seq_len=2048
    )
    
    print("   Allocating model in float16...")
    with torch.device("cpu"):
        torch.set_default_dtype(torch.float16)
        local_model = Llama(args)
        torch.set_default_dtype(torch.float32)

    local_state_dict = local_model.state_dict()

    print("4. Mapping Weights...")
    mapping = {
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "norm.weight":           "model.norm.weight",
        "output.weight":         "lm_head.weight",
    }
    
    for i in range(args.n_layers):
        local_prefix = f"layers.{i}"
        hf_prefix = f"model.layers.{i}"
        layer_mapping = {
            f"{local_prefix}.attention_norm.weight": f"{hf_prefix}.input_layernorm.weight",
            f"{local_prefix}.ffn_norm.weight":       f"{hf_prefix}.post_attention_layernorm.weight",
            f"{local_prefix}.attention.wq.weight":   f"{hf_prefix}.self_attn.q_proj.weight",
            f"{local_prefix}.attention.wk.weight":   f"{hf_prefix}.self_attn.k_proj.weight",
            f"{local_prefix}.attention.wv.weight":   f"{hf_prefix}.self_attn.v_proj.weight",
            f"{local_prefix}.attention.wo.weight":   f"{hf_prefix}.self_attn.o_proj.weight",
            f"{local_prefix}.feed_forward.w1.weight": f"{hf_prefix}.mlp.gate_proj.weight",
            f"{local_prefix}.feed_forward.w2.weight": f"{hf_prefix}.mlp.down_proj.weight",
            f"{local_prefix}.feed_forward.w3.weight": f"{hf_prefix}.mlp.up_proj.weight",
        }
        mapping.update(layer_mapping)

    count = 0
    for local_name, hf_name in mapping.items():
        if hf_name in hf_state_dict:
            hf_param = hf_state_dict[hf_name]
            local_state_dict[local_name].copy_(hf_param)
            count += 1
        else:
            print(f"!! Missing key: {hf_name}")

    print(f"5. Weights Loaded! ({count} tensors copied)")
    
    del hf_model, hf_state_dict
    import gc
    gc.collect()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Moving model to {device}...")
    local_model = local_model.to(device)
    local_model.eval()
    
    return local_model, args