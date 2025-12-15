# -*- coding: utf-8 -*-
import torch
from model import Llama, ModelArgs

def test_model_structure():
    print("Initializing TinyLlama configuration...")
    # 使用一个小一点的配置进行快速测试
    args = ModelArgs(
        dim=256,        # 缩小维度
        n_layers=2,     # 减少层数
        n_heads=4,
        n_kv_heads=2,
        vocab_size=1000
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on {device}...")
    
    model = Llama(args).to(device)
    
    # 构造一个 dummy 输入: batch_size=2, seq_len=10
    x = torch.randint(0, args.vocab_size, (2, 10)).to(device)
    
    print("Running forward pass...")
    try:
        logits = model(x)
        print(f"Output shape: {logits.shape}")
        
        # 验证输出形状应该为 (Batch, SeqLen, VocabSize)
        expected_shape = (2, 10, args.vocab_size)
        assert logits.shape == expected_shape
        print("\n[SUCCESS] Model forward pass works! Shapes align correctly.")
        
    except Exception as e:
        print(f"\n[ERROR] Forward pass failed: {e}")
        raise e

if __name__ == "__main__":
    test_model_structure()