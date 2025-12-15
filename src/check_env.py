# -*- coding: utf-8 -*-
import torch
import sys

def check_env():
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # 检查 CUDA
    if torch.cuda.is_available():
        print(f"\n[SUCCESS] CUDA is available!")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.get_device_name(0)}")
        
        # 做一个简单的矩阵乘法测试 GPU 速度
        print("\nRunning a simple matrix multiplication on GPU...")
        a = torch.randn(4096, 4096).cuda()
        b = torch.randn(4096, 4096).cuda()
        
        # 预热
        torch.matmul(a, b)
        
        # 计时
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        c = torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        
        print(f"Compute finished. Time: {start.elapsed_time(end):.2f} ms")
    else:
        print("\n[WARNING] CUDA is NOT available. You are running on CPU.")
        print("For an AI Infra project, a GPU is highly recommended.")

if __name__ == "__main__":
    check_env()