# -*- coding: utf-8 -*-
import torch
import triton
import triton.testing
from kernel import triton_rms_norm

# 定义 PyTorch 原生实现作为对比
def torch_rms_norm(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 用作 x 轴的变量名 (这里是 token 总数)
        x_vals=[128 * i for i in range(2, 100)],  # 从 256 到 12800
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['PyTorch', 'Triton'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='rms-norm-performance',
        args={'D': 2048, 'dtype': torch.float16},
    )
)
def benchmark(N, D, provider, dtype):
    # N: Batch * SeqLen (总 Token 数)
    # D: Hidden Dim
    x = torch.randn(N, D, device='cuda', dtype=dtype)
    w = torch.randn(D, device='cuda', dtype=dtype)
    eps = 1e-5
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_rms_norm(x, w, eps), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_rms_norm(x, w, eps), quantiles=quantiles)
        
    # 计算有效带宽 GB/s
    # 读 x (2 bytes) + 读 w (2 bytes) + 写 y (2 bytes) = 理论上至少读写 2N*D + 2N*D (简化)
    # 这里的公式只是近似，用于对比相对性能
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=False)