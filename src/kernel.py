# -*- coding: utf-8 -*-
import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_x_row,
    stride_out_row,
    N_COLS,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate start pointers
    x_row_start_ptr = x_ptr + row_idx * stride_x_row
    out_row_start_ptr = out_ptr + row_idx * stride_out_row
    
    # Generate offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_COLS

    # 1. Load x
    x = tl.load(x_row_start_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # 2. Load weight
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # 3. Calculate RMS
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / N_COLS
    rstd = tl.rsqrt(mean_sq + eps)

    # 4. Normalize and scale
    out = x * rstd * w

    # 5. Store result
    tl.store(out_row_start_ptr + offsets, out, mask=mask)


def triton_rms_norm(x, weight, eps):
    x = x.contiguous()
    y = torch.empty_like(x)
    x_flat = x.view(-1, x.shape[-1])
    
    M, N = x_flat.shape
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    

    rms_norm_kernel[(M,)](
        x, weight, y,
        x_flat.stride(0),  
        y.view(-1, y.shape[-1]).stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y