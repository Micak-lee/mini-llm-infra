# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 在 import torch 下面加：
try:
    from kernel import triton_rms_norm
    USE_TRITON = True
    print("[INFO] Triton loaded successfully!")
except ImportError:
    USE_TRITON = False
    print("[WARN] Triton not found, using PyTorch fallback.")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class ModelArgs:
    dim: int = 2048
    n_layers: int = 22
    n_heads: int = 32
    n_kv_heads: int = 4
    vocab_size: int = 32000
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    max_batch_size: int = 1  # 新增：支持的最大 Batch Size
    
    @property
    def head_dim(self):
        return self.dim // self.n_heads

# -----------------------------------------------------------------------------
# Infrastructure Component: KV Cache
# -----------------------------------------------------------------------------
class KVCache(nn.Module):
    """
    Pre-allocated Key-Value Cache.
    Fixed memory buffer to avoid dynamic allocation overhead during generation.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, args.n_kv_heads, args.head_dim),
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, args.n_kv_heads, args.head_dim),
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16
        )

    def update(self, xk: torch.Tensor, xv: torch.Tensor, layer_idx: int, start_pos: int):
        """
        Update the cache with new tokens and return the full history.
        xk, xv: (Batch, SeqLen, n_kv_heads, HeadDim)
        """
        bsz, seqlen, _, _ = xk.shape
        
        # 写入 Cache (Filling the slot)
        # 注意：这里我们假设 start_pos 是当前新 token 的起始位置
        self.cache_k[:bsz, start_pos : start_pos + seqlen, :, :] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen, :, :] = xv
        
        # 读取历史 (Reading back history)
        # 只要读到 start_pos + seqlen 即可
        keys = self.cache_k[:bsz, : start_pos + seqlen, :, :]
        values = self.cache_v[:bsz, : start_pos + seqlen, :, :]
        
        return keys, values

# -----------------------------------------------------------------------------
# Basic Components
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
            # 动态分发策略 (Dynamic Dispatch)
            # 这里的 128 是根据你的 benchmark 数据推断出的"甜点"阈值
            # 当总 Token 数 (Batch * SeqLen) 超过 128 时，Triton 的高带宽优势抵消了启动开销
            
            # x.numel() 是元素总数，除以 hidden_dim 得到 token 数 (行数)
            num_tokens = x.numel() // self.weight.numel()

            if USE_TRITON and x.is_cuda and num_tokens > 128:
                return triton_rms_norm(x.contiguous(), self.weight, self.eps)
            else:
                # N <= 128 (通常是 Decoding 阶段)，走 PyTorch 原生路径
                output = self._norm(x.float()).type_as(x)
                return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    ndim = xq_.ndim
    assert 0 <= 1 < ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
    freqs_cis = freqs_cis.view(*shape)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# -----------------------------------------------------------------------------
# Core Layers
# -----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads 
        self.head_dim = args.head_dim
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], 
                kv_cache: Optional[KVCache] = None, layer_idx: int = 0, start_pos: int = 0):
        bsz, seqlen, _ = x.shape
        
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # === KV Cache Logic ===
        if kv_cache is not None:
            # Update cache and get full history
            keys, values = kv_cache.update(xk, xv, layer_idx, start_pos)
        else:
            # Training mode or No-Cache mode
            keys, values = xk, xv
        
        # GQA handling
        if self.n_kv_heads < self.n_heads:
             n_rep = self.n_heads // self.n_kv_heads
             keys = keys[:, :, :, None, :].expand(bsz, keys.size(1), self.n_kv_heads, n_rep, self.head_dim).reshape(bsz, keys.size(1), self.n_heads, self.head_dim)
             values = values[:, :, :, None, :].expand(bsz, values.size(1), self.n_kv_heads, n_rep, self.head_dim).reshape(bsz, values.size(1), self.n_heads, self.head_dim)

        xq = xq.transpose(1, 2)   # (B, H, Seq, Dim)
        keys = keys.transpose(1, 2) # (B, H, CacheSeq, Dim)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis, mask, kv_cache=None, layer_idx=0, start_pos=0):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, kv_cache, layer_idx, start_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Llama(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        self.freqs_cis = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2)
        
        # Initialize Cache Storage
        # 每一层都需要一个独立的 Cache
        self.kv_caches = nn.ModuleList([KVCache(args) for _ in range(args.n_layers)])

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

        for i, layer in enumerate(self.layers):
            # 将对应层的 cache 传进去
            h = layer(h, freqs_cis, mask, self.kv_caches[i], layer_idx=i, start_pos=start_pos)
            
        h = self.norm(h)
        output = self.output(h)
        return output