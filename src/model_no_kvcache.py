# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class ModelArgs:
    dim: int = 2048             # Embedding dimension
    n_layers: int = 22          # Number of transformer layers
    n_heads: int = 32           # Number of query heads
    n_kv_heads: int = 4         # Number of key/value heads (GQA)
    vocab_size: int = 32000     # Vocabulary size
    multiple_of: int = 256      # FFN hidden dimension alignment
    norm_eps: float = 1e-5      # Epsilon for RMSNorm
    max_seq_len: int = 2048     # Maximum context length
    
    @property
    def head_dim(self):
        return self.dim // self.n_heads

# -----------------------------------------------------------------------------
# Basic Components: RMSNorm & RoPE
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Unlike LayerNorm, it does not re-center the mean, only normalizes variance.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for Rotary Position Embeddings (RoPE).
    Returns complex tensors.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # Convert to polar form: cos(theta) + i*sin(theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply RoPE to Query and Key states.
    """
    # 1. View last dim as complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 2. Reshape freqs_cis to match xq dimensions for broadcasting
    ndim = xq_.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (xq_.shape[1], xq_.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
    freqs_cis = freqs_cis.view(*shape)
    
    # 3. Rotate via complex multiplication and flatten back
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

# -----------------------------------------------------------------------------
# Core Layers: Attention & FeedForward
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

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        
        # 1. QKV Projections
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # 2. Reshape heads
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # 3. Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 4. Handle GQA (Grouped Query Attention)
        # If KV heads < Query heads, we need to repeat KV heads
        if self.n_kv_heads < self.n_heads:
             n_rep = self.n_heads // self.n_kv_heads
             xk = xk[:, :, :, None, :].expand(bsz, seqlen, self.n_kv_heads, n_rep, self.head_dim).reshape(bsz, seqlen, self.n_heads, self.head_dim)
             xv = xv[:, :, :, None, :].expand(bsz, seqlen, self.n_kv_heads, n_rep, self.head_dim).reshape(bsz, seqlen, self.n_heads, self.head_dim)

        # 5. Attention Calculation
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    SwiGLU FeedForward Layer.
    Formula: w2(F.silu(w1(x)) * w3(x))
    """
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

# -----------------------------------------------------------------------------
# Transformer Block & Llama Model
# -----------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
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
        
        # Initialize RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
            
        h = self.norm(h)
        output = self.output(h)
        return output