# Mini-LLM-Infra: High-Performance Inference System

**Technologies:** Python 3.10+ | PyTorch 2.0+ | OpenAI Triton | CUDA | FastAPI

> **A white-box, high-performance LLM inference engine built from scratch.**
>
> *Demonstrates advanced AI Infrastructure concepts: Zero-Copy KV Cache, Kernel Fusion, and Latency-Aware Dynamic Dispatch.*

---

## Project Overview

This project implements a Llama-2 architecture inference engine completely from the ground up, bypassing high-level abstractions like Hugging Face's `generate()` pipeline. It is engineered to optimize **memory bandwidth utilization** and **inference latency** on consumer-grade GPUs.

The system addresses two critical bottlenecks in Large Language Model serving:
1.  **Memory Bound Decoding:** Solved via a pre-allocated Key-Value Cache, reducing memory complexity from O(N^2) to O(1).
2.  **Kernel Overhead:** Solved via custom OpenAI Triton kernels for RMSNorm, achieving ~3x bandwidth improvement under high load.

### Key Features

* **Custom Llama Implementation:** Manual implementation of RoPE (Rotary Positional Embeddings), SwiGLU, and RMSNorm.
* **Zero-Copy KV Cache:** A static memory management system that eliminates redundant computations and memory fragmentation during auto-regressive generation.
* **Triton Kernel Fusion:** Custom fused kernels written in Triton DSL for RMSNorm, capable of reaching **218 GB/s** effective bandwidth (vs. 66 GB/s with native PyTorch).
* **Dynamic Backend Dispatch:** An intelligent routing layer that automatically switches between **Triton** (for high-throughput prefill) and **PyTorch Native** (for low-latency decoding) to minimize kernel launch overheads.
* **Production-Ready Hydration:** Supports downloading and loading model weights from a local directory (`./weight`), enabling fully offline air-gapped deployments.
* **Streaming API:** A FastAPI-based server implementing Server-Sent Events (SSE) for real-time token streaming, compatible with OpenAI client protocols.

---

## Performance Benchmarks

**Environment:** NVIDIA GPU (T4/3090) | PyTorch 2.1 | CUDA 12.1

### 1. Throughput Comparison (Tokens/sec)

| Method | Throughput | Improvement | Note |
| :--- | :--- | :--- | :--- |
| Naive Implementation | ~32 t/s | - | Re-computes full history |
| **With KV Cache** | **~66 t/s** | **+106%** | O(1) step complexity |

### 2. Operator Bandwidth (RMSNorm)

| Token Count (N) | PyTorch Native | Triton Fused Kernel | Conclusion |
| :--- | :--- | :--- | :--- |
| N < 1024 (Decoding) | ~103 GB/s | ~128 GB/s | Comparable due to launch overhead |
| **N > 4096 (Prefill)** | **~78 GB/s** | **~214 GB/s** | **Triton is ~2.7x Faster** |

*Note: The engine uses these thresholds to dynamically select the optimal backend at runtime.*

---

## Installation

This project uses `uv` for high-speed dependency management.

### 1. Prerequisites
* Linux environment with NVIDIA Drivers installed.
* Python 3.10 or higher.

### 2. Setup

```bash
# Clone the repository
git clone [https://github.com/yourusername/mini-llm-infra.git](https://github.com/yourusername/mini-llm-infra.git)
cd mini-llm-infra

# Install dependencies
uv init
uv add torch transformers triton fastapi uvicorn accelerate sentencepiece huggingface_hub

```
---

## Usage
### 1. Start the Inference Server
The server automatically handles model hydration. On the first run, it will download the **TinyLlama-1.1B** weights to the local `./weight` directory. Subsequent runs will load directly from disk.

```bash
# Optional: Set mirror for faster download if in China
export HF_ENDPOINT=[https://hf-mirror.com](https://hf-mirror.com)

# Run the server
uv run src/api_server.py

```

*Expected Output:*

```text
[INFO] Loading model into GPU memory...
1. Checking model weights in ./weight...
   [Info] Weights not found. Downloading to ./weight...
   [Success] Download complete.
2. Loading HF model from local storage...
...
[INFO] Model loaded successfully! Ready to serve.

```

### 2. Client Request (Streaming)Test the API using `curl`:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "The future of AI infrastructure is", "max_tokens": 50}'

```

---

## Project Structure

```text
mini-llm-infra/
??? src/
?   ??? model.py           # Core Llama modeling + KV Cache logic
?   ??? kernel.py          # Custom OpenAI Triton kernels
?   ??? load_weights.py    # Offline weight loading & mapping logic
?   ??? api_server.py      # FastAPI server with SSE
?   ??? inference_kv.py    # CLI inference script for testing
?   ??? benchmark_rmsnorm.py # Performance profiling scripts
??? weight/                # Local model storage (created on first run)
??? pyproject.toml         # Dependency config
??? README.md              # Documentation
```

### The KV Cache Mechanism
Standard Transformer decoding requires re-calculating attention scores for all previous tokens at every step.

* **Implementation:** We allocate a static tensor `(Batch, Max_Seq_Len, n_kv_heads, Head_Dim)` in VRAM during initialization.
* **Benefit:** Allows the model to perform "in-place" updates, reducing memory allocation overhead and ensuring constant-time access to past context.

### Dynamic Dispatch Strategy
Triton kernels provide massive throughput benefits but incur a small Python-to-GPU launch overhead (~5-10us).

* **Strategy:** The `RMSNorm` layer checks the input tensor size.
* If `num_tokens > 128` (Prefill phase), it uses the **Triton** path for maximum bandwidth.
* If `num_tokens <= 128` (Decoding phase), it falls back to **PyTorch** C++ implementations to minimize latency.



---

## Future Roadmap
* **PagedAttention:** Implement block-table memory management to support continuous batching and reduce internal fragmentation.
* **Quantization:** W8A16 support using Triton bitwise operations.
* **Speculative Decoding:** Implement a draft model to further accelerate generation speed.

---