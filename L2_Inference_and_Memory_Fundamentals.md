# Inference and Memory Fundamentals
### Course: Fast & Efficient LLM Inference with vLLM

---

## 1. What Is Inference?

- **Inference** is the process of using a trained model to generate a response.
- It is triggered whenever a user interacts with an AI application — asking a question, getting a code suggestion, summarizing a document, or having an AI agent act on their behalf.

---

## 2. The Three-Layer Inference Stack

To run inference in production, three components must work together:

| Layer | Component | Role |
|-------|-----------|------|
| Top | **Model** | File containing billions of learned parameters (e.g., Llama, Qwen) |
| Middle | **Inference Server** | Software like vLLM that loads the model, manages requests, and handles optimizations |
| Bottom | **Hardware Accelerator** | Typically a GPU that performs heavy numerical computation |

> **Note:** A model can be run directly on a GPU using a library like PyTorch (e.g., in a notebook or for a single user), but for serving many users concurrently and efficiently, the **inference server is essential**. It's what makes the GPU usable at production scale.

**Flow:** User prompt travels **down** the stack → gets processed → response travels **back up**.

---

## 3. How LLMs Generate a Response: Autoregressive Generation

- LLMs do **not** produce a whole sentence at once.
- They generate **one token at a time**, where a token is roughly a word or part of a word.

### The Generation Loop

1. User sends a prompt (e.g., *"The quick brown"*)
2. Model processes the input and **predicts the next token** (e.g., *"fox"*)
3. The new token is **appended** to the input
4. The model runs again on the extended input to predict the next token (e.g., *"jumps"*)
5. This continues until a special **end-of-sequence token** is generated

This process is called **autoregressive generation** — each new token depends on all tokens before it, including those the model just generated.

> **Key implication:** Every token in a response requires a **full forward pass** through the model. A 500-token answer means the model runs **500 times**.

---

## 4. Inside a Forward Pass

When a prompt is received, the model:

1. Converts each token into a **token embedding** (a series of numbers)
2. Passes embeddings through a **stack of transformer layers**
3. Each layer has two main parts:
   - **Self-Attention block** — tokens exchange information with each other
   - **Feed-Forward Network (FFN)** — processes each token's representation independently (no token-to-token interaction)
4. After the final layer, the output goes through the **LM Head**, which produces a score for every possible next token
5. The **highest-scoring token** is the prediction

---

## 5. Inside a Transformer Layer: Linear Layers

Both the Self-Attention block and FFN are built from **linear layers**.

- A **linear layer** is a matrix multiplication: takes an input vector → multiplies by a weight matrix → produces an output vector.
- Linear layers are where **almost all model parameters live** and where **almost all computation happens**.

### Self-Attention Block — 4 Linear Layers

| Projection | Name | Meaning |
|------------|------|---------|
| Q | Query | "What do I want to know from the context?" |
| K | Key | "Here's my label and the kind of information I contain" |
| V | Value | "If my label matches, here is my actual content" |
| O | Output | Final output projection of the attention block |

### Feed-Forward Network — 3 Linear Layers

- **Gate projection**
- **Up projection**
- **Down projection**

> These weight matrices are **learned once during training** and remain fixed during inference.

---

## 6. How Self-Attention Works (Step by Step)

Using the example token "fox" in *"The quick brown fox"*:

1. Compute **Q, K, V vectors** for the current token (fox) by passing its vector representation through the Q, K, V projection linear layers
2. Take the **query (Q)** of fox and compute a **dot product** against the **key (K)** of every token so far (including itself)
   - High dot product → token is relevant
   - Low dot product → token is not relevant
3. **Divide scores** by the square root of the key dimension (for numerical stability)
4. Apply **softmax** to convert scores into weights that sum to 1
5. Compute a **weighted sum** of all value (V) vectors using those weights → produces a single context-enriched vector
6. Pass through **O projection** → final output of the attention block

---

## 7. The KV Cache

### Why It Exists

> To generate a new token, we need the **Keys and Values of every previous token** in the sequence. Only the **Query** is needed for the current token.

Since K and V of past tokens **do not change** between steps, recomputing them is wasteful.

### How It Works

- After computing K and V for a token, they are **saved (cached) in GPU memory**
- On every subsequent inference step, only K and V for the **new token** are computed and appended to the cache
- Attention reads K and V of all previous tokens **from the cache**
- This caching happens at **every transformer layer**, so savings multiply by N layers

### KV Cache Size Formula

```
KV Cache per token = 2 × num_layers × num_KV_heads × head_dimension × dtype_bytes
```

- **2** = storing both K and V
- For **Llama 3 70B**: 80 layers, 8 KV heads, 128 head dimension, 2 bytes per number
- → **~320 KB per token**

### KV Cache at Scale (Llama 3 70B)

| Context Length | Use Case | KV Cache Size |
|----------------|----------|---------------|
| 2,000 tokens | Typical chat turn | ~640 MB |
| 8,000 tokens | Standard production tier | ~2.5 GB |
| 32,000 tokens | Long document / codebase | ~10 GB |
| 128,000 tokens | Llama 3's maximum | ~40 GB |

> **Critical insight:** Llama 3 70B model weights ≈ **140 GB**. A single 128K-token request needs **~40 GB** of KV cache on top of that — nearly a third of the model's own size, **per user**. Serving 10 concurrent long-context users = **400+ GB of KV cache alone**.

### Why KV Cache Is the Dominant Memory Concern

- Lives in **GPU memory**
- Grows **linearly** with sequence length and number of concurrent requests
- Managing it efficiently is the **single biggest job** of a production inference server

---

## 8. GPU Memory Hierarchy

### Vocabulary

- **Tensor**: the general term for any multi-dimensional array of numbers
  - Scalar = 0-dimensional tensor
  - Vector = 1-dimensional tensor
  - Matrix = 2-dimensional tensor
- Q, K, V vectors, KV cache, and model weights are all tensors

### Three Memory Tiers

| Tier | Name | Location | Characteristics |
|------|------|----------|-----------------|
| Bottom | **CPU DRAM** | Host machine | Largest; far from GPU; slowest transfer to GPU |
| Middle | **HBM (High Bandwidth Memory)** | On the GPU card | What people mean by "GPU memory" or "VRAM"; smaller than DRAM but much faster |
| Top | **SRAM** | On-chip, next to compute units | Tiniest; extraordinarily fast; feeds the tensor cores |

### The Tensor Cores

- **Tensor cores** are specialized GPU hardware that perform matrix multiplications extremely fast
- They read input from **SRAM**, do the math, and write results back to SRAM
- SRAM is tiny but fast enough to keep tensor cores continuously fed

### NVIDIA A100 Specs (Example)

| Memory Type | Size | Bandwidth |
|-------------|------|-----------|
| SRAM | ~20 MB | ~19 TB/s |
| HBM | 40 GB | 1.5 TB/s |
| Host ↔ GPU transfer | — | ~12 GB/s |

> **The tradeoff:** The closer memory sits to compute units, the **faster** it is but the **less** of it there is.

---

## 9. How Inference Tensors Move Through Memory

| Tensor | Where It Lives | Lifecycle |
|--------|---------------|-----------|
| **Model weights** | HBM | Loaded from disk/DRAM once at startup; stays for server lifetime |
| **KV Cache** | HBM | Grows as each request processes more tokens |
| **Weight/KV chunks** | Pulled HBM → SRAM | During every forward pass, for every linear layer, every layer, every token |
| **Transient tensors** (Q, K, V for current token; attention output; FFN output) | SRAM | Computed and used immediately by tensor cores; then discarded (do not persist) |

---

## 10. The Two Governing Factors of Inference Speed

1. **How fast data can move from HBM into SRAM**
2. **How fast tensor cores can compute on that data once it arrives**

> **Core optimization principle:** Every optimization technique comes back to this — move **less data**, move it **more efficiently**, or **manage memory better**.

---

## 11. What's Coming Next

- **Quantization**: A technique that shrinks the model by storing weights in **lower precision formats**, resulting in less data to move through the memory hierarchy.

---

*Notes based solely on the lecture transcript from the Coursera course: Fast & Efficient LLM Inference with vLLM — Lecture: Inference and Memory Fundamentals.*
