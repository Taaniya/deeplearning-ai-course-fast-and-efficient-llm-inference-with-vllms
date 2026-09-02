# LLM Optimization Fundamentals
**Course:** Fast & Efficient LLM Inference with vLLM

---

## 1. The Growing Gap Between Model Size and Hardware Capability

### Model Size Growth
- Model sizes have **roughly doubled every year** — from the original Transformer in 2017 at 50 million parameters to today's frontier models in the hundreds of billions, some pushing past a trillion.
- GPU memory growth has not kept pace with model size growth.
- This widening gap makes compression critically important.

### Four Real Problems Caused by This Gap

| Problem | Description |
|---|---|
| **GPU & Infrastructure Cost** | Bigger models require more hardware accelerators, often spread across multiple nodes — expensive to operate. |
| **User Experience Tradeoffs** | More parameters can mean slower responses, lower throughput, and less room for long context in the KV cache. |
| **Energy & Carbon Footprint** | Every extra GPU draws power; at scale, this adds up to a real environmental cost. |
| **Risk of Model Obsolescence** | Risk of investing in infrastructure for a model that gets superseded quickly. |

---

## 2. Quantization — The Core Compression Technique

### What Is Quantization?
- Quantization reduces the number of bits used to store model weights.
- Analogy: Instead of storing pi as `3.14159...` (many bits), store it as `3.14` (fewer bits).
- Most LLMs today are released in **BF16** (Brain Float 16 — 16 bits per number).
- Quantization converts numbers into lower bit formats: **FP8, INT8, or INT4**.

### Numeric Formats Explained

| Format | Full Name | Notes |
|---|---|---|
| **FP32** | Floating-Point 32 | Huge range, fine-grained precision |
| **BF16** | Brain Floating-Point 16 | Same range as FP32 but less precision; developed by Google; more stable for large models |
| **FP16** | Floating-Point 16 | Narrower range than BF16 |
| **INT8** | Integer 8 | Whole numbers, smaller range, larger gaps between values |
| **INT4** | Integer 4 | Most aggressive; largest gaps between representable values |

- As you move from FP32 → BF16 → FP16 → INT8 → INT4: **range shrinks, gaps between values grow** — you trade precision for size.

### What Gets Quantized?
Quantization specifically targets the **linear layers** inside transformer blocks, because:
- Most forward pass time is spent inside linear layers.
- The main matrix multiplications happen there.
- The bulk of the model's weights live there.

**Note:** The embedding layer and LM head are typically excluded from quantization to preserve accuracy.

Two things inside a linear layer can be quantized:
1. **Weights** — the model's learned parameters.
2. **Input Activations** — intermediate tensors computed during a forward pass that flow through the linear layers (e.g., the tensor multiplied by weights to produce Q, K, V, or the weighted sum that flows into the O projection).

---

## 3. Sparsification

- Zeros out weights that contribute least to the model's predictions so they can be skipped entirely during inference.
- A common approach is **2:4 sparsity** — two out of every four values in a weight tensor are set to zero, reducing both memory and computation.
- Used together with quantization to reduce a model's memory footprint.

---

## 4. Memory Savings: A Concrete Example (Llama 4 Scout — 109B Parameters)

| Precision | Bits per Parameter | Bytes per Parameter | Total Weight Size | GPUs Required (80 GB each) |
|---|---|---|---|---|
| BF16 (baseline) | 16 | 2 | ~220 GB | 3 GPUs |
| INT8 / FP8 | 8 | 1 | ~109 GB (50% reduction) | 2 GPUs |
| INT4 / FP4 | 4 | 0.5 | ~55 GB (75% reduction) | 1 GPU |

---

## 5. How Quantization Maps to the GPU Memory Hierarchy

### Two Distinct Performance Effects

**1. Weight Quantization → Lower Latency (faster data movement)**
- Every forward pass, the GPU pulls weights from HBM (High Bandwidth Memory) into SRAM so Tensor Cores can do the math.
- 8-bit weights = half the data to move → moves faster → faster inference.

**2. Activation Quantization → Higher Throughput (faster compute via Tensor Cores)**
- Tensor Cores are specialized GPU hardware for matrix multiplications.
- They perform more operations per second with lower precision inputs.
- Modern GPUs (Hopper, Ada Lovelace): dedicated **FP8 Tensor Cores**.
- Older Ampere GPUs: **INT8 Tensor Cores**.

> **Quantizing both weights and activations unlocks the full speedup.**

---

## 6. Two Quantization Schemes

### Weight-Only Quantization (e.g., W8A16)
- Only weights are quantized (e.g., to INT8); activations stay in higher precision (e.g., BF16).
- At inference: weights loaded from HBM in compressed form, dequantized back to BF16 just before multiplication.
- **Win:** Less data movement from HBM to SRAM.
- **No win:** Tensor Core speedup is not achieved.

### Weight + Activation Quantization (e.g., W8A8)
- Both weights and activations are quantized (e.g., to INT8 or FP8).
- At inference: math runs on lower precision Tensor Cores (e.g., FP8 Tensor Cores on Hopper).
- **Win:** Less data movement AND Tensor Cores doing more operations per second.
- Reduces both memory cost and compute cost.

---

## 7. Five Practical Benefits of Quantization

1. **Fewer GPU Resources Needed** — Quantized model fits on fewer GPUs (e.g., Llama 4 Scout: 3 GPUs → 1 GPU).
2. **Reduced Deployment Costs** — Fewer GPUs and smaller nodes = lower cloud or hardware bill.
3. **Decreased Latency** — Less data moving from HBM to SRAM = faster forward passes = faster responses for users.
4. **Higher Throughput and Longer Context** — More GPU memory freed up = more concurrent users in KV cache, or longer context per user.
5. **Lower Energy Consumption** — Fewer GPUs running for less time = less power draw and smaller carbon footprint at scale.

---

## 8. Real-World Performance Results (RAG Use Case)

### Workload Setup
- **Total input:** 1,024 tokens
  - 50 tokens — system prompt (e.g., "answer using the provided context")
  - 20 tokens — user question
  - 900 tokens — retrieved context (documents/PDFs)
- **Output:** ~128 tokens (the generated answer)
- Model: **Llama 3 70B** on **two H100 GPUs**
- Comparison: **FP16 (baseline) vs. FP8 (W8A8 quantized)**

### Results

| Metric | FP16 (Baseline) | FP8 (Quantized) | Improvement |
|---|---|---|---|
| **Throughput** (input tokens/sec, at peak load) | ~158 tokens/sec | ~474 tokens/sec | **~3× improvement** |
| **Time to First Token** (at high load) | >30,000 ms (30 seconds) | ~4,800 ms | **~6.7× reduction in latency** |

---

## 9. Does Quantization Hurt Model Accuracy?

### Key Finding
When quantization is done correctly with **calibrated techniques**, it does **not meaningfully degrade accuracy**.

### Naive vs. Calibrated Quantization
- **Naive quantization** (blindly rounding every number) does hurt the model.
- **Calibrated techniques** (GPTQ, AWQ, SmoothQuant) use a small representative dataset to identify which weights and values matter most and protect those during the quantization process.

### Benchmark Results (Three Reasoning Benchmarks)
Benchmarks used:
- **AIME 2024** — 30 expert-level competition math problems
- **MATH-500** — 500 challenging math problems
- **GPQA-Diamond** — Expert-validated science questions

Metric: **Average Pass@1** (percentage of problems solved correctly on the first attempt — higher is better)

#### Example: Qwen-14B Model

| Format | Description | Score | Difference vs. BF16 |
|---|---|---|---|
| **BF16** (baseline) | Original full precision | 73.6 | — |
| **INT W4A16** | 4-bit weights (most aggressive) | 72.8 | −0.8 (less than 1 point) |
| **FP W8A8** | 8-bit float weights & activations | 74.3 | +0.7 (within random variation) |

- The **INT W4A16** result: weights shrunk by 4× with less than 1-point accuracy drop.
- The **FP W8A8** result: model shrunk by 2× with no meaningful accuracy loss; the marginal score increase is attributed to random variation between runs, not actual improvement.

### Key Takeaway
> Done correctly, quantization gives you the speed and memory wins **without giving up the model's quality**.

---

## 10. Techniques Mentioned for Further Study

The following calibrated quantization techniques are introduced and will be covered in detail in the next lesson:

- **GPTQ**
- **AWQ**
- **SmoothQuant**

These techniques replace naive rounding with data-informed approaches that protect the most important weights during quantization.

---

*Notes compiled from the lecture transcript of "LLM Optimization Fundamentals" — Coursera course: Fast & Efficient LLM Inference with vLLM.*
