# Lecture Notes: Why Efficient Deployment Matters
**Course:** Fast & Efficient LLM Inference with vLLM

---

## 1. Core Premise: Inference Cost Dominates AI Spending

- Most discussion around AI cost focuses on **training** (massive GPU clusters, weeks of compute, billions of dollars).
- In reality, **inference is the larger cost driver in production**, because:
  - Training happens **once**.
  - Inference happens **every single time a user sends a message** — i.e., it scales with usage, not with a one-time event.

**Key takeaway:** Efficient deployment is an ongoing, recurring cost problem, not a one-time engineering problem.

---

## 2. The Shift in the Open-Source LLM Landscape

- **Early 2023:** essentially no competitive open-source models.
- **Now:** thousands to millions of models available on Hugging Face, from virtually every organization (including OpenAI itself).
- **Consequence:** The central question has shifted from:
  - ❌ "Can I get a good model?"
  - ✅ "Can I run that model efficiently?"

---

## 3. Why Self-Host Models Instead of Using an API

Four primary reasons:

| Reason | Explanation |
|---|---|
| **1. Cost Savings** | Pay for infrastructure, not per-token API pricing — allows matching model size to task difficulty. |
| **2. Security** | Data never leaves your environment — critical for regulated industries like healthcare and financial services. |
| **3. Control** | You decide when to upgrade/deprecate models; no third-party rate limits or outages. |
| **4. Customization** | Ability to fine-tune models for accuracy and cost control. |

---

## 4. Measuring Deployment Success: Service Level Objectives (SLOs)

LLM deployments need **measurable targets**, tracked across **two dimensions**: Accuracy and Inference Performance.

### 4.1 Accuracy
- A model that is wrong (hallucinates, gives off-brand answers) is not useful — accuracy must clear a **usable threshold specific to the use case**.
- **Model cards** are used to verify accuracy before deployment.
  - Example: Llama 3.1 70B model card compares the original model vs. an optimized/smaller version.
  - Uses standardized benchmarks (e.g., **MMLU** for general knowledge, **GSM8K** for math).
  - **Recovery %** = how much of the original accuracy the optimized version retains (example given: **99.88% average recovery**).
  - This recovery metric is the mechanism for validating a model meets its accuracy SLO *before* deployment.

### 4.2 Inference Performance
Three key **latency metrics**:

1. **Time to First Token (TTFT)** — time until the first output token is generated; reflects how long a user waits before seeing any response.
2. **Inter-Token Latency (ITL)** — average time between generating consecutive tokens (excludes the first token); reflects smoothness/speed of generation.
3. **Request Latency** — total end-to-end time for a request.

Plus one key **throughput metric**:

4. **Throughput** — average number of output tokens generated per second, across all requests; indicates whether the system can handle production-scale load.

**Key takeaway:** A production-viable LLM must be **both fast enough AND accurate enough** — small gaps in either dimension compound into major issues at scale.

---

## 5. Hardware Fundamentals: GPU Memory Requirements

GPU memory must hold **two things**:

1. **Model Weights** — fixed size regardless of number of users (1 or 100).
2. **KV Cache** — working memory that **grows with every token and every active request**.

*(Both are covered in more depth in the next lesson.)*

### Worked Example: Llama 3 70B

| Item | Value |
|---|---|
| Parameter size | ~2 bytes/parameter |
| Total weight memory | ~140 GB (70B params × 2 bytes) |
| Minimum GPUs needed | 2× 80GB GPUs (to fit weights) |
| **Standard production deployment** | **4× 80GB GPUs** |
| Total GPU memory (4×80GB) | 320 GB |
| Memory remaining after weights | ~180 GB (for KV cache + overhead) |
| KV cache cost per long-context request (32,000 tokens) | ~10 GB |
| Long-context users servable in parallel (theoretical) | **18 users** |

### Critical Insight: Naive Serving Wastes Capacity
- A **naive serving setup with no memory optimization** can reduce the number of parallel users from a theoretical **18 down to just 2–3**.
- **This gap — between theoretical and actual capacity — is exactly what the course aims to close.**

---

## 6. Two Categories of Optimization

| Category | When Applied | Techniques | Goal |
|---|---|---|---|
| **Model Optimizations** | Before deployment (to the model itself) | Quantization, Sparsification | Reduce memory footprint & compute requirements while preserving accuracy |
| **Inference Optimizations** | At runtime (in the inference engine) | Continuous Batching, Prefix Caching, PagedAttention | Don't change the model — change *how efficiently* it's run |

*(These techniques are the focus of later lessons in the course.)*

---

## 7. The Fundamental Trade-off Triangle

Every LLM deployment balances **three competing forces**:

1. **Performance** (latency + throughput) — real-time latency is critical, but high throughput requires more compute.
2. **Accuracy** — needed for trust, but higher-accuracy models tend to cost more.
3. **Cost** — infrastructure costs must be controlled, but aggressive model optimization can hurt accuracy.

> **Rule of thumb:** Most deployments can only "pick two" of the three — **unless** better tooling/techniques are used to push past this constraint. This is the stated goal of the course.

---

## 8. Cost Comparison: Illustrating the Impact of Optimization

Three deployment scenarios for serving Llama at scale (thousands of users):

| Deployment Approach | Description | Relative Cost |
|---|---|---|
| **1. Naive deployment** | Full precision weights, one request processed at a time | **Hundreds of thousands to millions of $/month** |
| **2. + Continuous Batching + PagedAttention (via vLLM)** | Batches multiple requests together; manages KV cache efficiently | **~10x cost reduction** vs. naive |
| **3. + Model Optimization (quantization)** | Same vLLM deployment, but using a quantized model | Further reduces memory footprint, speeds up weight loading, more efficient GPU hardware utilization — largest savings |

**Key takeaway:** Combining **inference optimization** (vLLM's continuous batching + PagedAttention) with **model optimization** (quantization) compounds savings dramatically — this combination is the core value proposition of the course.

---

## 9. Looking Ahead

The next lesson covers **inference and memory fundamentals**, specifically:
- What happens when a model generates a token.
- Where model weights and the KV cache live on the GPU.
- How data moves inside the GPU.

This foundational understanding sets up the deeper dive into model and inference optimization techniques (quantization, sparsification, continuous batching, prefix caching, PagedAttention).

---

## Quick-Reference Glossary

| Term | Definition |
|---|---|
| **SLO (Service Level Objective)** | Measurable target for accuracy and/or performance in production |
| **TTFT** | Time to First Token — latency before first output token appears |
| **ITL** | Inter-Token Latency — avg. time between subsequent tokens |
| **Throughput** | Output tokens/sec across all requests |
| **KV Cache** | Growing memory structure holding key/value attention states per token/request |
| **Quantization** | Model optimization technique reducing numeric precision of weights to save memory |
| **Sparsification** | Model optimization technique that removes/reduces redundant weights |
| **Continuous Batching** | Inference optimization that dynamically batches multiple requests together |
| **Prefix Caching** | Inference optimization reusing cached computation for shared prompt prefixes |
| **PagedAttention** | Inference optimization for efficient, non-contiguous KV cache memory management (used in vLLM) |
| **Recovery %** | Metric showing how much of a base model's accuracy an optimized version retains |

---

## Study/Discussion Questions (for review)

1. Why does inference cost outweigh training cost over the lifetime of a deployed model?
2. What are the trade-offs between the four reasons for self-hosting vs. using a hosted API?
3. Why is accuracy recovery (e.g., 99.88%) an important metric when evaluating a quantized/optimized model?
4. Walk through the math of GPU memory allocation for a 70B model — why do you need 4×80GB GPUs rather than 2×80GB?
5. Explain why a "naive" serving setup can reduce parallel capacity from 18 users to 2–3, even though the underlying hardware hasn't changed.
6. What is the difference between model optimization and inference optimization, and why are both needed to achieve maximum cost savings?
7. In the cost comparison example, which optimization step contributed the larger share of savings — batching/PagedAttention or quantization — and why might they be multiplicative rather than additive?
