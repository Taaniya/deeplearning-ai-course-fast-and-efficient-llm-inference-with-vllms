# Fast & Efficient LLM Inference with vLLM
## Lecture Notes: Serving LLMs Efficiently with vLLM — Part 1

---

## Overview

This lecture covers three core techniques that power modern LLM serving engines, specifically vLLM:

1. [**Continuous Batching** — keeps the GPU busy](3-continuos-batching)
2. [**PagedAttention** — manages KV cache memory without waste](#5-pagedattention)
3. [**Prefix Caching** — skips KV recomputation when requests share content](#6-prefix-caching)

---

## 1. Why Batching Matters

### The Problem: Single-Request Serving Is Inefficient
- LLM generation is **iterative** — every token requires a full forward pass through the model.
- Each forward pass pulls all model weights from HBM (High Bandwidth Memory) into GPU compute units.
- Serving one request at a time leaves the GPU **dramatically underutilized**.
- The compute needed for a single token is tiny, but you still pay the full cost of moving the entire model through memory.
- Tensor cores spend most of their time waiting for data, not doing math.
- This directly limits **throughput** (number of tokens/requests processed per second across all users).

### The Solution: Batching
- Process multiple requests together.
- Read the model's weights **once** and use them for many users simultaneously.
- Same memory cost, but much more work done per memory read.

---

## 2. Static Batching

### How It Works
- Collect a fixed group of requests, process them together, and wait until **every single one finishes** before starting the next batch.

### When It Works Well
- Works well for traditional models like **BERT** or **YOLO**, where input/output sizes are predictable.
- Example: A classification model takes one image and returns one label — fixed runtime per request. Batch 10 images → all finish at roughly the same time → GPU stays busy.

### Why It Fails for LLMs
- LLMs have **unpredictable context lengths**.
  - One user might ask "What's 2 + 2?" → 5-token answer.
  - Another might ask for a 2000-word essay.
- In a static batch, once a request finishes, its GPU slot sits **idle** until the longest request in the batch is done.
- Short requests are stuck waiting for long ones → a lot of idle GPU time.

**Example:**
| Request | Finishes At |
|---------|-------------|
| Request 3 | T5 |
| Request 1 | T6 |
| Request 2 | T8 (longest) |

> Requests 3 and 1 finish early, but their GPU slots waste capacity until T8.

---

## 3. Continuous Batching

### How It Works
- The scheduler works at the **token level**, not the batch level.
- The moment a request finishes, a **new request immediately takes its slot** in the batch.
- The batch is never idle.

**Example:**
- Request 3 finishes at T5 → Request 5 jumps in immediately.
- Request 1 finishes at T6 → Request 6 takes over immediately.

### Key Difference from Static Batching
| Static Batching | Continuous Batching |
|-----------------|---------------------|
| GPU slots idle until entire batch finishes | New requests dynamically fill slots as others complete |
| Fixed batch, wait for slowest request | Token-level scheduling |
| Poor GPU utilization with variable-length outputs | High GPU utilization |

---

## 4. KV Cache Memory Management

### Why KV Cache Is a Bottleneck
- Every active request has its own **KV cache**, growing one token at a time.
- More concurrent users → more KV cache memory needed.
- Poor memory management → fewer requests fit in a batch → lower throughput, even if the GPU has compute to spare.

### Two Challenging Properties of the KV Cache
1. It **grows and shrinks dynamically**.
2. You **don't know in advance** how long a request will be.

### The Old Approach: Contiguous Pre-allocation
- Systems pre-allocated one contiguous memory block sized to the **maximum possible length** (e.g., 2048 slots).
- This leads to three types of waste:

| Waste Type | Description |
|------------|-------------|
| **Internal Fragmentation** | Wasted space inside an allocation — slots pre-allocated but never used (e.g., 2048 slots reserved, only a few used). |
| **External Fragmentation** | Gaps between allocations that are physically free but too small to fit a new pre-allocated chunk. |
| **Over-reservation** | Slots a request will eventually use sit reserved and empty for most of its lifetime, blocking other requests from using that space. |

> The vLLM paper reports that only **20–40% of KV cache memory** was actually used to store real tokens. The rest was lost to fragmentation and over-reservation.

---

## 5. PagedAttention

### Core Innovation of vLLM
- Instead of storing the KV cache as one large continuous block, it is broken into **fixed-size blocks** (also called **pages**).
- Each block holds the keys and values for a **small number of tokens**.
- The system keeps a **block table** that maps each request's tokens to the physical blocks holding them.

### Inspiration
- Borrowed from **virtual memory and paging** in operating systems.
- The OS splits memory into small pages and scatters them wherever there is room, using a page table to track locations.
- PagedAttention applies the same trick to the KV cache.

### Step-by-Step Walkthrough

**Prompt:** `"Artificial Intelligence is"`

1. System grabs one free block (e.g., **Block 3**) and stores KV cache for 3 prompt tokens. Block table records: Block 3, 3 slots filled.
2. Model generates token `"the"` → Block 3 still has one empty slot → used. Block table updates: 4 slots filled.
3. Model generates `"future"` → Block 3 is now full → system grabs **Block 6**, stores `"future"` there.
   - Blocks do **not** need to be adjacent in memory.
4. Model generates `"of"`, then `"technology"` → each fills the next slot in Block 6.

> Memory is allocated **only as the request needs it**, filling one block at a time — never more.

### How Attention Works with PagedAttention
- To generate the next token (`"technology"`), the model uses a query for the current token.
- It needs to attend to keys and values of **all previous tokens**.
- System reads Block 3 from the block table, fetches it from GPU memory, computes attention.
- Then does the same for Block 6 — **one block at a time**.
- The block table enables stitching of non-contiguous memory blocks together for attention computation.

### Benefits
- No pre-allocation
- No wasted slots
- No fragmentation
- Two requests share the same physical memory pool, with blocks scattered wherever there is space.
- Both requests get exactly the memory they need, when they need it.
- Result: vLLM fits **more requests into the same GPU**, pushing throughput up.

---

## 6. Prefix Caching

### Core Idea
- When requests share the same **prefix** (e.g., a system prompt), they share KV cache blocks.
- **Compute once, reuse across users.**
- Reuses the KV cache when requests share the same starting tokens, instead of recomputing from scratch.

### Two Common Patterns

#### Pattern 1: Shared Prompts Across Users
- Three users send different questions but all use the same system prompt.
- **Without Prefix Caching:** The prompt's KV cache is recomputed for every user.
- **With Prefix Caching:** The prompt is computed once and reused for everyone.
- Also applies to few-shot examples or shared RAG context.

#### Pattern 2: Multi-Turn Conversations
- Round 2's prompt includes everything from Round 1 plus a new question.
- Since the Round 1 part is identical, its KV cache is pulled straight from memory — not recomputed.
- The model only does new work on the **new tokens**.

### Performance Impact
- As cache hit rate climbs, throughput climbs with it.
- At a **75% cache hit rate**, throughput is roughly **4× higher**.
- That is compute the system simply does not have to redo.

---

## 7. vLLM: Bringing It All Together

### What Is vLLM?
- An **open-source inference engine** built to be the fastest and easiest to use.
- Combines continuous batching, PagedAttention, and prefix caching.

### Adoption & Momentum (as of January 2025)
- **100,000 daily installs**
- Usage grew **10× in 2024**
- One of the **top AI and ML repositories on GitHub** by contributor count

### Model Support
- Llama, Qwen, DeepSeek, Gemma, Mistral, Granite, and more.

### Hardware Accelerator Support
- NVIDIA GPUs
- AMD Instinct
- Intel Gaudi
- Google TPUs
- AWS Neuron
- IBM Spyre

### Deployment Environments
- Edge
- Private Cloud
- Public Cloud

> One platform for all models and all hardware.

---

## Summary Table: The Three Core Techniques

| Technique | Problem It Solves | How It Works |
|-----------|-------------------|--------------|
| **Continuous Batching** | GPU idle time from variable-length outputs | Fills GPU slots at the token level as requests complete |
| **PagedAttention** | KV cache memory fragmentation and over-reservation | Breaks KV cache into fixed-size pages; uses a block table for non-contiguous memory |
| **Prefix Caching** | Redundant KV recomputation for shared content | Caches and reuses KV blocks for shared prefixes across requests |
