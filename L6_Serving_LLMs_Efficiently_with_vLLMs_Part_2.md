# Lecture Notes: Serving LLMs Efficiently with vLLM — Part 2

**Course:** Fast & Efficient LLM Inference with vLLM
**Lecture:** Serving LLMs Efficiently with vLLM — Part 2

---

## 1. Overview

This lecture is a hands-on continuation of Part 1. It covers:

- Connecting to a running vLLM inference server
- Sending requests through the OpenAI-compatible API
- Observing vLLM optimizations (Continuous Batching, PagedAttention, Prefix Caching) live via metrics

---

## 2. Launching a vLLM Server

### Command

```bash
vllm serve <model-id>
```

**Example model used:** `Qwen3-0.6B` (from Hugging Face Hub)

### What Each Part of the Command Does

| Argument / Component | Purpose |
|---|---|
| `vllm serve` | Launches vLLM's built-in inference server |
| Model identifier | Hugging Face Hub ID; vLLM downloads weights, tokenizer, and config on first run and caches them locally |
| `--dtype bfloat16` | Loads model weights in bfloat16 precision |
| `--max-model-len 4096` | Caps context window (prompt + generation) at 4096 tokens; used to size the KV cache block pool up front |

### Default Behaviors When Server Starts

- PagedAttention is enabled by default
- Continuous Batching is enabled by default
- Prefix Caching is enabled by default
- Server is exposed over HTTP on **port 8000**

---

## 3. OpenAI-Compatible API

vLLM wraps the model in an OpenAI-compatible HTTP API, implementing the same routes as the OpenAI SDK:

- `GET /v1/models` — list available models
- `POST /v1/completions` — text completions
- `POST /v1/embeddings` — embeddings

### Key Benefit

The same client code and request format used with OpenAI hosted models works with vLLM by simply changing the `base_url` to `localhost:8000/v1`. This makes it easy to:

- Prototype against a hosted model
- Swap to a self-hosted model without rewriting the application

---

## 4. Connecting to the Server

### Health Check Pattern

```python
import requests

VLLM_URL = "http://localhost:8000"

# Poll until server is ready
while True:
    response = requests.get(f"{VLLM_URL}/v1/models")
    if response.status_code == 200:
        break
    time.sleep(5)
```

- Sends a GET request to `/v1/models` every ~5 seconds
- A `200` response confirms the server has finished loading weights

---

## 5. Sending Requests

### Using the OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="...")

response = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[{"role": "user", "content": "What is PagedAttention?"}],
    max_tokens=...,
    temperature=...
)
```

### Notes on Qwen 3

- Qwen 3 is a **thinking model** (chain-of-thought reasoning)
- Thinking mode can be turned off to keep responses short

---

## 6. Log Probabilities (Logprobs)

### What Are Logprobs?

Log probabilities expose the model's internal confidence scores for each token it selects during generation.

### How to Use

- Request logprobs in the API call
- Inspect `choices[i].logprobs` in the response
- Convert log probabilities to confidence percentages

### Example

For the prompt *"The capital of France is"*, the model returns:

- **Paris** with a **92.5% confidence score**

### Why Logprobs Are Useful

- Understand when the model is **sure** vs. when it is **guessing**
- Useful for reliability analysis and filtering low-confidence outputs

---

## 7. Metrics Endpoint

### Endpoint

```
GET http://localhost:8000/metrics
```

### What It Exposes (Prometheus-Compatible Metrics)

| Metric | Description |
|---|---|
| Number of running requests | How many requests are currently active |
| Number of queued requests | How many requests are waiting to be processed |
| GPU cache usage (%) | KV cache memory pressure |
| Prompt tokens total | Cumulative prompt tokens processed |
| Generation tokens total | Cumulative generated tokens produced |

---

## 8. Continuous Batching in Action

### Experiment

- Send **5 concurrent requests** simultaneously
- Poll the metrics endpoint while requests are in-flight to observe running vs. queued counts

### Observation

- vLLM's scheduler handles all 5 requests at the same time via Continuous Batching
- Total time is **faster than running them one by one**
- The scheduler manages requests effectively, improving throughput

---

## 9. PagedAttention in Action

### How It Works (Recap from Part 1, observed here)

- The KV cache is divided into **fixed-size blocks**
- Blocks can be placed **anywhere in memory** (non-contiguous)
- When a request finishes, its blocks are **immediately freed and reused**
- No memory is wasted

### Role in Concurrent Request Handling

- PagedAttention is what enables Continuous Batching to work **at scale**
- Efficient memory reuse allows more requests to fit in GPU memory simultaneously

---

## 10. Prefix Caching in Action

### The Problem It Solves

Many applications send the **same system prompt** with every request. Without prefix caching, vLLM would recompute the KV cache for that shared prefix on every single request.

### How It Works

- First request: pays the **full prefill cost** for the system prompt
- Subsequent requests with the same system prompt: vLLM **recognizes the shared prefix** and skips recomputation

### Experiment

- Set up a fixed system prompt
- Send **5 different questions** using the same system prompt
- Track the `prefix_cache_queries` metric before and after

### Observed Result

- Prefix cache query count increased from **235 → 550** across the 5 requests
- Confirms vLLM is checking and reusing the cached prefix KV states

### Production Impact

- With short prompts, time savings are not very visible
- In production (thousands of instruction tokens, long few-shot examples), this **eliminates a huge amount of compute**

---

## 11. Summary of What Was Covered

| Topic | Key Action |
|---|---|
| vLLM server setup | Launched via `vllm serve`, understood each argument |
| OpenAI-compatible API | Used the same Python client with a different `base_url` |
| Log probabilities | Inspected model confidence scores per token |
| Concurrent requests | Sent 5 requests simultaneously; observed continuous batching |
| Metrics endpoint | Scraped Prometheus metrics to observe cache usage and request counts |
| PagedAttention | Observed memory-efficient KV cache block reuse |
| Prefix caching | Confirmed cache hit counts increasing across repeated system prompts |

---

## 12. What Comes Next

In the next lesson:

- **Benchmarking** the model with **GuideLLM**
- **Evaluating model quality** with **LM-Eval**
