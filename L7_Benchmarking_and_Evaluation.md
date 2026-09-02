# Lecture Notes: Measuring What Matters — Benchmarking and Evaluation

**Course:** Fast & Efficient LLM Inference with vLLM
**Lecture:** Measuring what matters: Benchmarking and Evaluation

---

## 1. The Core Problem: Why Measurement Matters

- After optimizing and serving a model, you need to determine:
  - Is the deployment **fast enough**?
  - Are the **responses good enough**?
- Without clear measurements, you cannot make informed deployment decisions.

### The Accuracy–Performance–Cost Triangle

LLM production deployments involve navigating tradeoffs between three corners:

| Optimize For | What You Sacrifice |
|---|---|
| High accuracy + Low latency | High cost |
| Low cost + High accuracy | High latency |
| Low cost + Low latency | Accuracy |

> The key insight: you can optimize for any two corners, but the third one pays the price.

---

## 2. Two Complementary Types of Measurement

### Model Evaluation (Broader)
- The overall process of assessing whether a model is **fit for purpose**.
- Covers criteria like: accuracy, safety, and task suitability.
- Key question: *"Is this model good enough for what I need?"*

### Model Benchmarking (Narrower, sits inside Evaluation)
- Standard comparison of a model against **predefined datasets, tasks, or other models** using objective metrics.
- One of the tools used to answer the evaluation question.

---

## 3. Service Level Objectives (SLOs)

Before benchmarking, you must define **what you're benchmarking against** — your SLOs.

### Example 1: E-commerce Chatbot
- Users expect **conversational speed** → tight targets required.

| Metric | Target | Percentile |
|---|---|---|
| Time To First Token (TTFT) | < 200 ms | p99 |
| Inter-Token Latency (ITL) | < 50 ms | p99 |

### Example 2: RAG (Retrieval-Augmented Generation) System
- Users are willing to wait longer for **thoughtful, grounded answers** → relaxed targets.

| Metric | Target |
|---|---|
| Time To First Token (TTFT) | < 300 ms |
| Inter-Token Latency (ITL, streamed) | < 100 ms |
| End-to-End Latency | < 3 seconds |

> **Key lesson:** Same metrics, different thresholds. Always define SLOs *before* benchmarking — numbers only mean something relative to your targets.

---

## 4. GuideLLM — Performance Benchmarking Tool

### What It Is
- Open-source tool from the **vLLM project**.
- Purpose-built for LLM serving benchmarking.
- Puts your inference server under **controlled load** and measures responses.

### Why It's Better Than Generic Load Testers
- Generic tools measure request latency as **one number**.
- GuideLLM understands **streaming responses** and captures LLM-specific metrics:
  - Time To First Token (TTFT)
  - Inter-Token Latency (ITL)
  - End-to-end latency
- Can run **ad hoc** from the command line or wired into **CI (Continuous Integration)** to catch regressions automatically.

---

## 5. Four Scenarios Where Benchmarking Is Essential

### 1. Pre-Deployment
- Determine if a model will work on your hardware at the required quality level.
- Example: On an NVIDIA H200 GPU, should you use Llama 3.1 8B or 70B for a customer service chatbot?
- The bigger model is more capable, but can your hardware serve it within your latency budget?
- Benchmarking answers this **before you're stuck with the wrong choice in production**.

### 2. Cost and Capacity Planning
- Determine how much hardware you need.
- Example: How many servers do I need to keep service running under peak load?
- Benchmarking gives you **throughput per server**, which feeds into provisioning and cost estimates.

### 3. Regression & A/B Testing
- Models change frequently: quantization, version swaps, serving configuration tuning — each can shift performance in non-obvious ways.
- Example: How much more traffic can the INT8 version handle compared to the baseline?
- Benchmarking lets you **catch regressions before users do**.

### 4. Hardware Evaluation
- Find the **breaking point**: the load level where latency starts to climb sharply.
- Example: What is the maximum requests per second my hardware can handle before performance degrades?
- Critical for **autoscaling** and setting honest capacity limits.

---

## 6. GuideLLM Traffic Patterns (Load Simulation Modes)

GuideLLM provides five traffic patterns, each revealing something different:

| Pattern | Description | Best For |
|---|---|---|
| **Synchronous** | One request at a time; waits for each to finish | Clean baseline; single request latency with no queuing |
| **Concurrent** | Fixed number of parallel streams | Shows server behavior under simultaneous users |
| **Constant** | Requests sent asynchronously at a fixed, specified rate | Simulating steady, predictable traffic |
| **Poisson** | Fixed rate but with random spacing (Poisson distribution) | Closest match to **real user traffic** (unpredictable arrivals) |
| **Sweep** | Runs the full spectrum automatically (synchronous → concurrent, with constant-rate runs in between) | Full performance curve in one go; great for capacity planning |

> **Important:** Benchmark numbers are shaped by everything in the stack — model architecture/size, quantization, serving engine, hardware, and batching settings. The benchmark measures their **combined effect**.

---

## 7. Interpreting Benchmark Results

### Key Metrics Reported by GuideLLM
- **Time To First Token (TTFT)**
- **Inter-Token Latency (ITL)**
- **End-to-End Latency**
- Output formats: **JSON and CSV** with pre-computed statistics (mean, percentiles, min, max).

### Statistical Distribution: Mean vs. Percentiles

| Statistic | Meaning |
|---|---|
| Mean | Average across all requests |
| p50 | 50% of requests are faster than this value |
| p95 | 95% of requests are faster than this value |
| p99 | 99% of requests are faster than this value |

> **Always look at p95 and p99**, not just the mean — averages hide outliers.
> - A big gap between mean and p95 = **tail latency problems** that users will feel.
> - Example: 5 in 100 users could be waiting many times longer than the average.

---

## 8. lm_eval — Accuracy Evaluation Tool

### Why Accuracy Evaluation Is Separate from Performance Benchmarking

- A deployment can be blazing fast but give **wrong answers** — that's not useful.
- A quantization technique that wins on throughput but drops accuracy is **not a win**.
- GuideLLM answers: *"How well does this deployment perform?"*
- lm_eval answers: *"How well does this model answer?"*

### What It Is
- **LM Evaluation Harness** from **Eleuther AI**.
- Open-source, standardized evaluation framework.
- Supports a huge range of built-in benchmarks covering:
  - General knowledge
  - Reasoning
  - Math
  - Coding

### Built-in Benchmark Examples
- **MMLU** — general knowledge
- **HellaSwag** — commonsense reasoning
- **GSM8K** — math word problems
- **ARC** — science questions
- **TruthfulQA** — truthfulness

### Compatibility
- Works with both **local models** and **remote API endpoints** (including a running vLLM server via the OpenAI completions endpoint).
- Supports **custom tasks**: define your own domain-specific evaluation using a small YAML config file.

### Practical Usage Note
- Running 20 examples gives a **noisy but quick starting point**.
- Production evaluations should use the **full test set** (e.g., ~10,000 examples for HellaSwag) with in-context examples per prompt for reliable results.
- The gap between a small sample run (e.g., 30% accuracy on 20 examples) and a full benchmark run (e.g., 43% on 10,000 examples) is **expected and normal**.

---

## 9. Published Model Cards as a Third Evidence Source

- Quantized model publishers include **accuracy tables** on their model cards (e.g., on Hugging Face).
- Users can evaluate tradeoffs **without running every benchmark themselves**.

### Example: Red Hat AI's Qwen3-0.6B Quantized Model Card (W4A16)
- **W4A16**: Weights at INT4, activations at base (FP16) configuration.
- The **recovery column** shows how much of the base model's accuracy is retained.
- Most benchmarks with meaningful base scores show **93–100% recovery**.
- HellaSwag example:
  - Base model: 43.04% accuracy
  - W4A16 quantized: 41.02% accuracy
  - Recovery rate: **95.3%**

> Check recovery specifically on the **tasks that matter to your use case**, not just overall averages.

---

## 10. The Three Sources of Evidence for Deployment Decisions

| Source | What It Tells You |
|---|---|
| **GuideLLM** | How the deployment performs: latency, throughput, consistency |
| **lm_eval** | How the model answers: accuracy on various task types |
| **Published Model Card** | How the model performs across many benchmarks (from the model publisher) |

---

## 11. Making the Deployment Decision

When deciding whether to deploy a quantized model, **both dimensions are required**:

- An optimization that **doubles throughput but drops accuracy by 15%** may not be worth it.
- An accurate model that **cannot meet latency SLOs is not deployable**.

### Example Trade-off for W4A16 Model
- **50% model size reduction**
- **~4% average accuracy loss** on OpenLLM v1 benchmarks
- Whether this is acceptable **depends on your use case**.

> Always align deployment decisions with your specific SLOs and the tasks that matter for your application.

---

## Key Takeaways Summary

1. Define **SLOs before benchmarking** — numbers are meaningless without targets.
2. Use **GuideLLM** for performance measurement (latency, throughput) under realistic load.
3. Use **lm_eval** for accuracy measurement against standardized and custom benchmarks.
4. Always examine **p95 and p99 percentiles**, not just the mean, to detect tail latency issues.
5. Consult **published model cards** as a quick, pre-computed source of accuracy tradeoff data.
6. **Neither performance nor accuracy alone is sufficient** — you need both for a production-ready deployment.
