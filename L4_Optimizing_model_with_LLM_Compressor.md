# Lecture Notes: Optimizing a Model with LLM Compressor

**Course:** Fast & Efficient LLM Inference with vLLM
**Lecture:** Optimizing a Model with LLM Compressor

---

## Overview

This lecture covers the practical application of quantization using the open-source **LLM Compressor** tool. The goal is to take a full-precision model, compress it, compare sizes before and after, and measure whether the compressed model still performs well.

---

## The Four Steps of LLM Compression

1. **Choose a Model** — Select a model, typically from Hugging Face (described as "the GitHub of models") or an internal organizational registry.
2. **Pick the Algorithm** — Different algorithms have different tradeoffs in compression speed and accuracy retention.
3. **Choose the Quantization Scheme** — Decide the precision for weights and activations.
4. **Inference with vLLM** — Once compressed, inference engines like vLLM can load the model directly.

---

## Step 1: Choosing a Model & Calibration Data

- Most quantization algorithms require **representative calibration data** to guide the compression.
- Calibration data helps the algorithm understand:
  - Which weights matter most
  - How to minimize accuracy loss during quantization

---

## Step 2: Choosing the Algorithm

### Algorithms Available in LLM Compressor

| Algorithm | Description | Key Tradeoff |
|---|---|---|
| **Round-to-Nearest** | Simplest method; rounds each weight to nearest target precision value | Fastest, no calibration needed, but accuracy degrades at lower bits (e.g., INT4) |
| **AWQ** (Activation Aware Weight Quantization) | Best balance of accuracy and speed, especially on NVIDIA hardware | Computationally lighter than GPTQ; calibration runs faster and needs less VRAM |
| **GPTQ** | Industry standard; mathematically rigorous | Higher accuracy but requires more compute and VRAM during compression |
| **SparseGPT** | Handles sparsification | Only useful with specific hardware like NVIDIA H100 |

### Pre-Processing Methods (Used Before Compression)

- **Smoothing** — Flattens spikes in the model's weight distribution to reduce information loss.
- **Transformations** — Apply rotations to weights so less information is lost during quantization.

---

### Algorithm Deep Dives

#### Round-to-Nearest
- Each weight is rounded to the nearest value in the target precision.
- No calibration data needed; very fast.
- Accuracy degradation occurs specifically at lower bits like INT4.
- Good as a **baseline**, but not recommended for production use.

#### AWQ (Activation Aware Weight Quantization)
- Based on the observation that **not all weights are equally important**.
- Some weights, when changed even slightly, cause large changes in model outputs; others can be rounded aggressively with minimal effect.
- Uses **activation magnitudes** during a calibration pass to identify which weights are which.
- Weights corresponding to **large activations** are treated more carefully; the rest are compressed more aggressively.
- Computationally lighter than GPTQ — calibration runs faster and requires less VRAM.

#### GPTQ
- Takes a **mathematically rigorous** approach.
- Core idea: given that quantization will introduce error, how can we compensate for that error in the remaining weights so the overall output changes as little as possible?
- Computes the **Hessian of the loss** with respect to the weights — a measure of curvature indicating how sensitive model output is to changes in each specific weight.
- Works through weights **layer by layer**: quantizes each one and updates remaining weights to compensate.
- Computing and inverting Hessians is expensive → more compute and memory needed.
- Result: very high accuracy, often better than AWQ on certain benchmarks.
- Most **widely supported** algorithm; a safe choice for sharing quantized models.

---

## Step 3: Choosing the Quantization Scheme

### W4A16 Scheme (Used in This Lecture)
- **4-bit weights (W4)** and **16-bit activations (A16)**
- Targets **Linear layers**, where the vast majority of parameters live
- The **lm_head** (output layer that maps tokens to vocabulary) is excluded to keep it at full precision
- Expected to lead to roughly **50% total size reduction** for smaller models

---

## Practical Notebook Walkthrough

### Setup
- Libraries used: `torch`, Hugging Face `transformers`
- Base model used: **Qwen3-0.6B** (small but capable; suitable for the demo environment)
- Two folders provided: one for original weights, one for quantized weights

### Defining the Quantization Recipe
- A **recipe** tells LLM Compressor how to quantize.
- This lecture uses a single algorithm: **GPTQ**
- Recipe specifies:
  - Algorithm: GPTQ
  - Precision: 4-bit weights, 16-bit activations
  - Target layers: Linear layers only (lm_head excluded)

### The `oneshot` API
- Imported from LLM Compressor
- Takes the **model**, **calibration data**, and **recipe** together
- Compresses the model in a **single pass**

### Calibration Dataset: WikiText-2
- A standard benchmark of Wikipedia articles
- Same dataset used later for perplexity evaluation
- Key parameters:
  - **`num_calibration_samples`**: Number of sequences run through the model during calibration. More samples give a better picture of weight importance, but gains become tiny past a few hundred while runtime keeps growing. **Default used: 256**
  - **`max_seq_length`**: Max token length per sample. Longer sequences let the quantizer see how weights behave across realistic context lengths; samples beyond this are truncated.

---

## Model Size Comparison

### Expected vs. Actual Reduction
- Going from **16-bit to 4-bit** might suggest a **75% reduction** in size.
- Actual reduction observed: **~42%**

### Why Less Than 75%?
- Only the **Linear layer weights** are quantized.
- The rest of the model — including the lm_head and normalization layers — stays at higher precision.
- These unquantized pieces pull the overall reduction down.

### Scaling Insight
- This ratio **improves with larger models** where linear weights make up an even bigger share of total size.
- Example: A **70B model** quantized the same way gets much closer to the theoretical 4× compression.

---

## Evaluating Model Quality

### Qualitative Check: Same Prompt Test
- Both the base model and the quantized model are given the **same prompt** with the **same generation settings**.
- The only difference: quantized model has 4-bit weights instead of 16-bit.
- Expected result: similar (not necessarily word-for-word identical) output.

### Quantitative Metric: Perplexity
- **Perplexity** is a standard metric for language models.
- Measures how well the model predicts text.
- **Lower perplexity = better performance.**
- If quantization has degraded the model, perplexity will be noticeably higher.

#### How Perplexity Is Calculated (in this lecture)
- A **sliding window** is applied over a chunk of the WikiText-2 test set.
- At each position, the **cross-entropy loss** is computed between the model's predicted next-token distribution and the actual next token (i.e., how "surprised" the model is by the actual token).
- **Exponentiating the average loss** gives perplexity.
- The window moves forward by **stride tokens**, overlapping with the previous window.
- This allows evaluation of long text **without feeding it all at once**, while still giving each token a reasonable amount of left context.
- The **test split** of WikiText-2 is used (held-out portion of the calibration dataset) to avoid **data leakage**.

#### Results
| Model | Perplexity |
|---|---|
| Base model (BFloat16) | 32.79 |
| Quantized model (W4A16) | 35.48 |
| Difference | ~8% increase |

#### Interpreting the Results
- Lower perplexity is better; the quantized model's perplexity is slightly higher.
- For most production deployments, **a few percent increase in perplexity is acceptable** given the significant reduction in model size and infrastructure savings.

---

## Key Takeaways

1. **LLM Compressor's `oneshot` API** applies post-training quantization using a specified recipe (e.g., GPTQ) in a single pass.
2. **W4A16 quantization** (4-bit weights, 16-bit activations) delivered a **~42% reduction** in model size (not the theoretical 75%, due to unquantized layers).
3. **Qualitative output comparison** on the same prompt shows the quantized model produces similar results to the base model.
4. **Perplexity** is the standard quantitative metric to measure accuracy tradeoff — an ~8% increase was observed, which is generally acceptable for production use.
5. The compression-to-accuracy tradeoff **scales better with larger models** (e.g., 70B), where linear layers dominate total parameter count.
