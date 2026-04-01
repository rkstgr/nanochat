# Atlas: Implementation Details from arXiv 2505.23735

> **Atlas: Learning to Optimally Memorize the Context at Test Time**
> Behrouz, Li, Kacham, Daliri, Deng, Zhong, Razaviyayn, Mirrokni (Google Research)

This document extracts all concrete implementation details from the paper for reference when implementing/refining Atlas in nanochat.

---

## 1. Core Architecture Components

### 1.1 Memory Module (Deep MLP)

- **Architecture**: 2-layer MLP with residual connection and GELU activation:
  ```
  M(x) = x + W1 * GELU(W2 * x)
  ```
- **Expansion factor**: 4 (i.e., hidden dim = 4 * input dim)
- **Layer norm** applied at the end of each chunk
- **Atlas++** variant uses a **gated MLP** memory:
  ```
  M(x) = x + W1 * (GELU(W2 * x) ⊗ W3 * x)
  ```
  where W1, W2, W3 are learnable linear matrices. This is analogous to SwiGLU-style gating.

### 1.2 Input Projections

- Standard linear projections for keys, values, and queries: Q = x W_Q, K = x W_K, V = x W_V
- **Short causal convolution** (kernel size 4) applied after projection on Q, K, V
- **Normalization on keys and queries** to stabilize training

### 1.3 Polynomial Feature Mapping

- Keys (and queries) are lifted via polynomial kernel: φ_p(x) = [x^β]_{|β| ≤ p}
- This maps d_k-dimensional keys to D = C(d_k + p, p) = O(d_k^p) dimensions
- The polynomial kernel approximates softmax via Taylor series:
  ```
  exp(q^T k) ≈ a_0 + a_1 q^T k + a_2 (q^T k)^2 + ... + a_p (q^T k)^p
  ```
- Coefficients a_i are **learnable parameters**, initialized at a_i = 1/i!
- Acts as **input feature gating**: a_i → 0 excludes degree-i features, a_i → 1 retains them

### 1.4 Memory Capacity Theorems

| Memory Type | Capacity (max independent KV pairs) |
|---|---|
| Matrix-valued (linear) with delta rule | O(d_k) — sub-linear in parameters |
| Deep MLP (L ≥ 2 layers) | O(d_k * d_v) to O(d_k * d_v * Σ ...) — grows with depth |
| Polynomial mapping degree p | O(d_k^p) — super-linear for p ≥ 2 |

Key insight: deeper memory AND polynomial features both independently increase capacity.

---

## 2. Omega Rule (Sliding Window Memory Update)

### 2.1 Formulation

Instead of the online/Delta rule (optimizing memory w.r.t. only the current token), the Omega rule optimizes over a **sliding window** of c past tokens:

```
min_M  Σ_{i=t-c+1}^{t}  γ_i^(t) * ||M(φ(k_i)) - v_i||^2_2
```

where:
- c is the **local context length** (window size)
- γ_i^(t) ∈ [0, 1] are **input-dependent** decay/gating parameters
- φ(·) is the polynomial feature mapping

### 2.2 Special Cases

- **c = 1**: reduces to the standard online Delta rule (equivalent to Titans without momentum)
- **c = ∞** (or context length): global optimization over all past tokens
- **c = 1 with momentum**: equivalent to Titans' long-term neural memory

### 2.3 OmegaNet Update Rule

Using gradient descent on the Omega objective:
```
M_t = α_t * M_{t-1} - Σ_{i=t-c+1}^{t} η_i^(t) * ∇||M_{t-1}(φ(k_i)) - v_i||^2_2
```

For **linear memory** (M_t = W_t):
```
M_t = (diag(α_t) - Σ_{i=t-c+1}^{t} γ_i^(t) * φ(k_i)φ(k_i)^T) * M_{t-1}
      + Σ_{i=t-c+1}^{t} γ_i^(t) * v_i * φ(k_i)^T
```

---

## 3. Atlas: Muon Optimizer for Memory Management

### 3.1 Update Rule

Atlas uses Muon optimizer (with weight decay) instead of plain GD:

```
S_t = θ_t * S_{t-1} - Σ_{i=t-c+1}^{t} η_i^(t) * ∇||M_{t-1}(φ(k_i)) - v_i||^2_2
M_t = α_t * M_{t-1} - η_t * NewtonSchulz-k(S_t)
```

where:
- S_t is the **momentum** state
- θ_t is the **momentum decay** (input-dependent)
- α_t is the **weight decay / forget gate** (input-dependent)
- η_t is the **learning rate** (input-dependent)
- NewtonSchulz-k is k steps of Newton-Schulz iteration (Polar Express)

### 3.2 Newton-Schulz / Polar Express Details

- k steps of iteration; paper uses **k = 5** (called NS-5)
- Converges to nearest semi-orthogonal matrix to momentum S_t
- Approximates second-order information
- k acts as an **internal test-time compute parameter**: more steps → better memorization
- Uses optimized Polar Express coefficients (per-step (a, b, c) tuples)
- Algorithm per step: X_{i+1} = a_i X_i + b_i (X_i @ X_i^T) @ X_i + c_i ((X_i @ X_i^T)^2) @ X_i
- Input is Frobenius-normalized before iteration

### 3.3 Input-Dependent Gates

All gates are **input-dependent** (data-dependent):
- **α_t**: weight decay / forget gate for memory
- **η_t**: learning rate for memory update
- **θ_t**: momentum decay
- **γ_i^(t)**: per-token gates within the sliding window (hard context gating)

---

## 4. Parallel Training Algorithm

### 4.1 Chunk-Wise Computation

- Sequence of length L is divided into chunks of size **b ≥ 1**
- **Intra-chunk**: parallel computation
- **Inter-chunk**: recurrent computation
- Gradients are computed w.r.t. the **last state of the previous chunk** (not the current evolving state)

### 4.2 Chunk Update Rule

For timesteps t' ≤ t < t' + b within a chunk:
```
M_t = α_t...α_{t'} * M_{t'} - Σ_{n=t'}^{t} (α_t...α_{t'})/(α_n...α_{t'}) * η_n * G_t
```
where G_t = Σ_{i=n-c+1}^{n} ∇ℓ(M_{t'}; k_i, v_i)

### 4.3 Sliding Window Mask

- For gradient computation within chunks, a **sliding window mask M_s** is applied during einsum operations
- c = 1: M_s is the identity matrix
- c > 1: M_s is identity with c-1 positions before each diagonal also set to 1
- This avoids materializing c separate gradient matrices

### 4.4 Parallel Momentum + Newton-Schulz

Key insight: the momentum recurrence is **independent of memory state** within a chunk:
```
S_t = β_t * S_0 - Θ ⊙ E ⊙ G
```
where G is the gradient matrix, E and Θ are diagonal matrices of η and θ values. Since all S_t values can be computed in parallel, NewtonSchulz-k(S_t) can also be computed in parallel across the chunk.

---

## 5. DeepTransformer Variants

### 5.1 Deep Linear Attention (DLA)

- Replace matrix-valued memory in GLA with a deep MLP
- Uses dot-product similarity as attentional bias: ℓ = ⟨M(φ(k_t)), v_t⟩
- Update: M_t = α_t M_{t-1} - η_t ∇ℓ(M_{t-1}; φ(k_t), v_t)
- When linear memory: reduces to gated linear attention

### 5.2 DeepTransformer

- Uses exponential (infinite-dimensional) kernel φ*(x) instead of polynomial
- Unnormalized formulation: M_t = M_{t-1} - ∇⟨M_{t-1}(φ*(k_t)), v_t⟩
- With linear memory: reduces to unnormalized Transformer
- Therefore: **strict generalization of Transformers**

### 5.3 Deep Omega Transformer (Dot)

- DeepTransformer + Omega rule (sliding window):
  ```
  M_t = M_{t-1} - ∇ Σ_{i=t-c+1}^{t} γ_i^(t) ||M(φ*(k_i)) - v_i||^2_2
  ```
- With linear memory and c=1: generalizes Transformers with Delta rule

---

## 6. Hybrid Architectures

### 6.1 MAG (Memory Attention Gate)

Atlas memory output gated with sliding window attention output. Follows the Titans convention.

### 6.2 MAL (Memory Attention Layer)

Atlas memory and sliding window attention as separate layers. Follows the Titans convention.

### 6.3 MAC (Memory as Context)

Used in BABILong experiments — follows the Titans MAC architecture (without persistent memory tokens).

---

## 7. Training Details

### 7.1 Hyperparameters

| Scale | Layers | Dim | Heads | Peak LR | Tokens |
|-------|--------|-----|-------|---------|--------|
| 170M  | 12     | 768  | 16   | 3e-3    | 15B    |
| 340M  | 24     | 1024 | 16   | 1.5e-3  | 15B    |
| 760M  | 24     | 1536 | 16   | 1.25e-3 | 30B    |
| 1.3B  | 18     | 2048 | 8    | 7e-4    | 100B   |

### 7.2 Optimizer & Schedule

- **Outer loop**: AdamW with lr=4e-4, cosine annealing, weight decay=0.1
- **Batch size**: 0.5M tokens
- **Tokenizer**: T5 with vocabulary size 32K
- **Training sequence length**: 4K tokens (2K for SWA baselines)

### 7.3 Memory Architecture Defaults

- 2-layer MLP, expansion factor 4, GELU activation
- Residual connections
- Layer norm at end of each chunk

---

## 8. Key Experimental Results

### 8.1 Language Modeling (760M / 30B tokens)

| Model | Wiki PPL | LMB PPL | Avg Downstream |
|-------|----------|---------|----------------|
| Transformer++ | 25.21 | 27.64 | 48.69 |
| Titans (LMM) | 20.04 | 21.96 | 51.56 |
| OmegaNet | 19.16 | 20.14 | 52.56 |
| **Atlas** | **18.92** | 21.01 | **52.77** |
| Atlas++ | 19.04 | **20.03** | 53.09 |
| Atlas (MAG) | **18.62** | 21.18 | 53.08 |

### 8.2 Ablation Study (760M)

| Variant | PPL | Downstream Acc |
|---------|-----|----------------|
| Atlas (full) | 19.97 | 52.77 |
| + Gated MLP Memory | 19.53 | 53.09 |
| + Attention (MAG) | 19.90 | 53.08 |
| Linear Memory (no deep) | 21.03 | 49.74 |
| w/o Muon (GD only) | 19.65 | 52.56 |
| c = 1 (online/no Omega) | 21.98 | 49.26 |
| w/o Polynomial Mapping | 22.14 | 50.57 |

Key takeaways:
- **Omega rule (c > 1)** is the single most impactful component (21.98 → 19.97 PPL)
- **Polynomial mapping** is second most impactful (22.14 → 19.97 PPL)
- **Muon** provides moderate improvement over GD (19.65 → 19.97 is actually slightly worse PPL but better downstream)
- **Deep memory** matters: linear memory degrades significantly

### 8.3 Long Context (BABILong)

- Atlas maintains **80%+ accuracy at 10M context length**
- Titans drops off at 10M
- Both competitive until 1M context

### 8.4 Needle-in-Haystack (RULER S-NIAH)

- Atlas outperforms all recurrent baselines
- Extrapolates to 4x training context (trained on 4K, tested up to 16K)
- DeepTransformer and Dot variants also very strong on NIAH

---

## 9. Relation to Existing nanochat Implementation

The current `nanochat/atlas.py` implements:
- Linear (matrix-valued) memory per head — **paper also evaluates deep MLP memory**
- Polar Express orthogonalization (NS-5) — **matches paper**
- Input-dependent forgetting, learning rate, momentum gates — **matches paper**
- Short causal convolution on Q, K, V — **matches paper**
- Chunk-parallel computation — **matches paper**

### Gaps / Potential Improvements from Paper

1. **Deep MLP memory**: The paper shows deep memory (2-layer MLP with residual) significantly outperforms linear memory (21.03 vs 19.97 PPL). Current nanochat uses linear memory.

2. **Gated MLP memory (Atlas++)**: Further improves over standard deep memory (19.53 vs 19.97 PPL). Uses SwiGLU-style gating.

3. **Polynomial feature mapping on keys/queries**: The paper shows this is the second most impactful component. Maps keys to higher-dimensional space via polynomial kernels. Learnable coefficients initialized at 1/i!.

4. **Sliding window context (c > 1)**: The Omega rule's sliding window is the most impactful single component. The paper uses sliding window masking during chunk-parallel gradient computation. Check if the current implementation supports c > 1.

5. **Per-token context gates γ_i^(t)**: Input-dependent gates within the sliding window that allow the model to prune irrelevant context within the window.

6. **Hybrid variants (MAG/MAL)**: Combining Atlas memory with sliding window attention provides additional gains. Atlas (MAG) achieves the best perplexity overall.

7. **DeepTransformer / Dot**: Entirely separate architecture family that generalizes Transformers by replacing the standard KV cache with a deep neural memory. Could be implemented as a separate model in nanochat.
