# microgpt-extended

An extension of Andrej Karpathy's [microgpt](https://karpathy.github.io/2026/02/12/microgpt/) — a single-file, dependency-free GPT implementation — augmented with four modern large language model techniques:

| Algorithm | File location | Toggle flag |
|-----------|--------------|-------------|
| GELU activation | `microgpt_extended.py` | `USE_GELU` |
| LoRA (Low-Rank Adaptation) | `microgpt_extended.py` | `USE_LORA` |
| RoPE (Rotary Position Embedding) | `microgpt_extended.py` | `USE_ROPE` |
| Mixture of Experts | `microgpt_extended.py` | `USE_MOE` |

All four algorithms are implemented in `microgpt_extended.py` using the same scalar autograd engine from the original. Feature flags at the top of the file let you toggle each one independently.

---

## How to run

```bash
# Original (unmodified)
python microgpt.py

# Extended version (all four algorithms active)
python microgpt_extended.py
```

No dependencies beyond the Python standard library.

---

## Algorithm documentation

### 1. Gaussian Error Linear Units (GELU)

**Paper:** Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)*, arXiv:1606.08415

#### Underlying idea

ReLU activation is a hard gate: it passes a value through only if it is positive, and otherwise outputs exactly zero. GELU is a soft, probabilistic version of this gate. It weights each input value by the probability that a standard normal random variable is less than that value — in other words, by Φ(x), the CDF of N(0,1):

```
GELU(x) = x · Φ(x)
```

Intuitively, large positive activations are kept almost entirely (Φ(x) ≈ 1), large negative activations are suppressed almost entirely (Φ(x) ≈ 0), and activations near zero are attenuated proportionally to how likely they are to be meaningful signal. This smooth, stochastic gating encourages neurons to be active only when their input is statistically significant, which tends to improve training on tasks involving noise.

#### Implementation

The exact CDF is expensive to compute, so we use the tanh approximation from the paper:

```python
GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

Because this is expressed as a composition of existing `Value` operations (`exp`, `+`, `*`, `**`), the autograd engine propagates gradients through it automatically — no custom backward pass required.

In the code, `gelu(xi)` replaces `xi.relu()` in every MLP block.

---

### 2. LoRA — Low-Rank Adaptation

**Paper:** Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685

#### Underlying idea

Fine-tuning a large pretrained model requires updating all its weight matrices, which is memory- and compute-intensive. LoRA observes that the *change* to a weight matrix W during adaptation has low intrinsic rank — it lives in a much smaller subspace than the full d × d matrix. Rather than updating W directly, LoRA freezes W and adds a trainable low-rank decomposition alongside it:

```
W_effective = W + ΔW = W + B · A
```

where A is (r × d_in) and B is (d_out × r), with r ≪ d. The number of new parameters is 2·r·d instead of d², a dramatic reduction for large d. B is initialised to zero so that ΔW = 0 at the start of training and the model's initial behaviour is identical to the pretrained model.

#### Implementation

LoRA is applied to the Query (Q) and Key (K) attention projection matrices, which are the most impactful targets in practice. A helper function `linear_lora` performs the augmented matrix-vector multiply:

```python
def linear_lora(x, w, la, lb):
    base  = linear(x, w)    # W·x  (frozen conceptually; trained here since we have no pretrained checkpoint)
    mid   = linear(x, la)   # A·x  (r-dim intermediate)
    delta = linear(mid, lb) # B·(A·x)
    return [b + d for b, d in zip(base, delta)]
```

`la` (matrix A) is initialized with small random Gaussian values. `lb` (matrix B) is initialized to zeros, ensuring the model starts from the same point as the non-LoRA baseline.

---

### 3. RoPE — Rotary Position Embedding

**Paper:** Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864

#### Underlying idea

The original microgpt encodes position by adding a learned position embedding vector to the token embedding at the start of each forward pass. This is a global, absolute encoding baked into the residual stream. RoPE takes a different approach: position information is injected *directly into the attention mechanism* by rotating the Query and Key vectors.

For each pair of dimensions (2i, 2i+1) in a head's Q or K vector, a 2D rotation is applied using a frequency θᵢ that depends on the dimension index:

```
θᵢ = 1 / 10000^(2i / d_head)

q'[2i]   = q[2i]·cos(m·θᵢ) − q[2i+1]·sin(m·θᵢ)
q'[2i+1] = q[2i]·sin(m·θᵢ) + q[2i+1]·cos(m·θᵢ)
```

where m is the sequence position. The key insight is that when the attention score q^T k is computed between a query at position m and a key at position n, the rotation matrices cancel in a way that leaves only the *relative* displacement (m − n) in the dot product. This means the model learns relative, not absolute, positional relationships — a desirable property that generalises better to sequence lengths unseen during training.

Because cos/sin are applied to constant scalars (the angles θᵢ·m are not learnable), they appear in the autograd graph only as fixed scaling factors, and gradients propagate through them correctly.

#### Implementation

```python
def rope_rotate(x, pos):
    result = list(x)
    dim = len(x)
    for i in range(0, dim - 1, 2):
        theta = pos / (10000 ** (i / dim))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x0, x1 = x[i], x[i + 1]
        result[i]     = x0 * cos_t - x1 * sin_t
        result[i + 1] = x0 * sin_t + x1 * cos_t
    return result
```

`rope_rotate` is called once per head for both Q and K at every time step, using the current sequence position. The rotated K is stored in the KV cache so that when later queries attend to it, the relative position difference is automatically encoded in the dot product.

When `USE_ROPE = True`, the learned position embedding table `wpe` is removed from the model — its parameters are replaced by the parameter-free rotation.

---

### 4. Mixture of Experts (MoE)

**Reference:** Shazeer et al. (2017); Fedus et al., *Switch Transformers* (2021); [HuggingFace MoE blog](https://huggingface.co/blog/moe)

#### Underlying idea

In a standard Transformer, the MLP block is a single feed-forward network applied to every token equally. Mixture of Experts scales this up by maintaining multiple parallel expert MLPs (each with its own parameters) and routing each token to only a subset of them:

1. A small **router** network maps the hidden state to a probability distribution over experts.
2. One or more experts are selected (**sparse gating** — most experts are skipped for any given token).
3. The final output is a weighted combination of the selected experts' outputs.

This decouples *model capacity* (total number of parameters across all experts) from *compute per token* (only k experts are evaluated). A model can therefore be much larger in total parameter count while keeping per-token FLOPs the same as a dense model with fewer parameters.

The router learns to specialise experts. In practice, some experts learn to handle certain types of input (e.g., punctuation, specific syntactic roles) while others handle different patterns. Load-balancing auxiliary losses are often added to prevent all tokens from routing to the same expert, though that is omitted here for clarity.

#### Implementation

```python
# Router: n_experts logits from the current hidden state
router_logits = linear(x, state_dict[f'layer{li}.router'])
router_probs  = softmax(router_logits)

# Top-1 sparse gating
expert_idx  = max(range(n_experts), key=lambda e: router_probs[e].data)
gate_weight = router_probs[expert_idx]

# Run the selected expert
h = linear(x, state_dict[f'layer{li}.mlp_fc1_e{expert_idx}'])
h = [gelu(hi) if USE_GELU else hi.relu() for hi in h]
h = linear(h, state_dict[f'layer{li}.mlp_fc2_e{expert_idx}'])

# Scale by routing probability (differentiable; gradient flows through gate_weight)
x = [gate_weight * hi for hi in h]
```

The gate weight multiplied onto the expert output keeps the routing decision differentiable: the router learns which expert to prefer based on how much it reduces the loss.

---

## References

- Hendrycks, D. & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*. arXiv:1606.08415
- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685
- Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864
- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. arXiv:1701.06538
- Fedus, W., Zoph, B. & Shazeer, N. (2021). *Switch Transformers*. arXiv:2101.03961
- Karpathy, A. (2026). *microgpt*. https://karpathy.github.io/2026/02/12/microgpt/
