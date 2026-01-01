"""
microgpt_extended.py
Extended version of Karpathy's microgpt.py with four modern LLM algorithms:
  1. GELU activation (replaces ReLU in the MLP block)
  2. LoRA - Low-Rank Adaptation (applied to Q and K attention projections)
  3. RoPE - Rotary Position Embedding (replaces learned position embeddings)
  4. Mixture of Experts (replaces the single MLP with multiple sparse expert MLPs)

All additions preserve the scalar-autograd style of the original.
"""

import os
import math
import random
random.seed(42)

# ─── Feature flags ────────────────────────────────────────────────────────────
USE_GELU = True   # Gaussian Error Linear Units in the MLP blocks
USE_LORA = True   # Low-Rank Adaptation on Q and K weight matrices
USE_ROPE = True   # Rotary Position Embeddings (removes learned wpe)
USE_MOE  = True   # Mixture of Experts replacing each MLP block
# ──────────────────────────────────────────────────────────────────────────────

# ─── Dataset ──────────────────────────────────────────────────────────────────
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ─── Tokenizer ────────────────────────────────────────────────────────────────
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ─── Autograd engine (same as original, with GELU added) ──────────────────────
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self):  return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self):  return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self):          return self * -1
    def __radd__(self, other):  return self + other
    def __sub__(self, other):   return self + (-other)
    def __rsub__(self, other):  return other + (-self)
    def __rmul__(self, other):  return self * other
    def __truediv__(self, other):  return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# ─── GELU activation ──────────────────────────────────────────────────────────
# Implements the tanh approximation of GELU from Hendrycks & Gimpel (2016):
#   GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
# Composed from existing Value ops so autograd handles gradients automatically.
def gelu(xi):
    k = 0.7978845608  # sqrt(2 / pi)
    c = (xi + xi**3 * 0.044715) * k          # inner argument of tanh
    e2c = (c * 2).exp()                        # e^(2c)
    tanh_c = (e2c - 1) / (e2c + 1)            # tanh(c) = (e^2c - 1)/(e^2c + 1)
    return xi * (1 + tanh_c) * 0.5

# ─── Hyperparameters ──────────────────────────────────────────────────────────
n_layer    = 1
n_embd     = 16
block_size = 16
n_head     = 4
head_dim   = n_embd // n_head
lora_rank  = 4          # rank r for LoRA decomposition
n_experts  = 4          # number of MoE experts per layer

# ─── Parameter initialization ─────────────────────────────────────────────────
matrix  = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
zeros   = lambda nout, nin:           [[Value(0.0)                  for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte':     matrix(vocab_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd),
}

# Position embedding: only added when RoPE is disabled
if not USE_ROPE:
    state_dict['wpe'] = matrix(block_size, n_embd)

for i in range(n_layer):
    # Attention projections
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)

    # LoRA: low-rank adaptation matrices for Q and K
    # A is initialised Gaussian (the "down" projection), B initialised to zero so
    # ΔW = B @ A starts at zero and the model begins identical to the base model.
    if USE_LORA:
        state_dict[f'layer{i}.lora_aq'] = matrix(lora_rank, n_embd)  # r × n_embd
        state_dict[f'layer{i}.lora_bq'] = zeros(n_embd, lora_rank)   # n_embd × r  (init 0)
        state_dict[f'layer{i}.lora_ak'] = matrix(lora_rank, n_embd)
        state_dict[f'layer{i}.lora_bk'] = zeros(n_embd, lora_rank)

    # MLP or Mixture of Experts
    if USE_MOE:
        # Router: projects hidden state to n_experts logits
        state_dict[f'layer{i}.router'] = matrix(n_experts, n_embd)
        # Each expert has its own up-projection (fc1) and down-projection (fc2)
        for e in range(n_experts):
            state_dict[f'layer{i}.mlp_fc1_e{e}'] = matrix(4 * n_embd, n_embd)
            state_dict[f'layer{i}.mlp_fc2_e{e}'] = matrix(n_embd, 4 * n_embd)
    else:
        state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# ─── Neural network building blocks ───────────────────────────────────────────
def linear(x, w):
    """Standard matrix-vector multiply: y = W x."""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def linear_lora(x, w, la, lb):
    """
    LoRA-augmented linear: y = (W + B A) x
    la : r × d_in   (A matrix, random init)
    lb : d_out × r  (B matrix, zero init)
    The low-rank delta B@A starts at zero so initial behaviour matches the
    base model; A and B are learned jointly with all other parameters.
    """
    base  = linear(x, w)                                 # W x
    mid   = linear(x, la)                                 # A x  (length r)
    delta = linear(mid, lb)                               # B (A x)
    return [b + d for b, d in zip(base, delta)]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

# ─── RoPE helper ──────────────────────────────────────────────────────────────
def rope_rotate(x, pos):
    """
    Apply rotary position embedding to vector x at sequence position pos.
    Pairs consecutive dimensions (2i, 2i+1) and rotates each pair by θ_i·pos
    where θ_i = 1 / 10000^(2i / dim).  Because cos/sin are evaluated on
    constant scalars (not Value nodes), autograd sees them only as scaling
    factors and propagates through them correctly.
    """
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

# ─── GPT forward pass ─────────────────────────────────────────────────────────
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]

    # Positional encoding: learned table (original) or RoPE (applied later to Q/K)
    if USE_ROPE:
        x = list(tok_emb)          # no position embedding added to the residual stream
    else:
        pos_emb = state_dict['wpe'][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]

    x = rmsnorm(x)

    for li in range(n_layer):
        # ── 1) Multi-head Attention ──────────────────────────────────────────
        x_residual = x
        x = rmsnorm(x)

        # Q and K projections (optionally augmented with LoRA)
        if USE_LORA:
            q = linear_lora(x, state_dict[f'layer{li}.attn_wq'],
                               state_dict[f'layer{li}.lora_aq'],
                               state_dict[f'layer{li}.lora_bq'])
            k = linear_lora(x, state_dict[f'layer{li}.attn_wk'],
                               state_dict[f'layer{li}.lora_ak'],
                               state_dict[f'layer{li}.lora_bk'])
        else:
            q = linear(x, state_dict[f'layer{li}.attn_wq'])
            k = linear(x, state_dict[f'layer{li}.attn_wk'])

        v = linear(x, state_dict[f'layer{li}.attn_wv'])

        # Apply RoPE to each head's Q and K slices before caching / attending
        if USE_ROPE:
            q_rotated = []
            k_rotated = []
            for h in range(n_head):
                hs = h * head_dim
                q_rotated.extend(rope_rotate(q[hs:hs + head_dim], pos_id))
                k_rotated.extend(rope_rotate(k[hs:hs + head_dim], pos_id))
            q, k = q_rotated, k_rotated

        # Cache already-rotated K and plain V
        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs + head_dim]
            k_h = [ki[hs:hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs + head_dim] for vi in values[li]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim ** 0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        # ── 2) MLP / Mixture-of-Experts block ───────────────────────────────
        x_residual = x
        x = rmsnorm(x)

        if USE_MOE:
            # Router: compute soft weights over experts
            router_logits = linear(x, state_dict[f'layer{li}.router'])
            router_probs  = softmax(router_logits)

            # Top-1 sparse gating: pick the expert with highest probability
            expert_idx = max(range(n_experts), key=lambda e: router_probs[e].data)
            gate_weight = router_probs[expert_idx]   # scalar routing weight

            # Forward through the selected expert
            h = linear(x, state_dict[f'layer{li}.mlp_fc1_e{expert_idx}'])
            h = [gelu(hi) if USE_GELU else hi.relu() for hi in h]
            h = linear(h, state_dict[f'layer{li}.mlp_fc2_e{expert_idx}'])

            # Scale expert output by its routing probability (encourages load balance)
            x = [gate_weight * hi for hi in h]
        else:
            # Standard single MLP
            x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
            x = [gelu(xi) if USE_GELU else xi.relu() for xi in x]
            x = linear(x, state_dict[f'layer{li}.mlp_fc2'])

        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# ─── Adam optimizer ───────────────────────────────────────────────────────────
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_buf = [0.0] * len(params)
v_buf = [0.0] * len(params)

# ─── Training loop ────────────────────────────────────────────────────────────
num_steps = 1000
for step in range(num_steps):
    doc    = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n      = min(block_size, len(tokens) - 1)

    keys_cache   = [[] for _ in range(n_layer)]
    values_cache = [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits  = gpt(token_id, pos_id, keys_cache, values_cache)
        probs   = softmax(logits)
        loss_t  = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_buf[i] / (1 - beta1 ** (step + 1))
        v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# ─── Inference ────────────────────────────────────────────────────────────────
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys_cache   = [[] for _ in range(n_layer)]
    values_cache = [[] for _ in range(n_layer)]
    token_id = BOS
    sample   = []
    for pos_id in range(block_size):
        logits   = gpt(token_id, pos_id, keys_cache, values_cache)
        probs    = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
