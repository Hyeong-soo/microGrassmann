"""
benchmark.py — microGPT (Attention) vs microGrassmann (Plucker) Performance Comparison
==========================================================================
Trains both models under identical conditions (data, hyperparameters, number of training steps)
and compares loss curves, parameter counts, training speed, and generation quality.

Usage:
  python3 benchmark.py
==========================================================================
"""

import os
import math
import random
import time

# ============================================================================
#  Shared Infrastructure: Autograd, Data, Utilities
# ============================================================================

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
    def __pow__(self, other):
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))
    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))
    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self):        return self * -1
    def __radd__(self, o):    return self + o
    def __sub__(self, o):     return self + (-o)
    def __rsub__(self, o):    return o + (-self)
    def __rmul__(self, o):    return self * o
    def __truediv__(self, o): return self * o ** -1
    def __rtruediv__(self, o):return o * self ** -1
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children: build(c)
                topo.append(v)
        build(self)
        self.grad = 1
        for v in reversed(topo):
            for c, lg in zip(v._children, v._local_grads):
                c.grad += lg * v.grad

# --- Data Loading ---
if not os.path.exists('input.txt'):
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(url, 'input.txt')

random.seed(42)
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1

# --- Shared Hyperparameters ---
n_embd     = 16
n_layer    = 1
block_size = 16
n_head     = 4               # Attention only
head_dim   = n_embd // n_head
r          = 4               # Grassmann only
plucker_dim = r * (r - 1) // 2
window     = [1, 2, 4]

lr, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
num_steps  = 1000
temperature = 0.5

# --- Utility Functions ---
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]

def softmax(logits):
    mx = max(v.data for v in logits)
    exps = [(v - mx).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def sigmoid(x):
    return Value(1.0) / (Value(1.0) + (-x).exp())

def compute_plucker(u, v):
    coords = []
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            coords.append(u[i] * v[j] - u[j] * v[i])
    return coords


# ============================================================================
#  Model A: microGPT (Multi-Head Attention)
# ============================================================================
#  Faithfully reproduces Karpathy's original microGPT architecture.
#  Core: Q*K^T -> softmax -> V weighted sum (attends to all previous tokens)
# ============================================================================

def build_attention():
    """Initializes the Attention model parameters."""
    matrix = lambda nout, nin: \
        [[Value(random.gauss(0, 0.08)) for _ in range(nin)] for _ in range(nout)]
    sd = {
        'wte': matrix(vocab_size, n_embd),
        'wpe': matrix(block_size, n_embd),
        'lm_head': matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        sd[f'L{i}.wq'] = matrix(n_embd, n_embd)   # Query projection
        sd[f'L{i}.wk'] = matrix(n_embd, n_embd)   # Key projection
        sd[f'L{i}.wv'] = matrix(n_embd, n_embd)   # Value projection
        sd[f'L{i}.wo'] = matrix(n_embd, n_embd)   # Output projection
        sd[f'L{i}.fc1'] = matrix(4 * n_embd, n_embd)
        sd[f'L{i}.fc2'] = matrix(n_embd, 4 * n_embd)
    params = [p for mat in sd.values() for row in mat for p in row]
    return sd, params

def forward_attention(token_id, pos_id, kv_cache, sd):
    """Forward pass for the Attention model (Karpathy microGPT original)."""
    x = [t + p for t, p in zip(sd['wte'][token_id], sd['wpe'][pos_id])]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_res = x
        x = rmsnorm(x)
        q = linear(x, sd[f'L{li}.wq'])
        k = linear(x, sd[f'L{li}.wk'])
        v = linear(x, sd[f'L{li}.wv'])
        kv_cache[li]['k'].append(k)
        kv_cache[li]['v'].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in kv_cache[li]['k']]
            v_h = [vi[hs:hs+head_dim] for vi in kv_cache[li]['v']]
            attn = [sum(q_h[j]*k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                    for t in range(len(k_h))]
            w = softmax(attn)
            head_out = [sum(w[t]*v_h[t][j] for t in range(len(v_h)))
                        for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, sd[f'L{li}.wo'])
        x = [a + b for a, b in zip(x, x_res)]
        x_res = x
        x = rmsnorm(x)
        x = linear(x, sd[f'L{li}.fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, sd[f'L{li}.fc2'])
        x = [a + b for a, b in zip(x, x_res)]
    return linear(x, sd['lm_head'])


# ============================================================================
#  Model B: microGrassmann (Plucker Coordinates)
# ============================================================================
#  Causal Grassmann Layer from the "Attention Is Not What You Need" paper.
#  Core: Dimensionality reduction -> Plucker encoding -> Gated mixing (local window only)
# ============================================================================

def build_grassmann():
    """Initializes the Grassmann model parameters."""
    matrix = lambda nout, nin: \
        [[Value(random.gauss(0, 0.08)) for _ in range(nin)] for _ in range(nout)]
    sd = {
        'wte': matrix(vocab_size, n_embd),
        'wpe': matrix(block_size, n_embd),
        'lm_head': matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        sd[f'L{i}.W_red']    = matrix(r, n_embd)
        sd[f'L{i}.W_plu']    = matrix(n_embd, plucker_dim)
        sd[f'L{i}.W_gate_h'] = matrix(n_embd, n_embd)
        sd[f'L{i}.W_gate_g'] = matrix(n_embd, n_embd)
        sd[f'L{i}.fc1']      = matrix(4 * n_embd, n_embd)
        sd[f'L{i}.fc2']      = matrix(n_embd, 4 * n_embd)
    params = [p for mat in sd.values() for row in mat for p in row]
    return sd, params

def forward_grassmann(token_id, pos_id, z_cache, sd):
    """Forward pass for the Grassmann model."""
    x = [t + p for t, p in zip(sd['wte'][token_id], sd['wpe'][pos_id])]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_res = x
        x = rmsnorm(x)
        z = linear(x, sd[f'L{li}.W_red'])
        z_cache[li].append(z)
        geo_feats = []
        for delta in window:
            pp = pos_id - delta
            if pp < 0: continue
            plucker = compute_plucker(z, z_cache[li][pp])
            nsq = sum(p * p for p in plucker)
            ni = (nsq + Value(1e-8)) ** -0.5
            plucker = [p * ni for p in plucker]
            geo_feats.append(linear(plucker, sd[f'L{li}.W_plu']))
        if geo_feats:
            nf = len(geo_feats)
            g = [sum(geo_feats[k][j] for k in range(nf)) / nf for j in range(n_embd)]
        else:
            g = [Value(0.0)] * n_embd
        gh = linear(x, sd[f'L{li}.W_gate_h'])
        gg = linear(g, sd[f'L{li}.W_gate_g'])
        alpha = [sigmoid(h + g_) for h, g_ in zip(gh, gg)]
        x = [a*xi + (Value(1.0)-a)*gi for a, xi, gi in zip(alpha, x, g)]
        x = [a + b for a, b in zip(x, x_res)]
        x_res = x
        x = rmsnorm(x)
        x = linear(x, sd[f'L{li}.fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, sd[f'L{li}.fc2'])
        x = [a + b for a, b in zip(x, x_res)]
    return linear(x, sd['lm_head'])


# ============================================================================
#  Training & Evaluation Loop
# ============================================================================

def train_model(name, build_fn, forward_fn, make_cache, seed):
    """
    Trains a model and returns the results.

    Args:
        name:       Model name (for display)
        build_fn:   Parameter builder function
        forward_fn: Forward pass function
        make_cache: Cache creation lambda (Attention: KV cache, Grassmann: z cache)
        seed:       Random seed (for parameter initialization)

    Returns:
        (loss_history, samples, n_params, elapsed_sec)
    """
    random.seed(seed)
    sd, params = build_fn()
    n_params = len(params)

    m_buf = [0.0] * n_params
    v_buf = [0.0] * n_params
    loss_history = []

    print(f"\n  [{name}] Number of parameters: {n_params}")
    print(f"  [{name}] Starting training...")

    t0 = time.time()
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        cache = make_cache()
        losses = []

        for pos_id in range(n):
            logits = forward_fn(tokens[pos_id], pos_id, cache, sd)
            probs = softmax(logits)
            losses.append(-probs[tokens[pos_id + 1]].log())

        loss = (1 / n) * sum(losses)
        loss.backward()

        lr_t = lr * (1 - step / num_steps)
        for i, p in enumerate(params):
            m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
            v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
            mh = m_buf[i] / (1 - beta1 ** (step + 1))
            vh = v_buf[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * mh / (vh ** 0.5 + eps_adam)
            p.grad = 0

        loss_history.append(loss.data)
        if step % 100 == 0 or step == num_steps - 1:
            print(f"    step {step+1:4d}/{num_steps} | loss {loss.data:.4f}")

    elapsed = time.time() - t0

    # --- Inference: Name Generation ---
    random.seed(999)  # Fix sampling seed (for fair comparison)
    samples = []
    for _ in range(20):
        cache = make_cache()
        tid = BOS
        chars = []
        for pos_id in range(block_size):
            logits = forward_fn(tid, pos_id, cache, sd)
            probs = softmax([l / temperature for l in logits])
            tid = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if tid == BOS: break
            chars.append(uchars[tid])
        samples.append(''.join(chars))

    return loss_history, samples, n_params, elapsed


# ============================================================================
#  Execution & Results Comparison
# ============================================================================

print("=" * 64)
print("  microGPT (Attention)  vs  microGrassmann (Plucker)")
print("  Same-Condition Performance Comparison")
print("=" * 64)
print(f"\n  Common settings: d={n_embd}, layers={n_layer}, block_size={block_size}, "
      f"steps={num_steps}")
print(f"  Attention: {n_head} heads, head_dim={head_dim}")
print(f"  Grassmann: r={r}, plucker_dim={plucker_dim}, window={window}")

# --- Run Training ---
attn_cache_fn = lambda: [{'k': [], 'v': []} for _ in range(n_layer)]
grass_cache_fn = lambda: [[] for _ in range(n_layer)]

loss_a, samp_a, np_a, time_a = train_model(
    "Attention", build_attention, forward_attention, attn_cache_fn, seed=1234)

loss_g, samp_g, np_g, time_g = train_model(
    "Grassmann", build_grassmann, forward_grassmann, grass_cache_fn, seed=1234)


# ============================================================================
#  Results Output
# ============================================================================

def avg(lst, start, end):
    """Computes the average over a range."""
    segment = lst[start:end]
    return sum(segment) / len(segment) if segment else 0

print(f"\n{'=' * 64}")
print(f"  Comparison Results")
print(f"{'=' * 64}")

# --- Basic Statistics ---
print(f"\n  {'Metric':<24}{'Attention':>16}{'Grassmann':>16}")
print(f"  {'─' * 56}")
print(f"  {'Parameters':<24}{np_a:>16,}{np_g:>16,}")
print(f"  {'Training time (sec)':<24}{time_a:>16.1f}{time_g:>16.1f}")
print(f"  {'Time per step (ms)':<24}{time_a/num_steps*1000:>16.1f}{time_g/num_steps*1000:>16.1f}")

# --- Loss Curve Comparison (Range Averages) ---
print(f"\n  {'Loss Comparison (avg)':<24}{'Attention':>16}{'Grassmann':>16}")
print(f"  {'─' * 56}")
ranges = [
    ("Early (1-100)",       0,   100),
    ("Mid (401-500)",       400, 500),
    ("Late (901-1000)",     900, 1000),
]
for label, s, e in ranges:
    la = avg(loss_a, s, e)
    lg = avg(loss_g, s, e)
    diff = ((lg - la) / la) * 100
    marker = "<--" if lg < la else ("-->" if lg > la else "==")
    print(f"  {label:<24}{la:>16.4f}{lg:>16.4f}  {diff:+.1f}% {marker}")

print(f"\n  {'Final loss':<24}{loss_a[-1]:>16.4f}{loss_g[-1]:>16.4f}")
print(f"  {'Lowest loss':<24}{min(loss_a):>16.4f}{min(loss_g):>16.4f}")

# --- Loss Curve ASCII Visualization ---
print(f"\n  Loss Curve (50-step moving average):")
print(f"  {'─' * 56}")

bucket = 50
n_buckets = num_steps // bucket
ma_a = [avg(loss_a, i*bucket, (i+1)*bucket) for i in range(n_buckets)]
ma_g = [avg(loss_g, i*bucket, (i+1)*bucket) for i in range(n_buckets)]

all_vals = ma_a + ma_g
lo, hi = min(all_vals), max(all_vals)
chart_w = 40

for i in range(n_buckets):
    step_label = f"{(i+1)*bucket:>4}"
    if hi > lo:
        pos_a = int((ma_a[i] - lo) / (hi - lo) * (chart_w - 1))
        pos_g = int((ma_g[i] - lo) / (hi - lo) * (chart_w - 1))
    else:
        pos_a = pos_g = chart_w // 2

    line = [' '] * chart_w
    # Place the one behind first, then the one in front (so both are visible when overlapping)
    line[pos_a] = 'A'
    if pos_g == pos_a:
        line[pos_g] = '*'  # Overlap shown as *
    else:
        line[pos_g] = 'G'
    print(f"  {step_label} |{''.join(line)}|")

print(f"  {'':>5}  {lo:.2f}{' ' * (chart_w - 10)}{hi:.2f}")
print(f"        A = Attention,  G = Grassmann,  * = overlap")

# --- Generation Samples Comparison ---
print(f"\n  {'Generation Samples Comparison':}")
print(f"  {'─' * 56}")
print(f"  {'#':<4}  {'Attention':<20}  {'Grassmann':<20}")
print(f"  {'─' * 56}")
for i in range(20):
    print(f"  {i+1:<4}  {samp_a[i]:<20}  {samp_g[i]:<20}")

# --- Complexity Comparison ---
print(f"\n  Theoretical Complexity Comparison (sequence length L, embedding dim d):")
print(f"  {'─' * 56}")
print(f"  {'Attention':<16}  O(L^2 * d_head  +  L * d^2)")
print(f"  {'':>16}  Computes similarity for all token pairs (L x L matrix)")
print(f"  {'Grassmann':<16}  O(L * |W| * r^2  +  L * d^2)")
print(f"  {'':>16}  Uses only fixed window offsets (no L^2 term)")
print(f"  {'':>16}  |W|={len(window)}, r={r} -> effectively O(L * d^2)")

# --- Conclusion ---
final_a, final_g = loss_a[-1], loss_g[-1]
diff_pct = ((final_g - final_a) / final_a) * 100
speed_ratio = time_a / time_g if time_g > 0 else float('inf')
param_ratio = np_a / np_g if np_g > 0 else float('inf')

print(f"\n{'=' * 64}")
print(f"  Summary")
print(f"{'=' * 64}")
print(f"  Grassmann's final loss is {diff_pct:+.1f}% relative to Attention")
print(f"  Parameters: Grassmann is {np_g/np_a*100:.0f}% of Attention")
print(f"  Training speed: Grassmann takes {time_g/time_a*100:.0f}% of Attention's time")
if abs(diff_pct) < 15:
    print(f"  --> Consistent with paper's claim: at similar scale, Grassmann achieves")
    print(f"      performance within 10-15% of Attention (without L^2!)")
print(f"{'=' * 64}")
