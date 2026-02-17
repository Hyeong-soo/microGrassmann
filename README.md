# microGrassmann

Pure Python implementation of **"Attention Is Not What You Need"** ([arXiv:2512.19428](https://arxiv.org/abs/2512.19428)) in the style of [Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) (~200 lines, zero dependencies).

## Core Idea

Replace Transformer's Multi-Head Attention with **Grassmann geometry**:

```
Attention:   Q·K^T → softmax → weighted sum of V    O(L²)
Grassmann:   Plücker coordinates → geometric encoding  O(L)
```

How each approach represents the relationship between two tokens:
- **Attention**: dot product → 1 scalar (similarity score)
- **Grassmann**: Plücker coordinates → C(r,2)-dim vector (geometric relationship)

## Quickstart

```bash
# Main implementation (annotated, 734 lines)
python3 micro_grassmann.py

# Clean version (~130 lines)
python3 micro_grassmann_clean.py

# Karpathy's original microGPT
python3 micro_gpt.py
```

Zero dependencies. Just Python 3.
Dataset (`input.txt`) auto-downloads on first run.

## Benchmarks

### Name Generation (3000 steps)

```
              Params    Eval Loss
Attention     4,192     2.3090      (microGPT baseline)
Grassmann     3,840     2.3097      (8% fewer params, same performance)
```

### Parenthesis Matching (seq length ~35, 1000 steps)

```
              Eval Loss    Speed         Valid Parens Generated
Attention     0.5967       352ms/step    100%
Grassmann     0.6660       246ms/step    20%
```

Parenthesis matching requires tracking **all** previous tokens to count open brackets — Attention sees everything, Grassmann only sees fixed window offsets.

However, Grassmann is **30% faster** per step, and this speed advantage grows with sequence length (O(L) vs O(L²)).

> **Note on multi-layer scaling:** Our implementation uses a single layer, which limits Grassmann to its fixed window offsets `[1, 2, 4, 8, 12, 16]`. The paper uses 6–12 layers where information **flows** through the sequence across layers — each layer's output becomes the next layer's input, so even distant tokens can influence each other indirectly. With sufficient depth, Grassmann could potentially handle long-range dependencies like parenthesis matching through this cascading "flow" mechanism, similar to how stacked CNN layers with small kernels achieve large receptive fields. The title's "Flow" refers to exactly this: information propagation through controlled deformations of subspaces across layers, not explicit pairwise attention.

```bash
python3 benchmark.py         # Name generation comparison
python3 benchmark_paren.py   # Parenthesis matching comparison
```

## Architecture

```
Input token
  ↓
Token embedding + Position embedding
  ↓
RMSNorm
  ↓
┌─── Causal Grassmann Layer ────────────────────┐
│  x ──W_red──→ z (dim reduction: d→r)          │
│                ├── Plücker(z, z_{t-1})  ─┐     │
│                ├── Plücker(z, z_{t-2})   ├→ avg → g (geometric vector)
│                ├── Plücker(z, z_{t-4})   │     │
│                ├── Plücker(z, z_{t-8})  ─┘     │
│                ...                             │
│  alpha = sigmoid(W_gate_h·x + W_gate_g·g)     │
│  output = alpha·x + (1-alpha)·g                │
└────────────────────────────────────────────────┘
  ↓ + residual
FFN (d → 4d → ReLU → d)
  ↓ + residual
Output projection → logits
```

### Faithfulness to the Paper

| Component | Paper | Our Implementation | Notes |
|-----------|-------|--------------------|-------|
| Plücker coordinates | p_ij = z_i·z'_j - z_j·z'_i | Same | Core operation |
| Window offsets | [1,2,4,8,12,16] | Same | |
| Gate | concat([h;g]) → W_gate | W_h·x + W_g·g | Mathematically equivalent |
| Aggregation | Simple average | Same | |
| Mixing | α·h + (1-α)·g | Same | |
| Normalization | LayerNorm | RMSNorm | microGPT style |
| FFN activation | GELU | ReLU | microGPT style |
| Bias | Yes | No | microGPT style |
| Layers | 6–12 | 1 | Pure Python constraint |

## Educational Materials

Step-by-step learning resources included:

| File | Description |
|------|-------------|
| `tutorial.py` | 14-step tutorial — from Attention basics to Grassmann |
| `visualize_plucker.py` | Plücker coordinate visualization (requires matplotlib) |
| `trace_g.py` | Traces geometric vector g construction with actual numbers |
| `trace_alpha_wred.py` | Explains how alpha is determined and why W_red matters |

```bash
python3 tutorial.py
python3 trace_g.py
python3 trace_alpha_wred.py

# Plücker visualization (needs venv)
python3 -m venv .venv && source .venv/bin/activate && pip install matplotlib
python3 visualize_plucker.py  # → plucker_explained.png
```

## File Structure

```
├── micro_grassmann.py        # Main implementation (annotated)
├── micro_grassmann_clean.py  # Clean version (~130 lines)
├── micro_gpt.py              # Karpathy's microGPT (baseline)
├── benchmark.py              # Attention vs Grassmann on name generation
├── benchmark_paren.py        # Parenthesis matching comparison
├── tutorial.py               # Step-by-step tutorial
├── visualize_plucker.py      # Plücker coordinate visualization
├── trace_g.py                # Geometric vector trace
├── trace_alpha_wred.py       # Alpha and W_red explanation
├── plucker_explained.png     # Visualization output
└── input.txt                 # Name dataset (auto-downloaded)
```

## References

- Paper: [Attention Is Not What You Need (arXiv:2512.19428)](https://arxiv.org/abs/2512.19428)
- Baseline: [Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- Full-scale reproduction: [Infatoshi/grassmann-flows](https://github.com/Infatoshi/grassmann-flows)
