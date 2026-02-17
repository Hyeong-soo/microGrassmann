"""
micro_grassmann.py
================================================================================
Pure Python implementation of the "Attention Is Not What You Need" paper
(arXiv:2512.19428 — Grassmann Flows as an Attention-Free Alternative)

Following the style of Andrej Karpathy's microGPT, this implements a
Grassmann Flow-based language model using only pure Python, with no
external dependencies.

[Core Idea]
  Traditional Transformer:  Q·K^T -> softmax -> weighted sum(V) -> O(L^2) complexity
  This paper:               Plucker coordinates -> geometric encoding  -> O(L) complexity

  "Information flows through sequences not via explicit pairwise weights,
   but through controlled deformations of low-rank subspaces."
   — From the paper

[Architecture Comparison]
  microGPT (Karpathy)          micro_grassmann (this file)
  ─────────────────────        ─────────────────────────
  Value autograd engine        (same — reused)
  Token/position embeddings    (same)
  Multi-Head Attention  ──→    Causal Grassmann Layer ★
    Q = W_q · x                  z = W_red · x (dimension reduction)
    K = W_k · x                  Plucker(z_t, z_{t-delta})
    V = W_v · x                  g = W_plu · plucker
    softmax(QK^T/sqrt(d))·V      alpha = sigmoid(gate)
    W_o · attn_out                mix = alpha*x + (1-alpha)*g
  Feed-Forward Network         (same)
  Adam Optimizer               (same)

Dependencies: None (pure Python + math + random)
================================================================================
"""

import os
import math
import random

random.seed(42)


# ============================================================================
#  PART 1: Autograd Engine (Automatic Differentiation)
# ============================================================================
#
#  Neural network training = "adjusting parameters incrementally to reduce loss"
#  To do this, we need to know "how sensitive is the loss to each parameter?"
#  (= gradient).
#
#  Autograd records all math operations, and when backward() is called,
#  it uses the Chain Rule to automatically compute gradients for all parameters.
#
#  [Chain Rule Example]
#    When loss = f(g(x)),
#    d(loss)/dx = d(loss)/d(g) * d(g)/dx
#    i.e., propagate by multiplying "upstream gradient" x "local gradient".
#
#  A single Value object = a wrapper around a single scalar value:
#    .data         -> actual value (computed during forward pass)
#    .grad         -> gradient of loss w.r.t. this value (computed during backward pass)
#    ._children    -> input Values that produced this value
#    ._local_grads -> local gradient for each input (for chain rule)
#
#  [Example] If c = a * b:
#    c.data = a.data * b.data
#    c._children = (a, b)
#    c._local_grads = (b.data, a.data)   <- d(a*b)/da = b, d(a*b)/db = a
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
        # d(a+b)/da = 1, d(a+b)/db = 1
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a*b)/da = b, d(a*b)/db = a  (derivative of product)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        # d(a^n)/da = n * a^(n-1)  (power rule)
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        # d(ln a)/da = 1/a
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        # d(e^a)/da = e^a
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        # ReLU(x) = max(0, x)
        # Gradient: 1 if x > 0, 0 if x <= 0
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):        return self * -1
    def __radd__(self, o):    return self + o
    def __sub__(self, o):     return self + (-o)
    def __rsub__(self, o):    return o + (-self)
    def __rmul__(self, o):    return self * o
    def __truediv__(self, o): return self * o ** -1
    def __rtruediv__(self, o):return o * self ** -1

    def backward(self):
        """
        Backpropagation:
        Traverses the computation graph in reverse order to compute
        gradients for all Values.

        1. Determine node order via topological sort
        2. Start from the loss node (grad = 1, because d(loss)/d(loss) = 1)
        3. Visit each node in reverse order, applying the chain rule
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1  # Starting point: d(loss)/d(loss) = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad  # Chain Rule: local_grad * upstream_grad


# ============================================================================
#  PART 2: Data Loading & Tokenizer
# ============================================================================
#
#  Same dataset as Karpathy's microGPT: a list of English names (names.txt)
#  Each name is one "document", tokenized at the character level.
#
#  Tokenization example:
#    "henry" -> [BOS, h, e, n, r, y, BOS]
#
#  BOS (Beginning/End of Sequence) token:
#    - Marks the start and end of a sequence
#    - A special token that tells the model "start/end here"
# ============================================================================

if not os.path.exists('input.txt'):
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(url, 'input.txt')

docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"Number of documents (names): {len(docs)}")

uchars = sorted(set(''.join(docs)))    # Full list of unique characters (a-z, etc.)
BOS = len(uchars)                       # BOS token ID = last index
vocab_size = len(uchars) + 1            # Total vocabulary size = num_chars + BOS
print(f"Vocabulary size: {vocab_size}")


# ============================================================================
#  PART 3: Hyperparameters — This Is Where We Diverge from Attention!
# ============================================================================
#
#  [microGPT's Attention-related hyperparameters]
#    n_head = 4       (number of attention heads)
#    head_dim = 4     (dimension per head)
#    -> Requires Q, K, V matrices, computes an L x L attention matrix
#
#  [microGrassmann's Grassmann-related hyperparameters]
#    r = 4            (target reduced dimension — compresses token vectors to this size)
#    plucker_dim = 6  (Plucker coordinate dimension = r*(r-1)/2 = 4*3/2 = 6)
#    window = [1,2,4] (local window offsets)
#    -> No L x L attention matrix at all! Only local windows are used
#
#  [What are window offsets?]
#    They determine how many steps back the current token (at position t) looks.
#    If window = [1, 2, 4]:
#      delta=1: Compare with the immediately previous token (t-1) -> adjacent context
#      delta=2: Compare with the token two steps back (t-2)       -> slightly wider context
#      delta=4: Compare with the token four steps back (t-4)      -> distant context
#    This captures sequence information at multiple scales (multi-scale).
#    Attention looks at all pairs (O(L^2)), but here only fixed offsets are used (O(L)).
# ============================================================================

n_embd     = 16       # Embedding dimension (size of the vector representing each token)
n_layer    = 1        # Number of Grassmann layers (depth)
block_size = 16       # Maximum sequence length
r          = 4        # ** Grassmann reduced dimension (d=16 -> compressed to r=4)
plucker_dim = r * (r - 1) // 2   # C(r,2) = 4*3/2 = 6  (Plucker coordinate dimension)
window     = [1, 2, 4, 8, 12, 16] # Local window offset set (same as the paper)

print(f"Embedding dim d={n_embd}, reduced dim r={r}, "
      f"Plucker dim={plucker_dim}, window={window}")


# ============================================================================
#  PART 4: Model Parameter Initialization
# ============================================================================
#
#  Understanding the role of each parameter matrix is key.
#
#  ┌─────────────────────────────────────────────────────────────────┐
#  │  Parameter     Size              Role                           │
#  ├─────────────────────────────────────────────────────────────────┤
#  │  wte         (vocab, d)     Token embedding (word -> vector)    │
#  │  wpe         (block, d)     Position embedding (position -> vector) │
#  │  lm_head     (vocab, d)     Output projection (vector -> word probs) │
#  │                                                                 │
#  │  --- Below completely replaces Attention's Q,K,V,O matrices --- │
#  │                                                                 │
#  │  W_red       (r, d)         Dimension reduction (16-dim -> 4-dim) │
#  │  W_plu       (d, plucker)   Restore Plucker coords to model dim │
#  │  W_gate_h    (d, d)         Gate: weight for original pathway   │
#  │  W_gate_g    (d, d)         Gate: weight for geometry pathway   │
#  │  mlp_fc1     (4d, d)        FFN expansion (same as Transformer) │
#  │  mlp_fc2     (d, 4d)        FFN contraction (same as Transformer) │
#  └─────────────────────────────────────────────────────────────────┘
#
#  [Parameter Count Comparison]
#  microGPT Attention:  W_q + W_k + W_v + W_o = 4 * d * d = 4 * 16 * 16 = 1024
#  microGrassmann:      W_red + W_plu + W_gate_h + W_gate_g
#                       = r*d + d*C(r,2) + d*d + d*d
#                       = 4*16 + 16*6 + 16*16 + 16*16
#                       = 64 + 96 + 256 + 256 = 672
#  -> Grassmann uses fewer parameters while leveraging geometric structure!
# ============================================================================

matrix = lambda nout, nin, std=0.08: \
    [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte':     matrix(vocab_size, n_embd),    # Token embedding table
    'wpe':     matrix(block_size, n_embd),    # Position embedding table
    'lm_head': matrix(vocab_size, n_embd),    # Output projection
}

for i in range(n_layer):
    # -- Grassmann layer parameters (replaces Attention!) --
    state_dict[f'L{i}.W_red']    = matrix(r, n_embd)           # Dimension reduction
    state_dict[f'L{i}.W_plu']    = matrix(n_embd, plucker_dim) # Plucker -> d
    state_dict[f'L{i}.W_gate_h'] = matrix(n_embd, n_embd)      # Gate (original)
    state_dict[f'L{i}.W_gate_g'] = matrix(n_embd, n_embd)      # Gate (geometry)
    # -- Feed-Forward Network (same as Transformer) --
    state_dict[f'L{i}.mlp_fc1']  = matrix(4 * n_embd, n_embd)  # d -> 4d
    state_dict[f'L{i}.mlp_fc2']  = matrix(n_embd, 4 * n_embd)  # 4d -> d

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"Total number of parameters: {len(params)}")


# ============================================================================
#  PART 5: Utility Functions
# ============================================================================

def linear(x, w):
    """
    Matrix-vector product: y = W * x
    Computes the dot product of each row of W with x.

    Example: If W is 3x2 and x is [x0, x1], then
        y = [w00*x0 + w01*x1,
             w10*x0 + w11*x1,
             w20*x0 + w21*x1]  -> 3-dimensional vector
    """
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]


def softmax(logits):
    """
    Softmax: Converts a real-valued vector into a probability distribution.
    All values become between 0 and 1, and they sum to 1.

    For numerical stability, we subtract the maximum value (the result is
    mathematically identical).
    softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
    """
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    """
    RMS Normalization (Root Mean Square Normalization):
    Normalizes the magnitude (scale) of a vector to a consistent level.

    Formula: x_norm = x / sqrt(mean(x^2) + epsilon)

    Why is this needed?
      During training, if vector values grow too large or small, gradients
      can explode or vanish. Normalization keeps training stable.

    Difference from LayerNorm:
      LayerNorm = subtract mean, divide by std (2 statistics)
      RMSNorm   = divide by RMS only (1 statistic) -> simpler and faster
    """
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def sigmoid(x):
    """
    Sigmoid function: Maps any real number to the (0, 1) range.

    Formula: sigma(x) = 1 / (1 + exp(-x))

    Graph shape:
      x = -inf -> approaches 0
      x = 0    -> 0.5
      x = +inf -> approaches 1

    Used in gates to determine "how much to let through".
    alpha = sigmoid(score)  ->  0 means "block", 1 means "fully pass through"
    """
    return Value(1.0) / (Value(1.0) + (-x).exp())


# ============================================================================
#  PART 6: Plucker Coordinate Computation — The Mathematical Core of This Paper!
# ============================================================================
#
#  [How does Attention represent the relationship between two tokens?]
#    Q * K^T = scalar (a single number)
#    "How related are these two tokens?" -> compressed into a single similarity score
#
#  [How does Grassmann represent the relationship between two tokens?]
#    Plucker(u, v) = a C(r,2)-dimensional vector
#    "What 2D plane do these two tokens form?" -> a rich geometric representation
#
#  ────────────────────────────────────────────────
#  Intuitive explanation of Plucker coordinates:
#  ────────────────────────────────────────────────
#
#  In 3D space, two vectors u, v define a plane.
#  Plucker coordinates are the mathematical way to represent this plane.
#
#  In r-dimensional space, the 2D subspace (plane) spanned by two vectors u, v
#  corresponds to a point on the Grassmann manifold Gr(2,r).
#
#  Plucker coordinates are the coordinate system for this point:
#    p_ij = u_i * v_j  -  u_j * v_i     (for all pairs i < j)
#
#  Mathematically, this is the coordinates of the "wedge product" u ^ v,
#  which can be interpreted as the "signed area" of the parallelogram formed
#  by the two vectors, projected onto each coordinate plane.
#
#  ────────────────────────────────────────────────
#  Why can this be better than the dot product?
#  ────────────────────────────────────────────────
#
#  Dot product: u . v = |u||v|cos(theta)
#    -> Only angle information remains (1D scalar)
#
#  Wedge product/Plucker: u ^ v
#    -> Preserves angle + direction + area information (C(r,2)-dimensional vector)
#    -> Represents the relationship between two vectors much more richly
#
#  ────────────────────────────────────────────────
#  Concrete computation example (r=4):
#  ────────────────────────────────────────────────
#
#  u = [u0, u1, u2, u3]
#  v = [v0, v1, v2, v3]
#
#  Plucker coordinates (all pairs where i < j):
#    p_01 = u0*v1 - u1*v0    <- area projected onto the (i=0, j=1) plane
#    p_02 = u0*v2 - u2*v0    <- (i=0, j=2)
#    p_03 = u0*v3 - u3*v0    <- (i=0, j=3)
#    p_12 = u1*v2 - u2*v1    <- (i=1, j=2)
#    p_13 = u1*v3 - u3*v1    <- (i=1, j=3)
#    p_23 = u2*v3 - u3*v2    <- (i=2, j=3)
#
#  -> A vector of 6 values (C(4,2) = 6)
# ============================================================================

def compute_plucker(u, v):
    """
    Computes the Plucker coordinates of two r-dimensional vectors u and v.

    Args:
        u: list of r Values (reduced vector of the current token)
        v: list of r Values (reduced vector of a previous token)

    Returns:
        list of C(r,2) Values (Plucker coordinates)
    """
    coords = []
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            # p_ij = u_i * v_j  -  u_j * v_i
            # This is each component of the wedge product
            coords.append(u[i] * v[j] - u[j] * v[i])
    return coords


# ============================================================================
#  PART 7: Model Forward Pass — Causal Grassmann Layer
# ============================================================================
#
#  The full process of predicting the next token given an input token.
#
#  ┌─────────────────────────────────────────────────┐
#  │  Input: token_id (current token), pos_id (position) │
#  │                                                  │
#  │  1. Embedding: token + position -> vector        │
#  │  2. RMS normalization                            │
#  │  3. *** Causal Grassmann Layer ***               │
#  │     a. Dimension reduction: x(16-dim) -> z(4-dim)│
#  │     b. Store z in cache                          │
#  │     c. Compute Plucker coords with prev tokens   │
#  │     d. Restore Plucker to model dimension        │
#  │     e. Average results across offsets             │
#  │     f. Gate-mix original and geometric info       │
#  │  4. Residual connection                          │
#  │  5. FFN (Feed-Forward Network)                   │
#  │  6. Output: vector -> vocabulary probabilities    │
#  │                                                  │
#  │  Output: logits (vocab_size dimensions)          │
#  └─────────────────────────────────────────────────┘
#
#  [What is the Causal constraint?]
#    Language models must "not be able to see the future".
#    When predicting the next character after "hel", the model must not peek at
#    'l' and 'o' from "hello".
#    Attention uses masking to enforce this.
#    Grassmann naturally guarantees this because window offsets are positive
#    (delta > 0).
#    -> delta=1 only references t-1, delta=2 only references t-2 (always looks
#       at the past)
# ============================================================================

def forward(token_id, pos_id, z_cache):
    """
    Forward pass for a single token.

    Args:
        token_id: ID of the current input token (integer)
        pos_id:   Current position index (starting from 0)
        z_cache:  Cache storing reduced vectors of previous tokens.
                  z_cache[layer_idx] = [z_0, z_1, ..., z_{t-1}]
                  Plays a similar role to the Key/Value cache in Attention.

    Returns:
        logits: list of vocab_size Values (next token prediction scores)
    """

    # -- Step 1: Embedding --
    # Convert token ID and position ID to vectors and add them together.
    # This produces a vector representing "the character 'h' at position 3".
    tok_emb = state_dict['wte'][token_id]    # Token embedding: vocab -> R^d
    pos_emb = state_dict['wpe'][pos_id]      # Position embedding: position -> R^d
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    # -- Step 2: Iterate through Grassmann layers --
    for li in range(n_layer):
        x_res = x      # Save current state for residual connection
        x = rmsnorm(x)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  (A) Linear Reduction (Dimension Reduction)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # x (16-dim) -> z (4-dim)
        #
        # Why reduce dimensions?
        #   Plucker coordinate dimension = C(r, 2) = r*(r-1)/2
        #   If r=16, C(16,2)=120... too large and computationally expensive
        #   If r=4, C(4,2)=6, which is manageable
        #
        # Comparison with Attention:
        #   Attention: x -> Q (via W_q), x -> K (via W_k) -> two projections
        #   Grassmann: x -> z (via W_red) -> a single projection is enough!
        #   (Since relationships are extracted via Plucker from a single z,
        #    instead of needing separate Q and K)
        z = linear(x, state_dict[f'L{li}.W_red'])

        # Add current position's z to cache (so future tokens can reference it)
        z_cache[li].append(z)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  (B) Multi-Scale Plucker Encoding (The Paper's Core Operation!)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Pairs the current token's z with previous tokens' z values
        # and computes Plucker coordinates for each pair.
        #
        # Example: At position t=5, window=[1,2,4]:
        #   delta=1: Plucker(z_5, z_4) -> "geometric relationship with the previous token"
        #   delta=2: Plucker(z_5, z_3) -> "geometric relationship with the token 2 steps back"
        #   delta=4: Plucker(z_5, z_1) -> "geometric relationship with the token 4 steps back"
        #
        # Comparison with Attention:
        #   Attention: Computes Q*K similarity with all previous tokens -> O(L) per token
        #   Grassmann: Computes only a fixed number of offsets             -> O(|window|) per token
        #   Since |window| is constant, the overall complexity is O(L) vs Attention's O(L^2)

        geo_features = []

        for delta in window:
            prev_pos = pos_id - delta
            if prev_pos < 0:
                # Skip if out of range (e.g., at position 0 with delta=1 -> position -1 doesn't exist)
                continue

            z_prev = z_cache[li][prev_pos]   # Reduced vector of the token delta steps back

            # --- Plucker coordinate computation ---
            plucker = compute_plucker(z, z_prev)

            # --- Normalization ---
            # Plucker coordinates are "projective coordinates" on the Grassmann manifold,
            # so scale (magnitude) doesn't matter — only direction matters.
            # Normalization also improves numerical stability.
            norm_sq = sum(p * p for p in plucker)
            norm_inv = (norm_sq + Value(1e-8)) ** -0.5
            plucker = [p * norm_inv for p in plucker]

            # --- Projection to model dimension ---
            # 6-dim Plucker coordinates -> restored to 16-dim model space
            # This projection transforms "geometric relationships" into a
            # representation the model can understand
            g_delta = linear(plucker, state_dict[f'L{li}.W_plu'])
            geo_features.append(g_delta)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  (C) Geometric Feature Aggregation
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Averages the geometric features from multiple offsets.
        #
        # Feature from delta=1: "adjacent context information"
        # Feature from delta=2: "slightly wider context information"
        # Feature from delta=4: "distant context information"
        # -> Their average = "information synthesizing context at various scales"
        #
        # Comparison with Attention:
        #   Attention: Softmax-weighted average (mixes V using attention weights)
        #   Grassmann: Simple average (equally mixes geometric info from each scale)

        if geo_features:
            nf = len(geo_features)
            g = [sum(geo_features[k][j] for k in range(nf)) / nf
                 for j in range(n_embd)]
        else:
            # At position 0, there are no previous tokens to reference, so use a zero vector.
            # The gate learns to pass through the original x in this case.
            g = [Value(0.0) for _ in range(n_embd)]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  (D) Gated Mixing
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # "How much should we mix original information (x) with geometric
        #  information (g)?"
        # A learnable gate makes this decision per dimension.
        #
        # alpha_j = sigmoid(W_gate_h * x + W_gate_g * g)_j
        #
        # If alpha_j is close to 1: dimension j retains original x
        #   -> "This dimension's information is sufficient without geometry"
        #
        # If alpha_j is close to 0: dimension j uses geometric g
        #   -> "This dimension needs relationship info with previous tokens"
        #
        # Final output: x_mixed = alpha * x + (1-alpha) * g
        #
        # Comparison with Attention:
        #   Attention: Single output (attention weighted sum)
        #   Grassmann: Gate mixes original and geometric info "per dimension"
        #              -> Enables finer-grained control

        gate_h = linear(x, state_dict[f'L{li}.W_gate_h'])
        gate_g = linear(g, state_dict[f'L{li}.W_gate_g'])
        alpha = [sigmoid(h_j + g_j) for h_j, g_j in zip(gate_h, gate_g)]

        x = [a * xi + (Value(1.0) - a) * gi
             for a, xi, gi in zip(alpha, x, g)]

        # Residual Connection
        # Add back the original input -> helps gradients flow through deep layers
        x = [a + b for a, b in zip(x, x_res)]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  (E) Feed-Forward Network (Same as Transformer)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FFN is a stage that independently transforms each token.
        # Structure: d -> 4d (expand) -> ReLU (nonlinearity) -> 4d -> d (contract)
        #
        # If the Grassmann layer handled "inter-token relationships",
        # FFN refines the "internal representation of each token".
        # (This part is completely identical to the original Transformer)
        x_res = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'L{li}.mlp_fc1'])    # d -> 4d expansion
        x = [xi.relu() for xi in x]                      # Nonlinear activation
        x = linear(x, state_dict[f'L{li}.mlp_fc2'])    # 4d -> d contraction
        x = [a + b for a, b in zip(x, x_res)]            # Residual connection

    # -- Step 3: Output Projection --
    # Transforms the 16-dim hidden vector to vocab_size dimensions.
    # Applying softmax to these values (logits) gives "the probability of each
    # character being the next token".
    return linear(x, state_dict['lm_head'])


# ============================================================================
#  PART 8: Training Loop
# ============================================================================
#
#  [Training Objective]
#    "Accurately predict the next character given the previous characters"
#
#  Example: Training on the name "emma"
#    Input BOS -> Target 'e'  (predict the first character of the name)
#    Input 'e' -> Target 'm'  (predict the second character)
#    Input 'm' -> Target 'm'  (predict the third character)
#    Input 'm' -> Target 'a'  (predict the fourth character)
#    Input 'a' -> Target BOS  (predict the end of the name)
#
#  [Training Steps]
#    1. Tokenize one name
#    2. Forward pass at each position -> predict next token
#    3. Compute cross-entropy loss (how far off is the prediction from the target)
#    4. backward() -> compute gradients for all parameters
#    5. Update parameters with Adam optimizer
#
#  [Adam Optimizer]
#    A faster and more stable optimization algorithm than plain SGD:
#    - m (1st moment): moving average of gradients -> smooths the "direction"
#    - v (2nd moment): moving average of gradient^2 -> auto-adjusts the "scale"
#    - Learning rate warmup/decay: fast at the beginning, slow at the end
# ============================================================================

lr, beta1, beta2, eps = 0.01, 0.85, 0.99, 1e-8
m_buf = [0.0] * len(params)    # Adam 1st moment
v_buf = [0.0] * len(params)    # Adam 2nd moment
num_steps = 1000

print(f"\n{'=' * 60}")
print(f"  microGrassmann training started")
print(f"  Sequence modeling with Grassmann geometry, no Attention!")
print(f"  Plucker coordinates replace the Q*K dot product.")
print(f"{'=' * 60}\n")

for step in range(num_steps):
    # --- Data preparation ---
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # z_cache: "reduced vector cache" corresponding to Attention's KV-cache
    z_cache = [[] for _ in range(n_layer)]
    losses = []

    # --- Forward pass: predict next token at each position ---
    for pos_id in range(n):
        token_id  = tokens[pos_id]
        target_id = tokens[pos_id + 1]

        logits = forward(token_id, pos_id, z_cache)
        probs  = softmax(logits)

        # Cross-entropy loss: -log(probability of the target token)
        # If probability is close to 1, loss ~ 0 (good prediction)
        # If probability is close to 0, loss -> large value (bad prediction)
        losses.append(-probs[target_id].log())

    loss = (1 / n) * sum(losses)    # Average loss
    loss.backward()                  # Backpropagation -> compute gradients

    # --- Adam optimizer update ---
    lr_t = lr * (1 - step / num_steps)    # Linear learning rate decay
    for i, p in enumerate(params):
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_buf[i] / (1 - beta1 ** (step + 1))   # Bias correction
        v_hat = v_buf[i] / (1 - beta2 ** (step + 1))   # Bias correction
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps)
        p.grad = 0    # Reset gradient (for the next step)

    if step % 100 == 0 or step == num_steps - 1:
        print(f"  step {step + 1:4d}/{num_steps} | loss {loss.data:.4f}")


# ============================================================================
#  PART 9: Inference — Generating New Names with the Trained Model
# ============================================================================
#
#  Now that training is complete, the model generates "new names it has never
#  seen before".
#
#  [Generation Process (Autoregressive Sampling)]
#    1. Start with BOS token
#    2. Model outputs a probability distribution over next tokens
#    3. Sample one token according to the probabilities
#    4. Feed the sampled token as input and repeat from step 2
#    5. Stop when BOS (= end) token is generated
#
#  [Temperature]
#    Low temperature (< 1): Favors high-probability tokens -> safe names
#    High temperature (> 1): Probabilities become more uniform -> creative (unusual) names
#    temperature = 0.5: Generates somewhat conservatively
#
#  Key point: All this generation happens without Attention, purely using
#  Grassmann geometry!
# ============================================================================

temperature = 0.5

print(f"\n{'=' * 60}")
print(f"  Inference: New names generated by the Grassmann model")
print(f"  (Without Attention, using only geometric flow of Plucker coordinates)")
print(f"{'=' * 60}\n")

for idx in range(20):
    z_cache = [[] for _ in range(n_layer)]
    token_id = BOS
    chars = []

    for pos_id in range(block_size):
        logits = forward(token_id, pos_id, z_cache)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(
            range(vocab_size), weights=[p.data for p in probs]
        )[0]

        if token_id == BOS:
            break
        chars.append(uchars[token_id])

    print(f"  sample {idx + 1:2d}: {''.join(chars)}")

print(f"\n  Done! Generated names using only Grassmann geometry, without Attention.")
print(f"  Paper: https://arxiv.org/abs/2512.19428")
print(f"  Reference: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95")
