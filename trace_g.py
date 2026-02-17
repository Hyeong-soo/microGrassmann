"""
trace_g.py — Numerically tracing how the geometry vector g is constructed
================================================================
The answer to "What is g?":
  g = A vector summarizing the "geometric relationships" with previous tokens

This file traces, number by number, how 'o' at position 4 in "hello"
forms geometric relationships with previous characters.

Run: python3 trace_g.py
================================================================
"""

import random
import math
random.seed(42)

def section_header(title):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}\n")

def fmt(vec, n=3):
    """Format a vector for display"""
    return "[" + ", ".join(f"{v:+.{n}f}" for v in vec) + "]"

def mat_vec(W, x):
    """Matrix-vector multiplication"""
    return [sum(wi*xi for wi, xi in zip(row, x)) for row in W]


# ================================================================
section_header("Setup: Processing 'hello', current position is 'o' (position 4)")
# ================================================================

print("""
  Sequence: h  e  l  l  o
  Position: 0  1  2  3  4 <- current

  window = [1, 2, 4] so 'o' references the following previous tokens:
    delta=1 -> position 3 'l'  (immediately preceding)
    delta=2 -> position 2 'l'  (two steps back)
    delta=4 -> position 0 'h'  (four steps back)

  Goal: From these 3 relationships, construct the geometry vector g
        to enrich the representation of the current 'o'
""")

# Embedding dimension d=6 (small for understanding), reduced dimension r=3
d = 6
r = 3
plucker_dim = r * (r - 1) // 2  # C(3,2) = 3

print(f"  Embedding dim d={d}, reduced dim r={r}, Plucker dim={plucker_dim}")


# ================================================================
section_header("STEP 0: Embedding vectors (x) for each character")
# ================================================================

print("""
  In a real model these are determined by training,
  but here we use fixed example vectors.
  Each vector has d=6 dimensions.
""")

embeddings = {
    'h': [ 0.5, -0.3,  0.8,  0.2, -0.1,  0.4],
    'e': [-0.2,  0.7, -0.1,  0.5,  0.3, -0.6],
    'l': [ 0.3,  0.4,  0.6, -0.2,  0.5,  0.1],  # position 2
    'l2':[ 0.3,  0.4,  0.6, -0.2,  0.5,  0.1],  # position 3 (same character)
    'o': [ 0.1, -0.5,  0.3,  0.9, -0.4,  0.7],
}

for ch, vec in embeddings.items():
    name = ch if ch != 'l2' else 'l'
    pos = {'h':0, 'e':1, 'l':2, 'l2':3, 'o':4}[ch]
    marker = " <- current" if ch == 'o' else ""
    print(f"  position {pos} '{name}':  x = {fmt(vec)}{marker}")


# ================================================================
section_header("STEP 1: Dimensionality reduction — x(6D) -> z(3D)")
# ================================================================

print("""
  We compress each token's vector using W_red (3x6 matrix).
  z = W_red @ x

  Why reduce?
    In Plucker computation we need to examine all i<j pairs,
    6 dimensions -> C(6,2)=15 pairs -> too many
    3 dimensions -> C(3,2)=3 pairs -> manageable
""")

W_red = [
    [ 0.4, -0.2,  0.3,  0.1, -0.5,  0.2],
    [ 0.1,  0.5, -0.3,  0.4,  0.2, -0.1],
    [-0.3,  0.1,  0.6, -0.2,  0.3,  0.4],
]

z_all = {}
for ch in ['h', 'e', 'l', 'l2', 'o']:
    z = mat_vec(W_red, embeddings[ch])
    z_all[ch] = z
    name = ch if ch != 'l2' else 'l'
    pos = {'h':0, 'e':1, 'l':2, 'l2':3, 'o':4}[ch]
    print(f"  position {pos} '{name}':  x{fmt(embeddings[ch][:3])}... -> z={fmt(z)}")

z_o = z_all['o']
print(f"\n  Current 'o' reduced vector: z_o = {fmt(z_o)}")
print(f"  We will compare this z_o with z vectors of previous tokens.")


# ================================================================
section_header("STEP 2: Plucker coordinate computation — the 'relationship' with each previous token")
# ================================================================

print("""
  Now the key part!
  We compute Plucker coordinates between z_o (current 'o')
  and the z of each previous token.

  Since r=3, Plucker has 3 components:
    p_01 = z_o[0]*z_prev[1] - z_o[1]*z_prev[0]  (xy shadow)
    p_02 = z_o[0]*z_prev[2] - z_o[2]*z_prev[0]  (xz shadow)
    p_12 = z_o[1]*z_prev[2] - z_o[2]*z_prev[1]  (yz shadow)

  Each Plucker vector = "What plane do the current token
  and the previous token span in 3D space?"
""")

window = [1, 2, 4]
prev_map = {1: ('l2', 3, 'l'), 2: ('l', 2, 'l'), 4: ('h', 0, 'h')}

plucker_results = {}

for delta in window:
    ch_key, pos, ch_name = prev_map[delta]
    z_prev = z_all[ch_key]

    print(f"  --- delta={delta}: 'o'(pos 4) and '{ch_name}'(pos {pos}) ---\n")
    print(f"    z_o    = {fmt(z_o)}")
    print(f"    z_prev = {fmt(z_prev)}")

    p01 = z_o[0]*z_prev[1] - z_o[1]*z_prev[0]
    p02 = z_o[0]*z_prev[2] - z_o[2]*z_prev[0]
    p12 = z_o[1]*z_prev[2] - z_o[2]*z_prev[1]
    plucker = [p01, p02, p12]

    print(f"\n    Plucker computation:")
    print(f"      p_01 = {z_o[0]:+.3f}*{z_prev[1]:+.3f} - {z_o[1]:+.3f}*{z_prev[0]:+.3f} = {p01:+.4f}  (xy shadow)")
    print(f"      p_02 = {z_o[0]:+.3f}*{z_prev[2]:+.3f} - {z_o[2]:+.3f}*{z_prev[0]:+.3f} = {p02:+.4f}  (xz shadow)")
    print(f"      p_12 = {z_o[1]:+.3f}*{z_prev[2]:+.3f} - {z_o[2]:+.3f}*{z_prev[1]:+.3f} = {p12:+.4f}  (yz shadow)")
    print(f"\n    Plucker = {fmt(plucker, 4)}")

    # Normalization
    norm = math.sqrt(sum(p*p for p in plucker) + 1e-8)
    plucker_normed = [p / norm for p in plucker]
    print(f"    Normalize (to unit length): {fmt(plucker_normed, 4)}")
    print(f"    -> Keep only 'direction', discard 'magnitude' (only which plane matters)")

    plucker_results[delta] = plucker_normed
    print()


# ================================================================
section_header("STEP 3: Plucker -> restore to model dimension (g_delta)")
# ================================================================

print("""
  Plucker coordinates are 3D, but the model vectors are 6D.
  We use W_plu (6x3 matrix) to transform 3D Plucker to 6D.

  g_delta = W_plu @ plucker_normalized

  What this does:
    Translates the 3D information "these two tokens have this
    geometric relationship" into a 6D representation the model
    can understand.

  Analogy: a translator from French (3D) to English (6D)
""")

W_plu = [
    [ 0.3, -0.4,  0.2],
    [ 0.5,  0.1, -0.3],
    [-0.2,  0.6,  0.4],
    [ 0.1, -0.3,  0.5],
    [ 0.4,  0.2, -0.1],
    [-0.1,  0.3,  0.6],
]

g_deltas = {}

for delta in window:
    plucker_n = plucker_results[delta]
    g_delta = mat_vec(W_plu, plucker_n)
    g_deltas[delta] = g_delta

    ch_key, pos, ch_name = prev_map[delta]
    print(f"  delta={delta} (relationship with '{ch_name}'):")
    print(f"    Plucker(3D) {fmt(plucker_n, 3)}")
    print(f"    -> transform via W_plu ->")
    print(f"    g_delta(6D) {fmt(g_delta, 3)}")
    print(f"    This = \"the geometric relationship with '{ch_name}' expressed in 6D\"")
    print()


# ================================================================
section_header("STEP 4: Average of g_deltas -> final geometry vector g")
# ================================================================

print("""
  We computed the relationship with each of the 3 previous tokens:
    g_delta1 = relationship with 'l'(pos 3)  (immediately preceding, closest)
    g_delta2 = relationship with 'l'(pos 2)  (two steps back)
    g_delta4 = relationship with 'h'(pos 0)  (four steps back, farthest context)

  We average these 3 into a single vector g:
    g = (g_delta1 + g_delta2 + g_delta4) / 3
""")

print("  Contribution from each delta:")
for delta in window:
    ch_key, pos, ch_name = prev_map[delta]
    print(f"    delta={delta} ('{ch_name}'): {fmt(g_deltas[delta], 3)}")

# Compute average
g = [0.0] * d
for delta in window:
    for j in range(d):
        g[j] += g_deltas[delta][j]
for j in range(d):
    g[j] /= len(window)

print(f"\n  Average ->  g = {fmt(g, 4)}")
print(f"""
  +----------------------------------------------------------+
  |                                                          |
  |  g = "A summary of the geometric relationships          |
  |       that the current 'o' has with surrounding tokens"  |
  |                                                          |
  |  Each dimension of g contains a mixture of:              |
  |    - Info from nearby context ('l', delta=1)             |
  |    - Info from slightly wider context ('l', delta=2)     |
  |    - Info from distant context ('h', delta=4)            |
  |                                                          |
  +----------------------------------------------------------+
""")


# ================================================================
section_header("STEP 5: x vs g — What does each one know?")
# ================================================================

x_o = embeddings['o']

print(f"  x (original vector):  {fmt(x_o, 3)}")
print(f"  g (geometry vector):  {fmt(g, 3)}")
print()
print("""
  What x knows:
    "I am 'o'. Here are my intrinsic characteristics."
    -> Information about the token itself (independent, context-free)

  What g knows:
    "Before me there were 'l', 'l', 'h',
     and my ('o') geometric relationships with them are like this."
    -> Relationship info with surrounding tokens (context-dependent)

  Analogy:
    x = a resume (my own qualifications)
    g = a recommendation letter (how people around me see me)

    You need both to fully represent "me"!
""")


# ================================================================
section_header("STEP 6: Gated mixing — blending x and g dimension by dimension")
# ================================================================

print("""
  alpha = sigmoid(W_gate_h @ x + W_gate_g @ g)
  output = alpha * x + (1 - alpha) * g

  alpha decides, for each dimension, "how much of x vs g to use."
""")

# Simple gate weights (example)
alpha_example = [0.8, 0.3, 0.6, 0.2, 0.9, 0.5]

print(f"  x (original):  {fmt(x_o, 3)}")
print(f"  g (geometry):  {fmt(g, 3)}")
print(f"  alpha:         {fmt(alpha_example, 1)}")
print()

output = [0.0] * d
print(f"  {'dim':>4}  {'alpha':>6}  {'alpha*x':>9}  {'(1-a)*g':>9}  {'= output':>9}  verdict")
print(f"  {'─'*64}")

for j in range(d):
    a = alpha_example[j]
    ax = a * x_o[j]
    ag = (1 - a) * g[j]
    out = ax + ag
    output[j] = out
    verdict = "mostly original x" if a >= 0.5 else "mostly geometry g"
    print(f"  [{j}]   {a:.1f}    {ax:+.4f}    {ag:+.4f}    {out:+.4f}   {verdict}")

print(f"\n  Final output = {fmt(output, 4)}")
print(f"""
  Interpretation:
    dim 0: alpha=0.8 -> 80% original + 20% geometry (this dim has enough 'o' self-info)
    dim 1: alpha=0.3 -> 30% original + 70% geometry (this dim needs surrounding relationships)
    dim 3: alpha=0.2 -> 20% original + 80% geometry (this dim heavily relies on context)
    dim 4: alpha=0.9 -> 90% original + 10% geometry (this dim is dominated by 'o' intrinsic features)
""")


# ================================================================
section_header("STEP 7: Full comparison with Attention")
# ================================================================

print(f"""
  Same situation: how 'o' at position 4 in "hello" gathers context

  +--- Attention's method --------------------------------------+
  |                                                             |
  |  Create Query Q from 'o',                                   |
  |  create Key K from each of 'h','e','l','l',                 |
  |  compute similarity scores via Q.K dot product:             |
  |                                                             |
  |    score with 'h': 0.12  -> weight 0.14                     |
  |    score with 'e': 0.08  -> weight 0.12                     |
  |    score with 'l': 0.35  -> weight 0.31                     |
  |    score with 'l': 0.42  -> weight 0.34  <- highest         |
  |                                   (sum = 1.0)               |
  |                                                             |
  |  -> Weighted average of all 4 tokens' Values                |
  |  -> 4 dot product computations (with every previous token)  |
  |  -> Each relationship = 1 scalar                            |
  +-------------------------------------------------------------+

  +--- Grassmann's method --------------------------------------+
  |                                                             |
  |  Only look at 3 nearby tokens (delta=1,2,4):                |
  |                                                             |
  |    'l'(pos 3) Plucker -> {fmt(plucker_results[1], 3):>22}   |
  |    'l'(pos 2) Plucker -> {fmt(plucker_results[2], 3):>22}   |
  |    'h'(pos 0) Plucker -> {fmt(plucker_results[4], 3):>22}   |
  |                                                             |
  |  Transform each to 6D via W_plu, then average -> g          |
  |  Gate to mix x and g dimension by dimension                 |
  |                                                             |
  |  -> 3 Plucker computations (fixed window only)              |
  |  -> Each relationship = a vector (not 1 scalar but 3!)      |
  +-------------------------------------------------------------+

  Key difference:
    Attention:  less info (scalar) x many tokens (all) = broad and shallow
    Grassmann:  rich info (vector) x few tokens (3)    = narrow and deep
""")


# ================================================================
section_header("Summary: What is g?")
# ================================================================

print("""
  g is "the story that surrounding tokens tell about me."

  How it is constructed:

    x(6D) --W_red--> z(3D)          dimensionality reduction
                       |
                       +-- Plucker with z_prev(delta=1) -> [3D relation] --W_plu--> g1(6D)
                       +-- Plucker with z_prev(delta=2) -> [3D relation] --W_plu--> g2(6D)
                       +-- Plucker with z_prev(delta=4) -> [3D relation] --W_plu--> g3(6D)
                                                                        |
                                                           average <----+
                                                            |
                                                            v
                                                       g(6D): geometry vector
                                                            |
                                    x --> gate(alpha) <-- g |
                                           |                |
                                           v                |
                                     output = alpha*x + (1-alpha)*g


  What each dimension of g contains:
    "nearby context (delta=1) relationship"   -+
    "medium context (delta=2) relationship"   -+-> averaged and blended
    "distant context (delta=4) relationship"  -+

  Without g: predict with x alone -> "I am 'o'" (ignoring context)
  With g:    predict with x+g     -> "I am the 'o' that comes after 'h-e-l-l'" (reflecting context)
""")
