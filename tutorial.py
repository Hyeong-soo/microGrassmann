"""
tutorial.py -- Understanding Attention and Grassmann from Scratch
=========================================================
This file is a tutorial meant to be read while running.
At each step, actual numbers are printed so you can see "what's happening" firsthand.

Usage:
  python3 tutorial.py

Prerequisites: Basic Python knowledge is all you need.
=========================================================
"""

import math
import random
random.seed(42)

def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

# ================================================================
separator("STEP 1: The Problem We're Trying to Solve")
# ================================================================

print("""
  Our goal: A "predict the next character" game

  Example: Let's say we're learning the name "emma".

    [start] -> next character? -> 'e'   (first letter of the name)
    'e'     -> next character? -> 'm'
    'e','m' -> next character? -> 'm'
    'e','m','m' -> next character? -> 'a'
    'e','m','m','a' -> next character? -> [end]

  Key question:
    "How do we predict the next character given the previous characters?"

  The way we use information from previous characters is exactly
  the difference between Attention and Grassmann.
""")


# ================================================================
separator("STEP 2: Converting Characters to Numbers (Embedding)")
# ================================================================

print("""
  Computers don't understand characters. They only understand numbers.
  So we need to convert each character into a "numeric vector".

  What's a vector? Just a list of numbers.
    'h' -> [0.3, -0.1, 0.7]    (3-dimensional vector)
    'e' -> [0.5,  0.2, -0.4]
    'l' -> [-0.2, 0.8, 0.1]

  These numbers are random at first,
  but through training, "similar characters get similar vectors".
""")

# Actual example
print("  Actual example (4-dimensional vectors):")
embeds = {
    'h': [0.3, -0.1, 0.7, 0.2],
    'e': [0.5,  0.2, -0.4, 0.6],
    'l': [-0.2, 0.8, 0.1, -0.3],
    'o': [0.1, -0.5, 0.3, 0.9],
}
for ch, vec in embeds.items():
    print(f"    '{ch}' -> {vec}")


# ================================================================
separator("STEP 3: The Core Problem -- How Do We Use Previous Character Information?")
# ================================================================

print("""
  Let's say we're predicting the next character after 'l' in "hello".
  The current character is 'l', and the previous characters were 'h', 'e', 'l'.

  Simplest approach: predict using only the current character 'l'
    -> Problem: what comes after 'l' depends on the preceding context!
       In "hello", the character after the first 'l' is 'l'
       In "help", the character after 'l' is 'p'.
    -> We need information from previous characters.

  So how do we combine previous character information with the current one?
  This is where three approaches diverge:

  Approach 1: Simple average  (too simple)
  Approach 2: Attention       (the core of Transformers)
  Approach 3: Grassmann       (proposed by this paper)
""")


# ================================================================
separator("STEP 4: Approach 1 -- Simple Average (Baseline)")
# ================================================================

print("""
  The simplest idea:
  "Just average the vectors of the previous characters and combine with the current one"
""")

h_vec = embeds['h']
e_vec = embeds['e']
l_vec = embeds['l']

avg = [sum(vs)/3 for vs in zip(h_vec, e_vec, l_vec)]
print(f"  'h' vector: {h_vec}")
print(f"  'e' vector: {e_vec}")
print(f"  'l' vector: {l_vec}")
print(f"  Simple average: {[round(x, 3) for x in avg]}")

print("""
  Problem:
    It treats all previous characters "equally".
    But in practice, some characters may be more important.
    Example: when predicting after 'l' in "hello", the immediately
        preceding 'l' may be more important than the distant 'h'.
""")


# ================================================================
separator("STEP 5: Approach 2 -- Attention (Weighted Average)")
# ================================================================

print("""
  The core idea of Attention:
  "Average the previous characters, but give higher weight to the important ones"

  How?
  1) The current character creates a signal: "I need this kind of info" -> Query (Q)
  2) Each previous character creates a signal: "I have this kind of info" -> Key (K)
  3) Compare Q and K (dot product) to compute a "relevance score"
  4) Take more Value (V) from characters with higher scores
""")

# -- Let's show this concretely --

print("  Concrete computation (current char: 'l', previous chars: 'h', 'e', 'l'):\n")

# Simple "projection" (in practice it's a matrix multiply, but here we keep it intuitive)
# Query: current character 'l' saying "I need this kind of info"
q = l_vec   # In practice: W_q @ l_vec
print(f"  1) Query (info that current 'l' needs): {q}")

# Key: each previous character saying "I have this kind of info"
keys = {'h': h_vec, 'e': e_vec, 'l': l_vec}  # In practice: W_k @ vec
print(f"  2) Keys (info each character has):")
for ch, k in keys.items():
    print(f"     '{ch}': {k}")

# Dot product: similarity scores between Q and K
print(f"\n  3) Similarity scores (dot product of Q with each K):")
scores = {}
for ch, k in keys.items():
    # Dot product = multiply corresponding elements and sum
    dot = sum(qi * ki for qi, ki in zip(q, k))
    scores[ch] = dot
    detail = " + ".join(f"{qi:.1f}*{ki:.1f}" for qi, ki in zip(q, k))
    print(f"     Q . K_{ch} = {detail} = {dot:.3f}")

print(f"\n     Results: {', '.join(f'{ch}:{s:.3f}' for ch, s in scores.items())}")

# Convert to weights with softmax
print(f"\n  4) Softmax (convert scores to 0~1 weights, sum=1):")
max_s = max(scores.values())
exp_scores = {ch: math.exp(s - max_s) for ch, s in scores.items()}
total = sum(exp_scores.values())
weights = {ch: e / total for ch, e in exp_scores.items()}
for ch, w in weights.items():
    print(f"     '{ch}': {w:.3f}  {'<-- highest!' if w == max(weights.values()) else ''}")

# Weighted average
print(f"\n  5) Weighted average (higher weight on important characters):")
weighted = [0.0] * 4
for ch, w in weights.items():
    vec = embeds[ch]
    for j in range(4):
        weighted[j] += w * vec[j]

print(f"     = {weights['h']:.3f}*{h_vec} (h)")
print(f"     + {weights['e']:.3f}*{e_vec} (e)")
print(f"     + {weights['l']:.3f}*{l_vec} (l)")
print(f"     = {[round(x, 3) for x in weighted]}")

print(f"\n  Comparison:")
print(f"     Simple average: {[round(x, 3) for x in avg]}")
print(f"     Attention:      {[round(x, 3) for x in weighted]}")
print(f"     -> Attention gives more weight to more relevant characters!")

print("""
  Summary:
  +----------------------------------------------+
  |  Attention = "Who should I ask, and how much?"|
  |                                                |
  |  Q (Query)  = "I'm curious about this"         |
  |  K (Key)    = "I know about this"               |
  |  V (Value)  = "Here's my information"            |
  |  Score      = Q . K (how relevant?)              |
  |  Weight     = softmax(Score) (convert to prob)   |
  |  Output     = Weight x V (weighted average)      |
  +----------------------------------------------+
""")


# ================================================================
separator("STEP 6: The Problem with Attention")
# ================================================================

print("""
  Attention works well, but it has a critical drawback:

  If the sequence length is L, we must compute scores for "all pairs".

    L=4  (hello):  4 x 4 = 16 computations    -> OK
    L=100:         100 x 100 = 10,000          -> manageable
    L=1000:        1,000,000                   -> getting slow
    L=100,000:     10,000,000,000              -> practically impossible

  This is the "O(L^2) complexity" problem.
  When the sequence is 2x longer, computation is 4x!

  Visualization (Attention matrix for L=8):

    current v  prev ->   t0  t1  t2  t3  t4  t5  t6  t7
    t0                  [.8  .   .   .   .   .   .   .]
    t1                  [.3  .7  .   .   .   .   .   .]
    t2                  [.1  .2  .7  .   .   .   .   .]
    t3                  [.1  .1  .3  .5  .   .   .   .]
    t4                  [.0  .1  .1  .3  .5  .   .   .]
    t5                  [.0  .0  .1  .2  .3  .4  .   .]
    t6                  [.0  .0  .0  .1  .2  .3  .4  .]
    t7                  [.0  .0  .0  .1  .1  .2  .3  .3]

    -> We must compute this entire matrix (L x L entries)!
""")


# ================================================================
separator("STEP 7: Approach 3 -- Grassmann (The Core of This Paper)")
# ================================================================

print("""
  The Grassmann idea:
  "We don't need to look at all previous characters!
   Look at only a few nearby ones, but extract richer information."

  Specifically:
  1) Attention: "similarity scores" with all previous tokens (1 scalar each)
  2) Grassmann: "geometric relationships" with nearby tokens (multiple vectors)

  An analogy:
    Attention  = asking all friends "how close are we?" on a 1-10 scale
    Grassmann  = having deep conversations with your 3 closest friends for rich info
""")

print("  Grassmann's 3 steps:\n")
print("  +------------------------------------------------------+")
print("  |  Step A: Dimensionality reduction                     |")
print("  |    Compress character vectors to smaller vectors       |")
print("  |    [0.3, -0.1, 0.7, 0.2] -> [0.5, -0.3]             |")
print("  |                                                       |")
print("  |  Step B: Plucker coordinates (geometric relationship) |")
print("  |    Express the 'relationship' between current and     |")
print("  |    previous characters as a vector                    |")
print("  |    Much richer info than a dot product (1 scalar)!    |")
print("  |                                                       |")
print("  |  Step C: Gated mixing                                 |")
print("  |    Blend original info and geometric info at a        |")
print("  |    learned ratio                                      |")
print("  +------------------------------------------------------+")


# ================================================================
separator("STEP 8: Grassmann Step A -- Dimensionality Reduction")
# ================================================================

print("""
  Why reduce?
  In the next step (Plucker), we compute all i<j pairs,
  and the number of pairs explodes with dimension:
    4 dims  -> C(4,2) = 6 pairs
    16 dims -> C(16,2) = 120 pairs
    64 dims -> C(64,2) = 2016 pairs!
  So we first reduce to a smaller dimension.
""")

# 4-dim -> 2-dim reduction example
W_red = [[0.5, -0.3, 0.2, 0.1],
         [0.1,  0.4, -0.2, 0.3]]

def mat_vec(W, x):
    return [sum(wi*xi for wi, xi in zip(row, x)) for row in W]

print("  Example: 4-dim -> 2-dim reduction\n")
for ch in ['h', 'e', 'l']:
    vec = embeds[ch]
    z = mat_vec(W_red, vec)
    detail = ", ".join(f"{v:.2f}" for v in z)
    print(f"    '{ch}' {vec} -> z=[{detail}]")


# ================================================================
separator("STEP 9: Grassmann Step B -- Plucker Coordinates (The Key!)")
# ================================================================

print("""
  Attention represents the relationship between two vectors via the "dot product".
  Dot product = a single number (scalar)

  Grassmann represents the relationship between two vectors via "Plucker coordinates".
  Plucker = multiple numbers (vector)

  What are Plucker coordinates?
  They are a mathematical representation of the "plane" spanned by two vectors.
""")

# 2D example (easy to understand)
print("  --- 2D example (simplest case) ---\n")

u = [3.0, 1.0]   # reduced vector of current character
v = [1.0, 2.0]   # reduced vector of previous character

print(f"    Current character's z: u = {u}")
print(f"    Previous character's z: v = {v}")

# Dot product (Attention approach)
dot = u[0]*v[0] + u[1]*v[1]
print(f"\n    [Attention approach] Dot product:")
print(f"      u . v = {u[0]}*{v[0]} + {u[1]}*{v[1]} = {dot}")
print(f"      -> Single number: {dot}")
print(f"      -> Meaning: 'How much do these two vectors point in the same direction?'")

# Plucker (Grassmann approach)
p01 = u[0]*v[1] - u[1]*v[0]
print(f"\n    [Grassmann approach] Plucker coordinates:")
print(f"      p = u[0]*v[1] - u[1]*v[0]")
print(f"        = {u[0]}*{v[1]} - {u[1]}*{v[0]}")
print(f"        = {p01}")
print(f"      -> Single number: {p01} (in 2D there's only 1 pair)")
print(f"      -> Meaning: 'Area of the parallelogram formed by these two vectors'")
print(f"              Positive = counterclockwise, Negative = clockwise")

# 3D example (Plucker's advantage becomes clear)
print("\n\n  --- 3D example (where Plucker's advantage shows) ---\n")

u3 = [3.0, 1.0, 2.0]
v3 = [1.0, 2.0, -1.0]
print(f"    u = {u3}")
print(f"    v = {v3}")

dot3 = sum(a*b for a, b in zip(u3, v3))
print(f"\n    [Attention] Dot product = {dot3}  (single number)")

p01 = u3[0]*v3[1] - u3[1]*v3[0]
p02 = u3[0]*v3[2] - u3[2]*v3[0]
p12 = u3[1]*v3[2] - u3[2]*v3[1]
print(f"\n    [Grassmann] Plucker coordinates:")
print(f"      p_01 = u[0]*v[1] - u[1]*v[0] = {u3[0]}*{v3[1]} - {u3[1]}*{v3[0]} = {p01}")
print(f"      p_02 = u[0]*v[2] - u[2]*v[0] = {u3[0]}*{v3[2]} - {u3[2]}*{v3[0]} = {p02}")
print(f"      p_12 = u[1]*v[2] - u[2]*v[1] = {u3[1]}*{v3[2]} - {u3[2]}*{v3[1]} = {p12}")
print(f"      -> Vector [{p01}, {p02}, {p12}]  (3 numbers!)")

print(f"""
    Comparison:
      Dot product ->  {dot3}            1 number (directional similarity only)
      Plucker     -> [{p01}, {p02}, {p12}]  3 numbers (rich representation of the relationship)

    The dot product only tells "how similar are they?"
    Plucker gives complete information about "what plane do they span?"
    -> Two pairs can have the same dot product but different Plucker coordinates!
""")


# ================================================================
separator("STEP 10: Same Dot Product, Different Plucker -- Why Plucker Is Richer")
# ================================================================

print("""
  An example where two pairs of vectors have the same dot product but different Plucker:
""")

pair_a = ([1.0, 2.0, 0.0], [2.0, 1.0, 0.0])
pair_b = ([1.0, 0.0, 2.0], [2.0, 0.0, 1.0])

for name, (a, b) in [("Pair A", pair_a), ("Pair B", pair_b)]:
    dot = sum(x*y for x, y in zip(a, b))
    p01 = a[0]*b[1] - a[1]*b[0]
    p02 = a[0]*b[2] - a[2]*b[0]
    p12 = a[1]*b[2] - a[2]*b[1]
    print(f"  {name}: u={a}, v={b}")
    print(f"    Dot product = {dot}")
    print(f"    Plucker = [{p01}, {p02}, {p12}]")
    print()

print("""  The dot product is the same for both: 4.0!
  But the Plucker coordinates are completely different.

  Pair A: relationship in the xy plane -> [-3, 0, 0]
  Pair B: relationship in the xz plane -> [0, -3, 0]

  -> Attention cannot distinguish these two situations,
    but Grassmann can!
""")


# ================================================================
separator("STEP 11: Window -- Why Not Look at All Tokens?")
# ================================================================

print("""
  Attention:  the current token looks at "all" previous tokens
  Grassmann:  the current token looks at only "a few nearby" ones (window)

  Our setting: window = [1, 2, 4]

  Token at position 7 references:
    delta=1 -> position 6 (immediately before)
    delta=2 -> position 5 (two steps back)
    delta=4 -> position 3 (four steps back)

    0   1   2   3   4   5   6   [7]
                *       *   *   current
                |       |   |
              delta=4  d=2 d=1

  Why is this still okay?
    1) In language, the most important information is usually nearby
    2) Stacking layers allows distant information to propagate indirectly
       (Layer 1 reaches 4 positions back via delta=4 -> Layer 2 another 4 back -> 8 positions total)
    3) Because Plucker is richer than dot product, fewer references suffice

  Complexity comparison:
    Attention:  all pairs    -> O(L^2)  -> explodes as L grows
    Grassmann:  fixed 3 refs -> O(3*L)  -> proportional to L (linear!)
""")


# ================================================================
separator("STEP 12: Gate -- Mixing Original and Geometric Information")
# ================================================================

print("""
  With Grassmann, we obtained the "geometric relationship with previous tokens (g)".
  We also have the current token's "original vector (x)".

  How do we combine them?

  Simplest: x + g (just add)
  -> Problem: for some dimensions the original is important,
     for others the geometric info is important

  Solution: "gate" -- learn a mixing ratio per dimension!
""")

# Simple gate example
x_example = [0.5, -0.3, 0.8, 0.1]
g_example = [0.1,  0.7, -0.2, 0.6]
alpha_example = [0.9, 0.2, 0.7, 0.4]  # sigmoid output (0~1)

print("  Example:")
print(f"    Original vector x:     {x_example}")
print(f"    Geometric vector g:    {g_example}")
print(f"    Gate alpha:            {alpha_example}")
print(f"    (alpha is a sigmoid output, between 0 and 1)")
print()
print("    Mixing formula: output = alpha * x + (1-alpha) * g\n")

output = []
for j in range(4):
    a = alpha_example[j]
    val = a * x_example[j] + (1-a) * g_example[j]
    output.append(round(val, 3))
    print(f"    dim {j}: {a:.1f}*{x_example[j]:+.1f} + {1-a:.1f}*{g_example[j]:+.1f} = {val:+.3f}"
          f"  {'<- keep original' if a > 0.5 else '<- adopt geometric'}")

print(f"\n    Final output: {output}")

print("""
  alpha = 0.9 -> 90% original + 10% geometric (dimension where original info is sufficient)
  alpha = 0.2 -> 20% original + 80% geometric (dimension where previous token relationships matter)

  These alpha values are determined automatically during training!
""")


# ================================================================
separator("STEP 13: Full Pipeline Summary")
# ================================================================

print("""
  +-------------------------------------------------------------+
  |                                                             |
  |  Input: "h e l l o"  ->  Predict next char at position 4 'o'|
  |                                                             |
  |  +---------------------------------------------+           |
  |  |  1. Embedding: 'o' -> numeric vector x       |           |
  |  +----------------------+-----------------------+           |
  |                         v                                   |
  |  +---------------------------------------------+           |
  |  |  2. Dim reduction: x(16-dim) -> z(4-dim)     |           |
  |  +----------------------+-----------------------+           |
  |                         v                                   |
  |  +---------------------------------------------+           |
  |  |  3. Plucker coordinate computation:           |           |
  |  |     z_current and z_(1 step back) -> rel vec 1|           |
  |  |     z_current and z_(2 steps back) -> rel vec 2|          |
  |  |     z_current and z_(4 steps back) -> rel vec 3|          |
  |  |     -> average = geometric vector g            |           |
  |  +----------------------+-----------------------+           |
  |                         v                                   |
  |  +---------------------------------------------+           |
  |  |  4. Gated mixing:                             |           |
  |  |     alpha = sigmoid(...)                      |           |
  |  |     output = alpha*x + (1-alpha)*g            |           |
  |  +----------------------+-----------------------+           |
  |                         v                                   |
  |  +---------------------------------------------+           |
  |  |  5. FFN: transform the vector once more (d->4d->d)|      |
  |  +----------------------+-----------------------+           |
  |                         v                                   |
  |  +---------------------------------------------+           |
  |  |  6. Output: vector -> "probability of next char being   ||
  |  |          a: 3%, b: 1%, ... z: 2%"             |           |
  |  +---------------------------------------------+           |
  |                                                             |
  +-------------------------------------------------------------+
""")


# ================================================================
separator("STEP 14: Attention vs Grassmann -- Final Comparison")
# ================================================================

print("""
  +----------------+---------------------+----------------------+
  |                |     Attention       |     Grassmann        |
  +----------------+---------------------+----------------------+
  | Reference      | All previous tokens | Only nearby N tokens |
  | range          | (full context)      | (local window)       |
  +----------------+---------------------+----------------------+
  | Relationship   | Dot prod -> 1 scalar| Plucker -> vector    |
  | representation | "How similar?"      | "What relationship?" |
  +----------------+---------------------+----------------------+
  | Mixing method  | Softmax weighted avg| Gated mixing         |
  |                |                     | (per-dim ratio ctrl) |
  +----------------+---------------------+----------------------+
  | Complexity     | O(L^2)              | O(L)                 |
  |                | 2x length -> 4x slow| 2x length -> 2x slow|
  +----------------+---------------------+----------------------+
  | Performance    | Baseline            | Comparable (within   |
  |                |                     | 10-15%)              |
  +----------------+---------------------+----------------------+
  | Required       | W_q, W_k, W_v, W_o | W_red, W_plu, W_gate |
  | matrices       | (4 matrices)        | (3 matrices)         |
  +----------------+---------------------+----------------------+

  One-line summary:
    Attention = asking all friends briefly           (broad and shallow)
    Grassmann = deep conversation with close friends (narrow and deep)
""")

print("  Now go re-read micro_grassmann.py!")
print("  You'll see which part of the explanation each step corresponds to.")
print()
