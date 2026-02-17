"""
trace_alpha_wred.py — How is alpha determined, and why can't any arbitrary matrix be used for dimensionality reduction?
=================================================================================

Question 1: Who determines alpha, when, and how?
Question 2: It seems like W_red can't just reduce dimensions with any matrix?

Run: python3 trace_alpha_wred.py
=================================================================================
"""

import math
import random
random.seed(42)

def section_header(title):
    print(f"\n{'='*68}")
    print(f"  {title}")
    print(f"{'='*68}\n")

def fmt(vec, n=3):
    return "[" + ", ".join(f"{v:+.{n}f}" for v in vec) + "]"

def mat_vec(W, x):
    return [sum(wi*xi for wi, xi in zip(row, x)) for row in W]

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# ================================================================
section_header("PART 1: How is alpha determined?")
# ================================================================

print("""
  First, the conclusion:
    alpha is NOT a "learned parameter" but a "computed value."

    alpha = sigmoid(W_gate_h @ x  +  W_gate_g @ g)
            -------  ----------     ----------
            squeeze  "score from     "score from
            to 0~1   examining x"    examining g"

    What IS learned are W_gate_h and W_gate_g (two matrices).
    alpha itself is freshly computed for every token, every dimension.
""")


# ================================================================
section_header("1-1. Following the alpha computation with numbers")
# ================================================================

d = 4  # 4 dimensions for understanding

x = [0.5, -0.3, 0.8, 0.2]   # Current token's vector (original)
g = [-0.1, 0.6, 0.1, -0.4]  # Geometry vector (relationships with previous tokens)

print(f"  x (original vector):  {fmt(x)}")
print(f"  g (geometry vector):  {fmt(g)}")

# Gate matrices (determined by training)
W_gate_h = [
    [ 0.3, -0.5,  0.2,  0.1],
    [ 0.1,  0.4, -0.3,  0.6],
    [-0.2,  0.1,  0.7, -0.4],
    [ 0.5, -0.1,  0.3,  0.2],
]

W_gate_g = [
    [ 0.2,  0.3, -0.1,  0.4],
    [-0.3,  0.5,  0.2, -0.1],
    [ 0.4, -0.2,  0.6,  0.3],
    [-0.1,  0.2, -0.3,  0.5],
]

print(f"""
  W_gate_h (4x4 matrix) -- the "eye" that examines x
  W_gate_g (4x4 matrix) -- the "eye" that examines g

  These two matrices are the trainable parameters.
  (Random at first, gradually improve through training)
""")

# Actual computation
score_h = mat_vec(W_gate_h, x)   # W_gate_h @ x
score_g = mat_vec(W_gate_g, g)   # W_gate_g @ g

print(f"  STEP 1: W_gate_h @ x = {fmt(score_h, 4)}")
print(f"          'Evaluate how useful x is for each dimension'\n")

print(f"  STEP 2: W_gate_g @ g = {fmt(score_g, 4)}")
print(f"          'Evaluate how useful g is for each dimension'\n")

combined = [h + g_val for h, g_val in zip(score_h, score_g)]
print(f"  STEP 3: Sum both scores = {fmt(combined, 4)}")
print(f"          'Overall judgment: a score considering both x and g'\n")

alpha = [sigmoid(c) for c in combined]
print(f"  STEP 4: Apply sigmoid    = {fmt(alpha, 4)}")
print(f"          'Convert to 0~1 range: alpha complete!'\n")

print(f"  Interpretation:")
for j in range(d):
    a = alpha[j]
    if a > 0.7:
        verdict = f"-> mostly original x ({a:.0%} x, {1-a:.0%} g)"
    elif a < 0.3:
        verdict = f"-> mostly geometry g ({a:.0%} x, {1-a:.0%} g)"
    else:
        verdict = f"-> evenly mixed ({a:.0%} x, {1-a:.0%} g)"
    print(f"    dim[{j}]: alpha={a:.3f}  {verdict}")


# ================================================================
section_header("1-2. Key insight: alpha 'varies depending on the situation'")
# ================================================================

print("""
  Even with the same model, different inputs produce different alphas.
  W_gate_h and W_gate_g are fixed, but x and g change.
""")

# Scenario A: 'o' in "hello"
x_A = [0.5, -0.3, 0.8, 0.2]
g_A = [-0.1, 0.6, 0.1, -0.4]
alpha_A = [sigmoid(h + gv) for h, gv in
           zip(mat_vec(W_gate_h, x_A), mat_vec(W_gate_g, g_A))]

# Scenario B: 'd' in "world" (different x, different g)
x_B = [-0.4, 0.7, 0.1, 0.5]
g_B = [0.3, -0.2, 0.5, 0.8]
alpha_B = [sigmoid(h + gv) for h, gv in
           zip(mat_vec(W_gate_h, x_B), mat_vec(W_gate_g, g_B))]

# Scenario C: first token in sequence (g = zero vector)
x_C = [0.5, -0.3, 0.8, 0.2]
g_C = [0.0, 0.0, 0.0, 0.0]
alpha_C = [sigmoid(h + gv) for h, gv in
           zip(mat_vec(W_gate_h, x_C), mat_vec(W_gate_g, g_C))]

print(f"  Scenario A: 'o' in 'hello' (has context)")
print(f"    x = {fmt(x_A)},  g = {fmt(g_A)}")
print(f"    alpha = {fmt(alpha_A)}")
print()

print(f"  Scenario B: 'd' in 'world' (different context)")
print(f"    x = {fmt(x_B)},  g = {fmt(g_B)}")
print(f"    alpha = {fmt(alpha_B)}")
print()

print(f"  Scenario C: first token in sequence (no context, g=0)")
print(f"    x = {fmt(x_C)},  g = {fmt(g_C)}")
print(f"    alpha = {fmt(alpha_C)}")

print(f"""
  Key observations:
    - Same W_gate matrices but alpha is completely different each time!
    - In Scenario C (first token), g=0 so W_gate_g @ g = 0
      -> alpha is determined solely by x
      -> the model can naturally learn "if no context, just use the original"
""")


# ================================================================
section_header("1-3. How is W_gate trained?")
# ================================================================

print("""
  W_gate_h and W_gate_g are trained via backpropagation.

  An analogy for the training process:

  +---------------------------------------------------------+
  |                                                         |
  |  [Day 1 -- early training]                              |
  |   W_gate_h, W_gate_g = random                          |
  |   alpha ~ around 0.5 (meaningless mixing)               |
  |   loss = high (predictions are terrible)                |
  |                                                         |
  |  [Day 100 -- mid training]                              |
  |   Backpropagation sends this signal to W_gate:          |
  |     "Using g more in dim[1] would reduce the loss!"     |
  |   -> W_gate adjusts to lower alpha in dim[1]            |
  |                                                         |
  |  [Day 1000 -- late training]                            |
  |   W_gate has learned patterns:                          |
  |     "When x looks like this, use g more,                |
  |      when x looks like that, use g less"                |
  |   -> alpha adapts well to each situation                |
  |   loss = low (predictions are accurate)                 |
  |                                                         |
  +---------------------------------------------------------+

  Gradient flow:
    loss -> output -> alpha*x + (1-alpha)*g -> alpha -> sigmoid
    -> W_gate_h @ x + W_gate_g @ g -> gradients reach W_gate_h, W_gate_g!

  alpha itself is a "result," not a "parameter."
  W_gate_h, W_gate_g are the "cause" and these are what get trained.

  Simple analogy:
    W_gate = the "intuition" for judging what to use in what situation
    alpha  = the "decision" made by that intuition at this moment

    Intuition (W_gate) develops through experience (training),
    decisions (alpha) are made fresh each moment.
""")


# ================================================================
# ================================================================
# ================================================================
section_header("PART 2: Dimensionality reduction — Why can't any arbitrary matrix work?")
# ================================================================

print("""
  Question: "It seems like it doesn't work to just multiply any matrix
  as long as it reduces dimensions"
  -> Correct intuition. Depending on what W_red preserves, the results
  are completely different.
""")


# ================================================================
section_header("2-1. The essence of dimensionality reduction: 'What to discard and what to keep'")
# ================================================================

print("""
  Reducing a 6D vector to 3D = discarding half the information

  The question is "which half to discard."

  Analogy:
    You want to compare students by reducing 6 subject scores
    (Korean, Math, English, Science, Social Studies, PE) down to 3.

    Method A: Keep only Korean, Math, English
      -> Can compare by major subjects
      -> Cannot distinguish science prodigies from athletic prodigies

    Method B: Keep only Science, Social Studies, PE
      -> Compare by minor subjects
      -> Differences in Korean/Math/English scores disappear

    Method C: Transform into "humanities aptitude", "science aptitude", "fitness"
      -> Combine multiple subjects to create new axes
      -> Compress the core patterns of all 6 subjects into 3!

  W_red is Method C.
  Rather than simply picking some dimensions to discard,
  it "combines" multiple dimensions to create new axes.
""")


# ================================================================
section_header("2-2. Experiment: same tokens but Plucker differs depending on W_red")
# ================================================================

d = 6
r = 3

# Embeddings for two tokens
x_cat = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3]   # 'cat' feel (odd-indexed dims high)
x_dog = [0.2, 0.8, 0.3, 0.7, 0.4, 0.6]   # 'dog' feel (even-indexed dims high)
x_cat2= [0.85, 0.15, 0.75, 0.25, 0.65, 0.35]  # similar to 'cat' (slightly different)

print(f"  Token A 'cat':     {fmt(x_cat)}")
print(f"  Token B 'dog':     {fmt(x_dog)}")
print(f"  Token C 'cat2':    {fmt(x_cat2)}  (similar to A)")

print(f"""
  Intuitively:
    A-C pair (cats together) -> should capture a "similar relationship"
    A-B pair (cat-dog)       -> should capture a "different relationship"

  A good W_red should make this difference clearly visible in Plucker coordinates.
""")

# --- W_red 1: Good matrix (projection that preserves inter-token differences) ---
# Each row looks at the vector from a different "perspective"
W_red_good = [
    [ 1.0, -1.0,  0.0,  0.0,  0.0,  0.0],   # dim0 vs dim1 difference
    [ 0.0,  0.0,  1.0, -1.0,  0.0,  0.0],   # dim2 vs dim3 difference
    [ 0.0,  0.0,  0.0,  0.0,  1.0, -1.0],   # dim4 vs dim5 difference
]
# cat: odd indices high -> differences are positive
# dog: even indices high -> differences are negative -> z directions are very different!

# --- W_red 2: Bad matrix (projection that collapses differences) ---
W_red_bad = [
    [ 0.3,  0.3,  0.3,  0.3,  0.3,  0.3],   # just the overall average
    [ 0.31, 0.29, 0.3,  0.3,  0.3,  0.3],   # nearly the same average
    [ 0.3,  0.3,  0.31, 0.29, 0.3,  0.3],   # yet another nearly identical average
]
# Sums all dimensions with similar weights -> cat and dog end up similar

def compute_plucker(u, v):
    coords = []
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            coords.append(u[i]*v[j] - u[j]*v[i])
    norm = math.sqrt(sum(p*p for p in coords) + 1e-8)
    return [p/norm for p in coords]

def plucker_distance(p1, p2):
    """Distance between two normalized Plucker vectors (smaller = more similar)"""
    return math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))

print(f"  -- Experiment: Good W_red vs Bad W_red --\n")

for name, W_red in [("Good W_red", W_red_good), ("Bad W_red", W_red_bad)]:
    z_A = mat_vec(W_red, x_cat)
    z_B = mat_vec(W_red, x_dog)
    z_C = mat_vec(W_red, x_cat2)

    plucker_AB = compute_plucker(z_A, z_B)  # cat-dog
    plucker_AC = compute_plucker(z_A, z_C)  # cat-cat2

    dist = plucker_distance(plucker_AB, plucker_AC)

    print(f"  [{name}]")
    print(f"    z_cat   = {fmt(z_A)}")
    print(f"    z_dog   = {fmt(z_B)}")
    print(f"    z_cat2  = {fmt(z_C)}")
    print(f"    Plucker(cat, dog)   = {fmt(plucker_AB, 4)}")
    print(f"    Plucker(cat, cat2)  = {fmt(plucker_AC, 4)}")
    print(f"    Distance between Pluckers = {dist:.4f}")
    if dist > 0.3:
        print(f"    -> Large distance! Distinguishes 'different relationship vs similar relationship' well")
    else:
        print(f"    -> Small distance! Even different relationships look similar")
    print()


# ================================================================
section_header("2-3. Why does this difference arise? — A geometric explanation")
# ================================================================

print("""
  Plucker coordinates = the 'direction of the plane' spanned by two vectors

  Key: When W_red reduces dimensions, vectors are 'projected' into 3D space.
  Depending on how the projection is done, relationships between vectors
  are either preserved or collapsed.

  +-------------------------------------------------------------+
  |                                                              |
  |  [Good W_red]                                                |
  |                                                              |
  |  Cat and dog pointed in different directions in the original |
  |  6D space, and they remain in different directions even      |
  |  after reduction to 3D.                                      |
  |                                                              |
  |  In 3D space:                                                |
  |    z_cat   -> upward direction                               |
  |    z_dog   -> sideways direction                             |
  |    z_cat2  -> upward direction (similar to z_cat)            |
  |                                                              |
  |  -> Plucker captures this difference!                        |
  |    (cat-dog) plane != (cat-cat2) plane                       |
  |                                                              |
  +-------------------------------------------------------------+
  |                                                              |
  |  [Bad W_red]                                                 |
  |                                                              |
  |  Averaging all dimensions with similar weights               |
  |  -> differences get collapsed                                |
  |                                                              |
  |  In 3D space:                                                |
  |    z_cat   -> one direction                                  |
  |    z_dog   -> nearly the same direction (!!)                 |
  |    z_cat2  -> nearly the same direction (!!)                 |
  |                                                              |
  |  -> All three vectors end up in similar directions           |
  |  -> Plucker cannot distinguish the relationship differences  |
  |                                                              |
  +-------------------------------------------------------------+

  Analogy:
    Good W_red = a good camera angle
      When photographing a 3D object as a 2D image, a good angle
      reveals the shape well

    Bad W_red = a bad camera angle
      Shooting only from the front makes a cylinder and a sphere
      look the same; you need a side view to see the difference,
      but you keep shooting from the front
""")


# ================================================================
section_header("2-4. So how does W_red become a 'good matrix'?")
# ================================================================

print("""
  Answer: Through training (backpropagation)!

  Path of gradient flow:

    loss
     | (the prediction was wrong!)
    output = alpha*x + (1-alpha)*g
     |
    g = W_plu @ plucker
     |
    plucker = compute_plucker(z, z_prev)
     | |
    z = W_red @ x        z_prev = W_red @ x_prev
     |                    |
    Gradient reaches W_red!

  What the gradient to W_red means:
    "Because of this W_red, Plucker came out like this,
     that Plucker produced g, and g was mixed into the output,
     which ultimately made the prediction off by this much.
     Adjusting W_red in this direction will reduce the loss."

  As training progresses:
    -> W_red evolves to perform "reduction that reveals inter-token relationships"
    -> Semantically similar tokens get similar z, different tokens get different z
    -> Plucker captures meaningful relationships
""")


# ================================================================
section_header("2-5. Three roles of dimensionality reduction")
# ================================================================

print("""
  W_red is not simply "reducing dimensions" -- it simultaneously performs 3 roles:

  +--------------------------------------------------------------+
  |                                                              |
  |  Role 1: Computational cost reduction                        |
  |    d=16 -> r=4 reduces Plucker dim from C(16,2)=120 to      |
  |    C(4,2)=6. A 20x reduction in computation.                 |
  |                                                              |
  |  Role 2: Extracting "features important for comparison"      |
  |    Picks out the 4 core patterns needed for relationship     |
  |    comparison from 16 features.                              |
  |    Noise dimensions are naturally removed.                   |
  |                                                              |
  |    Analogy: When comparing people, out of 16 traits,         |
  |    "height/weight/age/sex" -- just 4 -- are enough for       |
  |    body-type comparison. W_red learns to find the            |
  |    "4 things needed for relationship comparison."            |
  |                                                              |
  |  Role 3: Transforming into a space where Plucker works       |
  |    Plucker coordinates = "direction of the plane spanned     |
  |    by two vectors."                                          |
  |    For this to be meaningful, vectors must be in a proper    |
  |    space.                                                    |
  |                                                              |
  |    Bad space: all vectors nearly same direction -> no plane  |
  |               difference                                     |
  |    Good space: vectors in diverse directions -> clear plane  |
  |                differences                                   |
  |                                                              |
  |    W_red plays the role of sending vectors into a "space     |
  |    where Plucker works well."                                |
  |                                                              |
  +--------------------------------------------------------------+
""")


# ================================================================
section_header("2-6. Comparison with Attention's Q, K")
# ================================================================

print("""
  In fact, something similar happens in Attention.

  +--- Attention -------------------------------------------------+
  |                                                               |
  |  Q = W_q @ x     <- project x into "query space"             |
  |  K = W_k @ x     <- project x into "key space"               |
  |  score = Q . K    <- dot product in those spaces (scalar)     |
  |                                                               |
  |  W_q, W_k are also learned.                                   |
  |  Their role: sending to a "space suitable for comparison."     |
  |  Using arbitrary matrices won't work -- same principle!        |
  |                                                               |
  +--- Grassmann -------------------------------------------------+
  |                                                               |
  |  z = W_red @ x         <- project x into "geometry space"     |
  |  z_prev = W_red @ x_prev  <- into the same space              |
  |  plucker = Plucker(z, z_prev)  <- relationship (a vector!)    |
  |                                                               |
  |  W_red is also learned.                                        |
  |  Its role: sending to a "space suitable for geometric          |
  |  comparison."                                                  |
  |                                                               |
  +---------------------------------------------------------------+

  Differences:
    Attention: 2 matrices (W_q, W_k) project into separate spaces
    Grassmann: 1 matrix (W_red) projects into the same space

    Attention: relationship = dot product (1 scalar)
    Grassmann: relationship = Plucker (C(r,2) vector components)

  Why one W_red is sufficient:
    Plucker coordinates are inherently "asymmetric."
    In p_ij = u_i*v_j - u_j*v_i, u and v play different roles.
    (Swapping u and v flips the sign!)
    So even in the same space, directional information is preserved.
""")


# ================================================================
section_header("Final summary")
# ================================================================

print("""
  +------------------------------------------------------------+
  |                                                            |
  |  Q: How is alpha determined?                               |
  |                                                            |
  |  A: alpha = sigmoid(W_gate_h @ x + W_gate_g @ g)          |
  |     - W_gate_h, W_gate_g: learned matrices (develop        |
  |       through experience)                                  |
  |     - alpha: a value computed fresh for each token          |
  |       (a momentary decision)                               |
  |     - Looks at both x and g to decide mixing ratio         |
  |       "per dimension"                                      |
  |     - Early training: roughly 0.5 (mixed arbitrarily)      |
  |     - Late training: precisely tuned per situation          |
  |                                                            |
  +------------------------------------------------------------+
  |                                                            |
  |  Q: Can any matrix work for dimensionality reduction?      |
  |                                                            |
  |  A: No!                                                    |
  |     W_red simultaneously performs 3 things via training:    |
  |       1. Computational cost reduction (fewer dimensions)   |
  |       2. Extracting features critical for relationship     |
  |          comparison                                        |
  |       3. Transforming into a space where Plucker           |
  |          produces meaningful output                        |
  |                                                            |
  |     Random matrix -> meaningless reduction -> Plucker      |
  |       outputs useless values                               |
  |     Trained matrix -> core extraction -> Plucker captures  |
  |       relationships well                                   |
  |                                                            |
  |     Same principle as Attention's W_q, W_k improving       |
  |     through training!                                      |
  |                                                            |
  +------------------------------------------------------------+
""")
