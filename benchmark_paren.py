"""
benchmark_paren.py — Comparing Attention vs Grassmann on Parenthesis Matching
================================================================
Since names (5-6 characters) don't show a difference,
we compare using a parenthesis matching task that requires long-range dependencies.

To predict the next token in strings like "((()))(())",
the model must remember distant opening parentheses.

Usage: python3 benchmark_paren.py
================================================================
"""
import math, random, time
random.seed(42)

# ================================================================
# Data: Generating Balanced Parenthesis Strings
# ================================================================
def generate_parens(min_len=20, max_len=50):
    """Generate a random balanced parenthesis string (Dyck language)"""
    target_pairs = random.randint(min_len // 2, max_len // 2)
    seq = []
    open_count = 0
    remaining = target_pairs * 2
    for _ in range(target_pairs * 2):
        can_open = open_count < target_pairs and (remaining - open_count) > open_count
        can_close = open_count > 0
        if can_open and can_close:
            if random.random() < 0.5:
                seq.append(0)  # "("
                open_count += 1
            else:
                seq.append(1)  # ")"
                open_count -= 1
        elif can_open:
            seq.append(0)
            open_count += 1
        else:
            seq.append(1)
            open_count -= 1
        remaining -= 1
    return seq

BOS = 2
vocab_size = 3  # "(", ")", BOS
block_size = 64

docs = [generate_parens() for _ in range(10000)]
random.shuffle(docs)
avg_len = sum(len(d) for d in docs) / len(docs)
print(f"Parenthesis data: {len(docs)} samples, avg length: {avg_len:.1f}, block_size: {block_size}")
print(f"Example: {''.join('()B'[t] for t in docs[0][:40])}...")

# ================================================================
# Autograd (Shared)
# ================================================================
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data, self.grad = data, 0
        self._children, self._local_grads = children, local_grads
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return Value(self.data + o.data, (self, o), (1, 1))
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return Value(self.data * o.data, (self, o), (o.data, self.data))
    def __pow__(self, n):
        return Value(self.data ** n, (self,), (n * self.data ** (n - 1),))
    def log(self):  return Value(math.log(self.data), (self,), (1 / self.data,))
    def exp(self):  return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
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

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]

def softmax(logits):
    mx = max(v.data for v in logits)
    exps = [(v - mx).exp() for v in logits]
    s = sum(exps)
    return [e / s for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    return [xi * (ms + 1e-5) ** -0.5 for xi in x]

def sigmoid(x):
    return Value(1.0) / (Value(1.0) + (-x).exp())

matrix = lambda nout, nin, std=0.08: \
    [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# ================================================================
# Model A: Attention (microGPT)
# ================================================================
def build_attention():
    n_embd, n_layer, n_head = 16, 1, 4
    head_dim = n_embd // n_head
    sd = {
        'wte': matrix(vocab_size, n_embd),
        'wpe': matrix(block_size, n_embd),
        'lm_head': matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        sd[f'L{i}.wq'] = matrix(n_embd, n_embd)
        sd[f'L{i}.wk'] = matrix(n_embd, n_embd)
        sd[f'L{i}.wv'] = matrix(n_embd, n_embd)
        sd[f'L{i}.wo'] = matrix(n_embd, n_embd)
        sd[f'L{i}.fc1'] = matrix(4 * n_embd, n_embd)
        sd[f'L{i}.fc2'] = matrix(n_embd, 4 * n_embd)
    params = [p for mat in sd.values() for row in mat for p in row]

    def forward(tid, pid, keys, values):
        x = [t + p for t, p in zip(sd['wte'][tid], sd['wpe'][pid])]
        x = rmsnorm(x)
        for li in range(n_layer):
            xr = x
            x = rmsnorm(x)
            q = linear(x, sd[f'L{li}.wq'])
            k = linear(x, sd[f'L{li}.wk'])
            v = linear(x, sd[f'L{li}.wv'])
            keys[li].append(k)
            values[li].append(v)
            xa = []
            for h in range(n_head):
                hs = h * head_dim
                qh = q[hs:hs+head_dim]
                kh = [ki[hs:hs+head_dim] for ki in keys[li]]
                vh = [vi[hs:hs+head_dim] for vi in values[li]]
                al = [sum(qh[j]*kh[t][j] for j in range(head_dim)) / head_dim**0.5
                      for t in range(len(kh))]
                aw = softmax(al)
                ho = [sum(aw[t]*vh[t][j] for t in range(len(vh))) for j in range(head_dim)]
                xa.extend(ho)
            x = linear(xa, sd[f'L{li}.wo'])
            x = [a + b for a, b in zip(x, xr)]
            xr = x
            x = rmsnorm(x)
            x = linear(x, sd[f'L{li}.fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, sd[f'L{li}.fc2'])
            x = [a + b for a, b in zip(x, xr)]
        return linear(x, sd['lm_head'])

    def make_cache():
        return [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]

    return params, forward, make_cache

# ================================================================
# Model B: Grassmann
# ================================================================
def build_grassmann():
    n_embd, n_layer = 16, 1
    r = 4
    plucker_dim = r * (r - 1) // 2
    window = [1, 2, 4, 8, 16]  # Extended for longer sequences

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

    def compute_plucker(u, v):
        coords = []
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                coords.append(u[i] * v[j] - u[j] * v[i])
        return coords

    def forward(tid, pid, z_cache):
        x = [t + p for t, p in zip(sd['wte'][tid], sd['wpe'][pid])]
        x = rmsnorm(x)
        for li in range(n_layer):
            xr = x
            x = rmsnorm(x)
            z = linear(x, sd[f'L{li}.W_red'])
            z_cache[li].append(z)
            gf = []
            for delta in window:
                pp = pid - delta
                if pp < 0: continue
                zp = z_cache[li][pp]
                plk = compute_plucker(z, zp)
                nsq = sum(p * p for p in plk)
                ninv = (nsq + Value(1e-8)) ** -0.5
                plk = [p * ninv for p in plk]
                gf.append(linear(plk, sd[f'L{li}.W_plu']))
            if gf:
                nf = len(gf)
                g = [sum(gf[k][j] for k in range(nf)) / nf for j in range(n_embd)]
            else:
                g = [Value(0.0) for _ in range(n_embd)]
            gh = linear(x, sd[f'L{li}.W_gate_h'])
            gg = linear(g, sd[f'L{li}.W_gate_g'])
            alpha = [sigmoid(h + gv) for h, gv in zip(gh, gg)]
            x = [a * xi + (Value(1.0) - a) * gi for a, xi, gi in zip(alpha, x, g)]
            x = [a + b for a, b in zip(x, xr)]
            xr = x
            x = rmsnorm(x)
            x = linear(x, sd[f'L{li}.fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, sd[f'L{li}.fc2'])
            x = [a + b for a, b in zip(x, xr)]
        return linear(x, sd['lm_head'])

    def make_cache():
        return [[] for _ in range(n_layer)]

    return params, forward, make_cache

# ================================================================
# Training & Evaluation
# ================================================================
def train_and_eval(name, params, forward, make_cache, num_steps=1000):
    lr, beta1, beta2, eps = 0.01, 0.85, 0.99, 1e-8
    m_buf = [0.0] * len(params)
    v_buf = [0.0] * len(params)

    print(f"\n{'='*60}")
    print(f"  {name} | Parameters: {len(params)} | {num_steps} steps")
    print(f"{'='*60}")

    t0 = time.time()
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + doc + [BOS]
        n = min(block_size, len(tokens) - 1)
        cache = make_cache()
        losses = []

        for pos_id in range(n):
            logits = forward(tokens[pos_id], pos_id, *cache) if isinstance(cache, tuple) else forward(tokens[pos_id], pos_id, cache)
            probs = softmax(logits)
            losses.append(-probs[tokens[pos_id + 1]].log())

        loss = (1 / n) * sum(losses)
        loss.backward()

        lr_t = lr * (1 - step / num_steps)
        for i, p in enumerate(params):
            m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
            v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
            m_hat = m_buf[i] / (1 - beta1 ** (step + 1))
            v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps)
            p.grad = 0

        if step % 200 == 0 or step == num_steps - 1:
            print(f"  step {step+1:4d}/{num_steps} | loss {loss.data:.4f}")

    train_time = time.time() - t0

    # Eval
    eval_samples = 200
    eval_loss_sum, eval_tokens = 0.0, 0
    for ei in range(eval_samples):
        doc = docs[(num_steps + ei) % len(docs)]
        tokens = [BOS] + doc + [BOS]
        n = min(block_size, len(tokens) - 1)
        cache = make_cache()
        for pos_id in range(n):
            logits = forward(tokens[pos_id], pos_id, *cache) if isinstance(cache, tuple) else forward(tokens[pos_id], pos_id, cache)
            probs = softmax(logits)
            eval_loss_sum += -math.log(probs[tokens[pos_id + 1]].data)
            eval_tokens += 1
    eval_loss = eval_loss_sum / eval_tokens

    # Generation test: valid parentheses ratio
    valid_count = 0
    gen_samples = 50
    for _ in range(gen_samples):
        cache = make_cache()
        tid = BOS
        seq = []
        for pid in range(block_size):
            logits = forward(tid, pid, *cache) if isinstance(cache, tuple) else forward(tid, pid, cache)
            probs = softmax([l / 0.5 for l in logits])
            tid = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if tid == BOS: break
            seq.append(tid)
        # Validity check
        depth = 0
        valid = True
        for t in seq:
            if t == 0: depth += 1
            elif t == 1: depth -= 1
            if depth < 0:
                valid = False
                break
        if depth != 0: valid = False
        if len(seq) == 0: valid = False
        if valid: valid_count += 1

    print(f"\n  eval loss: {eval_loss:.4f}")
    print(f"  Training time: {train_time:.1f}s ({train_time/num_steps*1000:.0f}ms/step)")
    print(f"  Valid parentheses generated: {valid_count}/{gen_samples} ({valid_count/gen_samples*100:.0f}%)")

    # 5 generation examples
    print(f"  Generation examples:")
    for si in range(5):
        cache = make_cache()
        tid = BOS
        seq = []
        for pid in range(block_size):
            logits = forward(tid, pid, *cache) if isinstance(cache, tuple) else forward(tid, pid, cache)
            probs = softmax([l / 0.5 for l in logits])
            tid = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if tid == BOS: break
            seq.append('(' if tid == 0 else ')')
        print(f"    {''.join(seq)}")

    return eval_loss, train_time, valid_count / gen_samples

# ================================================================
# Execution
# ================================================================
num_steps = 1000

print("Building Attention model...")
a_params, a_fwd, a_cache = build_attention()
print("Building Grassmann model...")
g_params, g_fwd, g_cache = build_grassmann()

a_loss, a_time, a_valid = train_and_eval("Attention (microGPT)", a_params, a_fwd, a_cache, num_steps)
g_loss, g_time, g_valid = train_and_eval("Grassmann (Plucker)", g_params, g_fwd, g_cache, num_steps)

print(f"\n{'='*60}")
print(f"  Final Comparison (parenthesis matching, avg length {avg_len:.0f})")
print(f"{'='*60}")
print(f"  {'':15} {'Attention':>12} {'Grassmann':>12} {'Diff':>10}")
print(f"  {'─'*52}")
print(f"  {'Parameters':15} {len(a_params):>12} {len(g_params):>12} {len(g_params)/len(a_params)*100-100:>+9.0f}%")
print(f"  {'eval loss':15} {a_loss:>12.4f} {g_loss:>12.4f} {(g_loss-a_loss)/a_loss*100:>+9.1f}%")
print(f"  {'Training time':15} {a_time:>11.1f}s {g_time:>11.1f}s {(g_time-a_time)/a_time*100:>+9.0f}%")
print(f"  {'ms/step':15} {a_time/num_steps*1000:>11.0f}ms {g_time/num_steps*1000:>11.0f}ms")
print(f"  {'Valid parens':15} {a_valid*100:>11.0f}% {g_valid*100:>11.0f}%")
