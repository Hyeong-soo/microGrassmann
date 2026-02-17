import os, math, random
random.seed(42)

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

if not os.path.exists('input.txt'):
    import urllib.request
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt',
        'input.txt')

docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1

n_embd, n_layer, block_size = 16, 1, 16
r = 4
plucker_dim = r * (r - 1) // 2
window = [1, 2, 4, 8, 12, 16]

matrix = lambda nout, nin, std=0.08: \
    [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd),
}
for i in range(n_layer):
    state_dict[f'L{i}.W_red']    = matrix(r, n_embd)
    state_dict[f'L{i}.W_plu']    = matrix(n_embd, plucker_dim)
    state_dict[f'L{i}.W_gate_h'] = matrix(n_embd, n_embd)
    state_dict[f'L{i}.W_gate_g'] = matrix(n_embd, n_embd)
    state_dict[f'L{i}.mlp_fc1']  = matrix(4 * n_embd, n_embd)
    state_dict[f'L{i}.mlp_fc2']  = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]

def softmax(logits):
    mx = max(v.data for v in logits)
    exps = [(v - mx).exp() for v in logits]
    s = sum(exps)
    return [e / s for e in exps]

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

def forward(token_id, pos_id, z_cache):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_res = x
        x = rmsnorm(x)
        z = linear(x, state_dict[f'L{li}.W_red'])
        z_cache[li].append(z)

        geo_features = []
        for delta in window:
            prev_pos = pos_id - delta
            if prev_pos < 0:
                continue
            z_prev = z_cache[li][prev_pos]
            plucker = compute_plucker(z, z_prev)
            norm_sq = sum(p * p for p in plucker)
            norm_inv = (norm_sq + Value(1e-8)) ** -0.5
            plucker = [p * norm_inv for p in plucker]
            g_delta = linear(plucker, state_dict[f'L{li}.W_plu'])
            geo_features.append(g_delta)

        if geo_features:
            nf = len(geo_features)
            g = [sum(geo_features[k][j] for k in range(nf)) / nf
                 for j in range(n_embd)]
        else:
            g = [Value(0.0) for _ in range(n_embd)]

        gate_h = linear(x, state_dict[f'L{li}.W_gate_h'])
        gate_g = linear(g, state_dict[f'L{li}.W_gate_g'])
        alpha = [sigmoid(h_j + g_j) for h_j, g_j in zip(gate_h, gate_g)]
        x = [a * xi + (Value(1.0) - a) * gi for a, xi, gi in zip(alpha, x, g)]
        x = [a + b for a, b in zip(x, x_res)]

        x_res = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'L{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'L{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_res)]

    return linear(x, state_dict['lm_head'])

lr, beta1, beta2, eps = 0.01, 0.85, 0.99, 1e-8
m_buf = [0.0] * len(params)
v_buf = [0.0] * len(params)
num_steps = 3000

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    z_cache = [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        logits = forward(tokens[pos_id], pos_id, z_cache)
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

    if step % 100 == 0 or step == num_steps - 1:
        print(f"step {step+1:4d}/{num_steps} | loss {loss.data:.4f}")

eval_samples = 200
eval_loss_sum, eval_tokens = 0.0, 0
for ei in range(eval_samples):
    doc = docs[(num_steps + ei) % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    z_cache = [[] for _ in range(n_layer)]
    for pos_id in range(n):
        logits = forward(tokens[pos_id], pos_id, z_cache)
        probs = softmax(logits)
        eval_loss_sum += -math.log(probs[tokens[pos_id + 1]].data)
        eval_tokens += 1
print(f"\neval loss ({eval_samples} unseen samples): {eval_loss_sum / eval_tokens:.4f}")

temperature = 0.5
print()
for idx in range(20):
    z_cache = [[] for _ in range(n_layer)]
    token_id = BOS
    chars = []
    for pos_id in range(block_size):
        logits = forward(token_id, pos_id, z_cache)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS: break
        chars.append(uchars[token_id])
    print(f"  {''.join(chars)}")
