"""
benchmark.py — microGPT (Attention) vs microGrassmann (Plucker) 성능 비교
==========================================================================
동일한 조건(데이터, 하이퍼파라미터, 학습 스텝 수)에서 두 모델을 학습시켜
loss 곡선, 파라미터 수, 학습 속도, 생성 품질을 비교합니다.

사용법:
  python3 benchmark.py
==========================================================================
"""

import os
import math
import random
import time

# ============================================================================
#  공유 인프라: Autograd, 데이터, 유틸리티
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

# --- 데이터 로딩 ---
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

# --- 공유 하이퍼파라미터 ---
n_embd     = 16
n_layer    = 1
block_size = 16
n_head     = 4               # Attention 전용
head_dim   = n_embd // n_head
r          = 4               # Grassmann 전용
plucker_dim = r * (r - 1) // 2
window     = [1, 2, 4]

lr, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
num_steps  = 1000
temperature = 0.5

# --- 유틸리티 함수 ---
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
#  모델 A: microGPT (Multi-Head Attention)
# ============================================================================
#  Karpathy의 원본 microGPT 아키텍처를 그대로 재현합니다.
#  핵심: Q*K^T → softmax → V 가중합 (모든 이전 토큰을 참조)
# ============================================================================

def build_attention():
    """Attention 모델의 파라미터를 초기화합니다."""
    matrix = lambda nout, nin: \
        [[Value(random.gauss(0, 0.08)) for _ in range(nin)] for _ in range(nout)]
    sd = {
        'wte': matrix(vocab_size, n_embd),
        'wpe': matrix(block_size, n_embd),
        'lm_head': matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        sd[f'L{i}.wq'] = matrix(n_embd, n_embd)   # Query 프로젝션
        sd[f'L{i}.wk'] = matrix(n_embd, n_embd)   # Key 프로젝션
        sd[f'L{i}.wv'] = matrix(n_embd, n_embd)   # Value 프로젝션
        sd[f'L{i}.wo'] = matrix(n_embd, n_embd)   # Output 프로젝션
        sd[f'L{i}.fc1'] = matrix(4 * n_embd, n_embd)
        sd[f'L{i}.fc2'] = matrix(n_embd, 4 * n_embd)
    params = [p for mat in sd.values() for row in mat for p in row]
    return sd, params

def forward_attention(token_id, pos_id, kv_cache, sd):
    """Attention 모델의 forward pass (Karpathy microGPT 원본)."""
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
#  모델 B: microGrassmann (Plucker 좌표)
# ============================================================================
#  "Attention Is Not What You Need" 논문의 Causal Grassmann Layer.
#  핵심: 차원 축소 → Plucker 인코딩 → 게이트 혼합 (로컬 윈도우만 참조)
# ============================================================================

def build_grassmann():
    """Grassmann 모델의 파라미터를 초기화합니다."""
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
    """Grassmann 모델의 forward pass."""
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
#  학습 & 평가 루프
# ============================================================================

def train_model(name, build_fn, forward_fn, make_cache, seed):
    """
    모델을 학습시키고 결과를 반환합니다.

    Args:
        name:       모델 이름 (출력용)
        build_fn:   파라미터 빌더 함수
        forward_fn: forward pass 함수
        make_cache: 캐시 생성 람다 (Attention: KV캐시, Grassmann: z캐시)
        seed:       랜덤 시드 (파라미터 초기화용)

    Returns:
        (loss_history, samples, n_params, elapsed_sec)
    """
    random.seed(seed)
    sd, params = build_fn()
    n_params = len(params)

    m_buf = [0.0] * n_params
    v_buf = [0.0] * n_params
    loss_history = []

    print(f"\n  [{name}] 파라미터 수: {n_params}")
    print(f"  [{name}] 학습 시작...")

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

    # --- 추론: 이름 생성 ---
    random.seed(999)  # 샘플링 시드 고정 (공정 비교)
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
#  실행 & 결과 비교
# ============================================================================

print("=" * 64)
print("  microGPT (Attention)  vs  microGrassmann (Plucker)")
print("  동일 조건 성능 비교")
print("=" * 64)
print(f"\n  공통 설정: d={n_embd}, layers={n_layer}, block_size={block_size}, "
      f"steps={num_steps}")
print(f"  Attention: {n_head} heads, head_dim={head_dim}")
print(f"  Grassmann: r={r}, plucker_dim={plucker_dim}, window={window}")

# --- 학습 실행 ---
attn_cache_fn = lambda: [{'k': [], 'v': []} for _ in range(n_layer)]
grass_cache_fn = lambda: [[] for _ in range(n_layer)]

loss_a, samp_a, np_a, time_a = train_model(
    "Attention", build_attention, forward_attention, attn_cache_fn, seed=1234)

loss_g, samp_g, np_g, time_g = train_model(
    "Grassmann", build_grassmann, forward_grassmann, grass_cache_fn, seed=1234)


# ============================================================================
#  결과 출력
# ============================================================================

def avg(lst, start, end):
    """구간 평균 계산."""
    segment = lst[start:end]
    return sum(segment) / len(segment) if segment else 0

print(f"\n{'=' * 64}")
print(f"  비교 결과")
print(f"{'=' * 64}")

# --- 기본 통계 ---
print(f"\n  {'항목':<24}{'Attention':>16}{'Grassmann':>16}")
print(f"  {'─' * 56}")
print(f"  {'파라미터 수':<24}{np_a:>16,}{np_g:>16,}")
print(f"  {'학습 시간 (초)':<24}{time_a:>16.1f}{time_g:>16.1f}")
print(f"  {'step당 시간 (ms)':<21}{time_a/num_steps*1000:>16.1f}{time_g/num_steps*1000:>16.1f}")

# --- Loss 곡선 비교 (구간별 평균) ---
print(f"\n  {'Loss 비교 (구간 평균)':<24}{'Attention':>16}{'Grassmann':>16}")
print(f"  {'─' * 56}")
ranges = [
    ("초반 (1-100)",     0,   100),
    ("중반 (401-500)",   400, 500),
    ("후반 (901-1000)",  900, 1000),
]
for label, s, e in ranges:
    la = avg(loss_a, s, e)
    lg = avg(loss_g, s, e)
    diff = ((lg - la) / la) * 100
    marker = "<--" if lg < la else ("-->" if lg > la else "==")
    print(f"  {label:<24}{la:>16.4f}{lg:>16.4f}  {diff:+.1f}% {marker}")

print(f"\n  {'최종 loss':<24}{loss_a[-1]:>16.4f}{loss_g[-1]:>16.4f}")
print(f"  {'최저 loss':<24}{min(loss_a):>16.4f}{min(loss_g):>16.4f}")

# --- Loss 곡선 ASCII 시각화 ---
print(f"\n  Loss 곡선 (50 step 이동 평균):")
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
    # 먼저 뒤에 놓이는 것, 그 다음에 앞에 놓이는 것 (겹칠 때 둘 다 보이게)
    line[pos_a] = 'A'
    if pos_g == pos_a:
        line[pos_g] = '*'  # 겹치면 *
    else:
        line[pos_g] = 'G'
    print(f"  {step_label} |{''.join(line)}|")

print(f"  {'':>5}  {lo:.2f}{' ' * (chart_w - 10)}{hi:.2f}")
print(f"        A = Attention,  G = Grassmann,  * = 겹침")

# --- 생성 샘플 비교 ---
print(f"\n  {'생성 샘플 비교':}")
print(f"  {'─' * 56}")
print(f"  {'#':<4}  {'Attention':<20}  {'Grassmann':<20}")
print(f"  {'─' * 56}")
for i in range(20):
    print(f"  {i+1:<4}  {samp_a[i]:<20}  {samp_g[i]:<20}")

# --- 복잡도 비교 ---
print(f"\n  이론적 복잡도 비교 (시퀀스 길이 L, 임베딩 차원 d):")
print(f"  {'─' * 56}")
print(f"  {'Attention':<16}  O(L^2 * d_head  +  L * d^2)")
print(f"  {'':>16}  모든 토큰 쌍의 유사도를 계산 (L x L 행렬)")
print(f"  {'Grassmann':<16}  O(L * |W| * r^2  +  L * d^2)")
print(f"  {'':>16}  고정 윈도우 오프셋만 사용 (L^2 항 없음)")
print(f"  {'':>16}  |W|={len(window)}, r={r} → 사실상 O(L * d^2)")

# --- 결론 ---
final_a, final_g = loss_a[-1], loss_g[-1]
diff_pct = ((final_g - final_a) / final_a) * 100
speed_ratio = time_a / time_g if time_g > 0 else float('inf')
param_ratio = np_a / np_g if np_g > 0 else float('inf')

print(f"\n{'=' * 64}")
print(f"  요약")
print(f"{'=' * 64}")
print(f"  Grassmann의 최종 loss가 Attention 대비 {diff_pct:+.1f}%")
print(f"  파라미터 수: Grassmann이 Attention의 {np_g/np_a*100:.0f}%")
print(f"  학습 속도: Grassmann이 Attention의 {time_g/time_a*100:.0f}%")
if abs(diff_pct) < 15:
    print(f"  --> 논문 주장과 일치: 비슷한 규모에서 Grassmann이")
    print(f"      Attention 대비 10-15% 이내의 성능을 달성 (L^2 없이!)")
print(f"{'=' * 64}")
