"""
micro_grassmann.py
================================================================================
"Attention Is Not What You Need" 논문의 순수 Python 구현
(arXiv:2512.19428 — Grassmann Flows as an Attention-Free Alternative)

Andrej Karpathy의 microGPT 스타일을 따라, 외부 의존성 없이
순수 Python만으로 Grassmann Flow 기반 언어 모델을 구현합니다.

[핵심 아이디어]
  기존 Transformer:  Q·K^T → softmax → 가중합(V)  → O(L^2) 복잡도
  이 논문:           Plucker 좌표 → 기하학적 인코딩   → O(L) 복잡도

  "정보는 명시적인 쌍별 가중치가 아니라,
   저랭크 부분공간의 제어된 변형을 통해 시퀀스를 흐른다."
   — 논문 본문 중

[구조 비교]
  microGPT (Karpathy)          micro_grassmann (이 파일)
  ─────────────────────        ─────────────────────────
  Value autograd 엔진          (동일 — 재사용)
  토큰/위치 임베딩              (동일)
  Multi-Head Attention  ──→    Causal Grassmann Layer ★
    Q = W_q · x                  z = W_red · x (차원 축소)
    K = W_k · x                  Plucker(z_t, z_{t-delta})
    V = W_v · x                  g = W_plu · plucker
    softmax(QK^T/sqrt(d))·V      alpha = sigmoid(gate)
    W_o · attn_out                mix = alpha*x + (1-alpha)*g
  Feed-Forward Network          (동일)
  Adam Optimizer                (동일)

의존성: 없음 (순수 Python + math + random)
================================================================================
"""

import os
import math
import random

random.seed(42)


# ============================================================================
#  PART 1: Autograd 엔진 (자동 미분)
# ============================================================================
#
#  신경망 학습 = "파라미터를 조금씩 조정해서 loss를 줄이는 것"
#  이를 위해 "loss가 각 파라미터에 얼마나 민감한가?" (=gradient)를 알아야 합니다.
#
#  Autograd는 모든 수학 연산을 기록해두었다가, backward()를 호출하면
#  Chain Rule(연쇄법칙)을 써서 모든 파라미터의 gradient를 자동 계산합니다.
#
#  [Chain Rule 예시]
#    loss = f(g(x)) 일 때,
#    d(loss)/dx = d(loss)/d(g) * d(g)/dx
#    즉, "상위 gradient" x "국소 gradient"를 곱해 전파합니다.
#
#  Value 객체 하나 = 스칼라 값 하나를 감싸는 래퍼:
#    .data         → 실제 값 (forward pass에서 계산)
#    .grad         → 이 값에 대한 loss의 gradient (backward pass에서 계산)
#    ._children    → 이 값을 만들어낸 입력 Value들
#    ._local_grads → 각 입력에 대한 국소 gradient (chain rule용)
#
#  [예시] c = a * b 이면:
#    c.data = a.data * b.data
#    c._children = (a, b)
#    c._local_grads = (b.data, a.data)   ← d(a*b)/da = b, d(a*b)/db = a
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
        # d(a*b)/da = b, d(a*b)/db = a  (곱의 미분)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        # d(a^n)/da = n * a^(n-1)  (거듭제곱의 미분)
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        # d(ln a)/da = 1/a
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        # d(e^a)/da = e^a
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        # ReLU(x) = max(0, x)
        # 기울기: x > 0 이면 1, x <= 0 이면 0
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
        역전파(Backpropagation):
        계산 그래프를 역순으로 순회하며 모든 Value의 gradient를 계산합니다.

        1. 위상 정렬(topological sort)로 노드 순서를 정함
        2. loss 노드부터 시작 (grad = 1, 왜냐면 d(loss)/d(loss) = 1)
        3. 역순으로 각 노드를 방문하며 chain rule 적용
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

        self.grad = 1  # 시작점: d(loss)/d(loss) = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad  # Chain Rule: 국소gradient * 상위gradient


# ============================================================================
#  PART 2: 데이터 로딩 & 토크나이저
# ============================================================================
#
#  Karpathy의 microGPT와 동일한 데이터셋: 영어 이름 목록 (names.txt)
#  각 이름이 하나의 "문서"이고, 문자(character) 단위로 토큰화합니다.
#
#  토큰화 예시:
#    "henry" -> [BOS, h, e, n, r, y, BOS]
#
#  BOS(Beginning/End of Sequence) 토큰:
#    - 시퀀스의 시작과 끝을 나타냄
#    - 모델에게 "여기서 시작/끝" 이라고 알려주는 특수 토큰
# ============================================================================

if not os.path.exists('input.txt'):
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(url, 'input.txt')

docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"문서(이름) 수: {len(docs)}")

uchars = sorted(set(''.join(docs)))    # 전체 고유 문자 목록 (a~z 등)
BOS = len(uchars)                       # BOS 토큰의 ID = 마지막 인덱스
vocab_size = len(uchars) + 1            # 전체 어휘 크기 = 문자수 + BOS
print(f"어휘 크기: {vocab_size}")


# ============================================================================
#  PART 3: 하이퍼파라미터 — 여기서 Attention과 갈라집니다!
# ============================================================================
#
#  [microGPT의 Attention 관련 하이퍼파라미터]
#    n_head = 4       (어텐션 헤드 수)
#    head_dim = 4     (헤드당 차원)
#    → Q, K, V 행렬이 필요하고, L x L 크기의 어텐션 행렬을 계산
#
#  [microGrassmann의 Grassmann 관련 하이퍼파라미터]
#    r = 4            (차원 축소 목표 — 토큰 벡터를 이 크기로 압축)
#    plucker_dim = 6  (Plucker 좌표 차원 = r*(r-1)/2 = 4*3/2 = 6)
#    window = [1,2,4] (로컬 윈도우 오프셋)
#    → L x L 어텐션 행렬이 전혀 없음! 로컬 윈도우만 사용
#
#  [윈도우 오프셋이란?]
#    현재 토큰(위치 t)이 "몇 칸 뒤의 토큰"을 참조할지 결정합니다.
#    window = [1, 2, 4] 이면:
#      delta=1: 바로 이전 토큰 (t-1)과 비교  → 인접 문맥
#      delta=2: 두 칸 전 토큰 (t-2)과 비교   → 약간 넓은 문맥
#      delta=4: 네 칸 전 토큰 (t-4)과 비교   → 먼 문맥
#    이렇게 다중 스케일(multi-scale)로 시퀀스 정보를 포착합니다.
#    Attention은 모든 쌍을 보지만(O(L^2)), 여기선 정해진 오프셋만 봅니다(O(L)).
# ============================================================================

n_embd     = 16       # 임베딩 차원 (각 토큰을 나타내는 벡터의 크기)
n_layer    = 1        # Grassmann 레이어 수 (깊이)
block_size = 16       # 최대 시퀀스 길이
r          = 4        # ** Grassmann 축소 차원 (d=16 → r=4로 압축)
plucker_dim = r * (r - 1) // 2   # C(r,2) = 4*3/2 = 6  (Plucker 좌표 차원)
window     = [1, 2, 4, 8, 12, 16] # 로컬 윈도우 오프셋 집합 (논문과 동일)

print(f"임베딩 차원 d={n_embd}, 축소 차원 r={r}, "
      f"Plucker 차원={plucker_dim}, 윈도우={window}")


# ============================================================================
#  PART 4: 모델 파라미터 초기화
# ============================================================================
#
#  각 파라미터 행렬의 역할을 이해하는 것이 핵심입니다.
#
#  ┌─────────────────────────────────────────────────────────────────┐
#  │  파라미터        크기              역할                          │
#  ├─────────────────────────────────────────────────────────────────┤
#  │  wte         (vocab, d)     토큰 임베딩 (단어 -> 벡터)           │
#  │  wpe         (block, d)     위치 임베딩 (위치 -> 벡터)           │
#  │  lm_head     (vocab, d)     출력 프로젝션 (벡터 -> 단어 확률)     │
#  │                                                                 │
#  │  --- 아래가 Attention의 Q,K,V,O 행렬을 완전히 대체하는 것들 ---   │
#  │                                                                 │
#  │  W_red       (r, d)         차원 축소 (16차원 -> 4차원)          │
#  │  W_plu       (d, plucker)   Plucker 좌표를 모델 차원으로 복원     │
#  │  W_gate_h    (d, d)         게이트: 원본 경로의 가중치            │
#  │  W_gate_g    (d, d)         게이트: 기하학 경로의 가중치          │
#  │  mlp_fc1     (4d, d)        FFN 확장 (Transformer와 동일)       │
#  │  mlp_fc2     (d, 4d)        FFN 축소 (Transformer와 동일)       │
#  └─────────────────────────────────────────────────────────────────┘
#
#  [파라미터 수 비교]
#  microGPT Attention:  W_q + W_k + W_v + W_o = 4 * d * d = 4 * 16 * 16 = 1024
#  microGrassmann:      W_red + W_plu + W_gate_h + W_gate_g
#                       = r*d + d*C(r,2) + d*d + d*d
#                       = 4*16 + 16*6 + 16*16 + 16*16
#                       = 64 + 96 + 256 + 256 = 672
#  → Grassmann이 파라미터가 더 적으면서 기하학적 구조를 활용!
# ============================================================================

matrix = lambda nout, nin, std=0.08: \
    [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte':     matrix(vocab_size, n_embd),    # 토큰 임베딩 테이블
    'wpe':     matrix(block_size, n_embd),    # 위치 임베딩 테이블
    'lm_head': matrix(vocab_size, n_embd),    # 출력 프로젝션
}

for i in range(n_layer):
    # -- Grassmann 레이어 파라미터 (Attention을 대체!) --
    state_dict[f'L{i}.W_red']    = matrix(r, n_embd)           # 차원 축소
    state_dict[f'L{i}.W_plu']    = matrix(n_embd, plucker_dim) # Plucker -> d
    state_dict[f'L{i}.W_gate_h'] = matrix(n_embd, n_embd)      # 게이트 (원본)
    state_dict[f'L{i}.W_gate_g'] = matrix(n_embd, n_embd)      # 게이트 (기하학)
    # -- Feed-Forward Network (Transformer와 동일) --
    state_dict[f'L{i}.mlp_fc1']  = matrix(4 * n_embd, n_embd)  # d -> 4d
    state_dict[f'L{i}.mlp_fc2']  = matrix(n_embd, 4 * n_embd)  # 4d -> d

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"총 파라미터 수: {len(params)}")


# ============================================================================
#  PART 5: 유틸리티 함수들
# ============================================================================

def linear(x, w):
    """
    행렬-벡터 곱: y = W * x
    W의 각 행(row)과 x의 내적(dot product)을 계산합니다.

    예: W가 3x2, x가 [x0, x1] 이면
        y = [w00*x0 + w01*x1,
             w10*x0 + w11*x1,
             w20*x0 + w21*x1]  → 3차원 벡터
    """
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]


def softmax(logits):
    """
    소프트맥스: 실수 벡터를 확률 분포로 변환합니다.
    모든 값이 0~1 사이가 되고, 합이 1이 됩니다.

    수치 안정성을 위해 최댓값을 빼줍니다 (결과는 수학적으로 동일).
    softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
    """
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    """
    RMS Normalization (Root Mean Square Normalization):
    벡터의 크기(스케일)를 일정하게 맞춰줍니다.

    공식: x_norm = x / sqrt(mean(x^2) + epsilon)

    왜 필요한가?
      학습 중 벡터 값이 너무 크거나 작아지면 gradient가 폭발/소실합니다.
      정규화하면 학습이 안정적으로 진행됩니다.

    LayerNorm과의 차이:
      LayerNorm = 평균을 빼고 분산으로 나눔 (2번의 통계량 계산)
      RMSNorm   = RMS로만 나눔 (1번의 통계량 계산) → 더 간단하고 빠름
    """
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def sigmoid(x):
    """
    시그모이드 함수: 임의의 실수 -> (0, 1) 범위로 변환합니다.

    공식: sigma(x) = 1 / (1 + exp(-x))

    그래프 모양:
      x = -inf → 0에 가까움
      x = 0    → 0.5
      x = +inf → 1에 가까움

    게이트(gate)에서 "얼마나 통과시킬지" 비율을 결정하는 데 사용됩니다.
    alpha = sigmoid(score)  →  0이면 "차단", 1이면 "완전 통과"
    """
    return Value(1.0) / (Value(1.0) + (-x).exp())


# ============================================================================
#  PART 6: Plucker 좌표 계산 — 이 논문의 수학적 핵심!
# ============================================================================
#
#  [Attention은 어떻게 두 토큰의 관계를 표현하는가?]
#    Q * K^T = 스칼라 (하나의 숫자)
#    "이 두 토큰이 얼마나 관련 있는가?" → 유사도 점수 하나로 압축
#
#  [Grassmann은 어떻게 두 토큰의 관계를 표현하는가?]
#    Plucker(u, v) = C(r,2)차원의 벡터
#    "이 두 토큰이 어떤 2차원 평면을 형성하는가?" → 풍부한 기하학적 표현
#
#  ────────────────────────────────────────────────
#  Plucker 좌표의 직관적 설명:
#  ────────────────────────────────────────────────
#
#  3차원 공간에서 두 벡터 u, v가 있으면 하나의 평면을 정의합니다.
#  이 평면을 수학적으로 표현하는 것이 Plucker 좌표입니다.
#
#  r차원 공간에서 두 벡터 u, v가 만드는 2D 부분공간(평면)은
#  Grassmann 다양체 Gr(2,r) 위의 한 점에 대응됩니다.
#
#  Plucker 좌표는 이 점의 좌표계입니다:
#    p_ij = u_i * v_j  -  u_j * v_i     (모든 i < j 쌍에 대해)
#
#  이것은 수학적으로 "외적(wedge product)" u ^ v 의 좌표이며,
#  두 벡터가 이루는 평행사변형의 "부호 있는 넓이"를 각 좌표 평면에
#  사영한 것으로 볼 수 있습니다.
#
#  ────────────────────────────────────────────────
#  왜 내적(dot product)보다 나을 수 있는가?
#  ────────────────────────────────────────────────
#
#  내적: u . v = |u||v|cos(theta)
#    → 각도 정보만 남음 (1차원 스칼라)
#
#  외적/Plucker: u ^ v
#    → 각도 + 방향 + 면적 정보 모두 보존 (C(r,2)차원 벡터)
#    → 두 벡터의 관계를 훨씬 풍부하게 표현
#
#  ────────────────────────────────────────────────
#  구체적 계산 예시 (r=4):
#  ────────────────────────────────────────────────
#
#  u = [u0, u1, u2, u3]
#  v = [v0, v1, v2, v3]
#
#  Plucker 좌표 (i < j인 모든 쌍):
#    p_01 = u0*v1 - u1*v0    ← (i=0, j=1) 평면에 사영된 넓이
#    p_02 = u0*v2 - u2*v0    ← (i=0, j=2)
#    p_03 = u0*v3 - u3*v0    ← (i=0, j=3)
#    p_12 = u1*v2 - u2*v1    ← (i=1, j=2)
#    p_13 = u1*v3 - u3*v1    ← (i=1, j=3)
#    p_23 = u2*v3 - u3*v2    ← (i=2, j=3)
#
#  → 6개의 값으로 구성된 벡터 (C(4,2) = 6)
# ============================================================================

def compute_plucker(u, v):
    """
    두 r차원 벡터 u, v의 Plucker 좌표를 계산합니다.

    Args:
        u: r차원 Value 리스트 (현재 토큰의 축소 벡터)
        v: r차원 Value 리스트 (이전 토큰의 축소 벡터)

    Returns:
        C(r,2)차원 Value 리스트 (Plucker 좌표)
    """
    coords = []
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            # p_ij = u_i * v_j  -  u_j * v_i
            # 이것이 외적(wedge product)의 각 성분입니다
            coords.append(u[i] * v[j] - u[j] * v[i])
    return coords


# ============================================================================
#  PART 7: 모델 Forward Pass — Causal Grassmann Layer
# ============================================================================
#
#  한 토큰이 들어오면 다음 토큰을 예측하는 전체 과정입니다.
#
#  ┌─────────────────────────────────────────────────┐
#  │  입력: token_id (현재 토큰), pos_id (위치)       │
#  │                                                  │
#  │  1. 임베딩: 토큰 + 위치 → 벡터                    │
#  │  2. RMS 정규화                                    │
#  │  3. *** Causal Grassmann Layer ***                │
#  │     a. 차원 축소:  x(16차원) → z(4차원)           │
#  │     b. z를 캐시에 저장                             │
#  │     c. 이전 토큰들과 Plucker 좌표 계산             │
#  │     d. Plucker를 모델 차원으로 복원                 │
#  │     e. 여러 오프셋의 결과를 평균                    │
#  │     f. 게이트로 원본/기하학 정보 혼합               │
#  │  4. 잔차 연결                                      │
#  │  5. FFN (Feed-Forward Network)                    │
#  │  6. 출력: 벡터 → 어휘 확률                         │
#  │                                                    │
#  │  출력: logits (vocab_size 차원)                    │
#  └─────────────────────────────────────────────────┘
#
#  [Causal(인과적) 제약이란?]
#    언어 모델은 "미래를 볼 수 없어야" 합니다.
#    "hel" 다음에 올 글자를 예측할 때, "hello"의 'l', 'o'를 미리 보면 안 됩니다.
#    Attention은 이를 위해 마스킹(masking)을 사용합니다.
#    Grassmann은 윈도우 오프셋이 양수(delta > 0)이므로 자연스럽게 보장됩니다.
#    → delta=1이면 t-1만, delta=2이면 t-2만 참조 (항상 과거만 봄)
# ============================================================================

def forward(token_id, pos_id, z_cache):
    """
    한 토큰에 대한 forward pass.

    Args:
        token_id: 현재 입력 토큰의 ID (정수)
        pos_id:   현재 위치 인덱스 (0부터 시작)
        z_cache:  이전 토큰들의 축소 벡터를 저장하는 캐시.
                  z_cache[layer_idx] = [z_0, z_1, ..., z_{t-1}]
                  Attention에서의 Key/Value 캐시와 비슷한 역할입니다.

    Returns:
        logits: vocab_size 차원의 Value 리스트 (다음 토큰 예측 점수)
    """

    # ── Step 1: 임베딩 ──
    # 토큰 ID와 위치 ID를 각각 벡터로 변환하고 더합니다.
    # 이렇게 하면 "위치 3에 있는 문자 'h'" 라는 의미의 벡터가 됩니다.
    tok_emb = state_dict['wte'][token_id]    # 토큰 임베딩: vocab -> R^d
    pos_emb = state_dict['wpe'][pos_id]      # 위치 임베딩: position -> R^d
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    # ── Step 2: Grassmann 레이어 반복 ──
    for li in range(n_layer):
        x_res = x      # 잔차 연결을 위해 현재 상태 저장
        x = rmsnorm(x)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  (A) 차원 축소 (Linear Reduction)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # x (16차원) → z (4차원)
        #
        # 왜 차원을 줄이는가?
        #   Plucker 좌표의 차원 = C(r, 2) = r*(r-1)/2
        #   r=16이면 C(16,2)=120... 너무 크고 계산 비용이 큼
        #   r=4이면 C(4,2)=6 으로 관리 가능
        #
        # Attention과의 비교:
        #   Attention: x → Q (via W_q), x → K (via W_k) → 두 개의 프로젝션
        #   Grassmann: x → z (via W_red) → 하나의 프로젝션으로 충분!
        #   (Q,K 대신 하나의 z에서 Plucker로 관계를 추출하므로)
        z = linear(x, state_dict[f'L{li}.W_red'])

        # 현재 위치의 z를 캐시에 추가 (이후 토큰들이 참조할 수 있도록)
        z_cache[li].append(z)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  (B) 다중 스케일 Plucker 인코딩 (논문의 핵심 연산!)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 현재 토큰의 z와 이전 토큰들의 z를 쌍으로 묶어
        # 각 쌍의 Plucker 좌표를 계산합니다.
        #
        # 예: 현재 위치 t=5, window=[1,2,4] 이면
        #   delta=1: Plucker(z_5, z_4) → "바로 전 토큰과의 기하학적 관계"
        #   delta=2: Plucker(z_5, z_3) → "두 칸 전 토큰과의 기하학적 관계"
        #   delta=4: Plucker(z_5, z_1) → "네 칸 전 토큰과의 기하학적 관계"
        #
        # Attention과의 비교:
        #   Attention: 모든 이전 토큰과의 Q*K 유사도 계산 → O(L) per token
        #   Grassmann: 고정된 수의 오프셋만 계산           → O(|window|) per token
        #   |window|는 상수이므로 전체 시퀀스에 대해 O(L) vs Attention의 O(L^2)

        geo_features = []

        for delta in window:
            prev_pos = pos_id - delta
            if prev_pos < 0:
                # 범위 밖이면 건너뜀 (예: 위치 0에서 delta=1이면 위치 -1 → 없음)
                continue

            z_prev = z_cache[li][prev_pos]   # delta만큼 이전 토큰의 축소 벡터

            # --- Plucker 좌표 계산 ---
            plucker = compute_plucker(z, z_prev)

            # --- 정규화 ---
            # Plucker 좌표는 Grassmann 다양체 위의 "사영 좌표"이므로
            # 스케일(크기)은 중요하지 않고 방향만 중요합니다.
            # 정규화하면 수치적으로도 안정적입니다.
            norm_sq = sum(p * p for p in plucker)
            norm_inv = (norm_sq + Value(1e-8)) ** -0.5
            plucker = [p * norm_inv for p in plucker]

            # --- 모델 차원으로 프로젝션 ---
            # 6차원 Plucker 좌표 → 16차원 모델 공간으로 되돌림
            # 이 프로젝션이 "기하학적 관계"를 모델이 이해할 수 있는 표현으로 변환
            g_delta = linear(plucker, state_dict[f'L{li}.W_plu'])
            geo_features.append(g_delta)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  (C) 기하학 Feature 집약 (Aggregation)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 여러 오프셋에서 온 기하학 feature들을 평균냅니다.
        #
        # delta=1에서 온 feature: "인접 문맥 정보"
        # delta=2에서 온 feature: "약간 넓은 문맥 정보"
        # delta=4에서 온 feature: "먼 문맥 정보"
        # → 이들의 평균 = "다양한 스케일의 문맥을 종합한 정보"
        #
        # Attention과의 비교:
        #   Attention: softmax 가중 평균 (attention weight로 V를 혼합)
        #   Grassmann: 단순 평균 (각 스케일의 기하학 정보를 균등 혼합)

        if geo_features:
            nf = len(geo_features)
            g = [sum(geo_features[k][j] for k in range(nf)) / nf
                 for j in range(n_embd)]
        else:
            # 위치 0에서는 참조할 이전 토큰이 없으므로 영벡터 사용
            # 게이트가 이 경우 원본(x)을 그대로 통과시키도록 학습됩니다
            g = [Value(0.0) for _ in range(n_embd)]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  (D) 게이트 기반 혼합 (Gated Mixing)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # "원본 정보(x)와 기하학 정보(g)를 얼마나 섞을까?"
        # 이 결정을 학습 가능한 게이트(gate)가 차원별로 내립니다.
        #
        # alpha_j = sigmoid(W_gate_h * x + W_gate_g * g)_j
        #
        # alpha_j 가 1에 가까우면: j번째 차원은 원본 x를 유지
        #   → "이 차원의 정보는 기하학 없이도 충분해"
        #
        # alpha_j 가 0에 가까우면: j번째 차원은 기하학 g를 사용
        #   → "이 차원은 이전 토큰과의 관계 정보가 중요해"
        #
        # 최종 출력: x_mixed = alpha * x + (1-alpha) * g
        #
        # Attention과의 비교:
        #   Attention: 단일 출력 (attention weighted sum)
        #   Grassmann: 게이트가 원본과 기하학을 "차원별로" 혼합
        #              → 더 세밀한 제어가 가능

        gate_h = linear(x, state_dict[f'L{li}.W_gate_h'])
        gate_g = linear(g, state_dict[f'L{li}.W_gate_g'])
        alpha = [sigmoid(h_j + g_j) for h_j, g_j in zip(gate_h, gate_g)]

        x = [a * xi + (Value(1.0) - a) * gi
             for a, xi, gi in zip(alpha, x, g)]

        # 잔차 연결 (Residual Connection)
        # 원래 입력을 더해줌 → gradient가 깊은 레이어까지 잘 흐르도록
        x = [a + b for a, b in zip(x, x_res)]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  (E) Feed-Forward Network (Transformer와 동일)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FFN은 각 토큰을 독립적으로 변환하는 단계입니다.
        # 구조: d → 4d (확장) → ReLU (비선형) → 4d → d (축소)
        #
        # Grassmann 레이어가 "토큰 간 관계"를 처리했다면,
        # FFN은 "토큰 내부의 표현"을 정제하는 역할입니다.
        # (이 부분은 원래 Transformer와 완전히 동일합니다)
        x_res = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'L{li}.mlp_fc1'])    # d → 4d 확장
        x = [xi.relu() for xi in x]                      # 비선형 활성화
        x = linear(x, state_dict[f'L{li}.mlp_fc2'])    # 4d → d 축소
        x = [a + b for a, b in zip(x, x_res)]            # 잔차 연결

    # ── Step 3: 출력 프로젝션 ──
    # 16차원 히든 벡터를 vocab_size 차원으로 변환합니다.
    # 이 값(logits)에 softmax를 적용하면 "다음 토큰이 각 문자일 확률"이 됩니다.
    return linear(x, state_dict['lm_head'])


# ============================================================================
#  PART 8: 학습 루프
# ============================================================================
#
#  [학습의 목표]
#    "이전 문자들이 주어졌을 때 다음 문자를 정확히 예측하는 것"
#
#  예시:  이름 "emma" 학습 과정
#    입력 BOS → 정답 'e'  (이름의 첫 글자 예측)
#    입력 'e' → 정답 'm'  (두 번째 글자 예측)
#    입력 'm' → 정답 'm'  (세 번째 글자 예측)
#    입력 'm' → 정답 'a'  (네 번째 글자 예측)
#    입력 'a' → 정답 BOS  (이름의 끝 예측)
#
#  [학습 단계]
#    1. 이름 하나를 토큰화
#    2. 각 위치에서 forward pass → 다음 토큰 예측
#    3. Cross-entropy loss 계산 (예측이 정답과 얼마나 다른지)
#    4. backward() → 모든 파라미터의 gradient 계산
#    5. Adam optimizer로 파라미터 업데이트
#
#  [Adam Optimizer]
#    단순 SGD보다 빠르고 안정적인 최적화 알고리즘:
#    - m (1차 모멘트): gradient의 이동 평균 → "방향"을 부드럽게
#    - v (2차 모멘트): gradient^2의 이동 평균 → "스케일"을 자동 조절
#    - learning rate warmup/decay: 학습 초반엔 빠르게, 후반엔 천천히
# ============================================================================

lr, beta1, beta2, eps = 0.01, 0.85, 0.99, 1e-8
m_buf = [0.0] * len(params)    # Adam 1차 모멘트
v_buf = [0.0] * len(params)    # Adam 2차 모멘트
num_steps = 1000

print(f"\n{'=' * 60}")
print(f"  microGrassmann 학습 시작")
print(f"  Attention 없이 Grassmann 기하학으로 시퀀스 모델링!")
print(f"  Plucker 좌표가 Q*K 내적을 대체합니다.")
print(f"{'=' * 60}\n")

for step in range(num_steps):
    # --- 데이터 준비 ---
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # z_cache: Attention의 KV-cache에 대응하는 "축소 벡터 캐시"
    z_cache = [[] for _ in range(n_layer)]
    losses = []

    # --- Forward pass: 각 위치에서 다음 토큰 예측 ---
    for pos_id in range(n):
        token_id  = tokens[pos_id]
        target_id = tokens[pos_id + 1]

        logits = forward(token_id, pos_id, z_cache)
        probs  = softmax(logits)

        # Cross-entropy loss: -log(정답 토큰의 확률)
        # 확률이 1에 가까우면 loss ≈ 0 (잘 예측함)
        # 확률이 0에 가까우면 loss → 큰 값 (잘못 예측함)
        losses.append(-probs[target_id].log())

    loss = (1 / n) * sum(losses)    # 평균 loss
    loss.backward()                  # 역전파 → gradient 계산

    # --- Adam optimizer 업데이트 ---
    lr_t = lr * (1 - step / num_steps)    # 선형 학습률 감쇠
    for i, p in enumerate(params):
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_buf[i] / (1 - beta1 ** (step + 1))   # 편향 보정
        v_hat = v_buf[i] / (1 - beta2 ** (step + 1))   # 편향 보정
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps)
        p.grad = 0    # gradient 초기화 (다음 step을 위해)

    if step % 100 == 0 or step == num_steps - 1:
        print(f"  step {step + 1:4d}/{num_steps} | loss {loss.data:.4f}")


# ============================================================================
#  PART 9: 추론 — 학습된 모델로 새로운 이름 생성
# ============================================================================
#
#  학습이 끝났으니, 모델이 "본 적 없는 새로운 이름"을 만들어냅니다.
#
#  [생성 과정 (Autoregressive Sampling)]
#    1. BOS 토큰으로 시작
#    2. 모델이 다음 토큰의 확률 분포를 출력
#    3. 확률에 따라 토큰 하나를 샘플링
#    4. 샘플링된 토큰을 입력으로 넣고 2번 반복
#    5. BOS(=끝) 토큰이 나오면 종료
#
#  [Temperature]
#    temperature가 낮으면 (< 1): 확률이 높은 토큰을 더 선호 → 안전한 이름
#    temperature가 높으면 (> 1): 확률이 균등해짐 → 창의적(이상한) 이름
#    temperature = 0.5: 약간 보수적으로 생성
#
#  핵심: 이 모든 생성이 Attention 없이, 순수하게 Grassmann 기하학만으로!
# ============================================================================

temperature = 0.5

print(f"\n{'=' * 60}")
print(f"  추론: Grassmann 모델이 생성한 새로운 이름들")
print(f"  (Attention 없이, Plucker 좌표의 기하학적 흐름만으로)")
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

print(f"\n  완료! Attention 없이 Grassmann 기하학만으로 이름을 생성했습니다.")
print(f"  논문: https://arxiv.org/abs/2512.19428")
print(f"  참고: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95")
