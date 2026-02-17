# micro_grassmann

**"Attention Is Not What You Need"** 논문([arXiv:2512.19428](https://arxiv.org/abs/2512.19428))의 순수 Python 구현.

[Andrej Karpathy의 microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)(~200줄, 외부 의존성 없음) 스타일을 따라, Grassmann Flow 기반 언어 모델을 순수 Python만으로 구현합니다.

## 핵심 아이디어

Transformer의 Multi-Head Attention을 **Grassmann 기하학**으로 대체:

```
Attention:   Q·K^T → softmax → 가중합(V)    O(L^2)
Grassmann:   Plucker 좌표 → 기하학적 인코딩    O(L)
```

두 토큰의 관계를 표현하는 방식이 다릅니다:
- **Attention**: 내적(dot product) → 스칼라 1개 (유사도)
- **Grassmann**: Plucker 좌표 → 벡터 C(r,2)개 (기하학적 관계)

## 실행

```bash
# 메인 구현 (주석 포함, 734줄)
python3 micro_grassmann.py

# 주석 없는 버전 (~130줄)
python3 micro_grassmann_clean.py

# Karpathy의 원본 microGPT
python3 micro_gpt.py
```

외부 의존성 없음. Python 3만 있으면 됩니다.
데이터셋(`input.txt`)은 첫 실행 시 자동 다운로드됩니다.

## 벤치마크

### 이름 생성 (3000 steps)

```
              파라미터    eval loss    비고
Attention     4,192       2.3090      microGPT 원본
Grassmann     3,840       2.3097      파라미터 8% 적음, 성능 동등
```

### 괄호 매칭 (시퀀스 길이 ~35, 1000 steps)

```
              eval loss    속도         유효 괄호 생성
Attention     0.5967       352ms/step   100%
Grassmann     0.6660       246ms/step   20%
```

괄호 매칭은 모든 이전 토큰을 봐야 하는 과제 — Attention이 유리.
Grassmann은 고정 window만 보므로 30% 빠르지만 장거리 의존성에 약함.

```bash
# 벤치마크 실행
python3 benchmark.py         # 이름 생성 비교
python3 benchmark_paren.py   # 괄호 매칭 비교
```

## 아키텍처

```
입력 토큰
  ↓
토큰 임베딩 + 위치 임베딩
  ↓
RMSNorm
  ↓
┌─── Causal Grassmann Layer ────────────────────┐
│  x ──W_red──→ z (차원 축소: d→r)              │
│                ├── Plucker(z, z_{t-1})  ─┐     │
│                ├── Plucker(z, z_{t-2})   ├→ 평균 → g (기하학 벡터)
│                ├── Plucker(z, z_{t-4})   │     │
│                ├── Plucker(z, z_{t-8})  ─┘     │
│                ...                             │
│  alpha = sigmoid(W_gate_h·x + W_gate_g·g)     │
│  output = alpha·x + (1-alpha)·g                │
└────────────────────────────────────────────────┘
  ↓ + 잔차 연결
FFN (d → 4d → ReLU → d)
  ↓ + 잔차 연결
출력 프로젝션 → logits
```

### 논문 대비 구현 상태

| 항목 | 논문 | 우리 구현 | 비고 |
|------|------|-----------|------|
| Plucker 좌표 | p_ij = z_i·z'_j - z_j·z'_i | 동일 | 핵심 연산 |
| Window | [1,2,4,8,12,16] | 동일 | |
| Gate | concat([h;g]) → W_gate | W_h·x + W_g·g | 수학적 동치 |
| 집약 | 단순 평균 | 동일 | |
| 혼합 | alpha*h + (1-alpha)*g | 동일 | |
| 정규화 | LayerNorm | RMSNorm | microGPT 스타일 |
| FFN 활성화 | GELU | ReLU | microGPT 스타일 |
| Bias | 있음 | 없음 | microGPT 스타일 |
| Layers | 6~12 | 1 | 순수 Python 제약 |

## 학습 자료

이 프로젝트에는 단계별 학습 자료가 포함되어 있습니다:

| 파일 | 내용 |
|------|------|
| `tutorial.py` | 14단계 튜토리얼 — Attention 기초부터 Grassmann까지 |
| `visualize_plucker.py` | Plucker 좌표 시각화 (matplotlib 필요) |
| `trace_g.py` | 기하학 벡터 g의 계산 과정을 숫자로 추적 |
| `trace_alpha_wred.py` | alpha 결정 과정과 차원 축소 원리 설명 |

```bash
# 튜토리얼 실행
python3 tutorial.py

# Plucker 시각화 (venv 필요)
python3 -m venv .venv && source .venv/bin/activate && pip install matplotlib
python3 visualize_plucker.py  # → plucker_explained.png 생성

# 숫자 추적
python3 trace_g.py
python3 trace_alpha_wred.py
```

## 파일 구조

```
.
├── micro_grassmann.py        # 메인 구현 (주석 포함)
├── micro_grassmann_clean.py  # 주석 없는 버전
├── micro_gpt.py              # Karpathy microGPT 원본
├── benchmark.py              # Attention vs Grassmann 이름 생성 비교
├── benchmark_paren.py        # 괄호 매칭 비교
├── tutorial.py               # 단계별 튜토리얼
├── visualize_plucker.py      # Plucker 좌표 시각화
├── trace_g.py                # g 벡터 추적
├── trace_alpha_wred.py       # alpha, W_red 설명
├── plucker_explained.png     # 시각화 결과
└── input.txt                 # 이름 데이터셋 (자동 다운로드)
```

## 참고

- 논문: [Attention Is Not What You Need (arXiv:2512.19428)](https://arxiv.org/abs/2512.19428)
- 원본 microGPT: [Karpathy's gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- PyTorch 구현: [Infatoshi/grassmann-flows](https://github.com/Infatoshi/grassmann-flows)
