# CTC-KD 기존 연구 분석

> 주간 보고용 슬라이드 자료 — 기존 CTC KD의 문제점과 논문별 한계 정리

---

## 1. CTC-KD의 근본적인 문제

CTC는 일반적인 seq2seq와 달리, **동일한 발화에 대해 모델마다 다른 frame-level 정렬**을 가진다.  
이로 인해 teacher frame t의 분포를 student frame t'에 그대로 붙이는 것이 잘못된 supervision이 된다.

### 핵심 문제 3가지

| 문제 | 설명 | 언급 논문 |
|---|---|---|
| **Blank dominance** | CTC frame의 ~78%가 blank. Frame-level KD는 blank 분포 매칭에 지배됨 | Hilmes 2025 |
| **Temporal misalignment** | Teacher-student 간 CTC spike 위치가 다름. 같은 token이 다른 frame에서 발화됨 | Kurata 2019, Li 2025 |
| **No token-level structure** | Frame마다 독립적으로 KD → token 경계/정체성 정보 무시 | **(기존 논문 없음 — ours)** |

### 실측 Misalignment 수치 (dev_clean 256발화)

> 측정 도구: `scripts/diagnose_ctc_mismatch.py` — teacher: `stt_en_conformer_ctc_small`

**지표 설명**

| 지표 | 의미 |
|---|---|
| **Blank%** | Student argmax가 blank인 frame 비율 (teacher = 79.5%) |
| **Frame mismatch** | 전체 frame 중 teacher-student argmax가 다른 비율 |
| **NB→B miss** | Teacher non-blank frame 중 student가 blank 출력한 비율 |
| **NB spike error** | Teacher non-blank frame 중 student가 wrong output(blank 포함)한 비율 = NB→B + 다른 token |
| **Transition edit** | CTC transition sequence 간 edit distance (token boundary 불일치) |

> **NB spike error**가 핵심 지표: teacher가 token을 찍는 위치에서 student가 얼마나 틀리는지.  
> `NB→B miss` ⊂ `NB spike error` (miss는 blank로 찍은 것만, error는 다른 token도 포함)

**방법별 비교**

| 방법 | test_clean | test_other | Blank% | Frame mismatch | NB→B miss | **NB spike error** | Transition edit |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Teacher** (기준) | — | — | **79.5%** | — | — | — | — |
| No-KD | 14.97% | 34.23% | 78.1% | 24.2% | 51.7% | 59.8% | 14.6% |
| Vanilla Logit KD | 13.87% | 32.89% | 76.9% | 14.4% | 25.9% | 31.9% | 11.3% |
| KD-BE | 12.99% | 31.15% | 65.5% | 18.8% | 8.8% | 14.7% | 13.1% |
| Sym n=1 | 12.85% | 31.13% | 76.5% | **11.2%** | 17.0% | 22.7% | **9.6%** |
| Sym n=2 | 15.21%† | 33.93%† | 76.3% | 12.8% | 20.1% | 26.8% | 11.6% |
| Guide-CTC | 12.80% | 30.98% | 64.9% | 19.1% | 8.1% | 13.9% | 13.0% |
| Delayed-KD+BE | 13.68% | 32.28% | 75.7% | 31.5% | **63.0%** | **72.2%** | 12.1% |
| Self-KD l=4 | 13.06% | 31.90% | 78.0% | 23.3% | 50.1% | 56.6% | 11.4% |
| **Token-avg GT** (ours) | 12.72% | 31.01% | 63.7% | 19.7% | **6.4%** | **12.9%** | 12.8% |
| **Token-avg GT + Trans tw=5** (ours) | **12.57%** | 31.02% | — | — | — | — | — |
| **Token-avg GT + Trans tw=10** (ours) | 12.68% | **30.77%** | — | — | — | — | — |

† Sym n=2 조기 수렴 (epoch 50).

**주요 관찰**
- **NB spike error**: No-KD 59.8% → Token-avg GT **12.9%** (최저). Teacher가 token을 찍는 위치에서 student의 오류가 가장 적음. KD-BE(14.7%), Guide-CTC(13.9%)도 낮은 편.
- **Blank%**: KD-BE, Guide-CTC, Token-avg GT만 blank rate가 크게 감소(65~64%). 나머지는 No-KD(78.1%)와 거의 동일 — spike 위치 supervision이 없으면 blank 습관을 못 버림.
- **Delayed-KD**: NB spike error 72.2%로 No-KD(59.8%)보다 오히려 악화. Non-streaming 환경에 적용하면 spike를 더 억제하는 방향으로 수렴.
- **Self-KD**: NB spike error 56.6% ≈ No-KD — 외부 teacher 없이는 spike 위치 개선 없음.
- **Sym n=1 vs KD-BE**: KD-BE가 NB spike error(14.7%)는 낮지만 Sym n=1이 WER(12.85%) 더 좋음 → blank frame ±1 포함이 token identity 정보보다 temporal alignment 개선에 기여.

---

## 2. Vanilla Logit KD (baseline)

**방법**: Teacher softmax(T=2) → Student에 frame-level KL divergence

**문제**:
- 전체 frame에 동등하게 KD → blank frame이 loss를 지배
- Teacher-student 간 CTC alignment 불일치를 전혀 고려하지 않음
- Misaligned frame에 강제 supervision → 잘못된 gradient

**우리 실험 결과**: test_clean **13.87%**, test_other **32.89%** (No-KD 대비 −1.10 / −1.34)

---

## 3. Hilmes et al. 2025 — Blank Frame Selection

**논문 제목**: *"Blank Frame Selection for Efficient Knowledge Distillation in CTC-based ASR"* (추정)

**동기**: Blank frame이 KD를 방해한다 → blank를 선별적으로 제거/포함하자

### 방법들

| 방법 | 선택 기준 | 설명 |
|---|---|---|
| **KD-BE** | teacher argmax ≠ blank인 frame만 | Hard binary masking |
| **Sym n=1** | KD-BE + 각 non-blank frame 양쪽 ±1 frame | Boundary context 추가 |
| **Sym n=2** | KD-BE + ±2 frame | 더 넓은 context |
| **Trim** | 발화 내 첫~마지막 non-blank 사이 전체 | 발화 내부 blank 전체 포함 |
| **Threshold α** | p(blank) < α인 frame | Soft한 blank 선택 |
| **Random β** | Non-blank + β비율로 blank 랜덤 샘플링 | Stochastic 선택 |

**우리 실험 결과** (w=10, t=1):

| 방법 | test_clean | test_other | Δ clean |
|---|---|---|---|
| KD-BE | 12.99% | 31.15% | −1.98 |
| Sym n=1 | **12.85%** | **31.13%** | **−2.12** |
| Sym n=2 | 15.21%† | 33.93%† | +0.24 |

† Sym n=2 조기 수렴 (epoch 50). 논문 조건(λ=1.0, CTC loss 제거)과 다름.

### 한계점
- 여전히 **frame-level** supervision → token-level 구조 정보 없음
- Frame 선택 기준이 teacher **greedy argmax** 기반 → teacher 오류 전파
- Sym n=2처럼 blank를 많이 포함하면 오히려 성능 저하 (weight 조정 필요)
- Temporal misalignment 자체를 해결하지 않음 — 잘못된 frame에 KD가 들어가는 문제 잔존

---

## 4. Kurata & Audhkhasi 2019 — Guide-CTC

**논문 제목**: *"Guiding CTC Posterior Spikes for Improved Aligner-Free Sequence-to-Sequence Learning"*

**동기**: Teacher-student 간 CTC spike 위치가 다름 → student spike를 teacher spike 위치로 강제 유도

**방법**:
```
L_guide = -Σ_t  M_t · log P_student(t, argmax_teacher(t))

M_t = 1 if teacher argmax ≠ blank at frame t, else 0
```
- Teacher non-blank frame 위치에서 student가 teacher와 같은 token을 출력하도록 hard CE loss
- Soft KL(KD-BE) 대신 hard argmax target 사용

**KD-BE와의 차이**:
- KD-BE: Top-k soft distribution으로 KL
- Guide-CTC: 단일 argmax token으로 hard CE → spike timing 직접 최적화

**우리 실험 결과**: test_clean **12.80%**, test_other **30.98%** (No-KD 대비 −2.17 / −3.25)

### 한계점
- **Hard target** → teacher 예측의 불확실성(dark knowledge) 완전 무시
- Teacher argmax 오류 → student에 오류 전파 (pseudo-labeling과 동일한 문제)
- Temporal misalignment 해결 안 됨 — teacher frame t를 student frame t에 강제 적용
- Token-level 구조 없음

---

## 5. Li et al. 2025 — Delayed-KD (TAB)

**논문 제목**: *"Knowledge Distillation for Streaming CTC ASR"* (추정)

**원본 설정**: Streaming(chunk-based) student ← Full-context teacher. Student는 미래를 못 보므로 자연스럽게 spike가 지연됨.

**방법 (TAB: Token Alignment Boosting)**:
- Teacher frame t → student 측 대응 frame 구간 [s_t, s_t + d] 탐색 (d = TAB size)
- 해당 구간에서 KL divergence **최소** student frame 선택
- 강제로 잘못된 frame에 KD하는 대신, "가장 비슷한 frame"을 찾아서 supervision

**우리 적용** (non-streaming): 양쪽 다 full-context이나 spike timing mismatch는 여전히 존재함.  
TAB(d=2) = 40ms window → 논문의 최적 setting과 동일.

**우리 실험 결과**: test_clean **13.68%**, test_other **32.28%** (No-KD 대비 −1.29 / −1.95)  
> Non-streaming 적용 시 오히려 spike 억제 방향으로 수렴 → NB spike error 72.2%로 No-KD(59.8%)보다 악화

### 한계점
- **Local window 탐색** → mismatch가 크면(여러 frame 차이) 놓침
- Non-streaming 설정에서는 원래 동기(streaming delay)가 없어 효과가 제한적
- 여전히 frame-level KD → token-level 정보 없음
- TAB window 내 KL 최솟값 frame이 실제로 맞는 frame인지 보장 없음

---

## 6. Kim et al. 2024 — Self-KD

**논문 제목**: *"Self-Knowledge Distillation for CTC ASR"* (추정)

**동기**: 외부 teacher 없이 하나의 모델 내부에서 KD → alignment mismatch 자체를 제거

**방법**:
```
Teacher = 전체 encoder (8 layers) + CTC head
Student = 앞 l layer (intermediate) + 별도 CTC head (공유 가중치 사용)

Loss = (1-α)·L_CTC + α·(L_iCTC + L_SKD)
L_SKD = frame-level KL(intermediate output ‖ full output.detach())
```

- Teacher = Student가 같은 모델 → CTC alignment 완전 일치 → blank masking 불필요
- Intermediate layer가 상위 layer를 흉내내도록 학습

**우리 실험 결과**: test_clean **13.06%**, test_other **31.90%** (No-KD 대비 −1.91 / −2.33)  
> NB spike error 56.6% ≈ No-KD 59.8% — 외부 teacher 없이 spike 위치 개선 효과 없음

### 한계점
- **External teacher의 dark knowledge 없음** — teacher WER이 낮을수록 전이 불가
- Student capacity 증가 없음 (같은 모델 내부 사용)
- Intermediate layer가 얼마나 유용한지 layer depth에 민감
- CTC alignment 일치는 보장되나, token-level 구조는 여전히 없음
- 실제로 강력한 외부 teacher가 있을 때 적용 불가

---

## 7. 공통 한계 요약 (슬라이드 핵심)

| 논문 | Blank 문제 | Temporal misalignment | Token-level 구조 |
|---|:---:|:---:|:---:|
| Vanilla Logit KD | ❌ 미해결 | ❌ 미해결 | ❌ 없음 |
| Hilmes 2025 (KD-BE/Sym) | ✅ 부분 해결 | ❌ 미해결 | ❌ 없음 |
| Kurata 2019 (Guide-CTC) | ✅ 부분 해결 | ❌ 미해결 | ❌ 없음 |
| Li 2025 (Delayed-KD/TAB) | ✅ 부분 해결 | △ 부분 완화 | ❌ 없음 |
| Kim 2024 (Self-KD) | ✅ 불필요 (자체 정렬) | ✅ 완전 해결 (자체 정렬) | ❌ 없음 |
| **우리 (Token-avg KD)** | ✅ Viterbi alignment 기반 | ✅ Token segment 매핑 | ✅ Token-level avg |

**모든 기존 방법의 공통 한계**: **Token-level supervision 없음.**  
→ 어떤 frame이 어떤 token에 해당하는지 모르고 frame을 독립적으로 supervision.

---

## 8. 우리 방법의 포지셔닝

**관찰**: teacher-student CTC alignment mismatch의 핵심은 *frame-level이 아닌 token-level에서 정렬*해야 한다.

**우리 접근**:
1. Teacher posterior를 GT-Viterbi forced alignment로 token segment에 묶음
2. 각 segment 내 teacher 분포를 평균 → token identity 중심 분포 (top-8 renorm)
3. Student frame을 linear scaling으로 대응 token segment에 매핑
4. Token segment 단위로 CE loss

**GT-Viterbi의 필요성**: Teacher greedy로 alignment하면 teacher 오류가 그대로 KD target이 되어 pseudo-labeling과 동일. GT를 사용하면 teacher가 틀린 구간에서도 정답 token 위치에 teacher 분포를 anchor.

**결과 요약** (w=20):

| 방법 | test_clean | test_other | NB spike error |
|---|---:|---:|---:|
| Token-avg GT (base) | 12.72% | 31.01% | **12.9%** |
| Token-avg GT + Trans tw=5 | **12.57%** | 31.02% | — |
| Token-avg GT + Trans tw=10 | 12.68% | **30.77%** | — |
| Token-avg GT + Trans tw=20 | 13.12% | 31.31% | — |

→ **tw=5**: test_clean 최고 (12.57%). **tw=10**: test_other 최고 (30.77%). tw=20은 오히려 악화.  
→ Token-avg 단독 대비 transition 추가로 test_clean 기준 0.15%p 추가 개선.  
→ 전체 방법 중 **NB spike error 12.9% 최저** (Guide-CTC 13.9%, KD-BE 14.7% 대비).
