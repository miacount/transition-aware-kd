# ATD 검증 실험 (2026-07-21)

**ATD (Alignment-Tolerant Distillation)** — teacher의 blank-억제 occupancy(δ=6)를 KD 타깃의
support(span)로 쓰는 방법. 본 문서는 세 가지 검증 질문에 대한 **figure 2개 + 표**다.

셋업: test-clean, teacher `stt_en_conformer_ctc_small`, student Conformer-CTC d144×8L(~4.8M).
**δ는 6으로 고정**한다(§Q1 표의 δ 열은 그 선택의 근거로만 제시).

| 산출물 | 내용 |
|---|---|
| **`figures/fig1_atd_vs_mfa.png`** | (a) blank-suppressed posterior vs (b) MFA 강제정렬 — 같은 발화, frame축 공유 |
| **`figures/fig2_atd_misalignment.png`** | (a) teacher raw posterior / (b) student raw posterior / (c) ATD span |
| `analysis/mfa_validation/{summary,profile}.json`, `analysis/spike_coverage/summary.json` | 원자료 |

두 figure 모두 **같은 발화**(`1188-133604-0014`, "do not therefore think that the gothic school is an easy one")를 쓴다.
x축은 전부 **teacher encoder frame**(4× subsampling ≈ 40 ms)이다.

---

## Q1. ATD의 넓어진 구간은 음향학적으로 의미가 있는가 — MFA 대조

### → `figures/fig1_atd_vs_mfa.png`

2단 구성, frame축 공유, y축은 양쪽 다 **BPE 토큰**이라 행 단위로 직접 대조된다.

- **(a) Blank-suppressed teacher posterior** — `p̃(y_u | t)`, δ=6. FB를 거치기 전의 **원 증거**다.
  컬러바 = p̃(y_u|t).
- **(b) MFA 강제정렬** — 같은 발화의 실제 발화 구간. 음영 = 단어 구간, 세로 눈금 = 음소 경계
  (레이블은 난잡해서 뺐다).

**읽는 법**: (a)에서 blank를 누르자 스파이크 양옆으로 살아난 질량이, (b)의 **자기 단어 구간 안에**
들어간다. 이 발화는 96.3%, δ=6이 **새로 모집하는 질량**만 따지면 test-clean 전체에서 **98.7%**가
자기 단어 구간(±1프레임) 안이다.
→ blank 억제로 살아난 질량은 아무 데나 퍼진 게 아니라 **그 토큰이 실제로 발음되는 구간**에 있다.

### 표 1. "그냥 넓힌 것"과의 구별 — 폭을 맞춘 대칭 kernel 대조군

가장 중요한 대조군은 **kernel-ctl**: δ=6과 **평균 폭을 정확히 맞춘**(PR width 1.441, Gaussian σ=0.454)
대칭 smoothing이다. 폭이 같으므로 차이는 전부 *어느 프레임을 고르는가*에서 나온다.

| 진단 지표 | δ=0 | δ=3 | **δ=6 (ATD)** | δ=9 | δ=12 | kernel-ctl<br>(δ=6과 같은 폭) |
|---|---:|---:|---:|---:|---:|---:|
| Occupancy 폭 (frames) | 1.08 | 1.16 | **1.44** | 2.31 | 3.80 | 1.44 |
| ① 편입 질량의 자기-토큰 순도 ↑ | — | 0.87 | **0.68** | 0.50 | 0.35 | **0.32** |
| ② 편입 질량의 MFA 단어구간 내 비율 ↑ | — | 0.99 | **0.99** | 0.97 | 0.92 | 0.98 |
| ③ 편입 질량의 비대칭 (frames)<br><sub>0 = 기계적 대칭 확산</sub> | — | +0.24 | **+0.24** | +0.27 | +0.35 | 0.00 |
| ④ 이웃 토큰 오염 ↓ | — | 0.08 | **0.07** | 0.07 | 0.06 | **0.21** |
| ⑤ WHAT top-k 보존 (self share) ↑ | 0.978 | 0.978 | **0.977** | 0.974 | 0.963 | 0.972 |
| MFA 단어구간 coverage (±1프레임) | 0.996 | 0.996 | **0.995** | 0.985 | 0.954 | 0.994 |
| (참고) 학습 WER clean/other | — | 미학습 | **12.67/30.64** | 12.86/30.93 | 중단(val 14.31) | 미학습 |

> 지표 정의: ① 프레임 t가 토큰 u에 새로 모집됨(occupancy 증가 >0.01)일 때, 그 프레임 posterior에서
> blank를 제거·정규화한 뒤 y_u가 차지하는 비율을 모집질량으로 가중평균. ② 같은 모집 질량 중 MFA
> 단어 구간(±1프레임) 안의 비율. ③ 편입 질량 중심 − δ=0 스파이크 중심. ④ 편입 질량으로 가중평균한
> non-blank posterior 중 인접 transcript 토큰 비율. ⑤ γ-가중 WHAT top-k에서 y_u 자신의 비율.

**세 가지 논점**

1. **폭이 같아도 내용이 다르다 (①④③의 kernel 열).** 같은 폭 1.44에서 ATD의 편입 질량은 68%가
   그 토큰 자신의 숨은 확률인데, 기계적 smear는 32%(δ=12 수준)이고 이웃 오염은 3배(0.07 → 0.21)다.
   → 본질은 "얼마나 넓히나"가 아니라 **"어느 프레임을 고르나"**이고, 그 선택은 transcript-제약
   forward-backward가 한다. 비대칭 +0.24 vs kernel 0.00(구성상)이 그 증거 — **방향을 데이터가 정했다.**
2. **δ=6이 상한인 이유.** 복원 가능한 shoulder는 스파이크 ±1프레임에만 존재한다. 오프셋별로 잰
   잔여 identity r(y_u)는 −1에서 0.29, +1에서 0.41이지만 **±2에서 0.06으로 붕괴**한다.
   δ를 더 키우면 복원할 정보가 없는 프레임에 질량을 강제하게 되고, 실제 WER도 그 순서를 따른다
   (12.67 → 12.86 → 발산). **δ=6 고정의 근거.**
3. **⑤: dark knowledge는 불변.** blank만 억제해 non-blank 비율이 보존되므로 WHAT 타깃은 δ와 무관하다
   (0.978 → 0.977). shoulder 프레임의 WHAT 기여는 (1−p_blank) 가중 때문에 0.8%뿐 —
   **ATD의 이득은 타깃 내용이 아니라 정렬-관대한 WHERE에서 온다** (Q3가 직접 측정).

재현: `python3 scripts/fig1_atd_vs_mfa.py`,
`python3 scripts/analyze_mfa_occupancy.py --alignments data/mfa/test_clean_alignments.json --ctl-sigma 0.454`,
`python3 scripts/analyze_depeak_profile.py`

---

## Q2. 그럼 왜 MFA를 그냥 쓰지 않는가

**답: 써봤고 졌다. 그리고 진 이유가 "정렬기가 약해서"가 아니라 "외부 timing이라는 구조" 자체다.**
MFA는 **타깃 생성에는 쓰지 않고 검증 기준으로만 쓴다** — 그게 Q1과 Figure 1의 역할이다.

### 근거 1 — WHERE만 바꾼 통제 실험 (WHAT·손실·아키텍처·w=20·100ep 전부 동일)

| WHERE 좌표계 | test-clean | test-other | blank% | frames/tok | teacher와 spike offset | KD loss |
|---|---:|---:|---:|---:|---:|---:|
| **teacher 자신 (δ=6, ATD)** | **12.67** | **30.64** | 0.657 | 2.01 | **0.24** | **1.47** |
| teacher posterior (FB 제거) | 13.02 | 31.51 | — | — | — | 1.34 |
| 외부 시계 (MFA 단어 box) | 14.50 | 32.52 | 0.446 | 3.32 | 0.56 | 2.14 |
| student 자신 (dual, detached) | 15.30 | 33.43 | 0.430 | 3.37 | 0.80 | 1.58 |
| (no-KD 기준점) | 15.12 | 33.94 | 0.804 | 1.17 | 0.69 | — |

teacher의 sequential alignment에서 멀어질수록 **단조 악화**한다. MFA는 no-KD 대비 이득이 0.62뿐이고,
frame KD 최고치(13.41)에도 못 미친다.

### 근거 2 — 실패 원인은 정렬기 품질이 아니다

- MFA는 최강 외부 정렬기(단어 containment 99.5%)인데 **자체 학습한 약한 phoneme aligner(14.41/33.52)와
  사실상 동급**(14.50/32.52). → "정렬기가 약해서 졌다"는 반론이 봉쇄된다.
- Student는 MFA를 무시한 게 아니라 **성실히 복종했다**: blank 80→45%, 토큰당 3.32프레임,
  MFA 구간 내 질량 97.3%(전 모델 중 최고). 실패는 "타깃 무시"가 아니라 **"타깃을 따른 결과"**다.

### 근거 3 — 메커니즘: 지식과 시계가 분리되면 지식이 희석된다

- teacher와의 spike offset이 0.24 → **0.56**으로 후퇴(no-KD 0.69 쪽으로). dark knowledge를
  **teacher가 그 토큰을 말하지 않는 시점**에서 읽는다.
- KD loss가 끝까지 수렴하지 않는다(2.14 vs 1.47, +46%). CTC loss도 78.3 vs 75.2로 동반 악화 —
  균일 box는 CTC가 선호하는 정렬과 상시 충돌한다.
- **시간 스케일이 다르다**: MFA 단어 box 폭 4.62프레임 vs ATD support 1.44프레임.
  Figure 1을 보면 (b)의 단어 구간은 (a)의 support보다 3배 넓다. 그 폭으로 WHAT을 평균하면
  이웃 음소·묵음 프레임이 섞인다.
- **단위가 어긋난다**: MFA는 *단어/음소* 구간을 주는데 타깃은 *BPE 토큰* 단위다. 글자 수 비례 분할이라는
  근사가 강제로 들어간다(teacher occupancy엔 이 근사가 없다). Figure 1 (b)에서 `·go/th/ic`
  세 토큰이 "gothic" 한 구간을 통째로 공유하는 것이 그 문제다 — (a)에서는 셋이 각자 다른
  프레임에 앉아 있다.
- De-peaking 자체가 이득이 아님의 재확인: MFA student가 가장 덜 peaky(44.6%)인데 WER는 1.8 나쁘다.

### 근거 4 — 실용성

ATD support는 **teacher forward 1회**로 나온다. MFA는 발음사전 + 음향모델 + 정렬 파이프라인이 필요하고
언어·도메인을 바꿀 때마다(예: TEDLIUM) 다시 갖춰야 한다. 성능이 동률이어도 선택할 이유가 없는데,
성능도 1.8 나쁘다.

재현: `scripts/build_mfa_span_targets.py`, `analysis/eval_span-kd-mfa-support-w20.txt`

---

## Q3. misalignment는 ATD span 안에 들어오는가

### → `figures/fig2_atd_misalignment.png`

3단 구성, frame축 공유.

- **(a) Teacher raw posterior** — y축 1−p(blank). teacher가 각 토큰을 찍는 위치.
- **(b) Student raw posterior (no-KD)** — 같은 축. **점선 = teacher의 스파이크 위치**라서
  프레임 단위 어긋남이 바로 읽힌다. 이 발화 mean |offset| = 1.06프레임 (16토큰 중 12개가 어긋남).
- **(c) ATD span** — teacher occupancy γ(δ=6), 즉 KD 타깃의 support. y축 BPE 토큰,
  외곽선 = span 구간의 범위.

**읽는 법**: (a)와 (b)를 겹쳐 보면 student가 거의 모든 토큰을 다른 프레임에서 방출한다.
(c)의 span은 그 드리프트를 흡수할 폭을 갖는다 — 이 발화는 student 스파이크 **9/16**이 span 안이고,
δ=0(frame-KD의 암묵적 support)이었다면 **4/16**뿐이다 (2.25×).

### 표 2. no-KD student — 어긋남 분포와 span coverage (test-clean 200발화, 8,237토큰)

각 모델 **자신의** posterior에 transcript-제약 FB를 걸어 토큰별 스파이크를 잡는다.
`t_T(u)=argmax_t γ_teacher(t,u)`, `t_S(u)=argmax_t γ_student(t,u)`, `offset = t_S − t_T`.
coverage는 teacher 타깃 support가 **student가 실제로 쓴 프레임**에 얹어 준 질량
`cov_δ(u)=γ_δ^teacher(t_S(u),u)`, hit은 `cov_δ > 0.01`.

mean |offset| = **0.69**프레임, 정확히 일치 43.4%.

| \|offset\| | 토큰 | 비중 | δ=0<br><sub>질량 / hit%</sub> | δ=3 | **δ=6 (ATD)** | δ=9 | δ=12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3,577 | 43.4% | 0.967 / 100.0 | 0.936 / 100.0 | **0.834 / 100.0** | 0.615 / 100.0 | 0.404 / 99.6 |
| **1** | **3,818** | **46.4%** | 0.025 / **10.5** | 0.046 / 27.3 | **0.103 / 62.1** | 0.193 / 83.8 | 0.223 / 90.5 |
| 2 | 717 | 8.7% | 0.001 / 1.4 | 0.003 / 2.9 | 0.008 / 8.9 | 0.031 / 32.8 | 0.067 / 57.5 |
| 3 | 74 | 0.9% | 0.000 / 0.0 | 0.001 / 2.7 | 0.007 / 5.4 | 0.014 / 14.9 | 0.044 / 39.2 |
| ≥4 | 51 | 0.6% | 0.000 / 0.0 | 0.000 / 0.0 | 0.000 / 2.0 | 0.002 / 2.0 | 0.006 / 11.8 |
| **전체** | **8,237** | 100% | 0.432 / **48.4** | 0.428 / 56.4 | **0.411 / 73.0** | 0.360 / 85.3 | 0.285 / 90.6 |

**세 가지 논점**

1. **P2는 실재하며 ±1이 전부다.** 토큰의 **56.6%**가 teacher와 다른 프레임에서 발화하고,
   그중 **82%가 정확히 1프레임**이다(≥2는 10%뿐).
   → 문제의 크기가 Q1에서 잰 "복원 가능한 shoulder의 크기(±1)"와 **정확히 일치한다.**
   우연이 아니라 둘 다 CTC 스파이크의 프레임 양자화 오차라는 같은 현상이다.
2. **ATD가 사는 지점.** ±1 토큰 coverage 10.5% → **62.1% (5.9×)**, 전체로는 48.4% → **73.0%**.
   frame-KD는 어긋난 토큰의 약 90%에서 **타깃 질량이 0인 프레임과 student를 비교**한다 —
   "P2가 잘못된 gradient를 준다"는 주장의 직접 측정치다.
3. **δ=9/12가 coverage는 높은데 WER은 나쁜 이유.** coverage만 보면 δ=12(90.6%)가 최고지만,
   그 추가분은 Q1 표 ①에 따라 **identity가 없는 프레임**(±2 이상, r(y_u)=0.06)에서 산 것이다.
   **coverage는 δ와 함께 오르고 순도는 함께 내려가며, WER은 그 곱을 따른다** — δ=6이 교차점이다.
   이것이 δ 고정값의 정량적 정의다.

### 표 3. ATD student — 학습 후 어긋남이 실제로 줄었다

| | mean \|offset\| | 정확 일치 | \|off\|≤1 | 전체 coverage @δ=6 |
|---|---:|---:|---:|---:|
| no-KD | 0.69 | 43.4% | 89.8% | 73.0% |
| **ATD (δ=6, w=20)** | **0.22** | **78.4%** | **99.5%** | **94.5%** |

정렬을 **강제하지 않았는데도**(타깃은 정렬-관대하다) student가 teacher와 3배 가까이 잘 맞는다.
남은 어긋남도 21.1%가 ±1이고 ≥2는 0.5%뿐. → 관대한 배달이 오히려 정확한 정렬을 낳았다.

> 단, 이 지표는 타깃이 유래한 posterior 계열에서 측정한 것이라 학습 목적함수와 완전히 독립적이지 않다.
> METHOD_REPORT §6의 greedy-spike 지표(3.34→2.77)와 방향은 일치하지만, **증명이 아니라 메커니즘 근거**로 제시한다.

재현: `python3 scripts/fig2_atd_misalignment.py`, `python3 scripts/analyze_spike_coverage.py --limit 200`

---

## 세 결과의 연결

1. teacher 스파이크 **±1프레임에만** 복원 가능한 토큰 identity가 있다 (Q1: r(y_u) 0.29/0.41 → ±2에서 0.06).
2. student가 teacher와 어긋나는 것도 **거의 전부 ±1프레임**이다 (Q3: 어긋난 토큰의 82%).
3. 따라서 ±1을 덮되 그 이상은 덮지 않는 support가 최적이고, **δ=6이 정확히 그것**이다
   (폭 1.44프레임, ±1 coverage 62%, 순도 0.68). 더 키우면 coverage는 올라도 순도가 떨어져 WER이 악화한다.
4. 외부 정렬(MFA)은 이 ±1프레임 스케일과 무관한 좌표(폭 4.6프레임, 단어 단위)를 주므로,
   정렬기가 아무리 정확해도 teacher 지식과 단절되어 실패한다 (Q2).
