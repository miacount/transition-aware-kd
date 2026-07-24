# Span-KD: 정렬-관대한 구간 단위 지식 증류 (Alignment-Tolerant Span-Level Knowledge Distillation for CTC)

> **한 줄 주장.** CTC teacher의 peaky posterior가 지식 증류를 방해하는 두 근본 문제(신호 희소성·스파이크 정렬 불일치)를,
> **"토큰 구간 단위 비교"라는 단일 설계 원리**로 해소한다. 이를 위해 teacher 자신의 CTC occupancy를 blank-억제로 de-peak하여
> 각 토큰의 시간 구간을 정의하고, 그 구간 위에서 teacher·student를 각각 평균내어 dark knowledge를 증류한다.
> 설계의 모든 선택은 문제로부터 연역되며, ablation으로 그 필요성이 검증된다.

---

## 0. 실험 셋업

| 항목 | 값 |
|---|---|
| Teacher | `stt_en_conformer_ctc_small` (~13M, Conformer-CTC, 1024 BPE+blank, 4× subsampling) |
| Student | Conformer-CTC, d_model=144, 8 layers, ~4.8M, 1024 BPE+blank, 4× subsampling |
| 데이터 | LibriSpeech `train-clean-100` (~100h) |
| 평가 | dev/test-clean, dev/test-other, corpus-level WER, 100 epoch |

**Teacher WER (KD 상한, greedy)**: test-clean **3.70** / test-other **8.14**.
Teacher(13M·960h+)와 student(4.8M·100h)의 격차가 크므로, 성패는 절대 WER이 아니라
**동일 student·데이터에서 no-KD(15.12/33.94)를 teacher 쪽으로 얼마나 끌어오는가**로 판단한다.

> 소규모 셋업이므로 타 논문과 절대 WER 비교는 불가하며, 모든 비교는 **동일 셋업 내 상대 비교**다.

---

## 1. 문제 정의 — CTC-KD가 어려운 이유

CTC는 프레임 독립 가정으로 blank 포함 정렬을 학습하여 **극도로 peaky한 posterior**를 만든다. Teacher 실측(test-clean 200발화):

| | blank 프레임 비율 | 토큰당 프레임 | non-blank emit |
|---|---:|---:|---:|
| Teacher (CTC) | **81.6%** | **1.05** | 0.970 |

즉 teacher는 프레임의 82%를 blank로 채우고 각 토큰을 **1프레임 스파이크**로 방출한다. 여기서 두 문제가 나온다:

- **(P1) 신호 희소성.** 프레임별 KD의 신호 80%가 "여긴 blank"라는 무의미한 정보다. 실제 dark knowledge(토큰 identity·혼동 구조)는 극소수 non-blank 프레임에만 있다.
- **(P2) 스파이크 정렬 불일치.** teacher·student 스파이크가 다른 프레임에 찍히면 프레임별 비교가 **잘못된 gradient**를 준다. 실측상 no-KD student는 teacher와 평균 **3.34프레임** 어긋난다.

이 P1·P2가 본 연구가 푸는 문제이며, 이후 모든 설계는 여기서 연역된다.

---

## 2. 선행 연구 (재현한 논문) 와 한계

본 절은 **문헌의 방법을 재현한 baseline**만 다룬다. 각 방법이 P1·P2를 어떻게 다루는지, 그 한계는 무엇인지 본다.

### 2.1 Frame-level Logit KD (Hilmes et al. 2025)
프레임별 top-k posterior를 student가 프레임별로 맞춘다. P1을 완화하려 blank 프레임을 다루는 변형을 둔다.

| 재현 결과 (test-clean/other) | | 방식 |
|---|---|---|
| Vanilla logit KD | 14.29 / 33.03 | 전체 프레임 |
| KD-BE (blank elimination) | 14.00 / 32.86 | blank 프레임 제거 |
| Sym n=1 | 14.61 / 33.32 | non-blank ±1 프레임 포함 |
| KD-BE + OCC (best) | **13.41 / 31.90** | + blank/non-blank occupancy KL 보조항 |

- **한계 (P1)**: blank 제거·OCC 보조항은 근본 해법이 아닌 **수작업 패치**다. 특히 OCC가 최고 성능인 점은 "blank 처리가 병목"임을 방증하지만, 여전히 프레임 단위다.
- **한계 (P2)**: 어떤 변형도 **스파이크 정렬 불일치를 구조적으로 해결하지 못한다.** 프레임 1:1 비교라는 틀 자체가 P2에 취약하다.

### 2.2 CR-CTC (Yao et al. 2025, ICLR)
teacher 없이 같은 발화의 두 augmented view가 일관되도록 self-distillation. Peak suppression으로 일반화를 얻는다.

- **한계 (a)**: 학습 시 **forward 2회 → 2× compute**. 공정 비교하려면 epoch 절반(원논문 §4.1·Table 8: CTC 100ep vs CR-CTC 50ep).
- **한계 (b)**: 순수 self-regularization이라 **teacher의 dark knowledge를 주입하지 못한다.**
- **재현 결과 (공정성 유의)**:
  - 100 epoch (compute 2×, 불공정): 12.76 / 30.26
  - **50 epoch (compute 동일, 공정): 14.02 / 31.84** ← 표면적 우위는 상당 부분 2× 예산의 착시.

### 2.3 Label Priors (Huang et al. 2024) — 차용한 요소
CTC posterior에 label prior를 걸어 peakiness를 억제, forced alignment 정확도를 높인다. 본 연구는 이 **de-peaking 아이디어를 KD 타깃의 occupancy 생성**에 차용한다(§3.2, blank-only 형태로 단순화).

---

## 3. 제안 방법 — Span-KD (설계의 연역)

### 3.1 설계 원리 (P1·P2 → 요구조건 → 설계)

| 문제 | 요구조건 | 설계 결정 |
|---|---|---|
| **P2** 정렬 불일치 | 비교가 정렬에 무관해야 함 | 프레임 1:1이 아니라 **토큰 구간 단위로 뭉쳐** 비교 (D1) |
| **P1** 신호 희소성 | 토큰이 실제 존재하는 곳의 dark knowledge를 써야 함 | 각 토큰의 **시간 구간(occupancy)** 위에서 non-blank 지식을 집계 (D2·D4) |

핵심 원리는 하나다: **"토큰이 걸쳐 있는 구간에서 teacher와 student를 각각 평균내어 비교한다."** 아래 모든 요소가 여기서 파생된다.

### 3.2 각 설계 요소와 그 이유 (직관)

**(D1) 왜 구간 단위 비교인가.**
프레임 t에서 teacher가 "cat 0.8", student가 프레임 t+1에서 "cat 0.8"이면, 프레임별 비교는 t에서 불일치로 벌한다(P2). 그러나 두 모델을 **구간 [t-1..t+1] 전체로 평균**내면 둘 다 "cat"이 되어 일치한다. → 정렬이 몇 프레임 밀려도 견디는 **alignment-invariant 비교.**

**(D2) 왜 teacher 자신의 occupancy를 쓰나 (외부 정렬기 배제).**
"토큰이 어느 구간에 있나"는 teacher가 이미 안다(CTC 정렬). transcript-제약 forward-backward로 teacher의 occupancy `γ(t,u)`를 뽑으면 된다.
→ **teacher가 정렬원이자 지식원**이다. 외부 정렬기를 쓰면 timing이 teacher 지식과 단절된다(§4 ablation이 이를 확인).

**(D3) 왜 occupancy를 de-peak하나, 왜 blank만 누르나.**
teacher occupancy는 1프레임 스파이크(§1)라 그대로 평균내면 구간=1프레임, 즉 프레임-KD로 퇴화한다. 구간을 넓혀야 한다.
→ teacher posterior에서 **blank의 log확률에만 페널티 δ를 뺀다.** 이유:
  - blank를 누르면 인접 shoulder 프레임(토큰 확률이 낮게 깔린 곳)이 살아나 occupancy가 **자연스럽게 넓어진다.**
  - **blank만** 건드리므로 non-blank 토큰 간 비율은 불변 → **dark knowledge가 그대로 보존**된다.
  - per-token 빈도 prior를 안 쓰므로 **희귀 BPE tail explosion이 없다.**
  - δ는 **적당히**만 (과하게 넓히면 이웃 토큰 오염 — §4 확인).

**(D4) 왜 WHAT은 non-blank top-k인가.**
증류 대상은 "이 토큰이 무엇이고 무엇과 혼동되나"라는 **identity/혼동 구조**다. 이는 non-blank 부분에 있다. blank는 "gap"이지 identity가 아니므로 타깃에서 제외한다(§4: blank 유지 시 이득 없음).

**(D5) 학습 손실.**
student의 프레임별 posterior를 **같은 occupancy `γ`로 가중평균**한 `s_u`를 만들고, teacher의 occupancy-평균 top-k `q_u`와 cross-entropy:
```
L = L_CTC  +  w · Σ_u CE( s_u , q_u )
```
D1에 의해 이 비교는 정렬-무관하다.

### 3.3 하이퍼파라미터의 근거
- **δ (de-peak 강도)**: occupancy 폭을 결정. 너무 좁으면 프레임-KD로 퇴화, 너무 넓으면 이웃 오염 → 중간값(δ=6, ~1.8프레임).
- **w (KD weight)**: CTC와 KD의 균형. sweep 1/5/20/50 중 w=20 최적, w=50은 KD가 CTC를 압도해 정렬이 흔들리며 악화(12.67→13.25).

---

## 4. 설계 근거 검증 (Ablation) — "우리가 왜 이렇게 설계했는가"

각 설계 결정마다 **대안 가설을 세우고, 실험으로 그 대안이 열등함을 보여** 설계를 정당화한다. (본 절은 문헌 baseline이 아니라 **우리 설계 선택의 검증**이다.)

| 설계 결정 | 대안(가설) | 대안 결과 | Span-KD | 결론 |
|---|---|---|---|---|
| **D1/D2** teacher-self occupancy | 외부 phoneme aligner로 구간 정의 | 14.41 / 33.52 | **12.67 / 30.64** | timing이 지식과 단절되면 실패 → self-occupancy가 옳다 |
| **D1** soft occupancy 평균 | hard 경계(Viterbi)로 구간 평균 | 12.95 / 30.78 | **12.67 / 30.64** | 부드러운 경계가 hard 경계보다 우위 |
| **D3** 적당한 δ 폭 | 가중치-수준 de-peak로 4.2프레임까지 확대 | 14.42 / 32.29 | **12.67 / 30.64** | 넓은 구간은 이웃 토큰 오염 → 적당한 폭이 필수 |
| **D4** non-blank만 증류 | 타깃에 blank(occupancy) 유지 | 12.75 / 30.97 | **12.67 / 30.64** | blank는 identity 아님 → 넣으면 방해 |

네 ablation이 D1~D4 각각을 **연역적으로 세운 뒤 실험으로 확정**한다. "이것저것 해보니 됐다"가 아니라, **각 설계에 반례 대안을 두고 그 대안이 왜 나쁜지를 원리로 설명·검증**한 것이다.

---

## 5. 결과 — 재현 논문 baseline 대비

| 방법 | 계열 | test-clean | test-other |
|---|---|---:|---:|
| *Teacher (상한, 참고)* | — | *3.70* | *8.14* |
| no-KD | baseline | 15.12 | 33.94 |
| Vanilla logit KD (Hilmes 2025) | frame | 14.29 | 33.03 |
| KD-BE (Hilmes 2025) | frame | 14.00 | 32.86 |
| KD-BE + OCC (Hilmes 2025, best) | frame | 13.41 | 31.90 |
| CR-CTC (Yao 2025, 공정 50ep) | self | 14.02 | 31.84 |
| **Span-KD (ours, δ=6, w=20)** | **span** | **12.67** | **30.64** |

- **동일 compute에서 모든 재현 baseline을 능가.** 최고 frame-KD(KD-BE+OCC 13.41/31.90) 대비 clean −0.74·other −1.26, 공정 CR-CTC(14.02/31.84) 대비 clean −1.35·other −1.20.
- P1·P2를 프레임 틀 안에서 패치한 방법(Hilmes)과 teacher 지식을 아예 안 쓰는 방법(CR-CTC)의 한계를, **구간 단위라는 원리로 동시에 넘어선다.**

---

## 6. 왜 작동하는가 — 메커니즘 실측

학습된 student 직접 측정(test-clean 200발화):

| | blank 프레임% | 토큰당 프레임 | non-blank emit | teacher와 spike 어긋남 |
|---|---:|---:|---:|---:|
| Teacher | 81.6% | 1.05 | 0.970 | — |
| no-KD student | 79.8% | 1.18 | 0.922 | **3.34** |
| **Span-KD student** | **64.5%** | **2.04** | 0.915 | **2.77** |

1. **P2 해소의 직접 증거**: Span-KD student의 teacher 대비 spike 어긋남이 **3.34→2.77프레임**으로 감소. 정렬을 강제하지 않았는데도(관대한 타깃) 더 잘 맞는다 → dark knowledge가 올바른 위치에 주입됨(test-clean 우위와 직결).
2. **구간 학습의 흔적**: 토큰당 프레임이 1.18→**2.04**로 증가(blank 80%→64.5%). 구간-평균 타깃을 만족시키려 student가 토큰을 구간에 걸쳐 방출한 결과.
3. **범위의 명확화**: per-frame emit는 0.922→0.915로 거의 불변. Span-KD의 이득은 **정렬-관대한 dark knowledge 증류**(설계 목표)이지 vocab-level 과확신 억제가 아니다. 이 축은 §7에서 다룬다.

---

## 7. 발전 방향

- **Span-KD + α (calibration 요소, 개발 중).** §6.3대로 Span-KD는 vocab축(per-frame 과확신)엔 압력이 없다. 이를 보완할 추가 요소 α를 둔다. 현재 후보는 SR-CTC 계열 self-smoothing이나, **α는 특정 기법에 고정된 것이 아니라 "정렬-관대 dark knowledge(Span-KD) 위에 얹는 보완 요소"라는 슬롯**이다. 요구조건은 명확하다 — 노이즈가 아닌 **의미 있는 구조**로 softening해야 한다(§8: 노이즈 방향 softening은 실패).
- **occupancy 폭·δ의 이론적 최적점.** 현재 δ=6은 sweep으로 얻은 값. "정렬 관대함(넓힘) vs 이웃 오염(과확대)"의 trade-off를 정량 모델로 설명하는 것이 다음 과제.

---

## 8. 부정 결과 (방법의 경계)

시도했으나 기각한 것들. 각각 "무엇이 왜 안 되는가"로 방법의 경계를 정의한다.

| 시도 | 결과 | 원리적 이유 |
|---|---|---|
| 넓은 occupancy (4.2프레임) | 14.42/32.29 | 구간이 넓으면 이웃 토큰 posterior가 섞여 오염 |
| Temperature로 타깃 vocab softening | 14.38 (vs 14.17) | de-peaked teacher의 tail은 신뢰 가능한 dark knowledge가 아니라 노이즈 |
| 타깃에 blank 유지 | 12.75/30.97 | blank는 identity가 아니며 좁은 구간에선 신호도 약함 |
| 외부 aligner로 구간 정의 | 14.41/33.52 | timing이 teacher dark knowledge와 단절 |

원리: **살릴 정보 = non-blank identity + 적당한 구간 평균 / 버릴 정보 = 넓은 구간·노이즈 softening·외부 timing.**

---

## 9. 요약

CTC teacher의 peakiness(81.6% blank, 1.05프레임)는 KD의 두 문제 — 신호 희소성(P1)·스파이크 정렬 불일치(P2) — 를 낳는다.
**Span-KD**는 "토큰 구간 단위 비교"라는 단일 원리로 둘을 동시에 해소하며, 그 원리로부터
(D1) 구간 평균, (D2) teacher-self occupancy, (D3) blank-only de-peak, (D4) non-blank 증류를 **연역**하고, 네 ablation으로 각 선택을 **검증**한다.
결과적으로 동일 compute에서 재현한 모든 문헌 baseline(Hilmes frame-KD, CR-CTC)을 능가하며,
student가 teacher와 더 잘 정렬(3.34→2.77)되는 것이 성능 향상의 직접 근거다.

> **Novelty**: 조합이 아니라 **"teacher의 de-peaked self-occupancy를 정렬-관대한 span KD 타깃으로 쓴다"**는 단일 설계 원리에 있으며,
> 이 원리는 문제(P1·P2)로부터 연역되고 ablation으로 검증된다.
