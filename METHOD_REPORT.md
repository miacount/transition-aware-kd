# Span-KD: 정렬-관대한 구간 단위 지식 증류 (Alignment-Tolerant Span-Level Knowledge Distillation for CTC)

> 연구 보고 초안. CTC ASR에서 teacher의 peaky한 posterior가 지식 증류(KD)를 방해하는 문제를,
> **teacher 자신의 de-peaked soft occupancy를 이용해 토큰 구간 단위로 dark knowledge를 정렬-관대하게 증류**하는 방법으로 해결한다.
> 동일 학습 compute 조건에서 기존 최신 방법(CR-CTC, ICLR 2025)을 포함한 모든 baseline을 능가한다.

---

## 0. 실험 셋업 (모든 수치의 공통 조건)

| 항목 | 값 |
|---|---|
| Teacher | `stt_en_conformer_ctc_small` (~13M, Conformer-CTC, 1024 BPE + blank, 4× subsampling) |
| Student | Conformer-CTC, d_model=144, 8 layers, ~4.8M, 1024 BPE + blank, 4× subsampling |
| 데이터 | LibriSpeech `train-clean-100` (~100h) |
| 평가 | dev/test-clean, dev/test-other, corpus-level WER |
| 학습 | 100 epoch (별도 표기 시 예외) |

**Teacher WER (KD 상한, greedy)**: test-clean **3.70** / test-other **8.14** (dev-clean 3.67 / dev-other 8.11).
Teacher(13M·960h+ 학습)와 student(4.8M·100h)의 용량·데이터 격차가 크므로, KD의 목표는 이 상한에 근접하는 것이 아니라
**동일 student·데이터에서 no-KD(15.12/33.94) 대비 얼마나 teacher 쪽으로 끌어오는가**이다.

> **주의**: 소규모(4.8M student, 100h) 셋업이다. CR-CTC 원논문(Zipformer, 960h+)과 절대 WER은 비교 불가하며, 본 보고의 모든 비교는 **동일 셋업 내 상대 비교**다.

---

## 1. 배경 — CTC 지식 증류의 근본 문제: Peaky Posterior

CTC는 프레임 독립 가정하에 blank를 포함한 정렬을 학습하며, 그 결과 **극도로 peaky한 posterior**를 만든다.
본 연구에서 teacher를 직접 측정한 결과(test-clean 200발화):

| | blank 프레임 비율 | 토큰당 프레임 수 | non-blank emit 확률 |
|---|---:|---:|---:|
| Teacher (CTC) | **81.6%** | **1.05** | 0.970 |

즉 teacher는 전체 프레임의 ~82%를 blank로 채우고, 각 토큰을 **정확히 1프레임의 스파이크**로만 방출한다.
이 peakiness가 KD에서 두 가지 문제를 일으킨다:

1. **신호 희소성**: 프레임별 KD를 하면 신호의 80%가 "여긴 blank"라는 무의미한 정보다. 유용한 dark knowledge(토큰 identity·혼동 구조)는 극소수의 non-blank 스파이크에만 존재한다.
2. **정렬 불일치(spike position mismatch)**: teacher와 student의 스파이크가 서로 다른 프레임에 찍히면, 프레임별 비교가 잘못된 gradient를 준다. 본 연구 측정상 no-KD student의 스파이크는 teacher와 평균 **3.34프레임** 어긋난다.

이 두 문제가 "왜 CTC-KD가 잘 안 되는가"의 핵심이며, 선행 연구들은 각기 다른 방식으로 이를 우회하려 했다.

---

## 2. 선행 연구 분석과 한계

### 2.1 Frame-level Logit KD (Hilmes et al. 2025 계열)
- **방식**: teacher의 프레임별 top-k posterior를 student가 프레임별로 맞춘다. Blank 지배 문제를 완화하려 **KD-BE**(blank 프레임을 손으로 제거), Symmetric 등 변형을 둔다.
- **한계**: (a) blank 제거는 근본 해법이 아닌 **수작업 패치**이며, (b) 스파이크 정렬 불일치를 구조적으로 해결하지 못한다.
- **본 실험 결과**: vanilla 14.29/33.03, KD-BE 14.00/32.86 (test-clean/test-other). no-KD(15.12/33.94) 대비 개선폭이 작다.

### 2.2 Token-average KD (hard segment)
- **방식**: Viterbi 정렬로 각 토큰의 **hard 경계** `[start,end]`를 구하고, 그 구간에서 student·teacher를 평균내 비교. 프레임 불일치를 구간 평균으로 우회.
- **한계**: 경계가 **딱딱하게 잘린다**. 경계 근처의 부드러운 전이 정보를 버리고, Viterbi 정렬 오차에 취약.
- **본 실험 결과**: 12.95/30.78 — 강력하다. **"구간 평균" 아이디어 자체가 유효함**을 시사한다.

### 2.3 외부 정렬기 기반 Span KD (phoneme aligner)
- **방식**: 독립적으로 학습한 phoneme soft aligner로 토큰의 시간 구간(support)을 얻고, 그 위에서 teacher posterior를 집계.
- **한계**: aligner의 support가 **teacher의 dark knowledge와 단절**되어 있고(시간 정보만 제공), BPE↔phoneme 매핑이 부정확하다.
- **본 실험 결과**: 14.41/33.52 — **실패**. 데이터 커버리지를 61.5%→100%로 복구해도 개선 없음. "시간축 정보만" 쓰는 접근의 한계를 실증.

### 2.4 CR-CTC (Yao et al. 2025, ICLR) — self-consistency
- **방식**: teacher 없이, 같은 발화의 두 augmented view가 일관되도록 self-distillation. Peak suppression으로 일반화를 얻는다.
- **한계**: (a) 학습 시 **forward 2회 → 2× compute**, (b) 순수 self-regularization이라 **teacher의 dark knowledge를 주입하지 못한다**.
- **본 실험 결과 (⚠️ 공정성 핵심)**: 원논문 관례상 CR-CTC는 2× forward이므로 **동일 compute 비교를 위해 epoch을 절반으로** 학습해야 한다(원논문 §4.1, Table 8: CTC 100ep vs CR-CTC 50ep).
  - 100 epoch (compute 2×, **불공정**): 12.76 / 30.26
  - **50 epoch (compute 동일, 공정): 14.02 / 31.84**
  - 즉 CR-CTC의 표면적 우위는 상당 부분 **2× 학습 예산의 착시**였다.

### 2.5 Label Priors (Huang et al. 2024)
- CTC의 peakiness를 label prior로 억제해 forced alignment 정확도를 높인다. 본 연구는 이 아이디어를 **KD 타깃 생성용 de-peaking**으로 차용한다(아래 §3).

---

## 3. 제안 방법 — Span-KD (+ α smoothing)

### 3.1 핵심 아이디어
> **teacher 자신의 posterior를 사후에 적당히 de-peak하여 "soft occupancy"를 얻고, 그 구간 위에서
> teacher의 dark knowledge를 토큰 단위로, 정렬-관대하게 증류한다.**

별도 정렬기가 필요 없다(§2.3의 실패 원인 제거). Teacher가 곧 정렬기이자 지식원이다.

### 3.2 타깃 생성 (오프라인)
각 발화·토큰 u에 대해:
1. **De-peaked occupancy (WHERE)**: teacher log-posterior의 **blank에만** 페널티 δ를 빼고(non-blank 비율은 불변 → dark knowledge 보존, tail explosion 없음), transcript-제약 forward-backward로 occupancy `γ(t,u)`를 얻는다. δ가 클수록 토큰이 인접 프레임으로 퍼진다.
2. **Semantic target (WHAT)**: teacher의 sharp posterior를 `γ`로 가중평균해 non-blank top-k 분포 `q_u`를 만든다.

### 3.3 학습 손실
Student의 프레임별 posterior를 같은 support `γ`로 가중평균한 `s_u`를 만들고, `q_u`와 cross-entropy:
```
L = L_CTC  +  w · Σ_u CE( s_u , q_u )   +  (α) · L_SR
```
- **정렬 관대함**: teacher·student를 각각 구간 평균으로 뭉쳐 비교하므로, 스파이크가 프레임 단위로 어긋나도 견딘다(§2.1의 mismatch 문제 해소).
- **(α) L_SR (개발 중)**: SR-CTC(Yao 2025, A.1) smooth 정규화. student의 per-frame 분포를 자기 시간-이웃으로 smoothing해 과확신을 억제(§6 참조). "identity(Span-KD) + calibration(α)"의 2요소 구성.

### 3.4 하이퍼파라미터
δ=6 (사후 de-peak), w=20 (KD weight). 둘 다 sweep으로 선택 (w: 1/5/20/50에서 20이 최적, 50은 KD가 CTC를 압도해 악화).

---

## 4. 실험 결과 (동일 셋업·동일 compute)

| 방법 | 계열 | test-clean | test-other |
|---|---|---:|---:|
| *Teacher (상한, 참고)* | — | *3.70* | *8.14* |
| no-KD | baseline | 15.12 | 33.94 |
| Vanilla logit KD (Hilmes) | frame | 14.29 | 33.03 |
| KD-BE (Hilmes) | frame | 14.00 | 32.86 |
| KD-BE + OCC | frame | 13.41 | 31.90 |
| De-peaked frame-KD (ours 부산물) | frame | 14.17 | 32.18 |
| Span-KD (phoneme aligner) | span, 실패 | 14.41 | 33.52 |
| CR-CTC (공정, 50ep) | self | 14.02 | 31.84 |
| Token-average (hard 경계) | span | 12.95 | 30.78 |
| **Span-KD-δ (ours, w=20)** | **span** | **12.67** | **30.64** |
| *(참고) CR-CTC (불공정, 100ep, 2× compute)* | self | *12.76* | *30.26* |

**핵심 결론**
1. **Span-KD-δ가 동일 compute에서 전 방법 최고.** 공정 조건의 CR-CTC(14.02/31.84)를 clean/other 양쪽에서 ~1.3pp 능가.
2. **"구간 평균 KD" 계열(Span-KD, Token-avg)이 최강 계열.** 그 안에서 **soft occupancy(Span-KD) > hard 경계(Token-avg)** (12.67 vs 12.95).
3. 프레임 단위 방법과 self-consistency는 그 아래.

---

## 5. 다른 논문과의 차이점 (Positioning)

| | 정렬 처리 | 지식원 | Teacher 필요 | Compute |
|---|---|---|---|---|
| Frame-KD (Hilmes) | 프레임 1:1 (mismatch 취약) | teacher | ✓ | 1× |
| Token-avg | hard 경계 | teacher | ✓ | 1× |
| Phoneme-span | 외부 정렬기(단절된 timing) | teacher(불완전) | ✓+정렬기 | 1× |
| CR-CTC | 없음(프레임별 self-KL) | self만 | ✗ | 2× |
| **Span-KD (ours)** | **soft occupancy(정렬-관대)** | **teacher(자기 정렬)** | ✓ | **1×** |

- **vs Frame-KD**: 프레임을 구간으로 뭉쳐 mismatch를 근본적으로 흡수. blank 제거 같은 수작업 불필요.
- **vs Token-avg**: hard 경계 대신 teacher의 de-peaked soft occupancy → 경계가 부드럽고 정렬 오차에 강함(실측 우위).
- **vs Phoneme-span**: 외부 정렬기 제거. teacher 자신이 정렬·지식을 동시에 제공 → 단절/매핑 문제 소멸.
- **vs CR-CTC**: teacher의 dark knowledge를 실제로 주입하며 1× compute. 공정 비교 시 우위.

---

## 6. 왜 작동하는가 — Peakiness & Mismatch 분석

학습된 student를 직접 측정(test-clean 200발화):

| | blank 프레임% | 토큰당 프레임 | non-blank emit | teacher와 spike 어긋남 |
|---|---:|---:|---:|---:|
| Teacher | 81.6% | 1.05 | 0.970 | — |
| no-KD student | 79.8% | 1.18 | 0.922 | 3.34 |
| **Span-KD-δ student** | **64.5%** | **2.04** | 0.915 | **2.77** |

1. **시간축 de-peaking**: Span-KD student는 토큰을 1.18→**2.04프레임**에 걸쳐 방출(blank 80%→64.5%). 구간 평균 타깃을 만족시키려 토큰이 구간을 채운 결과.
2. **정렬 개선**: teacher와의 spike 어긋남이 3.34→**2.77프레임**으로 감소. 강제하지 않았는데도(관대한 타깃) 더 잘 정렬됨 → dark knowledge가 올바른 위치에 주입됨(test-clean 우위와 직결).
3. **정직한 한계**: emit 확률은 0.922→0.915로 **거의 불변**. 즉 Span-KD는 **시간축만** 부드럽게 했고 **vocab축(per-frame 확신)은 그대로**다. 본 방법의 이득은 "peak suppression"이 아니라 **정렬-관대한 dark knowledge 증류**에서 온다(이 점에서 CR-CTC의 vocab-softening 기반 일반화와 메커니즘이 다르다). → §7의 개발 방향(α)이 바로 이 축을 보완한다.

---

## 7. 발전 방향

1. **(진행 중) Span-KD + α (SR-CTC calibration)**: student per-frame 분포를 자기 시간-이웃으로 smoothing(SR-CTC, Yao 2025)해 vocab축 과확신을 억제. "identity(Span-KD) + calibration(α)" 2요소. 원논문에서 SR-CTC 단독으로 CTC 대비 test-other 6.02→5.22 개선을 보고 → 우리 방법의 부족한 vocab축을 메울 후보. β sweep(0.1/0.2/0.3) 예정.
2. **δ·occupancy 폭 최적화**: 사후 δ(현재 6, ~1.8프레임)가 스위트스팟인지 재검. 주의 — 가중치-수준 de-peak(teacher fine-tune)으로 support를 4.2프레임까지 넓힌 실험은 **오히려 악화**(14.42/32.29): 너무 넓으면 이웃 토큰 오염. 적당한 폭이 핵심.
3. **Teacher-free 확장**: 현재 self-distillation(CR-CTC)과 상보적. 두 신호(teacher dark knowledge + self-consistency)를 결합하는 방향은 novelty 관점에서 신중히 설계 필요.

---

## 8. 부정 결과 (연구 엄밀성)

다음은 **시도했으나 기각**한 것들이다. 각각 "무엇이 안 되는가"를 명확히 하여 방법의 경계를 정의한다.

| 시도 | 결과 | 교훈 |
|---|---|---|
| Teacher fine-tune으로 support 4.2프레임까지 확대 | 14.42/32.29 (악화) | **넓은 support는 이웃 토큰 오염** — 적당한 폭이 최적 |
| Temperature softening (de-peaked frame-KD T=2) | 14.38 vs T=1 14.17 (악화) | **노이즈 방향 softening은 해롭다** — tail이 신뢰 가능한 dark knowledge여야 |
| Span-KD 타깃에 blank(occupancy) 유지 | 12.75/30.97 (미미하게 악화) | 좁은 support에서 blank 신호(median 4%)가 약해 방해만 됨 |
| 외부 phoneme aligner span | 14.41/33.52 (실패) | 시간축만 쓰고 dark knowledge와 단절되면 무의미 |
| From-scratch BPE aligner | 학습 실패 | 1024클래스×100h는 데이터 굶주림; label-prior 부호 버그도 존재했음 |

이 부정 결과들은 **"살릴 정보(non-blank dark knowledge, 적당한 구간 평균) vs 버릴 정보(넓은 support, 노이즈 softening, 시간축-only)"**의 경계를 실증적으로 확정한다.

---

## 9. 요약

CTC teacher의 극단적 peakiness(81.6% blank, 1.05프레임 스파이크)는 지식 증류의 근본 병목이다.
**Span-KD**는 teacher 자신의 blank-억제 soft occupancy 위에서 dark knowledge를 **토큰 구간 단위·정렬 관대하게** 증류함으로써,
(1) 스파이크 정렬 불일치를 구조적으로 흡수하고, (2) 별도 정렬기 없이 teacher를 정렬원이자 지식원으로 활용하며,
(3) 동일 학습 compute에서 CR-CTC(ICLR 2025)를 포함한 모든 baseline을 능가한다.
Student는 실제로 덜 peaky(2.04프레임)하고 teacher와 더 잘 정렬(2.77프레임)되며, 이는 성능 향상의 직접 근거다.
남은 축(vocab-level calibration)은 **+α (SR-CTC)** 로 보완 중이다.

> 본 방법의 novelty는 조합이 아니라 **"teacher의 de-peaked self-occupancy를 정렬-관대한 span KD 타깃으로 쓴다"**는 단일 아이디어에 있으며,
> 정밀한 peakiness/mismatch 측정과 다수의 부정 결과가 그 필요성·유효성을 뒷받침한다.
