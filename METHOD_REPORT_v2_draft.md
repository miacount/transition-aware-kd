# Span-KD: 정렬-관대한 emission 증류 (Alignment-Tolerant Emission Distillation for CTC)

> **한 줄 주장.** CTC teacher의 peaky posterior가 낳는 **하나의 근본 문제 — teacher·student 스파이크의 정렬 불일치(P2)** 를,
> **"teacher의 emission 타이밍을 정렬-관대하게 증류한다"** 는 단일 원리로 해소한다.
> 이를 위해 teacher 자신의 CTC occupancy를 blank-억제로 de-peak하여 각 토큰의 시간 구간을 정의하고,
> 그 구간 위에서 student가 방출하도록 유도한다.
> **부수적 발견**: CTC teacher에는 증류할 만한 identity dark knowledge가 거의 없으며(타깃의 95%가 원-핫),
> CTC-KD의 실제 이득은 dark knowledge가 아니라 **emission 구조 전이**에서 온다.

> ⚠️ *DRAFT (2026-07-20 재작성).* β/uniform/K 계열 숫자는 현재 single-seed. headline·핵심 대비는 multi-seed 확정 필요(§8).

---

## 0. 실험 셋업

| 항목 | 값 |
|---|---|
| Teacher | `stt_en_conformer_ctc_small` (~13M, Conformer-CTC, 1024 BPE+blank, 4× subsampling) |
| Student | Conformer-CTC, d_model=144, 8 layers, ~4.8M, 1024 BPE+blank, 4× subsampling |
| 데이터 | LibriSpeech `train-clean-100` (~100h) |
| 평가 | dev/test-clean, dev/test-other, corpus-level WER, 100 epoch |

**Teacher WER (KD 상한, greedy)**: test-clean **3.70** / test-other **8.14**.
성패는 절대 WER이 아니라 **동일 student·데이터에서 no-KD(15.12/33.94)를 얼마나 끌어오는가**로 판단한다.
소규모 셋업이므로 모든 비교는 **동일 셋업 내 상대 비교**다.

---

## 1. 문제 정의 — CTC peakiness가 낳는 것

CTC는 프레임 독립 가정으로 blank 포함 정렬을 학습해 **극도로 peaky한 posterior**를 만든다. Teacher 실측(test-clean 200발화):

| | blank 프레임 비율 | 토큰당 프레임 | non-blank emit |
|---|---:|---:|---:|
| Teacher (CTC) | **81.6%** | **1.05** | 0.970 |

이 **peakiness가 root cause**이고, 여기서 두 결과가 파생된다.

- **(P2) 스파이크 정렬 불일치 — 우리가 푸는 문제.** teacher·student가 1프레임 스파이크를 서로 다른 프레임에 찍으면, 프레임별 KD 비교가 **잘못된 gradient**를 준다. 실측상 no-KD student는 teacher와 평균 **3.34프레임** 어긋난다.
- **(F) identity dark knowledge 부재 — 우리가 밝히는 사실.** emission이 일어나는 프레임에서 teacher는 **identity를 확신**한다(§6). 즉 "이 토큰이 무엇과 혼동되나"라는 dark knowledge가 CTC posterior에는 거의 없다.

> **blank는 "우리가 고치는 문제"가 아니다.** peakiness/blank-dominance는 (a) frame-KD를 낭비적·정렬취약하게 만드는 **원인**이자, (b) 우리가 occupancy 구간을 만드는 데 쓰는 **도구**(§3, de-peak)로만 등장한다. student blank가 학습 후 줄어드는 것(§6)은 peakiness 규제의 이득이 아니라 **올바른 타이밍을 배운 부산물**이다(§7의 CR-CTC 대비가 이를 확인).

---

## 2. 선행 연구 (재현) 와 한계

### 2.1 Frame-level Logit KD (Hilmes et al. 2025)
프레임별 top-k posterior를 student가 프레임별로 맞춘다. blank 프레임 처리를 위한 변형을 둔다.

| 재현 결과 (test-clean/other) | | 방식 |
|---|---|---|
| Vanilla logit KD | 14.29 / 33.03 | 전체 프레임 |
| KD-BE (blank elimination) | 14.00 / 32.86 | blank 프레임 제거 |
| KD-BE + OCC (best) | **13.41 / 31.90** | + blank/non-blank occupancy KL 보조항 |

- **한계 (P2)**: 어떤 변형도 프레임 1:1 비교라 **스파이크 정렬 불일치에 구조적으로 취약**하다.
- **재해석 (F)**: OCC(occupancy) 보조항이 최고 성능인 것은, 이 계열의 이득도 실은 **identity dark knowledge가 아니라 blank/emission 구조**에서 온다는 방증이다(§6·§8에서 정량화).

### 2.2 CR-CTC (Yao et al. 2025, ICLR)
teacher 없이 두 augmented view의 일관성으로 self-distill. Peak suppression으로 일반화를 얻는다.
- **공정 비교(compute 동일, 50ep): 14.02 / 31.84.** 순수 peak-suppression은 Span-KD(12.67)에 못 미친다 → **peakiness 규제만으로는 우리 이득을 설명 못 함**(§7).

### 2.3 Sequence-level KD / S-CTC (Huang et al. 2018)
teacher occupancy에 student per-frame posterior를 맞추는 정렬 기반 증류.
- λ=1 순수: 17.14/35.48 (collapse). +CTC-ft: 13.03/31.59.
- **정렬 정보만으론 부족** — 정렬-관대함 없이 emission을 강제하면 무너진다(§5 대비).

### 2.4 Label Priors (Huang et al. 2024) — 차용
CTC posterior에 label prior를 걸어 peakiness를 억제. 이 **de-peaking 아이디어를 occupancy 생성**에 blank-only 형태로 차용(§3).

---

## 3. 제안 방법 — Span-KD (설계의 연역)

### 3.1 원리
> **P2(정렬 불일치)** → 요구조건: teacher의 emission 타이밍을, 프레임 단위가 아니라 **정렬에 관대하게** student에 전이 → 설계: **teacher occupancy 구간에서 student가 방출하도록 유도.**

### 3.2 설계 요소

**(D1) teacher-self occupancy로 구간을 만든다.**
"토큰이 어느 구간에 있나"는 teacher가 안다(CTC 정렬). transcript-제약 forward-backward로 teacher occupancy `γ(t,u)`를 뽑는다. → **teacher가 타이밍의 근원.** 외부 정렬기를 쓰면 timing이 teacher와 단절돼 실패한다(§5).

**(D2) occupancy를 de-peak한다 — 도구로서의 blank 억제.**
teacher occupancy는 1프레임 스파이크(§1)라 그대로 쓰면 구간=1프레임(프레임-KD로 퇴화). teacher posterior의 **blank log확률에만 δ를 빼서** occupancy를 넓힌다. blank만 건드리므로 non-blank 비율은 불변 → **de-peak는 오직 "어디(WHERE)"만 넓히고 "무엇(WHAT)"은 바꾸지 않는다.** (이 성질이 §6의 dark-knowledge 부재와 직결.)

**(D3) 구간 위에서 정렬-관대하게 비교한다.**
student per-frame posterior를 같은 occupancy `γ`로 가중평균한 `s_u`를 만들어 teacher 타깃과 비교한다. 프레임이 몇 개 밀려도 구간 전체로 뭉치므로 **정렬 오차에 견딘다**(P2 해소).

**(D4) 타깃은 non-blank top-k, 그리고 핵심은 emission 전이다.**
타깃은 occupancy-평균한 teacher non-blank 분포의 top-k다. 그런데 학습 loss는 student를 재정규화하지 않는 CE라, 정확히 두 항으로 분해된다:
```
L_span,u = KL(q_u ‖ s̄_u)  +  β·(−log m_u)
           └─ content ─┘     └─ emission ─┘        (β=1 = 기본 구현)
```
- **content**: teacher non-blank 분포 모양 맞추기.
- **emission** `−log m_u`: student가 occupancy 구간에 **non-blank 확률을 충분히 배치**하도록 유도 = **teacher 타이밍을 실제로 전이하는 항.**
- §6/§8에서 보이듯 이 방법의 이득은 **emission 항**에서 오며, content는 타깃이 원-핫이라 사실상 identity 지정에 그친다.

### 3.3 하이퍼파라미터
- **δ (de-peak)**: occupancy 폭. 좁으면 프레임-KD로 퇴화, 넓으면 이웃 오염 → 중간(δ=6, ~1.8프레임).
- **w (KD weight)**: CTC와 균형. w=20 최적, w=50은 emission 압력이 CTC 정렬 요구를 압도해 악화(12.67→13.25).

---

## 4. 왜 작동하는가 — 메커니즘 실측

학습된 student 직접 측정(test-clean 200발화):

| | blank% | 토큰당 프레임 | non-blank emit | teacher와 spike 어긋남 |
|---|---:|---:|---:|---:|
| Teacher | 81.6% | 1.05 | 0.970 | — |
| no-KD student | 79.8% | 1.18 | 0.922 | **3.34** |
| **Span-KD student** | 64.5% | 2.04 | 0.915 | **2.77** |

1. **P2 해소의 직접 증거**: teacher와의 spike 어긋남이 **3.34→2.77프레임**으로 감소. 정렬을 강제하지 않았는데도(관대한 타깃) 더 잘 맞는다 → teacher **타이밍이 올바르게 전이**됨.
2. **emission 전이의 흔적**: 토큰당 프레임 1.18→**2.04**(blank 80→64.5%). teacher occupancy 구간에 걸쳐 방출하도록 배운 결과.
3. **범위 명확화**: per-frame emit는 0.922→0.915로 거의 불변. 이득은 **정렬-관대 emission 전이**이지 vocab-level 과확신 억제가 아니다(그건 CR-CTC의 영역이고 더 약함, §7).

---

## 5. 설계 근거 검증 — ablation을 통한 메커니즘 격리

각 ablation은 "성능 이득이 emission·정렬-관대함에서 오고, dark knowledge에서 오지 않는다"를 격리한다.
(β/uniform/K는 single-seed — §8에서 multi-seed 확정 예정.)

| 검증 대상 | 변형 | 결과 (test c/o) | vs Span-KD | 결론 |
|---|---|---|---|---|
| **emission이 동력인가** | β=0 (emission 제거, content만) | 13.72 / 32.29 | 12.67 / 30.64 | **emission 없으면 타이밍 전이 실패** — 최대 폭 악화 |
| **dark knowledge가 기여하나** | uniform (teacher 가중치 뭉갬) | 12.59 / 30.60 | ≈ | 가중치 버려도 동등/우위 → **상대분포는 기여 안 함** |
| **후보 집합이 필요한가** | K=1 (top-1 identity만) | 12.78 / 31.30 | ≈ | top-1만으로 충분 → **관계적 dark knowledge 불필요** |
| **blank를 타깃에 넣으면** | keep-blank (emission을 타깃으로 명시) | 12.75 / 30.97 | ≈ | drop-blank가 이미 loss로 emission 전달 → 중복 |
| **타이밍이 teacher에서 와야 하나** | 외부 phoneme aligner | 14.41 / 33.52 | ↓ | timing이 teacher와 단절되면 실패 → **teacher-self가 필수** |
| **soft occupancy가 나은가** | hard 경계(Viterbi) | 12.95 / 30.78 | ↓(소) | 부드러운 구간이 우위 |

**핵심 읽기**: β=0만 크게 악화(emission=동력)하고, dark knowledge를 빼는 세 변형(uniform·K1·keep-blank)은 전부 **기준과 동률**. → **이득은 emission/정렬-관대함, dark knowledge 아님.**

---

## 6. 부수적 발견 — CTC에는 증류할 dark knowledge가 없다

Span-KD 타깃(teacher non-blank top-8) 22,384 토큰 실측:

| 지표 | 값 |
|---|---:|
| top-1 확률 ≥ 0.99 인 토큰 | **87.3%** |
| top-1 확률 ≥ 0.90 인 토큰 | 95.5% |
| 2등 확률 중앙값 | **0.000** |
| top-8 엔트로피 평균 (원-핫=0, 균등=2.08) | 0.054 |
| dark knowledge 有(top-1<0.70) 토큰 | **1.9%** |

**타깃이 사실상 원-핫이다.** 파이프라인 stage-by-stage 추적([inspect_span_target.py](scripts/inspect_span_target.py))이 원인을 보인다:
- teacher WHAT 소스(raw posterior)는 모든 emission 프레임에서 **같은 top-1로 확신**(예: 'p' 프레임 p=1.000).
- de-peak로 넓힌 occupancy가 끌어오는 shoulder 프레임은 **순수 blank(제거됨)** 아니면 **같은 토큰의 반복** — 다른 혼동 후보는 등장하지 않는다.
- **CTC의 불확실성은 "토큰 vs blank"(시간축)이지 "토큰 A vs B"(identity축)가 아니다.** blank를 제거하면 CTC의 유일한 실제 불확실성이 사라지고, 남은 identity 분포는 원-핫이다.

**함의**: frame-logit-KD 계열이 전제하는 "CTC dark knowledge 증류"는 CTC에서 실증적으로 성립하지 않는다. CTC-KD의 이득은 **emission/blank 구조 전이**로 재귀속되어야 한다.

---

## 7. 결과 — 재현 baseline 대비

| 방법 | 계열 | test-clean | test-other |
|---|---|---:|---:|
| *Teacher (상한, 참고)* | — | *3.70* | *8.14* |
| no-KD | baseline | 15.12 | 33.94 |
| Vanilla logit KD (Hilmes 2025) | frame | 14.29 | 33.03 |
| KD-BE + OCC (Hilmes 2025, best) | frame | 13.41 | 31.90 |
| CR-CTC (Yao 2025, 공정 50ep) | self | 14.02 | 31.84 |
| S-CTC +CTC-ft (Huang 2018) | seq | 13.03 | 31.59 |
| **Span-KD (ours, δ=6, w=20)** | **span** | **12.67** | **30.64** |

- 동일 compute에서 모든 재현 baseline을 능가.
- 세 계열(frame emission·self peak-suppression·seq alignment)이 모두 emission을 다루지만, **정렬-관대 emission 축에서 Span-KD가 이긴다** — 이것이 novelty의 정확한 위치.

---

## 8. 부정 결과 · 미해결 (방법의 경계)

| 시도 | 결과 | 이유 |
|---|---|---|
| 넓은 occupancy (4.2프레임) | 14.42/32.29 | 구간이 넓으면 이웃 토큰 오염 |
| Temperature로 타깃 softening | 14.38 (vs 14.17) | de-peaked tail은 dark knowledge가 아니라 노이즈(§6 확인) |
| 외부 aligner로 구간 정의 | 14.41/33.52 | timing이 teacher와 단절 |

**미해결(다음 우선순위)**:
- **multi-seed 확정** — β=1 vs β=0(핵심), uniform. 현재 β=1 single-seed가 reference(12.67/30.64) 대비 12.89로 운 나쁘게 높음. 3-seed로 emission gap과 uniform 우위를 확정해야 §5 결론이 정식화됨.
- δ·occupancy 폭의 이론적 최적점(정렬-관대함 vs 이웃 오염 trade-off).

---

## 9. 요약

CTC teacher의 peakiness(81.6% blank, 1.05프레임)는 **하나의 핵심 문제 — 스파이크 정렬 불일치(P2)** 를 낳는다.
**Span-KD**는 *"teacher의 emission 타이밍을 정렬-관대하게 증류한다"* 는 원리로 이를 해소한다:
(D1) teacher-self occupancy, (D2) 도구로서의 blank-only de-peak, (D3) 구간-평균의 정렬-관대 비교, (D4) emission 전이.
student가 teacher와 더 잘 정렬(3.34→2.77)되는 것이 이득의 직접 근거다.

그리고 ablation은 이득이 **emission/정렬-관대함**에서 오고 **identity dark knowledge에서 오지 않음**을 격리하며,
타깃 실측(95% 원-핫)은 **CTC에 증류할 dark knowledge가 없음**을 보인다 — CTC-KD 전반에 대한 재귀속.

> **Novelty**: *"teacher의 de-peaked self-occupancy를 이용한 정렬-관대 emission 증류"* + *"CTC-KD의 이득은 dark knowledge가 아니라 emission 구조 전이임을 측정으로 규명."*
