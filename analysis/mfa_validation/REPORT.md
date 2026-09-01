# Blank-suppressed occupancy의 MFA 검증 (2026-07-11)

> 질문: δ(blank-only de-peak)로 넓어진 occupancy가 **의미 있게** 넓어진 것인가(H1),
> 아니면 그냥 blank만 죽여서 아무 질량이나 살아난 것인가(H2)?
> 특히 — 경계 blank 프레임에서 blank를 빼고 정규화하면 남는 질량이 **인접 토큰**인가, 무관한 non-blank인가?
>
> 셋업: teacher `stt_en_conformer_ctc_small`, test-clean 첫 200발화 중 190 유효
> (MFA transcript 불일치 10 skip), blank 프레임 34,404개. MFA 정렬: gilkeyio/librispeech-alignments
> (Montreal Forced Aligner, word+phone 구간). 스크립트: `scripts/analyze_mfa_occupancy.py`.
> 대조군: δ=0 γ를 δ=6과 **평균 폭이 같도록**(PR width 1.441) Gaussian σ=0.454로 smoothing한 대칭 kernel.
> sanity: mean p(blank)=0.806 (리포트의 0.816과 일치), fast FB는 레퍼런스 구현과 allclose 검증.

## 판정 요약

**H1 — 넓힘은 의미 있음. 단, "토큰 구간(span) 모델링"이 아니라 "스파이크 ±1프레임 shoulder의 선택적 복원"이다.**

1. 임의의 blank 프레임에서 blank를 빼고 정규화하면 잔여질량은 대부분 쓰레기다 (인접 토큰 18%, transcript 밖 59%).
2. 그러나 δ=6이 **실제로 모집한** 프레임에서는 잔여질량이 그 토큰 자신이다
   (mass-weighted 68% vs 같은 폭 대칭 kernel 32%, **2.1×**). transcript-제약 FB가
   "잔여가 해당 토큰인 프레임"만 골라 모집하고, junk 프레임은 δ=6에서도 억제된 채 남는다.
   → "그냥 blank만 죽인 것"이 아님. blank-suppression **단독**으로는 무의미하고,
   **FB의 선택성과 결합**될 때만 의미가 생긴다.
3. shoulder는 스파이크 ±1프레임에만 존재한다. dist≥2에는 복원할 identity가 없다
   → δ>6 단조 악화의 음향적 근거 확보.

## A. 모집 프레임 residual identity — 핵심 표

blank-제거 후 정규화한 residual에서, 모집된 (frame, token) 쌍의 **자기 토큰 점유율**:

| | n pairs | r(y_u) mean | r(y_u) mass-wtd | 모집질량의 MFA 단어 내 비율 |
|---|---:|---:|---:|---:|
| **δ=6** | 8,859 | 0.507 | **0.681** | **0.985** |
| kernel-ctl (같은 폭) | 15,593 | 0.328 | 0.322 | 0.978 |

같은 평균 폭인데 δ=6의 모집 질량은 kernel보다 2.1× 더 "자기 토큰"에 실려 있다.
kernel은 양쪽으로 무차별 확산해 **이웃 토큰 오염**이 3배다 (added target의 neighbour: δ6 0.073 vs ctl 0.211).

blank 프레임 전수(34,404) 층화:

| 층 | n | 인접 토큰 share | 인접 top-1 | transcript 밖 | MFA 단어 top-1 |
|---|---:|---:|---:|---:|---:|
| 전체 | 34,404 | 0.184 | 0.245 | 0.591 | 0.222 |
| dist=1 (스파이크 옆) | 13,716 | 0.392 | 0.526 | 0.429 | 0.481 |
| dist=2 | 7,804 | 0.084 | 0.103 | 0.638 | 0.099 |
| dist≥3 | 12,845 | ~0.02 | ~0.03 | ~0.73 | ~0.02 |
| **δ=6 모집 프레임** | 8,325 | **0.533** | **0.713** | 0.325 | **0.645** |
| 비모집 프레임 | 26,079 | 0.073 | 0.096 | 0.676 | 0.087 |

- top-1이 인접 토큰이 아닌 모집 프레임(28%)의 정체: BPE 표기 변형은 3%뿐, 대부분 apostrophe·단일 문자류 junk
  (인접-또는-변형 합산 72.1%).
- 순환성 주의: FB emission이 p(y_u)를 곱하므로 "모집 프레임에 y_u가 있음"은 절반은 구성상 당연.
  비순환 근거는 (i) 모집질량의 98.5%가 **MFA 단어 구간 내** (외부 기준), (ii) 같은 폭 kernel 대비 2.1× 선택성.

## B. MFA 구간 대비 coverage / 비대칭 / 폭 (토큰 단위, 단어 interval 기준)

| variant | cov(strict) | cov(±1) | added-mass cov(±1) | added-mass shift(frames) | width | r(width~dur) P/S |
|---|---:|---:|---:|---:|---:|---|
| δ=0 | 0.958 | 0.996 | — | — | 1.08 | −0.05/−0.28 |
| δ=3 | 0.958 | 0.996 | 0.989 | +0.24 | 1.16 | −0.17/−0.28 |
| **δ=6** | 0.952 | 0.995 | **0.987** | **+0.24** | 1.44 | −0.22/−0.27 |
| δ=9 | 0.930 | 0.985 | 0.977 | +0.27 | 2.31 | −0.09/−0.18 |
| δ=12 | 0.882 | 0.954 | 0.943 | +0.35 | 3.80 | +0.08/+0.06 |
| kernel-ctl | 0.947 | 0.994 | 0.979 | **−0.00** | 1.44 | −0.05/−0.28 |

- δ=6 추가 질량의 98.7%가 단어 구간(±1프레임) 안 — 넓힘이 음향 구간을 벗어나지 않는다.
- 추가 질량은 **비대칭**(+0.24프레임, 단어 뒤쪽 방향): 데이터가 결정한 방향성. kernel은 구성상 0.
  스파이크가 MFA 단어의 앞쪽(평균 0.39, 중앙값 0.35 지점)에 찍히므로, 뒤쪽 넓힘 = 단어 본체 방향.
- **폭-길이 상관 없음**(전 δ에서 ~0 또는 음): 넓힘은 duration 모델링이 아니다.
  δ=6 폭 1.44프레임 vs 단어 길이 수 프레임~수십 프레임 — occupancy는 토큰의 실제 구간 중
  스파이크 주변 일부만 덮는다. **"구간 정의"라는 서술은 과대주장이며 "shoulder 복원 + 정렬 관대"가 정확하다.**

## C. δ>6 실패의 정량 해부

| δ | 모집 occupancy 질량 | added의 self share | (참고) blank 프레임 dist≥2 identity |
|---|---:|---:|---|
| 3 | 0.034 | 0.487 | shoulder(dist=1)만 모집 |
| 6 | 0.135 | 0.470 | 주로 dist=1 |
| 9 | 0.353 | 0.439 | dist≥2 침범 시작 |
| 12 | 0.566 | 0.377 | dist≥2·3 대량 모집 |

dist=2부터 잔여 identity가 붕괴(0.39→0.08)하므로, δ>6은 **복원할 정보가 없는 프레임에 질량을 강제**한다.
RESEARCH_ANALYSIS §2.2의 "저증거 프레임 강제" 설명이 음향 데이터로 확정됨 (이웃 오염이 아님 — 이웃 share는 오히려 감소).

## D. WHAT 타깃은 δ와 사실상 무관 — 이득 경로는 WHERE

| variant | full target self | added part의 유효 기여(eff-wt) |
|---|---:|---:|
| δ=0 | 0.978 | — |
| δ=6 | 0.977 | 0.0076 |
| δ=12 | 0.963 | 0.0221 |

γ-가중 WHAT 평균에서 shoulder 프레임의 기여는 (1−p_blank)≈0.006 가중 때문에 **0.8%**에 불과 —
semantic top-k는 δ=0과 δ=6이 사실상 동일하다. 따라서 Span-KD의 이득은 WHAT 변화가 아니라
**student-side span-averaging support(WHERE)의 정렬 관대함**에서 온다. D3의 "non-blank 비율 보존
→ dark knowledge 불변" 주장이 실측으로 확인되는 동시에, 방법의 작동 축이 WHERE임이 분리 입증됨.

## 함의

1. **METHOD_REPORT 반영**: §3.2 D3 "각 토큰의 시간 구간을 정의" → "스파이크 주변 shoulder를 선택적으로
   복원해 정렬-관대한 support를 만든다"로 정정. §6에 본 분석(모집 identity 68% vs kernel 32%,
   added mass 98.7% in-word) 추가. §8 넓은 δ 실패 원인을 C표로 대체.
2. **Learnable/adaptive δ (아이디어 2) 사실상 기각**: dist≥2에 복원할 identity가 없으므로
   토큰 길이에 맞춰 넓히는 것은 데이터가 반대함. 폭의 상한은 음향적 shoulder(±1프레임)이고 δ=6이 이미 근접.
3. **Kernel-control 학습 런은 불필요해짐**: identity 지표에서 이미 2.1× 차이로 판별 완료.
   (돌린다면 "kernel도 WER은 비슷하냐"는 확인용일 뿐 — 예측: coverage는 비슷해도 이웃 오염 3×
   + junk 방향 gradient 때문에 δ=6보다 나쁠 것.)

산출물: `analysis/mfa_validation/{summary.json, blank_frame_records.npz, token_records.npz}`,
스크립트 `scripts/analyze_mfa_occupancy.py` (δ 목록·σ·eps 인자화, FB 벡터화 + 레퍼런스 검증 내장).

---

## 후속: MFA-support span-KD 학습 결과 (2026-07-11, D2 확정)

WHERE만 MFA 좌표(단어 구간 + 글자비례 BPE 분할, 균일 box)로 교체하고 WHAT·손실·아키텍처(sub4/d144)·
w=20·100ep 전부 동일하게 학습 (`scripts/build_mfa_span_targets.py`, kept 28,193/28,539, `<unk>` 와일드카드 회수).

| | test-clean | test-other | KD loss(최종) | blank% | frames/tok | teacher와 spike offset | occ-mass-in-MFA |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-KD | 15.12 | 33.94 | — | 0.804 | 1.17 | 0.69 | 0.958 |
| **span-d6 (ours)** | **12.67** | **30.64** | **1.47** | 0.657 | 2.01 | **0.24** | 0.966 |
| MFA-support | 14.50 | 32.52 | 2.14 | **0.446** | **3.32** | 0.56 | **0.973** |
| (구 자체 aligner) | 14.41 | 33.52 | — | — | — | — | — |

(spike offset은 transcript-제약 FB γ argmax 기준, test-clean 148발화; §6의 greedy-spike 지표와 별개 척도)

핵심 판독:
1. **Student는 MFA WHERE에 성실히 복종했다** — blank 80→45%, 토큰당 3.32프레임, occupancy 폭 3.50
   (타깃 box 폭 ~4.0), MFA 구간 내 질량 97.3%로 전 모델 중 최고. 실패는 "타깃 무시"가 아니라
   **"타깃을 따른 결과"**다.

### WHERE 좌표계 삼각측량 (2026-07-13, dual-occupancy 결과 추가)

WHERE의 출처만 바꾼 3점 비교 (WHAT·손실·아키텍처·w=20·100ep 동일; dual은 kd_start_step=2000):

| WHERE 좌표계 | test-clean | test-other | blank% | frames/tok | teacher와 spike offset | KD loss 수렴 |
|---|---:|---:|---:|---:|---:|---:|
| **teacher 자신 (d6, ours)** | **12.67** | **30.64** | 0.657 | 2.01 | **0.24** | 1.47 |
| 외부 시계 (MFA) | 14.50 | 32.52 | 0.446 | 3.32 | 0.56 | 2.14 |
| student 자신 (dual, detached) | 15.30 | 33.43 | 0.430 | 3.37 | **0.80** | 1.58 |
| (no-KD 기준점) | 15.12 | 33.94 | 0.804 | 1.17 | 0.69 | — |

- **Dual ≈ no-KD (clean에선 더 나쁨)**: 자기-좌표 pooling은 KD 전달을 사실상 0으로 만든다.
  KD loss는 1.58로 정상 수렴했는데 WER 이득이 없다 — loss 만족과 지식 전달이 분리된 사례.
- **폭주 self-widening**: 좌표가 매 스텝 student를 따라가므로(detach는 gradient만 끊지 스텝 간
  피드백은 못 끊음) "자기 span을 순수하게" 압력이 방출 확장→γ 확장→더 넓은 pooling의 양성 피드백을
  만들어 폭 3.58까지 비대해짐. anchored는 고정 외부 support(teacher γ)라 이 루프가 구조적으로 없음.
- **offset 0.80 (no-KD 0.69보다 후퇴)**: 위치 앵커가 없으니 teacher와의 동기가 오히려 악화.
  CTC loss도 81.7 vs 75.2(anchored)로 동반 악화 — 간섭 비용만 내고 지식은 못 받음.
- **결론**: span-KD의 이득은 "관대한 비교" 단독이 아니라 **"teacher 좌표에 관대하게 배달"**에서 온다.
  외부 좌표(MFA)는 지식을 희석시키고(약한 KD), 자기 좌표(dual)는 배달 자체를 없앤다(no KD).
  D1/D2가 3점 삼각측량으로 완결.

### no-FB ablation (2026-07-13): FB의 한계기여 = occurrence별 dark knowledge 분해능

WHERE를 FB γ 대신 **de-peaked posterior 그 자체** support(t,u)=p̃_δ6(y_u|t)로 교체 (FB 완전 제거,
`--where-mode posterior`). 결과: **13.02 / 31.51** — full span-KD(12.67/30.64)에 +0.35/+0.87,
그러나 여전히 모든 frame baseline(최고 13.41/31.90)보다 우위.

WHERE 좌표 사다리 최종형 (teacher의 sequential alignment에서 멀어질수록 단조 악화):

| WHERE 출처 | test-clean | test-other |
|---|---:|---:|
| teacher FB γ (full) | **12.67** | **30.64** |
| teacher posterior, FB 없음 | 13.02 | 31.51 |
| 외부 정렬 (MFA) | 14.50 | 32.52 |
| student 자신 (dual) | 15.30 | 33.43 |
| (no-KD) | 15.12 | 33.94 |

실패 채널 규명 (사전 예측은 절반만 적중 — 모집단은 맞았고 채널은 틀림):
- **WHERE는 무사**: 반복-id 토큰(32%)의 no-FB support가 multimodal(모든 occurrence로 분산,
  FB γ와의 overlap 0.365)임에도, 학습된 student의 스파이크 offset(0.248 vs 0.243)·집중도(0.997)는
  FB student와 동일 — CTC의 단조 정렬이 위치를 지켜준다.
- **진짜 채널 = WHAT 희석**: 타깃 직접 비교(16,591토큰)에서 고유-id 토큰은 FB/no-FB 타깃이
  사실상 동일(L1 0.0095)하지만, 반복-id 토큰은 L1 5.5×(0.0518)에 **dark-knowledge tail이
  36% 소실**(non-self 질량 0.0270→0.0174; cross-occurrence 평균이 occurrence별 혼동구조를
  상쇄해 타깃이 one-hot 쪽으로 뾰족해짐). KD loss가 더 낮게 수렴(1.34 vs 1.47)한 이유 —
  타깃이 쉬워진 것이지 지식이 더 전달된 게 아니다.
- **결론**: FB의 한계기여는 위치 배정이 아니라 **occurrence별 dark knowledge de-multiplexing**.
  "tail 32% 토큰에서 36% 소실 ↔ WER +0.35/+0.87"의 정량 대응은 sctc 교훈("유효 성분은
  dark knowledge")의 독립 재확인이기도 하다. 단순화 트레이드오프: FB 없이도 frame 계열은
  전부 이기므로, 저비용 구현이 필요하면 no-FB가 방법의 90%를 준다.
2. **외부 시계에 맞추자 teacher와의 동기가 깨졌다** — spike offset 0.24(d6) → 0.56(MFA), no-KD(0.69)
   쪽으로 후퇴. dark knowledge가 teacher가 말하는 시점이 아닌 곳에서 읽혔다.
3. **타깃이 끝까지 만족 불가** — KD loss 2.14 vs 1.47(+46%), CTC loss도 78.3 vs 75.2로 동반 악화
   (균일 box는 CTC가 선호하는 정렬과 상시 충돌).
4. **De-peaking 자체는 이득이 아님의 재확인** — MFA student가 가장 덜 peaky(44.6%)인데 WER는 d6보다
   1.8 나쁨. §6.3("이득은 정렬-관대 dark knowledge이지 평탄화가 아님")의 학습-수준 증거.
5. **정렬기 품질은 병목이 아니었다** — 최강 외부 정렬기(MFA, 단어 containment 99.5%)가 자체 학습한
   약한 aligner(14.41/33.52)와 같은 수준(14.50/32.52). 외부 timing이라는 **구조** 자체가 문제
   → D2("teacher가 정렬원이자 지식원이어야 한다") 확정, "정렬기가 약해서 진 것" 반론 봉쇄.
