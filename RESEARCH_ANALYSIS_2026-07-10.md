# 전체 실험 분석 & 향후 로드맵 (2026-07-10)

> 소스: `analysis/eval_*.txt`(공식 WER), `EXPERIMENT_SUMMARY.md`, `METHOD_REPORT.md`,
> `nemo_experiments/` 런 기록. 모든 수치는 corpus-level greedy WER (%), `test_clean / test_other` 기준.
> 셋업: teacher `stt_en_conformer_ctc_small`(13M, 3.70/8.14), student Conformer-CTC d144×8L(~4.8M), train-clean-100, 100ep.

---

## 1. 실험 전체 지도

### 1.1 Baseline + Frame-level Logit KD (Hilmes 2025 계열)

| Run | 설정 | test_clean | test_other | 판정 |
|---|---|---:|---:|---|
| student-no-kd | — | 15.12 | 33.94 | 기준점 |
| kd-vanilla-w1 | 전 frame, w=1 | 15.03 | 34.16 | w 부족 |
| kd-vanilla-w10 | 전 frame, w=10 | 14.29 | 33.03 | |
| kd-be-w10 | blank elimination | 14.00 | 32.86 | |
| kd-sym-n1-w3 | ±1 frame 포함 | 14.61 | 33.32 | BE보다 나쁨 |
| kd-sym-n1-w5 | | 15.62 | 34.11 | 악화 |
| kd-sym-n1-w10 / n2-w10 | | 발산 | 발산 | boundary blank+고weight 불안정 |
| kd-sym-n1-lam05 | λ=0.5 | 15.87 | 35.35 | 악화 |
| kd-be-occ | +OCC=1 | 13.51 | 31.73 | |
| **kd-be-occ-w2** | +OCC=2 | **13.41** | **31.90** | **frame 계열 최고** |
| kd-be-occ-w5 / w10 | | 13.54 / 14.84 | 32.38 / 33.61 | OCC 과대 시 붕괴 |
| kd-be-res / occ-res | +RES | 13.65 / 13.41 | 31.96 / 32.24 | OCC=2 대비 이점 없음 |
| guided-ctc-w1 | Kurata 2019 hard CE | 13.90 | 32.77 | soft KD보다 열세 |

**결론**: frame 틀 안에서는 BE+OCC(≈13.4/31.9)가 상한. blank 처리 패치로는 여기까지.

### 1.2 De-peaked Teacher (fine-tuning) 계열 — 전부 실패

| Run | test_clean | test_other |
|---|---:|---:|
| logit-kd-ft-d0.5-t1-w10 | 14.17 | 32.18 |
| logit-kd-ft-d0.5-t2-w10 | 14.38 | 32.95 |
| span-kd-ft-d0.5-w20 | 14.42 | 32.29 |

teacher 가중치 자체를 de-peak하도록 fine-tune하면 frame/span 모두에서 오히려 악화.
**de-peak은 타깃 생성 시점(posterior 연산)에서만, 지식원은 원본 teacher 그대로**가 맞다는 증거.

### 1.3 CR-CTC (Yao 2025)

| Run | test_clean | test_other | 비고 |
|---|---:|---:|---|
| tmf=2.5 (v1~v4) | 발산 ×4 | | 논문 기본 세기 재현 불가 (이 규모에선) |
| cr_ctc_w02_v5 (tmf=1.5, 100ep) | 12.76 | 30.26 | **2× compute — 불공정** |
| cr_ctc_w02_fair50 (50ep) | 14.02 | 31.84 | compute 공정 |

### 1.4 Token-avg / Span-KD (ours)

| Run | test_clean | test_other | 비고 |
|---|---:|---:|---|
| token-avg-gt-w20 (hard Viterbi 경계) | 12.95 | 30.78 | soft 경계 ablation 대조군 |
| span-kd-prop-e10 (외부 phoneme aligner) | 17.01 | 36.91 | 실패 |
| span-kd-prop-e10-v2 | 14.41 | 33.52 | no-KD보다도 나쁨 |
| span-kd-teacher-d6 w=1 / 5 | 13.74 / 13.34 | 32.70 / 31.31 | |
| **span-kd-teacher-d6 w=20** | **12.67** | **30.64** | **전체 최고 (현 SOTA)** |
| span-kd-teacher-d6 w=50 | 13.25 | 31.80 | KD 과대 |
| d6-keepblank-w20 | 12.75 | 30.97 | blank 유지 이득 없음 |
| d9-mask-w20 | 12.86 | 30.93 | 폭 확대 손해 (마스킹으로도) |
| d12-mask-w20 | ep62 중단 (val 14.31) | | 폭 확대 단조 악화 확정 |

### 1.5 Span-KD "+α" (calibration slot) 시도 — 전부 노이즈 수준

| Run | dev_clean | dev_other | test_clean | test_other |
|---|---:|---:|---:|---:|
| d6-w20 (기준) | 12.53 | 29.93 | 12.67 | 30.64 |
| + ts05 (teacher time-smooth) | 12.38 | 30.53 | 12.65 | 30.97 |
| + sr02 (SR-CTC aux) | 12.37 | **29.79** | 12.80 | 30.82 |
| labelprior-a08 (WHERE를 prior de-peak) | 12.41 | 29.99 | 12.61 | 30.88 |

4개 지표에서 순위가 서로 뒤집힘 (test_clean 최저는 labelprior, dev_other 최저는 sr02, test_other 최저는 기준).
**단일 런 노이즈(±0.2~0.3, §2.1) 안에서의 차이 → 이 축은 소진됨.**

### 1.6 Seq-KD (Huang 2018 S-CTC) & Label-prior CTC — 대조군/실패

| Run | test_clean | test_other | 비고 |
|---|---:|---:|---|
| sctc-lambda1 (λ=1, CTC 없음) | 17.14 | 35.48 | alignment-only 증류는 붕괴 수준 |
| sctc-lambda1-ctcft (+CTC 20ep) | 13.03 | 31.59 | 회복해도 span-KD 미달 |
| span-kd-d6-lpctc a03 / a05 / a10 | 13.24 / 14.70 / 발산 | 31.18 / 33.10 / — | student-side prior 폐기 (사용자 결정) |

sctc 대조군의 가치: teacher의 **정렬만** 증류하면 부족하고 (17.14), **non-blank top-k 분포(dark knowledge)**가
실제 기여함을 보여주는 논문용 증거. 단, λ=1이라는 극단 설정만 돌렸음 (§4 참고).

---

## 2. 확립된 사실과 그 위의 허점

### 2.1 방법론적으로 확립된 것 (계열 간 격차가 노이즈보다 큼)

1. **구간 > 프레임**: span 12.67 < token-avg 12.95 < BE+OCC 13.41 < BE 14.00 < vanilla 14.29 < no-KD 15.12.
2. **teacher-self occupancy ≫ 외부 aligner** (12.67 vs 14.41): timing과 지식원의 단절이 치명적.
3. **δ=6이 폭의 최적점**, 단조 악화 곡선 확보 (12.67 → 12.86 → ~14 → 14.42). 이웃 마스킹으로도 구제 안 됨
   → 실패 원인은 이웃 오염이 아니라 **저증거 프레임에 토큰 질량 강제**.
4. **WHAT은 non-blank만** (keepblank 무이득), **w=20 최적** (50에서 꺾임).
5. **teacher de-peak fine-tuning은 지식 파괴** (전 계열 악화).
6. 순수 alignment 증류(sctc λ=1)로는 안 됨 → dark knowledge가 핵심 성분.

### 2.2 아직 취약한 것

- **전부 단일 seed 단일 런.** 같은 런의 best↔last ckpt 차이만 봐도 test WER ±0.1~0.3
  (kd-be-occ 13.41↔13.51, occ-w2 13.18↔13.41, sym-n1-w3 14.61↔14.80).
  §1.5의 모든 결론, 그리고 "span 12.67 vs token-avg 12.95" 같은 근소한 격차는 이 노이즈와 같은 자릿수.
  **논문 headline (no-KD / BE+OCC / CR-fair / span-KD)은 multi-seed 없이는 방어 불가.**
- **checkpoint 선택 정책 비일관**: 어떤 결과는 best-val ckpt, 어떤 결과는 last ckpt로 기록됨
  (`analysis/`에 `-best` 접미사 파일이 별도로 존재). 표마다 어느 쪽인지 명시가 안 된 곳이 있음.
- **METHOD_REPORT의 낡은 서술**:
  - §4 D3 ablation의 "넓은 occupancy" 대안으로 **ft-d0.5 (14.42)** 를 인용 — 이건 폭 + teacher fine-tuning이
    섞인 confounded 비교. 정직한 폭 ablation은 d9-mask(12.86)/d12-mask 단조 곡선.
  - §8 "구간이 넓으면 이웃 토큰 posterior가 섞여 오염" — d9/d12는 **마스킹이 켜진 상태**에서도 악화됐고
    leakage 실측 ~0.03%였으므로 이 인과 설명은 틀림. "저증거 프레임 강제"로 정정 필요.
  - δ9/δ12, sctc, lpctc, +α 계열 결과가 리포트에 아직 없음.
- **CR-CTC 공정 비교의 남은 티끌**: 원논문은 epoch 절반 + batch 절반. fair50은 epoch만 절반.
- 한 셋업(4× sub, 4.8M, 100h)에서만 검증됨 — 일반화 주장 불가.

### 2.3 성능 병목의 위치

teacher 3.70/8.14 vs best student 12.67/30.64. 최근 5개 실험이 전부 ±0.15 안에서 진동
→ **현 셋업에서 KD 목적함수 개선의 한계수익은 소진 단계.** 남은 격차의 지배 요인은
student 용량(4.8M)과 데이터(100h)이며, 방법 연구의 다음 스텝은 "더 쥐어짜기"보다
**(a) 결론의 통계적 확정, (b) 다른 셋업으로의 일반화, (c) 방법이 열어주는 새 능력(무라벨 데이터)** 쪽이 수익이 큼.

---

## 3. 앞으로의 방향 — 우선순위별 실험 목록

### P0. 논문 방어력 (새 아이디어보다 먼저)

1. **Multi-seed 확정 런**: no-KD / kd-be-occ-w2 / span-kd-d6-w20 각 +2 seed (총 6런).
   CR-fair50까지 하면 +2. 이게 끝나야 §1의 어떤 표도 인용 가능한 수치가 됨.
   결과 보고를 mean±std로 전환. (비용: 6~8 × 100ep — 지금까지의 스윕 하나 값)
2. **METHOD_REPORT 정정** (§2.2 세 번째 항목): D3 ablation을 d6→d9→d12 곡선으로 교체,
   §8 인과 설명 수정, sctc/lpctc/+α 부정 결과 추가. 학습 필요 없음, 즉시 가능.
3. **ckpt 정책 통일**: "항상 best-val ckpt"로 고정하고 표 전체 재확인.

### P1. 일반화 + 비교표 완결

4. **두 번째 셋업 재현**: sub8 student(브랜치·프리셋 이미 존재)에서 no-KD / BE+OCC / span-KD 3점.
   frame KD는 sub8에서 teacher(sub4)와 frame rate가 달라 더 불리해지고, span-KD는 구간 비교라
   원리적으로 강건해야 함 — **방법의 차별점이 오히려 커지는 셋업**이라 스토리에 유리.
   (대안: train-clean-360으로 데이터 축 일반화)
5. **미완 baseline 2개**: `52_self_kd_l4_a05.sh` (Kim 2024), `51_delayed_kd_tab2_w10.sh` (Li 2025).
   프리셋이 이미 있고 비교표의 빈칸. Hilmes trim/threshold/random은 BE+OCC가 이미 대표라 선택사항.
6. **misalignment 진단표 채우기** (BASELINE_COMPARISON.md TBD): 학습 없이 `diagnose_ctc_mismatch.py`만
   돌리면 됨. §6 메커니즘 주장(3.34→2.77)을 baseline 전반으로 확장하는 저비용 고효율 작업.

### P2. 성능을 더 올릴 후보 (기대수익 순)

7. **span-KD + OCC 보조항**: frame 계열에서 유일하게 재현성 있게 유의미했던 보조항(OCC)은
   blank/non-blank **occupancy 축**을 조이는데, span-KD는 이 축에 무압력(§6.3 emit 0.915 불변).
   +α 계열(vocab softening)이 전멸한 것과 달리 이건 **직교 축**이라 아직 살아있는 가설.
   구현: sctc 타깃의 `1-Σγ`(blank occupancy)를 재활용하면 frame top-k 매니페스트 병합 없이 가능.
8. **student-side occupancy 집계 (dual-occupancy span-KD)**: 현재 student 평균은 teacher γ를
   시간축 리샘플해 쓰므로 사실 **teacher-anchored** 비교임. student 자신의 transcript-constrained
   γ(on-the-fly, detached)로 student 쪽을 집계하면 진짜 정렬-무관 비교가 됨.
   초기엔 student γ가 쓰레기이므로 `kd_start_step`(이미 구현됨)으로 워밍업. 방법적 novelty 한 단계 추가.
9. **sctc 정식 baseline 1런**: λ=1이 아니라 joint(λ=0.5 또는 legacy w 스윕 1점)로.
   지금은 "극단 설정의 실패"만 있어서 baseline 표에 넣기 애매함.
10. **iterative KD**: span-KD student(12.67)를 WHERE 소스로 쓰는 2라운드 — 저비용, 시도 가치는 중간.

### P3. 스케일/임팩트 확장 (논문 스토리를 키우는 방향)

11. **무라벨 데이터 확장 (가장 큰 잠재력)**: span 타깃은 transcript-constrained FB가 필요하지만,
    transcript를 **teacher greedy 출력**으로 대체하면 train-other-500 같은 무라벨 오디오에도 적용 가능.
    "정렬-관대 증류가 pseudo-label 노이즈에도 강건한가"라는 새 연구 질문이자,
    100h 한계를 실제로 뚫는 유일한 경로.
12. **더 큰 teacher** (`stt_en_conformer_ctc_medium/large`): 단, tokenizer가 현 1024 BPE와 다르면
    cross-tokenizer 증류가 필요해 대공사 — 착수 전 vocab 호환부터 확인.
13. span-KD + CR-CTC 결합: `training_step` 구조 변경 필요 + compute 2×라 공정 비교 프레임과 충돌.
    P3 이하 권장.

### 하지 말 것 (기각 확정 — 재제안 금지)

- δ>6 폭 확대 (마스킹 유무 무관, 단조 악화 확정)
- label-prior 계열 전부 (student-side lpctc, teacher-side labelprior — 사용자 결정으로 폐기)
- teacher de-peak fine-tuning (ft-d0.5 계열 전멸)
- vocab-axis softening 추가 변형 (temperature, time-smooth, SR-CTC 모두 무이득)
- 외부 aligner 기반 support (prop-e10 계열)

---

## 4. 코드 리뷰 — 오류·허점

### A. 결과에 영향 줄 수 있는 것

1. **`scripts/evaluate_student.py:load_model` — 조용한 아키텍처 불일치 위험 (가장 위험).**
   ckpt의 저장된 하이퍼파라미터가 아니라 **현재의** `configs/student_base.yaml`로 모델을 만들고
   `load_state_dict(..., strict=False)` 하면서 missing/unexpected 키를 **출력조차 안 함**.
   config가 학습 시점과 달라지면(d_model, subsampling, lpctc buffer 등) 일부 가중치가 랜덤 초기화된 채
   그대로 평가됨. sub8 실험을 재개하면 바로 밟는 지뢰.
   → missing/unexpected를 출력하고, encoder/decoder 키에 하나라도 missing이 있으면 abort하도록 수정 권장.

2. **`src/model.py` — `kd_lambda`가 `token_avg`/`aligned_token`/`combined` 브랜치에서 무시됨.**
   이 브랜치들은 `self._mix()` 대신 legacy 식을 하드코딩. `--kd-lambda`를 이 모드와 함께 주면
   init 검증(`kd_lambda is None and kd_weight==0`)은 통과하는데, `kd_weight=0`이면
   **KD 항이 0이 된 채 조용히 no-KD 학습**이 됨. `_mix` 사용으로 통일하거나 명시적 에러 필요.

3. **frame-KD 계열의 `teacher_frames == frames` 분기 (`_frame_logit_kd_loss`, `_frame_occ_res_loss`, `_guided_kd_loss`)**:
   1:1 프레임 대응 여부를 **padded 텐서 차원의 일치**로 판정. per-utterance 길이가 달라도 batch의
   max가 우연히 같으면 잘못 정렬된 채 통과. 현 4×/4× 셋업에선 per-utt 길이가 항상 같아 무해하지만,
   가정이 코드에 검증 없이 묻혀 있음. 최소한 assert 또는 per-utt 길이 비교로 전환 권장.

4. **`_span_kd_loss` — support가 전부 0인 토큰의 CE 오염**: 유효 프레임 범위에서 support 합이 0이면
   정규화 후 `s_avg=0 → clamp(1e-9).log()` → 토큰당 **상수 ≈20.7 CE**가 gate=1로 loss에 합산
   (gradient는 0). 드물지만 loss 통계를 왜곡하고 `total_w`를 부풀림. `sup_s.sum()==0`이면 skip해야 함.

5. **fp16 타깃 저장 (`support`, `token_gamma`)**: fp16은 ~6e-5 미만에서 정밀도 급락, ~6e-8 미만은 0으로
   flush. δ가 클수록 γ tail이 잘려나감 — d9/d12 해석에 소량의 교란 가능성 (결론을 뒤집을 크기는 아님).

### B. 견고성 / 재현성

6. **`_sr_ctc_loss` 경계 오염**: smoothing 타깃이 `enc_len` 밖 padding frame의 posterior를 한 프레임 끌어옴
   (KL 마스크는 있지만 타깃 생성엔 마스크 없음). 발화당 1프레임짜리 미세 오염.
7. **label-prior EMA buffer가 `alpha>0`일 때만 register** — lpctc ckpt ↔ 일반 config 간 load 시
   missing/unexpected key. lpctc가 폐기됐으니 **코드 자체를 제거**하는 게 깔끔함
   (`model.py`, `train.sh`, `student_base.yaml` 세 곳).
8. **`_eval_step` WER 누적**: rank-local corpus WER을 `sync_dist=True` log가 단순 평균 —
   멀티 GPU에서는 정확한 corpus WER이 아님 (denominator 불균등). 현재 단일 GPU라 실해 없음.
9. **`cr_ctc_weight=0.2`가 모든 run의 기본 config에 잔존**: `kd_mode!=cr_ctc`면 dead지만, cmd-args만 보고
   CR이 켜진 것으로 오독한 전례가 실제로 있음. 기본값을 0.0으로 바꾸고 cr_ctc 프리셋에서만 세팅 권장.
10. **builder들의 `--out-dir` 의미**: `out_dir.name`만 취해 항상 manifest 옆에 저장 — 절대경로를 줘도
    무시됨. 인자 이름과 동작이 불일치 (문서화 또는 수정).
11. **`--limit` + `--manifest-out`**: 잘린 rows만 기록되므로 기존 full manifest 경로를 실수로 주면 유실.
12. **`--resume` 검증이 `torch.load` 전체 로드**: 28k 파일 재개 시 매우 느림. 존재/크기 체크로 충분.

### C. 경미 (인지만)

13. `torch.load(..., weights_only=False)` 다수 — 자체 생성 파일이라 OK.
14. `student-no-kd`부터 지금까지 모든 런이 같은 val set(`dev_clean`)으로 best-ckpt 선택 —
    dev_other가 선택에 안 들어가므로 other 계열 수치는 약간 비관적일 수 있음 (일관적이므로 비교엔 무해).
15. `_ctc_viterbi_align`의 tie-breaking(stay 우선)은 표준적 — 문제없음.

---

## 5. 요약 — 이번 주에 하면 좋은 것

| 순서 | 작업 | 비용 | 성격 |
|---|---|---|---|
| 1 | METHOD_REPORT §4/§8 정정 + sctc/lpctc/+α 결과 반영 | 0 GPU | 문서 |
| 2 | evaluate_student strict-load 경고 추가 (코드 리뷰 A1) | 몇 분 | 코드 |
| 3 | multi-seed 6런 (no-KD / be-occ-w2 / span-d6-w20 × seed 2개 추가) | 6×100ep | P0 |
| 4 | misalignment 진단표 채우기 | 인퍼런스만 | P1 |
| 5 | self_kd / delayed_logit 프리셋 2런 | 2×100ep | P1 |
| 6 | (병렬로) span+OCC 1런 — 살아있는 유일한 "+α" 가설 | 1×100ep | P2 |
