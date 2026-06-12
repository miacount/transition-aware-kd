# 실험 정리

이 파일은 cleanup 이후의 공식 실험 결과를 기록하는 문서입니다.

이전 실험 산출물과 분석 파일은 정리되었고, 삭제 전 파일 목록은 `CLEANUP_MANIFEST.txt`에 보존했습니다.

## 공식 기준

Teacher:

```text
stt_en_conformer_ctc_small
1024 BPE + blank
4x subsampling
```

Student:

```text
Conformer CTC
1024 BPE + blank
d_model=144
layers=8
heads=4
4x subsampling
```

공식 평가 split:

```text
dev_clean
dev_other
test_clean
test_other
```

공식 WER는 corpus-level WER만 사용합니다.

## 실험 실행

No-KD:

```bash
bash experiments/presets/00_no_kd.sh
```

Transition KD:

```bash
bash experiments/presets/10_build_transition_targets.sh
bash experiments/presets/11_transition_kd_w025.sh
```

Vanilla logit KD:

```bash
bash experiments/presets/20_build_frame_topk_targets.sh
bash experiments/presets/21_vanilla_logit_kd_w01.sh
```

## 결과 기록

student: Conformer CTC d_model=144, layers=8, heads=4, 4x subsampling, 1024 BPE
teacher: stt_en_conformer_ctc_small

### kd_weight sweep

| Run | KD | kd_weight | dev_clean | dev_other | test_clean | test_other | 비고 |
|---|---|---:|---:|---:|---:|---:|---|
| student-none (epoch 98) | none | — | 14.69% | 33.58% | 14.97% | 34.23% | baseline |
| student-transition-w0.1 (epoch 97) | trans | 0.1 | 14.37% | 33.26% | 14.70% | 33.68% | |
| student-transition-w0.25 (epoch 98) | trans | 0.25 | **14.05%** | **32.89%** | **14.49%** | 33.62% | trans best |
| student-transition-w0.5 (epoch 98) | trans | 0.5 | 14.42% | 33.27% | 14.88% | 33.87% | |
| student-logit-t1-w1.0 (epoch 99) | logit T=1 | 1.0 | 14.17% | 32.59% | 14.71% | 33.44% | |
| student-logit-t1-w5.0 (epoch 92) | logit T=1 | 5.0 | 13.97% | 32.52% | 14.37% | 32.91% | |
| student-logit-t1-w10.0 (epoch 97) | logit T=1 | 10.0 | **13.74%** | **31.84%** | **13.87%** | **32.89%** | logit best |

logit KD (T=1, top-8)가 trans KD를 전 split에서 앞섬.
logit w=10.0 기준 baseline 대비 test_clean -1.10%p, test_other -1.34%p.
logit weight sweet spot이 10.0 이상일 가능성 있음 (추가 sweep 예정).

## Diagnostic 기록

| Run | frame mismatch | blank/nonblank mismatch | transition edit rate | 비고 |
|---|---:|---:|---:|---|
| TBD | - | - | - | cleanup 이후 재측정 필요 |
