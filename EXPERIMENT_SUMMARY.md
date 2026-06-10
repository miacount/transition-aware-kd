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

| Run | KD | dev_clean | dev_other | test_clean | test_other | 비고 |
|---|---|---:|---:|---:|---:|---|
| TBD | none | - | - | - | - | cleanup 이후 재실험 필요 |
| TBD | transition | - | - | - | - | cleanup 이후 재실험 필요 |
| TBD | logit | - | - | - | - | cleanup 이후 재실험 필요 |

## Diagnostic 기록

| Run | frame mismatch | blank/nonblank mismatch | transition edit rate | 비고 |
|---|---:|---:|---:|---|
| TBD | - | - | - | cleanup 이후 재측정 필요 |
