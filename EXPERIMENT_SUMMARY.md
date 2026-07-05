# 실험 정리

이 문서는 cleanup 이후 실제로 수행한 실험만 공식 기록으로 정리한다. 현재 완료된 실험 범위는 frame-level logit KD의 blank selection / blank-context 계열이다. Token-avg, 8x student, combined sweep은 cleanup 이후 공식 완료 실험이 아니므로 이 문서의 결과 표에서 제외한다.

## 기준

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

공식 WER는 `scripts/evaluate_student.py`가 출력한 corpus-level WER만 기록한다.

## Cleanup 이후 완료 실험 범위

완료된 run:

```text
student-no-kd
kd-vanilla-w10
kd-be-w10
kd-sym-n1-w3
kd-sym-n1-w5
kd-be-occ
kd-be-res
kd-be-occ-res
kd-be-occ-w2
kd-be-occ-w5
kd-be-occ-w10
```

실험 축:

| 축 | 설명 |
|---|---|
| No-KD | student baseline |
| Vanilla KD | 전체 frame top-k logit KD |
| KD-BE | teacher argmax non-blank frame만 사용 |
| Sym n=1 | KD-BE frame 주변 +/-1 frame까지 포함 |
| OCC | blank/non-blank occupancy KL 보조항 |
| RES | blank-normalized residual/token distribution KL 보조항 |

## 결과

| Run | KD 설정 | dev_clean | dev_other | test_clean | test_other | 비고 |
|---|---|---:|---:|---:|---:|---|
| `student-no-kd` | none | 14.92 | 33.21 | 15.12 | 33.94 | baseline |
| `kd-vanilla-w10` | vanilla logit KD, w=10 | 14.21 | 32.42 | 14.29 | 33.03 | frame 전체 KD |
| `kd-be-w10` | KD-BE, w=10 | 13.69 | 31.88 | 14.00 | 32.86 | blank elimination |
| `kd-sym-n1-w3` | Sym n=1, w=3 | 14.15 | 32.34 | 14.61 | 33.32 | boundary blank 포함 |
| `kd-sym-n1-w5` | Sym n=1, w=5 | 15.10 | 33.50 | 15.62 | 34.11 | 악화 |
| `kd-be-occ` | KD-BE + OCC=1 | 13.29 | 31.24 | 13.51 | 31.73 | OCC 추가 |
| `kd-be-res` | KD-BE + RES | 13.23 | 31.58 | 13.65 | 31.96 | residual KL 추가 |
| `kd-be-occ-res` | KD-BE + OCC=1 + RES=2.63 | 13.23 | 31.91 | 13.41 | 32.24 | 혼합 보조항 |
| `kd-be-occ-w2` | KD-BE + OCC=2 | 12.99 | 31.10 | 13.41 | 31.90 | dev 기준 최상, best ckpt 기준 |
| `kd-be-occ-w5` | KD-BE + OCC=5 | 13.20 | 31.56 | 13.54 | 32.38 | OCC 과대 가능 |
| `kd-be-occ-w10` | KD-BE + OCC=10 | 14.63 | 32.63 | 14.84 | 33.61 | 과대 weight로 악화 |

## 관찰

- Vanilla logit KD는 No-KD 대비 개선되지만, blank frame이 여전히 loss를 크게 지배한다.
- KD-BE는 vanilla보다 명확히 낫다. teacher non-blank frame만 쓰는 단순한 blank elimination만으로도 `test_other`가 33.03에서 32.86으로 내려간다.
- Sym n=1은 이번 sweep에서 안정적이지 않았다. w=3, w=5 모두 KD-BE/OCC 계열보다 낮다.
- OCC 보조항은 효과가 있다. `kd-be-occ-w2`가 dev 기준으로 가장 좋고, `kd-be-occ`도 test_other 31.73으로 강하다.
- OCC weight를 너무 키우면 악화된다. w=10은 vanilla KD와 큰 차이가 없을 정도로 무너진다.
- RES 단독 또는 OCC+RES는 나쁘지 않지만, 현재 표에서는 OCC=2보다 뚜렷한 이점이 없다.

## 현재 결론

현재 cleanup 이후 실험만 놓고 보면 가장 유망한 방향은 `KD-BE + 적당한 OCC 보조항`이다. 다만 이 계열은 여전히 teacher의 frame-level BPE spike 위치에 의존한다. 다음 실험은 frame 위치를 그대로 맞추는 대신 soft aligner로 BPE token과 acoustic span을 맞춰 teacher posterior를 더 안정적으로 집계하는 방향으로 진행한다.

## 다음 실험 계획: Soft Aligner + BPE Matching

목표:

```text
soft aligner로 acoustic span을 얻고,
BPE token sequence와 span을 맞춰,
frame-level spike 하나에 의존하지 않는 BPE-level KD target을 만든다.
```

계획:

1. soft aligner가 만든 phoneme posterior와 blank 분포를 확인한다.
2. transcript의 BPE token sequence를 word/character span을 거쳐 phoneme span에 매칭한다.
3. BPE별 aligner support `a(t,u)`를 만들고, teacher CTC occupancy `gamma_T(t,u)`와 곱해 teacher semantic target을 집계한다.
4. student posterior는 같은 BPE support `a(t,u)` 위에서 평균내어 token-level prediction을 만든다.
5. 기존 KD-BE/OCC best run과 비교할 수 있도록 같은 4x student 설정에서 학습한다.
6. 결과는 이 문서의 별도 `Soft aligner BPE KD` 섹션에 새 표로 추가한다.

현재 준비 상태:

| 항목 | 상태 |
|---|---|
| soft aligner checkpoint | `nemo_experiments/phoneme_soft_aligner/epoch_010.pt` |
| 기존 span target | `data/span_kd_prop_e10/`, gamma-only teacher target |
| 새 target 방식 | `q_T,u = sum_t normalize(gamma_T(t,u) * a(t,u)) P_T(t)` |
| 구현 옵션 | `scripts/build_span_kd_targets.py --teacher-target-weighting gamma_support` |
| smoke target | `data/dev_clean.span_kd_prop_e10_overlap_smoke.json`, usable 3/4 |

기록할 핵심 지표:

| 항목 | 이유 |
|---|---|
| support coverage | BPE token이 aligner span에 얼마나 잘 매칭되는지 확인 |
| skipped token ratio | alignment 실패/불확실 구간 비율 |
| avg span length | spike 1-frame 문제를 실제로 완화하는지 확인 |
| dev/test WER | KD-BE/OCC 대비 성능 비교 |
| mismatch diagnostic | blank/non-blank 및 transition mismatch 변화 확인 |

## 제외한 것

다음은 cleanup 이후 완료 실험이 아니므로 이 문서의 공식 결과 표에서 제외한다.

```text
token_avg / GT-Viterbi sweep
8x student 실험
combined logit + token_avg sweep
span_kd full experiment
```

관련 코드나 스크립트가 repo에 있어도, 실제 cleanup 이후 완료한 실험이 아니면 결과 표에 넣지 않는다.
