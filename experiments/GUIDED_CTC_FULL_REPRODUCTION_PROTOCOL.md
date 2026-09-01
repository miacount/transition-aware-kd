# Guided CTC full-pipeline controlled reproduction

## Scope and naming

The completed `paper-*-guided-exact-*` runs are preserved. They reproduce the
guide-loss objective from Kurata and Audhkhasi (Interspeech 2019),

`L = L_CTC + L_G`, where `L_G = -sum M_G(t,k) p_model(t,k)`,

but they do not reproduce the paper's teacher-alignment and posterior-fusion KD
pipeline. In tables, retain those runs as **Guided loss only**.

The new experiment is named **Guided CTC KD (full pipeline, controlled
reimplementation)**. It adapts the paper's full causal chain to the common
LibriSpeech-100h and CHiME-3 setup; it is not an exact Switchboard/LSTM
reproduction.

## Paper-faithful causal chain

For each dataset, train or reuse the following models.

1. `G`: a normally trained model with the final student architecture. This is
   frozen and used only to define desired spike timings.
2. `T_std_i`: a high-capacity teacher trained with ordinary CTC.
3. `T_guided_i`: the same high-capacity teacher, source initialization, seed,
   data order, optimizer, and exposure as `T_std_i`, except it is trained with
   `L_CTC + L_G(G)`.
4. `S_stdKL_N`: a fresh student trained with pure frame-wise KL from the
   probability-average of `N` standard teachers.
5. `S_guidedKL_N`: its paired student trained with pure frame-wise KL from the
   probability-average of `N` guided teachers.

Use `N=1` for the minimum defensible reproduction and `N=4` for the paper's
central posterior-fusion experiment. Average probabilities, not logits. The
final KD objective is pure frame-level KD (`kd_lambda=1`), because the paper
describes minimizing frame-wise KL and does not add a student CTC term at this
stage.

## Fixed local choices

| Item | LibriSpeech-100h | CHiME-3 |
|---|---|---|
| Guiding model `G` | best-dev `student-no-kd`, seed 1 | best-dev `paper-chime3-lr05-no-kd-s1`, seed 1 |
| Teacher source | `stt_en_conformer_ctc_small` | `stt_en_conformer_ctc_small` |
| Teacher architecture | source Conformer, 16x176, subsampling 4 | same |
| Student architecture | Conformer 8x144, subsampling 4 | same |
| Vocabulary | shared 1024 BPE + blank | same |
| Teacher adaptation | 10 epochs, best dev checkpoint | 10 epochs, best dev checkpoint |
| Final student KD | at most 100 epochs, best dev checkpoint | at most 100 epochs, best dev checkpoint |
| Evaluation | greedy and beam 16, no LM | greedy and beam 16, no LM |

Teacher adaptation starts from the same source model in every paired run. Seeds
change data order/dropout, while `T_std_i` and `T_guided_i` within a pair use the
same seed. Test sets are report-only; all checkpoint and continuation decisions
use dev WER and training-set spike diagnostics.

## Staged execution and gates

### Stage 0: implementation parity smoke test

- Export `G` as a loadable NeMo model or load its checkpoint with the exact
  student config.
- Cache only the guide top-1 token ID per frame; probabilities are unnecessary
  for `L_G`.
- Assert blank ID, BPE token ordering, sample rate, and subsampling compatibility.
- On 32 utterances, verify cached masks equal online guide argmax masks exactly.
- Numerically verify `L_G` is the raw selected-posterior sum, not `-log p` and
  not a frame mean.
- Run one training batch for standard teacher, guided teacher, and final pure-KL
  student; require finite loss and gradients.

### Stage 1: one guided teacher, seed 1

Train the paired `T_std_1` and `T_guided_1`. Before any final-student run,
measure:

- dev WER of both teachers;
- guide-to-teacher nonblank spike coverage at the identical frame index;
- teacher-to-guide coverage;
- nonblank rate and mean spike timing distance.

Proceed when the guided teacher is finite, does not collapse to blank, and its
guide-spike coverage improves by at least 10 percentage points over the paired
standard teacher. Teacher WER is allowed to degrade moderately, as it did in
the source paper; a relative degradation above 20% is treated as a failed
training recipe and investigated before expansion.

### Stage 2: one-teacher KD check

Train paired students from fresh identical initializations:

- `S_stdKL_1 <- T_std_1`
- `S_guidedKL_1 <- T_guided_1`

Both use the full posterior and pure frame KL for the same 100-epoch cap. No
CTC warm-up, CTC fine-tuning, temperature tuning, or KD-weight tuning is added.
Those would change the method under test. A 5-epoch health check may terminate
a non-finite or all-blank run, but cannot alter its recipe.

The core paper effect is considered reproduced locally if `S_guidedKL_1`
improves dev WER over `S_stdKL_1` and Stage 1 confirms increased spike
alignment. Test WER is inspected only after this dev-side decision.

### Stage 3: four-teacher posterior fusion

Only after Stage 2 passes, train seeds 2--4 for both standard and guided
teachers. Build streaming fused caches:

`P_fused(t,k) = (1/4) * sum_i P_i(t,k)`.

Do not retain four separate dense caches. Run inference for the four teachers
in one cache-building pass and write only the fused FP16 posterior, with
restartable per-utterance files. Abort on any frame-length or vocabulary
mismatch.

Train the paired students:

- `S_stdKL_4 <- fusion(T_std_1..4)`
- `S_guidedKL_4 <- fusion(T_guided_1..4)`

Report individual-teacher WER, fused-teacher WER, inter-teacher spike coverage,
and final-student WER. This paired standard-fusion control is mandatory: without
it, an improvement cannot be attributed to guiding rather than ensembling.

### Stage 4: robustness

Repeat only the two final `N=4` student runs with student seeds 2 and 3. Report
mean and standard deviation over the three final-student seeds. Teacher sets and
fused posterior caches remain fixed, so this measures student-training noise
rather than changing the teacher ensemble.

## Main comparison rows

Keep the old results and add rows without silently replacing them:

1. No KD.
2. Vanilla frame KD (`CTC + KD`, the common local baseline).
3. Guided loss only (the completed run).
4. Pure frame KL from one standard teacher (`S_stdKL_1`).
5. Guided CTC KD, one guided teacher (`S_guidedKL_1`).
6. Pure frame KL from four standard teachers (`S_stdKL_4`, appendix if space is
   limited).
7. Guided CTC KD, four-teacher fusion (`S_guidedKL_4`).

Rows 4--7 form the source-paper reproduction block. Row 2 remains the stronger
common-framework baseline but is not a substitute for the paper's paired
pure-KL control.

## Compute and disclosure

For `N=4`, the method requires one guiding-model training, four guided-teacher
trainings, four standard-teacher controls, two fused-posterior builds, and final
student training. Report both final-student compute and total pipeline compute.
Do not describe it as compute-matched to single-teacher KD.

Recommended paper wording:

> We retain a guide-loss-only ablation and separately implement the full Guided
> CTC distillation pipeline. A student-architecture guiding model first aligns
> one or four high-capacity teachers through the original selected-posterior
> objective. We then average the aligned teacher probabilities and train a fresh
> student by frame-wise KL. Standard, non-guided teachers with identical seeds
> and training schedules provide the paired control.

## Required implementation artifacts

Add, without modifying completed experiment directories:

- a checkpoint-to-NeMo exporter for `G`;
- a restartable top-1 guide-mask cache builder;
- a guided-teacher trainer initialized from the common source teacher;
- a restartable multi-model probability-fusion cache builder;
- LS and CHiME orchestration scripts with unique experiment names;
- tests for mask parity, raw guide-loss scaling, probability averaging,
  vocabulary/frame-length failure, pure-KL final loss, and resume behavior;
- a result summarizer containing WER and spike-coverage diagnostics.

