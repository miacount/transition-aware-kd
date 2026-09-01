# Ablation and mechanism-figure plan

## Goal

The main LS/CHiME-3 table establishes final WER. The ablation must separately
answer:

1. Does occurrence-aware temporal support alleviate CTC spike misalignment?
2. Does three-way mass calibration help independently of non-target knowledge?
3. Does conditional non-target distillation transfer information that ordinary
   CTC/KD does not expose?

All ablations retain the frozen teacher, student architecture, data,
augmentation, optimizer, 100-epoch cap, best-dev checkpoint selection, and
beam-16/no-LM evaluation from the main table. Component removal never triggers
weight retuning.

## Main ablation table

Run the following factorial ablation on both LibriSpeech-100h and CHiME-3.

| Variant | Occurrence FB | Blank penalty | Mass3 (weight 25) | NTDK (weight 8) |
|---|:---:|---:|:---:|:---:|
| No KD | - | - | - | - |
| Mass3 only | yes | 6 | yes | - |
| NTDK only | yes | 6 | - | yes |
| Full, narrow WHERE | yes | 0 | yes | yes |
| Full method | yes | 6 | yes | yes |

Report LS test-clean/test-other and CHiME-3 eval-real/eval-simu beam-16 WER in
one table. Reuse the completed No-KD and Full rows; the other three rows require
new runs per dataset (six new seed-1 trainings total).

Interpretation is pre-registered:

- `Mass3 only - No KD`: contribution of blank/GT/NT total-mass calibration.
- `Full - Mass3 only`: incremental contribution of conditional dark knowledge.
- `Full - NTDK only`: incremental contribution of emission/mass calibration.
- `Full(delta=6) - Full(delta=0)`: contribution of widening the
  transcript-constrained occurrence support to absorb local spike offsets.

The NTDK-only run must set the Mass3 coefficient to exactly zero. The current
constructor rejects `kd_weight=0` even when `span_ntdk_weight>0`; fix that guard
before launching rather than using an epsilon Mass3 weight.

### Statistical policy

First complete all rows with seed 1. Repeat Full, Mass3-only, and NTDK-only with
student seeds 2 and 3 on LibriSpeech. If a claimed incremental effect is below
0.3 absolute WER, report the three-seed mean and standard deviation and describe
overlapping uncertainty as a tie. CHiME-3 is retained as cross-condition
confirmation; repeat its critical pair only if the seed-1 ordering conflicts
with LibriSpeech or the margin is similarly small.

## Strong semantic control

Add one LibriSpeech-only control after the factorial table:

**Full + permuted NTDK weights.** For every occurrence, preserve the exact
teacher-selected top-32 token IDs, tail mass, probability multiset, entropy,
Mass3 target, temporal support, and loss weights, but apply a deterministic
random permutation to the top-32 probabilities. This destroys the teacher's
relative non-target ranking without changing candidate support or sharpness.

If Full outperforms this control, the NTDK gain cannot be explained only as
extra regularization, candidate count, or target entropy. Use a fixed offline
permutation seed and never permute validation/test targets.

## Appendix sensitivity

Do not expand the main ablation table with a large sweep. Put these compact
diagnostics in the appendix:

- blank penalty `delta`: 0/3/6/9/12, showing support hit rate and purity; train
  only 0 and 6 in the main factorial table;
- Mass3 weight: completed 15/25/35 runs at NTDK 8;
- NTDK weight: use only runs paired with Mass3 25; do not reuse the older
  legacy-primary NTDK sweep as if it were final-method sensitivity;
- top-M: report conditional-mass coverage/storage for M=8/16/32/64; a full
  training sweep is optional because M is a cache approximation, not the main
  methodological claim.

## Figure 1: misalignment recovery, not forced alignment

Use a three-panel figure on LS dev-other and optionally repeat its summary
numbers on CHiME dev-real. Keep test/eval splits for final WER reporting.

### (a) CTC spike-offset distribution

Histogram of absolute occurrence spike offset between teacher and No-KD
student. The existing test-clean pilot found that 56.6% of occurrences differ
and 82% of mismatches are one frame; recompute and annotate the corresponding
dev-other values for the paper. This establishes whether the dominant error is
a small local timing shift without using the test split for analysis choices.

### (b) Support recovery by delta

For delta=0/3/6/9/12, plot the fraction of student spike locations covered by
teacher occurrence support (`gamma(t_student,u)>0.01`). Use grouped bars for
offset 0, 1, 2, and all. The existing test-clean sanity check gives, for
one-frame offsets, 10.5% coverage at delta=0 and 62.1% at delta=6; overall
coverage rises from 48.4% to 73.0%. Use newly computed dev-other values in the
final caption.

### (c) Coverage-purity trade-off

Scatter support coverage against occurrence purity for transcript-constrained
FB and a matched-width symmetric temporal kernel. Marker size denotes support
width. This demonstrates that the method does not win by indiscriminate
smoothing: the FB support selects the correct occurrence shoulder with less
neighbor contamination.

Caption language must say **alignment-tolerant comparison** or **misalignment
recovery**, not that the method forces the student's spikes to match the
teacher's frame.

## Figure 2: conditional dark knowledge is present and learned

Use a three-panel figure, evaluated with the final Mass3+NTDK checkpoint rather
than an old Span-KD pilot.

### (a) Hidden distribution becomes visible after conditioning

Violin/box plots on a log y-axis for effective class count per occurrence:

- pooled full posterior;
- pooled posterior after excluding blank and occurrence GT (conditional NT).

Existing LS test-other diagnostics give medians about 1.01 and 5.78,
respectively. This shows that the teacher is globally almost one-hot while its
conditional alternatives remain structured.

### (b) The structure is stable and compressible

Plot top-M conditional mass for M=8/16/32/64, with a second small axis or inset
for clean-vs-30 dB perturbation stability. Existing M=32 values are mean mass
0.9765, median mass 0.9881, median top-32 Jaccard 0.939, and median JS 0.00362.
This justifies raw T=1 and top-32+tail without claiming lossless compression.

### (c) The student actually learns the relation

For No KD, Vanilla KD, Mass3 only, NTDK only, and Full, pool every student's
posterior with the same teacher occurrence support, remove blank and GT, and
compute teacher-to-student conditional top-32+tail KL (or JS). Plot mean with
bootstrap 95% confidence intervals. Full should reduce this divergence beyond
Mass3-only; otherwise do not claim that WER gains arise from transferred dark
knowledge.

As a supplementary bar chart, report the rank of each student's wrong token in
the teacher conditional NT distribution (MRR and recall@8). Existing pilot data
show the intended trend, but it must be recomputed using the frozen final
checkpoint to avoid mixing method versions.

## Optional qualitative figure

Select one utterance using a fixed rule (for example, the first test-other
utterance where Vanilla is wrong and Full is correct), not manual aesthetic
selection. Show:

1. teacher and student token posterior over time;
2. delta=0 and delta=6 occurrence support;
3. the occurrence's top-five conditional NT alternatives for teacher, Vanilla,
   and Full.

This is intuitive but remains supporting evidence; the aggregate figures above
carry the main claim.

## Recommended paper layout

- Main Table 1: existing LS/CHiME-3 method comparison.
- Main Table 2: five-row factorial ablation.
- Main Figure 1: method overview (WHERE versus WHAT, Mass3 versus NTDK).
- Main Figure 2: misalignment recovery panels.
- Main Figure 3: conditional dark-knowledge panels.
- Appendix: weight sensitivity, top-M storage/coverage, no-FB/MFA/matched-kernel
  controls, and qualitative example.

## Minimum new work

1. Add exact NTDK-only support to the model guard.
2. Build delta=0 Mass3+NTDK caches for LS and CHiME.
3. Train six seed-1 ablation runs.
4. Implement deterministic NTDK-weight permutation and train one LS control.
5. Extend the dark-knowledge diagnostic to compare final No-KD, Vanilla,
   Mass3-only, NTDK-only, and Full checkpoints using aggregate conditional KL.
6. Generate publication figures from JSON/NPZ outputs with fixed plotting
   scripts and no test-set model selection.
