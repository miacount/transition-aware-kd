# Baseline Comparison: CTC Knowledge Distillation Methods

## Setup

| | |
|---|---|
| **Teacher** | `stt_en_conformer_ctc_small` (NeMo pretrained, 16-layer Conformer, 4× sub) |
| **Student** | Conformer CTC, d=144, 8 layers, 4× subsampling, 1024 BPE + blank |
| **Data** | LibriSpeech train-clean-100 → dev-clean / test-clean / test-other |
| **Baseline (no-KD)** | test_clean = **14.97%**, test_other = **34.23%** |

---

## WER Comparison

### 4× Subsampling Student (main setup)

| Method | Paper | Config | test_clean | test_other | Δ clean | Δ other |
|--------|-------|--------|-----------|------------|---------|---------|
| **No-KD** | — | — | 14.97% | 34.23% | — | — |
| Transition KD | *ours* | w=0.25 | 14.34% | 33.52% | −0.63 | −0.71 |
| Vanilla Logit KD | Hilmes 2025 | t=1, w=10 | 13.87% | 32.89% | −1.10 | −1.34 |
| **KD-BE** (blank elimination) | Hilmes 2025 | t=1, w=10 | **12.99%** | 31.15% | −1.98 | −3.08 |
| **Sym n=1** (symmetric ±1) | Hilmes 2025 | t=1, w=10 | **12.85%** | **31.13%** | −2.12 | −3.10 |
| Sym n=2 (symmetric ±2) | Hilmes 2025 | t=1, w=10 | 15.21%‡ | 33.93%‡ | +0.24 | −0.30 |
| Trim | Hilmes 2025 | t=1, w=10 | TBD | TBD | — | — |
| Threshold α=0.9 | Hilmes 2025 | t=1, w=10 | TBD | TBD | — | — |
| Random β=0.5 | Hilmes 2025 | t=1, w=10 | TBD | TBD | — | — |
| Token-avg KD (v1, buggy blank_id) | *ours* | w=20 | 12.53% | 31.14% | −2.44 | −3.09 |
| Token-avg KD (v2, correct) | *ours* | w=20 | 12.72% | 30.96% | −2.25 | −3.27 |
| Guide-CTC | Kurata 2019 | t=1, w=10 | TBD | TBD | — | — |
| Delayed-KD TAB=2 | Li 2025* | t=1, w=10 | TBD | TBD | — | — |
| Delayed-KD TAB=2 + BE | Li 2025* | t=1, w=10 | TBD | TBD | — | — |
| Self-KD l=4, α=0.5 | Kim 2024 | — | TBD | TBD | — | — |
| Self-KD l=6, α=0.5 | Kim 2024 | — | TBD | TBD | — | — |

> ‡ Sym n=2 converged early (epoch 50 best); model may need investigation.  
> \* Delayed-KD adapted for non-streaming: only TAB search applied, student architecture unchanged.

### 8× Subsampling Student (sub-experiments)

| Method | Config | test_clean | test_other | Δ clean | Δ other |
|--------|--------|-----------|------------|---------|---------|
| No-KD | — | 15.97% | 35.55% | — | — |
| Vanilla Logit KD | t=1, w=10 | 15.84% | 35.07% | −0.13 | −0.48 |
| Token-avg KD | w=5 | **15.16%** | **34.22%** | −0.81 | −1.33 |

---

## Frame-level Misalignment Metrics

Measured on 256 utterances of dev_clean using `scripts/diagnose_ctc_mismatch.py`.

| Method | frame mismatch | blank/nonblank mismatch | teacher NB → student B | transition edit rate |
|--------|---------------|-------------------------|----------------------|---------------------|
| No-KD | 23.43% | 22.04% | 49.15% | 13.36% |
| Vanilla Logit KD (t=1, w=10) | TBD | TBD | TBD | TBD |
| KD-BE (t=1, w=10) | TBD | TBD | TBD | TBD |
| Sym n=1 (t=1, w=10) | TBD | TBD | TBD | TBD |
| Token-avg v2 (w=20) | TBD | TBD | TBD | TBD |
| Guide-CTC (t=1, w=10) | TBD | TBD | TBD | TBD |
| Delayed-KD TAB=2+BE | TBD | TBD | TBD | TBD |
| Self-KD l=4 | TBD | TBD | TBD | TBD |

**Metric definitions:**
- `frame mismatch`: fraction of frames where teacher argmax ≠ student argmax
- `blank/nonblank mismatch`: frames where teacher blank/non-blank decision differs from student
- `teacher NB → student B`: among teacher non-blank frames, fraction where student outputs blank (spike timing miss)
- `transition edit rate`: edit distance on CTC transition sequences (token boundary alignment)

To update: run `bash experiments/run_misalign_eval.sh` (see below).

---

## How to Run Missing Experiments

### Experiments using existing frame_topk targets (preset 20 already built)

```bash
# Hilmes 2025 additional variants
bash experiments/presets/41_logit_kd_trim_w10.sh
bash experiments/presets/42_logit_kd_thresh_09_w10.sh
bash experiments/presets/43_logit_kd_random_b05_w10.sh

# Kurata 2019 (Guide-CTC)
bash experiments/presets/50_guided_kd_w10.sh

# Li 2025 adapted (Delayed-KD)
bash experiments/presets/51_delayed_kd_tab2_w10.sh
bash experiments/presets/51_delayed_kd_tab2_be_w10.sh
```

### Self-KD (no external targets needed)

```bash
bash experiments/presets/52_self_kd_l4_a05.sh
bash experiments/presets/52_self_kd_l6_a05.sh
```

### Evaluate trained checkpoints

```bash
# After training completes, find best checkpoint and run:
bash experiments/eval.sh \
  --ckpt nemo_experiments/<run_dir>/checkpoints/<name>--val_wer=X.ckpt \
  --name <short_name>
```

---

## Alignment Visualization Examples

To visualize CTC alignment comparison across methods for a specific utterance:

```bash
python scripts/visualize_viterbi.py \
  --ckpt <checkpoint.ckpt> \
  --audio <audio.flac> \
  --text "the transcript here" \
  --out analysis/alignment_viz/
```

### Example 1 — Short utterance

> **Audio**: TBD (dev_clean utterance, ~3s)  
> **Transcript**: TBD  
> **Observations**: TBD

### Example 2 — Medium utterance

> **Audio**: TBD  
> **Transcript**: TBD  
> **Observations**: TBD

### Example 3 — Long utterance with rare words

> **Audio**: TBD  
> **Transcript**: TBD  
> **Observations**: TBD

---

## Method Summaries

### Hilmes et al. 2025 — Blank Selection for CTC KD

**Problem**: Vanilla logit KD supervises all frames equally. CTC has ~78% blank frames with no phonetic content, causing KD loss to be dominated by blank distribution matching.

**Methods**:
- **KD-BE**: Only compute KD loss on frames where teacher argmax ≠ blank (hard selection)
- **Symmetric n=k**: Extend non-blank frames by ±k frames to include boundary blanks
- **Trim**: Keep all frames between first and last non-blank in the sequence
- **Threshold α**: Keep frames where p(blank) < α
- **Random β**: Keep all non-blank frames + β-fraction of blank frames randomly

**Key results (paper)**: Symmetric n=2 + λ=1.0 (no CTC loss) gives best WER on TEDLIUMv2.  
**Our results**: Sym n=1 ≈ KD-BE (12.85% vs 12.99%); Sym n=2 underperforms — likely needs λ=1.0 or lower weight.

### Kurata & Audhkhasi 2019 — Guide-CTC

**Problem**: Different CTC-trained models produce spikes at different frame positions (alignment mismatch). Alignment mismatch causes student to learn incorrect token timings.

**Method**: Add a guide loss that forces student spikes to align with teacher spikes:
- `M_t = 1` if teacher argmax at frame t ≠ blank, else 0
- `L_guide = −Σ_t M_t · log P_student(t, teacher_argmax_t)`  
- Equivalent to cross-entropy with hard (argmax) teacher labels, restricted to non-blank frames

**Key difference from KD-BE**: KD-BE uses soft KL with top-k teacher probs. Guide-CTC uses hard CE with teacher argmax only. Guide-CTC directly optimizes spike timing alignment.

### Li et al. 2025 — Delayed-KD with TAB

**Original setting**: Non-streaming teacher → streaming (chunk-based) student. Student outputs are naturally delayed (hasn't seen future context), so TAB searches for the minimum-KL alignment within a forward window.

**Our adaptation** (non-streaming): Both teacher and student are full-context. TAB mechanism applied to handle spike timing mismatch:
- For each teacher frame t (mapped to student frame s_t), compute KL against student frames [s_t, s_t+d]
- Select student frame with minimum KL divergence
- Prevents forcing supervision at misaligned positions

**TAB size**: d=2 frames corresponds to 40ms latency at 10ms frame shift (matches paper's optimal 40ms setting).

### Kim et al. 2024 — Self-KD (No External Teacher)

**Problem**: KD from external teacher requires loading and running a large teacher model. Architecture mismatch between teacher and student complicates supervision.

**Method**: Self-distillation using shared encoder layers:
- Self-teacher = full model (all 8 layers + CTC head)  
- Self-student = first l layers + intermediate CTC head (shared weights with self-teacher)
- SKD loss = frame-level KL(intermediate output ‖ full output, detached)
- Total: `(1−α)·L_CTC + α·(L_iCTC + L_SKD)`

**No blank masking needed**: SKD loss at all frames because both teacher and student outputs come from the same model — alignment is guaranteed.

**Split layer**: l=4 (50%) or l=6 (75%) of encoder depth.

---

## Checklist for Paper Reproduction

- [x] Vanilla Logit KD (Hilmes 2025 baseline)
- [x] KD-BE / Blank Elimination (Hilmes 2025)
- [x] Symmetric n=1 (Hilmes 2025)
- [x] Symmetric n=2 (Hilmes 2025) — underperforms, needs λ=1.0 experiment
- [ ] Trim (Hilmes 2025)
- [ ] Threshold α=0.9 (Hilmes 2025)
- [ ] Random β=0.5 (Hilmes 2025)
- [ ] Guide-CTC (Kurata 2019)
- [ ] Delayed-KD TAB=2 (Li 2025 adapted)
- [ ] Delayed-KD TAB=2 + BE (Li 2025 adapted + Hilmes)
- [ ] Self-KD l=4 (Kim 2024)
- [ ] Self-KD l=6 (Kim 2024)
- [ ] Fill misalignment metrics table (run `diagnose_ctc_mismatch.py` on each)
- [ ] Add 3 alignment visualization examples
