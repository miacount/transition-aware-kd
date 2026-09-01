# Controlled CHiME-3 SNR stress-test protocol

## Purpose

This diagnostic tests whether stronger additive noise is associated with
larger teacher--No-KD CTC spike offsets and a larger WER advantage of AT-DKD
over Vanilla KD. It is not an official CHiME-3 evaluation condition.

## Fixed data design

- Source split: CHiME-3 `dt05` (development only).
- Clean source: channel 1 of each isolated BTH utterance.
- Noise source: channel 1 of the official CHiME-3 background recording named
  by `dt05_simu.json`.
- Conditions: clean, +5 dB, -5 dB, and -10 dB.
- Sample size: all 410 unique BTH utterances.
- Environment assignment: sort by `(speaker, wsj_name)` and assign
  BUS/CAF/PED/STR round-robin, yielding 103/103/102/102 utterances.
- Pairing: an utterance uses the same annotation-defined noise file and crop at
  every noisy SNR. Only noise amplitude changes.
- Speech level: normalize active-speech RMS to -30 dBFS once, before creating
  any condition.
- Active speech: 25 ms frames at a 10 ms hop whose RMS is within 35 dB of the
  utterance's 95th-percentile frame RMS.
- SNR: speech and noise power are measured on the same active-speech samples.
- Storage: 16 kHz mono PCM-16 WAV. Any anti-clipping gain is applied jointly to
  speech and noise and therefore preserves SNR.

Do not change SNRs or select utterances after inspecting AT-DKD results. If the
-10 dB condition saturates, report that outcome rather than silently replacing
the condition. A different severe condition may be selected using Vanilla-only
development results, but the rule and decision must be documented before
evaluating AT-DKD.

## Required corpus files

The official background directory is expected at:

```text
data/chime/CHiME3/data/audio/16kHz/backgrounds/
```

It must contain files such as `M04_141107_040_BUS.CH1.wav`. An alternative
location can be supplied with `--noise-dir`. Noise estimated by subtracting
noisy speech from a clean channel is deliberately rejected because residual
speech and channel-response mismatch would weaken a paper mechanism claim.

## Generation

Run a read-only preflight first:

```bash
python scripts/prepare_chime3_snr_stress.py --preflight-only
```

Generate all four conditions:

```bash
python scripts/prepare_chime3_snr_stress.py
```

The output is written under `data/chime3_snr_stress/dt05/`. Each condition has
one JSONL manifest under `manifests/`; every row records the source files, noise
crop, target and achieved SNR, normalization gains, channel, and environment.

## Reporting

Evaluate the frozen Teacher, No-KD, Vanilla KD, and AT-DKD checkpoints on the
same four manifests with the paper's no-LM beam-16 decoder. Report:

1. teacher--No-KD mean absolute occurrence spike offset;
2. Teacher, Vanilla KD, and AT-DKD corpus WER;
3. absolute WER reduction in percentage points;
4. relative WER reduction;
5. paired utterance-bootstrap 95% confidence intervals.

Bootstrap utterances, not individual tokens. Keep all models' predictions and
all SNR conditions paired within each resampled utterance.
