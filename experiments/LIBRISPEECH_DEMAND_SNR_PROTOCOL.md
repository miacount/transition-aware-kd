# LibriSpeech + DEMAND controlled SNR analysis

Use all 2,703 utterances (5.39 h) from LibriSpeech `dev-clean`. Sort utterance
IDs and assign the four 16 kHz DEMAND environments in round-robin order:

- `TBUS` (bus),
- `PCAFETER` (cafeteria),
- `SPSQUARE` (public square/pedestrian), and
- `STRAFFIC` (street traffic).

Generate clean, +10 dB, 0 dB, -5 dB, and -10 dB conditions. For a given utterance,
reuse the same noise environment, channel, and crop at every noisy SNR. Crop
starts are deterministic from SHA-256 of the fixed seed, utterance ID, and
environment. SNR is measured over clean-speech-active samples, defined as 25 ms
frames at a 10 ms hop whose RMS is within 35 dB of the utterance's
95th-percentile frame RMS. Before mixing, every clean utterance is normalized
to -40 dBFS active-speech RMS. This fixed headroom prevents clipping through
the -10 dB condition while keeping the speech component identical across all
four paired conditions.

Clean and noisy files are stored as 16 kHz, mono, PCM-16 FLAC. This is a controlled diagnostic set,
not a standard LibriSpeech evaluation condition.

## Reproduction

Download and extract the official DEMAND 16 kHz archives `TBUS`, `PCAFETER`,
`SPSQUARE`, and `STRAFFIC` under `data/noise/DEMAND/extracted`, then run:

```bash
python scripts/prepare_librispeech_demand_snr.py --preflight-only
python scripts/prepare_librispeech_demand_snr.py
```

Outputs and JSONL manifests are written under
`data/librispeech_dev_clean_demand/`. DEMAND should be cited with DOI
`10.5281/zenodo.1227121`.
