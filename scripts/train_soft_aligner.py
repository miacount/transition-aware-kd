#!/usr/bin/env python3
"""
Train a small TDNN-FFN soft aligner with CTC + label priors.

Ref: Huang et al. ICASSP 2024
     "Less Peaky and More Accurate CTC Forced Alignment by Label Priors"

Architecture: 3 Conv1d (TDNN-style) + 5 FFN layers, ~4M params
              stride=2 at first layer → 20ms frame rate

Loss: L = -log Σ_{π∈B^{-1}(W)} Π_t  p_t(π_t) / P(π_t)^α
     implemented as CTC on adjusted log-scores:
         adj[t,k] = log_probs[t,k] - α * log_prior[k]

Prior update: P(k) = mean p_t(k) over all valid frames, updated each epoch.
              Epoch 1 uses α=0 (warm-up, no prior).
"""
import os
import sys
import json
import argparse

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SoftAligner(nn.Module):
    """
    TDNN-FFN soft aligner.
    3 temporal conv layers (kernel 5,3,3 ; stride 2,1,1 ; dilation 1,2,4)
    5 feed-forward layers
    """
    def __init__(self, input_dim: int = 80, hidden: int = 512,
                 vocab: int = 1025, dropout: float = 0.1):
        super().__init__()
        self.tdnn = nn.Sequential(
            # layer 1 — stride 2 → 10ms→20ms frame rate
            nn.Conv1d(input_dim, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(), nn.Dropout(dropout),
            # layer 2 — dilation 2, effective context ±4 frames
            nn.Conv1d(hidden, hidden, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(), nn.Dropout(dropout),
            # layer 3 — dilation 4, effective context ±8 frames
            nn.Conv1d(hidden, hidden, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(), nn.Dropout(dropout),
        )
        ffn = []
        for _ in range(5):
            ffn += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
        self.ffn = nn.Sequential(*ffn)
        self.proj = nn.Linear(hidden, vocab)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x:       (B, T, 80)
        lengths: (B,)  input frame counts
        returns: log_probs (B, T', V), out_lengths (B,)
        """
        x = x.permute(0, 2, 1)           # → (B, 80, T)
        x = self.tdnn(x)                  # → (B, hidden, T')
        x = x.permute(0, 2, 1)           # → (B, T', hidden)
        x = self.ffn(x)
        log_probs = F.log_softmax(self.proj(x), dim=-1)
        # stride=2 length after first conv
        out_lengths = torch.div(lengths - 1, 2, rounding_mode="floor") + 1
        out_lengths = out_lengths.clamp(max=log_probs.shape[1])
        return log_probs, out_lengths


# ---------------------------------------------------------------------------
# CTC with label priors
# ---------------------------------------------------------------------------

def ctc_with_blank_prior(log_probs: torch.Tensor,
                         targets: torch.Tensor,
                         input_lengths: torch.Tensor,
                         target_lengths: torch.Tensor,
                         log_prior_blank: torch.Tensor,
                         alpha: float,
                         blank: int) -> torch.Tensor:
    """
    CTC loss with blank-only prior penalty (Huang et al. Variation A).

    Adjustment applied ONLY to the blank token:
        adj_t(blank) = log y_t(blank) - α * log P(blank)
        adj_t(k)     = log y_t(k)        for all k ≠ blank

    Compared to the full prior correction (applied to all 1024 BPE tokens):
    - Removes the ~15× tail boost on rare non-blank tokens that causes
      the CTC path probability to explode → stable training.
    - Only one scalar (P(blank)) to track; as P(blank) falls, the penalty
      weakens automatically → self-stabilising.
    - Still sufficient to push token durations longer (blank penalised
      at every frame, so optimal paths shift probability to non-blank).

    Implementation: splice the adjusted blank score into adj, keeping all
    other log-probs unchanged, then pass to standard F.ctc_loss.
    PyTorch's CTC treats inputs as log-probabilities but handles unnormalized
    log-scores correctly (gradient = exp(input) - backward_proportion).
    """
    if alpha == 0.0:
        return F.ctc_loss(
            log_probs.permute(1, 0, 2), targets,
            input_lengths, target_lengths,
            blank=blank, reduction="mean", zero_infinity=True,
        )
    # Build adj: copy log_probs, adjust blank column only
    adj = log_probs.clone()
    adj[:, :, blank] = log_probs[:, :, blank] - alpha * log_prior_blank.to(log_probs.device)
    return F.ctc_loss(
        adj.permute(1, 0, 2), targets,
        input_lengths, target_lengths,
        blank=blank, reduction="mean", zero_infinity=True,
    )


# keep old name as alias so call-sites don't break
ctc_with_priors = ctc_with_blank_prior


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _wav_to_mel(wav: np.ndarray, sr: int = 16000) -> torch.Tensor:
    """
    Compute log-mel spectrogram matching NeMo's AudioToMelSpectrogramPreprocessor.
    Settings: n_fft=512, win=400 (25ms), hop=160 (10ms), n_mels=80, normalize=per_feature.
    Pure numpy/librosa — safe to call in DataLoader worker processes (no CUDA).
    """
    import librosa
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=512, win_length=400, hop_length=160,
        n_mels=80, fmin=0, fmax=8000, power=2.0,
    )                                               # (80, T)
    mel = np.log(mel + 1e-6)
    # per-feature (per mel-bin) normalization across time
    mean = mel.mean(axis=1, keepdims=True)
    std  = mel.std(axis=1,  keepdims=True)
    mel  = (mel - mean) / (std + 1e-5)
    return torch.from_numpy(mel.T).float()          # (T, 80)


class ASRDataset(Dataset):
    def __init__(self, manifest: str, tokenizer, max_duration: float = 20.0):
        self.rows = []
        with open(manifest) as f:
            for line in f:
                row = json.loads(line.strip())
                if row.get("duration", 0) <= max_duration:
                    self.rows.append(row)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        wav, sr = sf.read(row["audio_filepath"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        mel = _wav_to_mel(wav, sr)                  # (T, 80)
        mel_len = torch.tensor(mel.shape[0])
        tokens = torch.tensor(self.tokenizer.text_to_ids(row["text"]), dtype=torch.long)
        return mel, mel_len, tokens


def collate_fn(batch):
    mels, mel_lens, tokens = zip(*batch)
    B = len(mels)
    T_max = max(m.shape[0] for m in mels)
    N_max = max(t.shape[0] for t in tokens)
    mel_pad  = torch.zeros(B, T_max, mels[0].shape[1])
    tok_pad  = torch.zeros(B, N_max, dtype=torch.long)
    ml = torch.stack(mel_lens)
    tl = torch.tensor([t.shape[0] for t in tokens])
    for i, (m, t) in enumerate(zip(mels, tokens)):
        mel_pad[i, :m.shape[0]] = m
        tok_pad[i, :t.shape[0]] = t
    return mel_pad, ml, tok_pad, tl


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",     default="data/train_clean_100.json")
    ap.add_argument("--val-manifest", default="data/dev_clean.json")
    ap.add_argument("--tokenizer",    default="tokenizer_1024")
    ap.add_argument("--save-dir",     default="nemo_experiments/soft_aligner")
    ap.add_argument("--epochs",  type=int,   default=20)
    ap.add_argument("--batch",   type=int,   default=32)
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--hidden",  type=int,   default=512)
    ap.add_argument("--alpha",   type=float, default=0.05,
                    help="label prior scaling factor (paper α=0.3 for phonemes; "
                         "use 0.03-0.1 for BPE with high P(blank))")
    ap.add_argument("--alpha-warmup", type=int, default=5,
                    help="linearly ramp α from 0 to --alpha over this many epochs after warmup")
    ap.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
    tokenizer = SentencePieceTokenizer(
        model_path=os.path.join(args.tokenizer, "tokenizer.model")
    )
    blank_id   = tokenizer.vocab_size   # 1024
    vocab_size = blank_id + 1           # 1025

    model = SoftAligner(input_dim=80, hidden=args.hidden, vocab=vocab_size).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.2f}M params | blank_id={blank_id}")

    train_ds = ASRDataset(args.manifest,     tokenizer)
    val_ds   = ASRDataset(args.val_manifest, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True,
                          drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Initialize priors uniformly (log(1/V))
    log_priors = torch.full((vocab_size,), -np.log(vocab_size), device=args.device)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # epoch 1: CTC warm-up (α=0)
        # epochs 2..alpha_warmup+1: linear ramp 0 → args.alpha
        # epochs alpha_warmup+2+: fixed at args.alpha
        if epoch == 1:
            alpha_eff = 0.0
        else:
            ramp = min(1.0, (epoch - 1) / max(1, args.alpha_warmup))
            alpha_eff = args.alpha * ramp

        # ---------- train ----------
        model.train()
        prior_acc    = torch.zeros(vocab_size, device=args.device)
        prior_frames = 0
        train_loss = 0.0; n_steps = 0

        for mel, mel_lens, tokens, tok_lens in train_dl:
            mel      = mel.to(args.device)
            mel_lens = mel_lens.to(args.device)
            tokens   = tokens.to(args.device)
            tok_lens = tok_lens.to(args.device)

            log_probs, out_lens = model(mel, mel_lens)

            loss = ctc_with_blank_prior(
                log_probs, tokens, out_lens, tok_lens,
                log_priors[blank_id], alpha_eff, blank_id,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # accumulate stats for prior update
            with torch.no_grad():
                probs = log_probs.exp()           # (B, T', V)
                for b in range(len(out_lens)):
                    T = int(out_lens[b].item())
                    prior_acc    += probs[b, :T].sum(dim=0)
                    prior_frames += T

            train_loss += loss.item(); n_steps += 1
            if n_steps % 200 == 0:
                print(f"  ep{epoch} step{n_steps:4d}  loss={train_loss/n_steps:.3f}", flush=True)

        # ---------- update label priors ----------
        log_priors = (prior_acc / max(prior_frames, 1)).clamp_min(1e-9).log()
        p_blank = log_priors[blank_id].exp().item()
        print(f"  [prior] P(blank)={p_blank:.3f}  (before training: ~0.78)")

        scheduler.step()

        # ---------- validate ----------
        model.eval()
        val_loss = 0.0; n_val = 0
        with torch.no_grad():
            for mel, mel_lens, tokens, tok_lens in val_dl:
                mel      = mel.to(args.device)
                mel_lens = mel_lens.to(args.device)
                tokens   = tokens.to(args.device)
                tok_lens = tok_lens.to(args.device)
                log_probs, out_lens = model(mel, mel_lens)
                # validate with α=0 to get "clean" CTC loss
                loss = ctc_with_blank_prior(log_probs, tokens, out_lens, tok_lens,
                                            log_priors[blank_id], 0.0, blank_id)
                val_loss += loss.item(); n_val += 1
        val_loss /= max(n_val, 1)

        print(f"Epoch {epoch:3d}/{args.epochs}"
              f"  train={train_loss/n_steps:.3f}"
              f"  val={val_loss:.3f}"
              f"  α_eff={alpha_eff:.2f}")

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "log_priors": log_priors.cpu(),
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"epoch_{epoch:03d}.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
            print(f"  → best checkpoint saved (val={best_val:.3f})")

    print(f"\nDone. Best val_loss={best_val:.3f}")
    print(f"Checkpoints in: {args.save_dir}/")


if __name__ == "__main__":
    main()
