"""
CTC path alignment visualization: teacher vs 4x students vs simulated 8x.

Generates two figures per run:
  1. Per-utterance CTC path heatmap rows:
       teacher | trans-KD 4x | logit-KD 4x | simulated 8x | mismatch overlay
  2. Aggregate alignment statistics bar chart (3 groups, 4 metrics)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import TransitionKDModel
from nemo.collections.asr.models import EncDecCTCModelBPE


# ── audio / model helpers ─────────────────────────────────────────────────────

def load_audio(path, sample_rate=16000):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != sample_rate:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
    return torch.from_numpy(wav)


@torch.no_grad()
def get_ctc_path(model, wav, device):
    wav_t = wav.unsqueeze(0).to(device)
    wav_len = torch.tensor([wav_t.shape[1]], dtype=torch.long, device=device)
    log_probs, enc_len, _ = model.forward(input_signal=wav_t,
                                           input_signal_length=wav_len)
    T = int(enc_len[0].item())
    return log_probs[0, :T].argmax(dim=-1).cpu().numpy()


def load_student(config_path, ckpt_path, device):
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model.copy()
    with open_dict(model_cfg):
        model_cfg.log_prediction = False
        if model_cfg.get("test_ds") is not None:
            del model_cfg.test_ds
    model = TransitionKDModel(cfg=model_cfg, trainer=None)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    return model.to(device).eval()


# ── path utilities ────────────────────────────────────────────────────────────

def resample_path(path, target_len):
    """Nearest-neighbour resample 1-D token path to target_len."""
    src = len(path)
    if src == target_len:
        return path.copy()
    pos = np.clip(
        np.floor((np.arange(target_len) + 0.5) / target_len * src).astype(np.int64),
        0, src - 1,
    )
    return path[pos]


def collapse_repeats(path):
    out, prev = [], None
    for x in path:
        x = int(x)
        if x != prev:
            out.append(x)
            prev = x
    return out


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != bj))
        prev = cur
    return prev[-1]


def safe(n, d):
    return float(n) / float(d) if d else 0.0


# ── alignment stats ───────────────────────────────────────────────────────────

def alignment_stats(teacher_path, other_path, blank_id):
    """Both paths must be the same length (already resampled to teacher T)."""
    assert len(teacher_path) == len(other_path), \
        f"length mismatch: {len(teacher_path)} vs {len(other_path)}"
    T = len(teacher_path)
    t_blank   = teacher_path == blank_id
    o_blank   = other_path  == blank_id
    mismatch  = teacher_path != other_path
    nb_mis    = (t_blank != o_blank)
    both_nb   = (~t_blank) & (~o_blank)
    nb_diff   = both_nb & mismatch
    t_trans   = collapse_repeats(teacher_path)
    o_trans   = collapse_repeats(other_path)
    ted       = edit_distance(t_trans, o_trans)
    return {
        "frame_mismatch":    safe(mismatch.sum(), T),
        "blank_nb_mismatch": safe(nb_mis.sum(),   T),
        "both_nb_diff":      safe(nb_diff.sum(),  max(both_nb.sum(), 1)),
        "trans_edit_rate":   safe(ted,             max(len(t_trans), 1)),
    }


# ── colour helpers ────────────────────────────────────────────────────────────

BLANK_COLOR = np.array([0.88, 0.88, 0.88])
_CMAP = plt.get_cmap("tab20")

def path_to_rgb(path, blank_id):
    rgb = np.zeros((len(path), 3))
    for i, tok in enumerate(path):
        rgb[i] = BLANK_COLOR if tok == blank_id else np.array(_CMAP(int(tok) % 20)[:3])
    return rgb


def mismatch_rgb(ref, other):
    """Red where other ≠ ref, else transparent black."""
    rgb = np.zeros((len(ref), 3))
    diff = (ref != other)
    rgb[diff, 0] = 1.0
    return rgb, diff


# ── per-figure helpers ────────────────────────────────────────────────────────

def _row(ax, path, blank_id, title, show_xlabel=False):
    rgb = path_to_rgb(path, blank_id)
    ax.imshow(rgb[np.newaxis], aspect="auto", interpolation="nearest",
              extent=[0, len(path), 0, 1])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, pad=2)
    if not show_xlabel:
        ax.set_xticks([])
    else:
        ax.set_xlabel("encoder frame index", fontsize=8)


def make_path_figure(teacher_path, trans_path, logit_path, blank_id, utt_id, out_path):
    T = len(teacher_path)

    sim8x_compressed = resample_path(teacher_path, max(1, T // 2))
    sim8x_path       = resample_path(sim8x_compressed, T)   # back to T for display
    trans_aligned    = resample_path(trans_path,  T)
    logit_aligned    = resample_path(logit_path,  T)

    # mismatch masks
    diff_trans  = (teacher_path != trans_aligned).astype(np.float32)
    diff_logit  = (teacher_path != logit_aligned).astype(np.float32)
    diff_8x     = (teacher_path != sim8x_path).astype(np.float32)

    # combined mismatch overlay: blue=trans, green=logit, red=8x; overlaps blended
    overlay = np.zeros((T, 3))
    overlay[:, 2] += diff_trans * 0.8   # blue
    overlay[:, 1] += diff_logit * 0.8   # green
    overlay[:, 0] += diff_8x            # red
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(6, 1, figsize=(15, 5),
                             gridspec_kw={"hspace": 0.65})

    _row(axes[0], teacher_path, blank_id,
         f"Teacher  (T={T} frames, 4× subsampling)")
    _row(axes[1], trans_aligned, blank_id,
         f"Trans-KD student  (4×, aligned to teacher)")
    _row(axes[2], logit_aligned, blank_id,
         f"Logit-KD student  (4×, aligned to teacher)")
    _row(axes[3], sim8x_path, blank_id,
         f"Simulated 8× student  (teacher compressed T/2={T//2} → upsampled back)")

    # mismatch rate annotations on the right
    for ax, diff, label in [
        (axes[1], diff_trans,  "trans"),
        (axes[2], diff_logit,  "logit"),
        (axes[3], diff_8x,     "8×sim"),
    ]:
        rate = diff.mean() * 100
        ax.text(1.002, 0.5, f"{rate:.1f}% mismatch", va="center",
                ha="left", transform=ax.transAxes, fontsize=7, color="#444")

    axes[4].imshow(overlay[np.newaxis], aspect="auto", interpolation="nearest",
                   extent=[0, T, 0, 1])
    axes[4].set_yticks([])
    axes[4].set_title(
        "Mismatch overlay  (blue=trans-KD, green=logit-KD, red=8×sim)", fontsize=8, pad=2)
    axes[4].set_xticks([])

    # blank density bar
    blank_mask = (teacher_path == blank_id).astype(np.float32)
    bar_rgb = np.stack([blank_mask * 0.6, blank_mask * 0.6, blank_mask * 0.6], axis=1)
    axes[5].imshow(bar_rgb[np.newaxis], aspect="auto", interpolation="nearest",
                   extent=[0, T, 0, 1])
    axes[5].set_yticks([])
    axes[5].set_title("Teacher blank frames (grey)", fontsize=8, pad=2)
    axes[5].set_xlabel("encoder frame index", fontsize=8)

    blank_pct = blank_mask.mean() * 100
    axes[5].text(1.002, 0.5, f"{blank_pct:.0f}% blank", va="center",
                 ha="left", transform=axes[5].transAxes, fontsize=7, color="#444")

    fig.suptitle(f"CTC path alignment — {utt_id}", fontsize=9, y=1.01)
    blank_patch = mpatches.Patch(color=BLANK_COLOR, label="blank token")
    fig.legend(handles=[blank_patch], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── aggregate stats figure ────────────────────────────────────────────────────

METRICS = ["frame_mismatch", "blank_nb_mismatch", "both_nb_diff", "trans_edit_rate"]
METRIC_LABELS = [
    "Frame\nmismatch",
    "Blank/non-blank\nmismatch",
    "Non-blank\ntoken diff",
    "Transition\nedit rate",
]
COLORS = {"Trans-KD 4×": "#4477AA", "Logit-KD 4×": "#228833", "8× sim": "#EE6633"}


def make_stats_figure(stats_dict, out_path):
    """stats_dict: {label: [list of stats dicts]}"""
    labels = list(stats_dict.keys())
    means  = {lbl: {k: np.mean([s[k] for s in v]) for k in METRICS}
              for lbl, v in stats_dict.items()}

    x      = np.arange(len(METRICS))
    width  = 0.22
    offsets = np.linspace(-(len(labels) - 1) / 2, (len(labels) - 1) / 2, len(labels))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for lbl, offset in zip(labels, offsets):
        vals = [means[lbl][k] * 100 for k in METRICS]
        bars = ax.bar(x + offset * width, vals, width,
                      label=lbl, color=COLORS.get(lbl, "#888888"))
        ax.bar_label(bars, fmt="%.1f%%", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=9)
    ax.set_ylabel("Rate (%)", fontsize=9)
    ax.set_title(
        "Teacher–Student CTC Path Alignment\n"
        "(4× students vs simulated 8× subsampling mismatch)",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="upper left")
    ymax = max(m for lbl_means in means.values() for m in lbl_means.values()) * 100
    ax.set_ylim(0, ymax * 1.35 + 3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Visualise CTC alignment: teacher vs trans/logit 4x students vs simulated 8x"
    )
    ap.add_argument("--manifest",   default="data/dev_clean.json")
    ap.add_argument("--config",     default="configs/student_base.yaml")
    ap.add_argument("--ckpt_trans", required=True,
                    help="best trans-KD 4x student checkpoint (.ckpt)")
    ap.add_argument("--ckpt_logit", required=True,
                    help="best logit-KD 4x student checkpoint (.ckpt)")
    ap.add_argument("--teacher",    default="stt_en_conformer_ctc_small")
    ap.add_argument("--limit",      type=int, default=50,
                    help="number of utterances to process for stats")
    ap.add_argument("--n_plot",     type=int, default=3,
                    help="number of per-utterance path figures to save")
    ap.add_argument("--out_dir",    default="analysis/alignment_viz")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import json
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] teacher : {args.teacher}")
    teacher    = EncDecCTCModelBPE.from_pretrained(args.teacher).to(device).eval()
    blank_id   = int(teacher.decoder.num_classes_with_blank - 1)
    sample_rate = int(teacher.cfg.preprocessor.sample_rate)

    print(f"[load] trans   : {args.ckpt_trans}")
    model_trans = load_student(args.config, args.ckpt_trans, device)

    print(f"[load] logit   : {args.ckpt_logit}")
    model_logit = load_student(args.config, args.ckpt_logit, device)

    rows = []
    with open(args.manifest) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.limit:
                break
    print(f"[data] {len(rows)} utterances from {args.manifest}")

    stats = {"Trans-KD 4×": [], "Logit-KD 4×": [], "8× sim": []}
    plotted = 0

    for idx, row in enumerate(rows):
        wav = load_audio(row["audio_filepath"], sample_rate)

        t_path     = get_ctc_path(teacher,     wav, device)
        trans_path = get_ctc_path(model_trans, wav, device)
        logit_path = get_ctc_path(model_logit, wav, device)

        T = len(t_path)

        # all aligned to teacher length
        trans_aln = resample_path(trans_path, T)
        logit_aln = resample_path(logit_path, T)
        sim8x_aln = resample_path(resample_path(t_path, max(1, T // 2)), T)

        stats["Trans-KD 4×"].append(alignment_stats(t_path, trans_aln, blank_id))
        stats["Logit-KD 4×"].append(alignment_stats(t_path, logit_aln, blank_id))
        stats["8× sim"].append(      alignment_stats(t_path, sim8x_aln, blank_id))

        if plotted < args.n_plot:
            utt_id = Path(row["audio_filepath"]).stem
            make_path_figure(
                t_path, trans_path, logit_path, blank_id,
                utt_id=utt_id,
                out_path=out_dir / f"path_{idx:03d}_{utt_id}.png",
            )
            plotted += 1

        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(rows)} done")

    make_stats_figure(stats, out_path=out_dir / "alignment_stats_4x_vs_8x.png")

    # terminal summary
    print("\n====== alignment summary ======")
    print(f"{'metric':30s}", end="")
    for lbl in stats:
        print(f"  {lbl:>14s}", end="")
    print()
    print("-" * (30 + 18 * len(stats)))
    for k in METRICS:
        print(f"{k:30s}", end="")
        for lbl in stats:
            m = np.mean([s[k] for s in stats[lbl]]) * 100
            print(f"  {m:13.2f}%", end="")
        print()


if __name__ == "__main__":
    main()
