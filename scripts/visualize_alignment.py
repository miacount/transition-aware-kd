"""
CTC path alignment visualization: 4x vs 8x subsampling.

Compares teacher CTC path alignment quality between:
  - 4x student  : same frame count as teacher → frame-level KD is reliable
  - 8x student  : half the frames → logit KD must resample, introducing errors
                  (simulated by nearest-neighbour compressing teacher path to T/2)

Generates two figures:
  1. Per-utterance CTC path heatmap (teacher / 4x student / simulated 8x)
  2. Aggregate alignment statistics bar chart
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


# ── helpers ──────────────────────────────────────────────────────────────────

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
    path = log_probs[0, :T].argmax(dim=-1).cpu().numpy()
    return path


def resample_path(path, target_len):
    """Nearest-neighbour resample path to target_len frames."""
    src_len = len(path)
    if src_len == target_len:
        return path.copy()
    pos = (np.arange(target_len) + 0.5) / target_len * src_len
    idx = np.clip(np.floor(pos).astype(np.int64), 0, src_len - 1)
    return path[idx]


def collapse_repeats(path):
    out = []
    prev = None
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
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ai != bj))
        prev = cur
    return prev[-1]


def safe(n, d):
    return float(n) / float(d) if d else 0.0


# ── alignment stats ───────────────────────────────────────────────────────────

def alignment_stats(teacher_path, other_path, blank_id):
    """Compare two same-length paths. Returns dict of rates."""
    assert len(teacher_path) == len(other_path)
    T = len(teacher_path)
    t_blank = teacher_path == blank_id
    o_blank = other_path == blank_id
    mismatch = teacher_path != other_path
    nb_mismatch = (t_blank != o_blank)
    both_nb = (~t_blank) & (~o_blank)
    both_nb_diff = both_nb & mismatch

    t_trans = collapse_repeats(teacher_path)
    o_trans = collapse_repeats(other_path)
    ted = edit_distance(t_trans, o_trans)

    return {
        "frame_mismatch":    safe(mismatch.sum(), T),
        "blank_nb_mismatch": safe(nb_mismatch.sum(), T),
        "both_nb_diff":      safe(both_nb_diff.sum(), max(both_nb.sum(), 1)),
        "trans_edit_rate":   safe(ted, max(len(t_trans), 1)),
    }


# ── plotting ──────────────────────────────────────────────────────────────────

BLANK_COLOR = (0.88, 0.88, 0.88)  # light grey

def path_to_rgb(path, blank_id, vocab_size, cmap_name="tab20"):
    cmap = plt.get_cmap(cmap_name)
    rgb = np.zeros((len(path), 3))
    for i, tok in enumerate(path):
        if tok == blank_id:
            rgb[i] = BLANK_COLOR
        else:
            rgb[i] = cmap(tok % 20)[:3]
    return rgb


def plot_utterance(ax, path, blank_id, vocab_size, title, show_xlabel=False):
    rgb = path_to_rgb(path, blank_id, vocab_size)
    ax.imshow(rgb[np.newaxis, :, :], aspect="auto",
              interpolation="nearest", extent=[0, len(path), 0, 1])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    if show_xlabel:
        ax.set_xlabel("frame index", fontsize=8)
    else:
        ax.set_xticks([])


def make_path_figure(teacher_path, student4x_path, blank_id, vocab_size,
                     utt_id, out_path):
    student8x_path = resample_path(teacher_path,
                                   max(1, len(teacher_path) // 2))
    student4x_resampled = resample_path(student4x_path, len(teacher_path))

    fig, axes = plt.subplots(4, 1, figsize=(14, 4),
                              gridspec_kw={"hspace": 0.55})

    plot_utterance(axes[0], teacher_path, blank_id, vocab_size,
                  f"Teacher (T={len(teacher_path)} frames, 4× subsampling)")
    plot_utterance(axes[1], student4x_resampled, blank_id, vocab_size,
                  f"4× Student (T={len(student4x_path)} frames) — aligned to teacher")
    plot_utterance(axes[2], student8x_path, blank_id, vocab_size,
                  f"8× Student simulation: teacher resampled to T/2={len(student8x_path)} frames")

    # mismatch overlay: show where 4x and 8x disagree with teacher
    diff4 = (teacher_path != student4x_resampled).astype(np.float32)
    diff8 = (teacher_path != resample_path(student8x_path, len(teacher_path))).astype(np.float32)
    diff_rgb = np.zeros((len(teacher_path), 3))
    diff_rgb[:, 0] = diff8          # red   = 8x mismatch
    diff_rgb[:, 2] = diff4 * 0.7    # blue  = 4x mismatch
    # overlap (both) → purple
    both = (diff4 > 0) & (diff8 > 0)
    diff_rgb[both] = [0.6, 0.0, 0.6]

    axes[3].imshow(diff_rgb[np.newaxis, :, :], aspect="auto",
                   interpolation="nearest", extent=[0, len(teacher_path), 0, 1])
    axes[3].set_yticks([])
    axes[3].set_title("Mismatch vs teacher  (blue=4× only, red=8× only, purple=both)",
                      fontsize=9)
    axes[3].set_xlabel("frame index", fontsize=8)

    fig.suptitle(f"CTC path alignment — {utt_id}", fontsize=10, y=1.01)

    blank_patch = mpatches.Patch(color=BLANK_COLOR, label="blank")
    fig.legend(handles=[blank_patch], loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def make_stats_figure(stats4x_list, stats8x_list, out_path):
    keys = ["frame_mismatch", "blank_nb_mismatch", "both_nb_diff", "trans_edit_rate"]
    labels = ["Frame\nmismatch", "Blank/non-blank\nmismatch",
              "Non-blank\ntoken diff", "Transition\nedit rate"]

    mean4 = {k: np.mean([s[k] for s in stats4x_list]) for k in keys}
    mean8 = {k: np.mean([s[k] for s in stats8x_list]) for k in keys}

    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    bars4 = ax.bar(x - width/2, [mean4[k]*100 for k in keys],
                   width, label="4× student (same as teacher)", color="#4477AA")
    bars8 = ax.bar(x + width/2, [mean8[k]*100 for k in keys],
                   width, label="8× student (resampled, half frames)", color="#EE6633")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Rate (%)", fontsize=9)
    ax.set_title("Teacher–Student CTC Path Alignment Statistics\n"
                 "(4× vs 8× subsampling mismatch with teacher)", fontsize=10)
    ax.legend(fontsize=9)
    ax.bar_label(bars4, fmt="%.1f%%", fontsize=7, padding=2)
    ax.bar_label(bars8, fmt="%.1f%%", fontsize=7, padding=2)
    ax.set_ylim(0, max(max(mean8.values()), max(mean4.values())) * 130 + 5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def load_student(config, ckpt, device):
    cfg = OmegaConf.load(config)
    model_cfg = cfg.model.copy()
    with open_dict(model_cfg):
        model_cfg.log_prediction = False
        if model_cfg.get("test_ds") is not None:
            del model_cfg.test_ds
    model = TransitionKDModel(cfg=model_cfg, trainer=None)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/dev_clean.json")
    ap.add_argument("--config", default="configs/student_base.yaml")
    ap.add_argument("--ckpt_4x", required=True,
                    help="4x student checkpoint")
    ap.add_argument("--teacher", default="stt_en_conformer_ctc_small")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--n_plot", type=int, default=3,
                    help="number of utterance path figures to save")
    ap.add_argument("--out_dir", default="analysis/alignment_viz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import json
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] teacher: {args.teacher}")
    teacher = EncDecCTCModelBPE.from_pretrained(args.teacher).to(device).eval()
    blank_id = int(teacher.decoder.num_classes_with_blank - 1)
    vocab_size = int(teacher.decoder.num_classes_with_blank)
    sample_rate = int(teacher.cfg.preprocessor.sample_rate)

    print(f"[load] 4x student: {args.ckpt_4x}")
    student4x = load_student(args.config, args.ckpt_4x, device)

    rows = []
    with open(args.manifest) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.limit:
                break

    stats4x_list, stats8x_list = [], []
    plotted = 0

    for idx, row in enumerate(rows):
        wav = load_audio(row["audio_filepath"], sample_rate)

        t_path  = get_ctc_path(teacher,   wav, device)
        s4_path = get_ctc_path(student4x, wav, device)

        # align to teacher length for stats
        s4_aligned = resample_path(s4_path, len(t_path))
        # simulate 8x: teacher path compressed to T/2, then back to T for fair comparison
        s8_compressed = resample_path(t_path, max(1, len(t_path) // 2))
        s8_aligned    = resample_path(s8_compressed, len(t_path))

        stats4x_list.append(alignment_stats(t_path, s4_aligned,   blank_id))
        stats8x_list.append(alignment_stats(t_path, s8_aligned,   blank_id))

        if plotted < args.n_plot:
            utt_id = Path(row["audio_filepath"]).stem
            make_path_figure(
                t_path, s4_path, blank_id, vocab_size,
                utt_id=utt_id,
                out_path=out_dir / f"path_{idx:03d}_{utt_id}.png",
            )
            plotted += 1

        if (idx + 1) % 10 == 0:
            print(f"  processed {idx+1}/{len(rows)}")

    make_stats_figure(
        stats4x_list, stats8x_list,
        out_path=out_dir / "alignment_stats_4x_vs_8x.png",
    )

    # print summary
    keys = ["frame_mismatch", "blank_nb_mismatch", "both_nb_diff", "trans_edit_rate"]
    print("\n====== alignment summary ======")
    print(f"{'metric':30s}  {'4x student':>12s}  {'8x (simulated)':>14s}")
    print("-" * 62)
    for k in keys:
        m4 = np.mean([s[k] for s in stats4x_list]) * 100
        m8 = np.mean([s[k] for s in stats8x_list]) * 100
        print(f"{k:30s}  {m4:11.2f}%  {m8:13.2f}%")


if __name__ == "__main__":
    main()
