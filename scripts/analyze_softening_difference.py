"""
Analyze: temperature-scaled logit-KD vs token-avg — are they encoding the same thing?

Shows that even at matched entropy, the two methods encode fundamentally different information:
- Temperature scaling: parametric, same frame, uniform spreading
- Token-avg: structural, segment-level, reflects actual temporal uncertainty

Outputs:
  analysis/softening_diff/
    entropy_comparison.png   — entropy of targets across segment types
    boundary_distribution.png — distribution comparison at token boundaries
    kl_vs_temperature.png    — KL(frame-T || token-avg) as T increases
    top2_overlap.png         — how often top-2 tokens agree between methods
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, "src")

OUT_DIR = Path("analysis/softening_diff")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_TAVG  = "data/train_clean_100.small_teacher.token_avg.json"
MANIFEST_FKD   = "data/train_clean_100.small_teacher.frame_top8_t1.json"
N_SAMPLES      = 200   # utterances to analyze
TEMPERATURES   = [1, 2, 4, 8]
BLANK          = 0


# ── helpers ───────────────────────────────────────────────────────────────────

def load_manifest(path, n=None):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if n and len(rows) >= n:
                break
    return rows


def load_tavg(row, manifest_dir):
    p = Path(row["teacher_token_avg_path"])
    if not p.is_absolute():
        p = Path(manifest_dir) / p
    return torch.load(p, map_location="cpu", weights_only=False)


def load_fkd(row, manifest_dir):
    p = Path(row["teacher_frame_kd_path"])
    if not p.is_absolute():
        p = Path(manifest_dir) / p
    return torch.load(p, map_location="cpu", weights_only=False)


def entropy(probs):
    """Shannon entropy of a probability vector (nats)."""
    p = probs.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)


def temperature_scale(probs_t1, T):
    """Re-apply temperature to T=1 softmax probabilities.
    Approximation: log(p_T1) / T then re-normalize.
    Exact when T=1 probabilities come from uniform logits, approximate otherwise.
    Valid because top-k probabilities sum to ~1 for large k and dominant peaks.
    """
    log_p = probs_t1.clamp_min(1e-12).log()
    scaled = (log_p / T).exp()
    return scaled / scaled.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def kl_div(p, q):
    """KL(p || q) per row, p and q are (*, V) prob vectors."""
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    return (p * (p.log() - q.log())).sum(dim=-1)


# ── load data ─────────────────────────────────────────────────────────────────

print("[load] manifests...")
tavg_rows = {r["audio_filepath"]: r for r in load_manifest(MANIFEST_TAVG)}
fkd_rows  = {r["audio_filepath"]: r for r in load_manifest(MANIFEST_FKD)}

common_keys = sorted(set(tavg_rows) & set(fkd_rows))[:N_SAMPLES]
print(f"  using {len(common_keys)} utterances")

manifest_dir = Path(MANIFEST_TAVG).parent


# ── per-segment statistics ────────────────────────────────────────────────────

stats = {
    "seg_len":       [],   # number of teacher frames in segment
    "tavg_entropy":  [],   # entropy of token-avg target
    "fkd_entropy":   {T: [] for T in TEMPERATURES},   # entropy of frame-KD at center frame
    "kl_fkd_tavg":  {T: [] for T in TEMPERATURES},   # KL(frame-T || tavg)
    "top1_match":   {T: [] for T in TEMPERATURES},   # does top-1 token agree?
    "top2_match":   {T: [] for T in TEMPERATURES},   # do top-2 sets agree?
}

# boundary frame analysis: last frame of each segment
boundary_stats = {
    "tavg_entropy": [],
    "fkd_t1_entropy": [],
    "fkd_t8_entropy": [],
    "kl_t8_tavg": [],
}

# collect boundary-frame distributions for Figure 2
boundary_examples = []   # list of dicts for a few clear boundary cases

print("[process] computing per-segment statistics...")
for key in common_keys:
    ta = load_tavg(tavg_rows[key], manifest_dir)
    fk = load_fkd(fkd_rows[key], manifest_dir)

    seg_starts  = ta["seg_starts"]   # (N,)
    seg_ends    = ta["seg_ends"]     # (N,)
    avg_ids     = ta["avg_ids"]      # (N, top_k)  int32
    avg_probs   = ta["avg_probs"].float()  # (N, top_k)

    fkd_ids   = fk["ids"].long()    # (T, top_k)
    fkd_probs = fk["probs"].float() # (T, top_k)
    T_frames  = fkd_ids.shape[0]
    vocab     = 1025  # 1024 BPE tokens + blank (id=1024)

    N = seg_starts.shape[0]

    for n in range(N):
        s, e = int(seg_starts[n]), int(seg_ends[n])
        seg_len = e - s
        if seg_len < 1 or s >= T_frames or e > T_frames:
            continue

        # ── token-avg distribution (full vocab scatter) ──
        tavg_full = torch.zeros(vocab)
        tavg_full.scatter_(0, avg_ids[n].long(), avg_probs[n])
        tavg_H = float(entropy(tavg_full))

        stats["seg_len"].append(seg_len)
        stats["tavg_entropy"].append(tavg_H)

        # ── center frame of segment ──
        center = min(s + seg_len // 2, T_frames - 1)

        # ── last (boundary) frame of segment ──
        boundary = min(e - 1, T_frames - 1)

        for T in TEMPERATURES:
            fkd_center_full = torch.zeros(vocab)
            fkd_center_full.scatter_(0, fkd_ids[center], fkd_probs[center])
            if T != 1:
                fkd_center_full = temperature_scale(fkd_center_full, T)

            fkd_H = float(entropy(fkd_center_full))
            kl    = float(kl_div(fkd_center_full, tavg_full))

            stats["fkd_entropy"][T].append(fkd_H)
            stats["kl_fkd_tavg"][T].append(kl)

            top1_t = int(fkd_ids[center, 0])
            top1_a = int(avg_ids[n, 0])
            stats["top1_match"][T].append(int(top1_t == top1_a))

            top2_t = set(fkd_ids[center, :2].tolist())
            top2_a = set(avg_ids[n, :2].tolist())
            stats["top2_match"][T].append(int(top2_t == top2_a))

        # ── boundary frame analysis ──
        fkd_b_full = torch.zeros(vocab)
        fkd_b_full.scatter_(0, fkd_ids[boundary], fkd_probs[boundary])
        fkd_b_t8 = temperature_scale(fkd_b_full.clone(), 8)

        boundary_stats["tavg_entropy"].append(tavg_H)
        boundary_stats["fkd_t1_entropy"].append(float(entropy(fkd_b_full)))
        boundary_stats["fkd_t8_entropy"].append(float(entropy(fkd_b_t8)))
        boundary_stats["kl_t8_tavg"].append(float(kl_div(fkd_b_t8, tavg_full)))

        # collect clear boundary examples for Figure 2 (short segments, high tavg entropy)
        if seg_len <= 3 and tavg_H > 0.5 and len(boundary_examples) < 5:
            boundary_examples.append({
                "tavg":    tavg_full.numpy(),
                "fkd_t1": fkd_b_full.numpy(),
                "fkd_t8": fkd_b_t8.numpy(),
                "top_ids": avg_ids[n, :8].tolist(),
                "seg_len": seg_len,
            })

print(f"  processed {len(stats['seg_len'])} segments")


# ── Figure 1: Entropy comparison across temperatures ─────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribution Softening: Temperature Scaling vs Token-Avg", fontsize=13)

# Left: mean entropy per method
methods  = [f"Logit-KD\nT={T}" for T in TEMPERATURES] + ["Token-Avg"]
entropies = [np.mean(stats["fkd_entropy"][T]) for T in TEMPERATURES] + \
            [np.mean(stats["tavg_entropy"])]
colors   = ["#4C72B0"] * len(TEMPERATURES) + ["#DD8452"]
errs     = [np.std(stats["fkd_entropy"][T]) for T in TEMPERATURES] + \
            [np.std(stats["tavg_entropy"])]

ax = axes[0]
bars = ax.bar(methods, entropies, color=colors, yerr=errs, capsize=4, alpha=0.85)
ax.set_ylabel("Target entropy (nats)")
ax.set_title("Mean KD target entropy\n(same entropy ≠ same information)")
ax.set_ylim(0, max(entropies) * 1.3)
for bar, v in zip(bars, entropies):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9)

# Right: KL divergence from token-avg across temperatures
kl_means = [np.mean(stats["kl_fkd_tavg"][T]) for T in TEMPERATURES]
kl_stds  = [np.std(stats["kl_fkd_tavg"][T])  for T in TEMPERATURES]
ax2 = axes[1]
ax2.errorbar(TEMPERATURES, kl_means, yerr=kl_stds, marker="o", linewidth=2,
             color="#4C72B0", capsize=5)
ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax2.set_xlabel("Temperature T")
ax2.set_ylabel("KL(frame-KD_T || token-avg) [nats]")
ax2.set_title("KL divergence from token-avg targets\n(large KL = fundamentally different, not just softer)")
ax2.set_xticks(TEMPERATURES)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "entropy_comparison.png", dpi=150)
plt.close()
print("[fig] entropy_comparison.png saved")


# ── Figure 2: Distribution at token boundary (qualitative) ───────────────────

if boundary_examples:
    ex = boundary_examples[0]
    top_ids = ex["top_ids"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig.suptitle(
        f"Token Boundary Frame (segment length = {ex['seg_len']} frames)\n"
        "Showing top-8 vocab positions from token-avg target",
        fontsize=11,
    )

    labels = [str(i) for i in top_ids]
    x = np.arange(len(top_ids))
    width = 0.25

    for ax, key, color, title in zip(
        axes,
        ["fkd_t1", "fkd_t8", "tavg"],
        ["#4C72B0", "#64B5CD", "#DD8452"],
        ["Logit-KD  T=1\n(boundary frame)", "Logit-KD  T=8\n(boundary frame, temperature-scaled)", "Token-Avg\n(segment average)"],
    ):
        vals = [ex[key][i] for i in top_ids]
        H = entropy(torch.tensor(ex[key] if key != "fkd_t8" else ex["fkd_t8"]))
        ax.bar(x, vals, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("Token ID (top-8 from token-avg)")
        ax.set_ylabel("Probability")
        ax.set_title(f"{title}\nH = {float(entropy(torch.tensor(vals))):.3f} nats")
        ax.set_ylim(0, max(max(vals)*1.2, 0.05))

    plt.tight_layout()
    plt.savefig(OUT_DIR / "boundary_distribution.png", dpi=150)
    plt.close()
    print("[fig] boundary_distribution.png saved")


# ── Figure 3: Top-1 / Top-2 token agreement ──────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Token Identity Agreement: Frame-KD vs Token-Avg")

for ax, key, title in zip(
    axes,
    ["top1_match", "top2_match"],
    ["Top-1 token match rate", "Top-2 token set match rate"],
):
    rates = [np.mean(stats[key][T]) * 100 for T in TEMPERATURES]
    ax.plot(TEMPERATURES, rates, marker="o", linewidth=2, color="#4C72B0")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Match rate (%)")
    ax.set_title(f"{title}\n(lower = more different information)")
    ax.set_xticks(TEMPERATURES)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    for T, r in zip(TEMPERATURES, rates):
        ax.text(T, r + 2, f"{r:.1f}%", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "top2_overlap.png", dpi=150)
plt.close()
print("[fig] top2_overlap.png saved")


# ── Summary statistics ────────────────────────────────────────────────────────

print("\n=== Summary ===")
print(f"{'Method':<22} {'Mean H (nats)':>15}  {'KL from tavg':>14}  {'Top-1 match':>12}  {'Top-2 match':>12}")
print("-" * 80)
for T in TEMPERATURES:
    H   = np.mean(stats["fkd_entropy"][T])
    kl  = np.mean(stats["kl_fkd_tavg"][T])
    t1  = np.mean(stats["top1_match"][T]) * 100
    t2  = np.mean(stats["top2_match"][T]) * 100
    print(f"  Logit-KD T={T:<4}      {H:>12.4f}   {kl:>12.4f}   {t1:>10.1f}%   {t2:>10.1f}%")

H_a = np.mean(stats["tavg_entropy"])
print(f"  Token-Avg          {H_a:>12.4f}   {'—':>12}   {'—':>10}   {'—':>10}")

print("\n[key insight]")
print("  If temperature scaling and token-avg were equivalent,")
print("  KL(frame-T || tavg) would approach 0 as T → ∞.")
print("  Persistent large KL means they encode structurally different information.")
