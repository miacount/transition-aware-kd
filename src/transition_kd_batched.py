"""
Batched Transition-KD loss (hard version) with padding + masking.

Shapes:
  log_probs     : (B, T_max, K)  log-softmax of student over vocab+blank
  targets       : (B, M_max)     teacher blank-retained dedup sequences, pad with 0
  input_lengths : (B,)           valid #frames per sequence (T_b)
  target_lengths: (B,)           valid #states per sequence (M_b), needs T_b >= M_b

Returns per-sequence loss (B,). Masking guarantees:
  - padded FRAMES (t >= T_b) never update alpha (DP freezes at last valid frame)
  - padded STATES (m >= M_b) can't corrupt the valid final state (DP only flows
    forward m-1 -> m, and we read out exactly state M_b-1)
"""
import torch

FLOOR = -1e9


# ---- single-sequence reference (oracle) --------------------------------
def transition_kd_loss_single(log_probs, target_seq):
    T, K = log_probs.shape
    target = torch.as_tensor(target_seq, device=log_probs.device, dtype=torch.long)
    M = target.numel()
    emit = log_probs[:, target]
    floor_col = torch.full((1,), FLOOR, device=log_probs.device, dtype=log_probs.dtype)
    alpha = torch.full((M,), FLOOR, device=log_probs.device, dtype=log_probs.dtype).clone()
    alpha[0] = emit[0, 0]
    for t in range(1, T):
        adv = torch.cat([floor_col, alpha[:-1]])
        alpha = emit[t] + torch.logaddexp(alpha, adv)
    return -alpha[M - 1]


# ---- batched version ---------------------------------------------------
def transition_kd_loss_batched(log_probs, targets, input_lengths, target_lengths):
    B, T_max, K = log_probs.shape
    M_max = targets.shape[1]
    device = log_probs.device
    dtype = log_probs.dtype

    # emit[b, t, m] = log_probs[b, t, targets[b, m]]      -> (B, T_max, M_max)
    idx = targets.unsqueeze(1).expand(B, T_max, M_max)
    emit = torch.gather(log_probs, 2, idx)

    floor_col = torch.full((B, 1), FLOOR, device=device, dtype=dtype)

    # init at t=0: state 0 reachable, rest FLOOR
    alpha = torch.full((B, M_max), FLOOR, device=device, dtype=dtype)
    alpha = alpha.clone()
    alpha[:, 0] = emit[:, 0, 0]

    for t in range(1, T_max):
        stay = alpha                                    # (B, M_max)
        adv = torch.cat([floor_col, alpha[:, :-1]], 1)  # shift right along states
        cand = emit[:, t, :] + torch.logaddexp(stay, adv)
        # only update sequences whose frame t is still valid
        update = (t < input_lengths).view(B, 1)
        alpha = torch.where(update, cand, alpha)

    # read out state (M_b - 1) for each sequence
    last_state = (target_lengths - 1).view(B, 1)
    final_alpha = alpha.gather(1, last_state).squeeze(1)  # (B,)
    return -final_alpha



# ---- tests -------------------------------------------------------------
def sharp_logprobs(path, K, peak=8.0):
    T = len(path)
    logits = torch.zeros(T, K)
    for t, c in enumerate(path):
        logits[t, c] = peak
    return torch.log_softmax(logits, dim=-1)


def dedup(path):
    out = []
    for c in path:
        if not out or out[-1] != c:
            out.append(c)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    H, E, L, b = 0, 1, 2, 3
    K = 4

    # three sequences with DIFFERENT T and DIFFERENT M, to stress padding
    paths = [
        [H, H, H, b, E, E, b, b, L, L],          # T=10, dedup H b E b L  (M=5)
        [H, b, E, E, E],                          # T=5,  dedup H b E      (M=3)
        [H, H, b, b, b, b, L],                    # T=7,  dedup H b L      (M=3)
    ]
    targets_list = [dedup(p) for p in paths]
    print("targets:", targets_list)

    # ----- reference: run each sequence alone -----
    ref = []
    for p, tgt in zip(paths, targets_list):
        ref.append(transition_kd_loss_single(sharp_logprobs(p, K), tgt).item())

    # ----- batched with padding -----
    B = len(paths)
    T_max = max(len(p) for p in paths)
    M_max = max(len(t) for t in targets_list)
    input_lengths = torch.tensor([len(p) for p in paths])
    target_lengths = torch.tensor([len(t) for t in targets_list])

    log_probs = torch.full((B, T_max, K), 0.0)
    for i, p in enumerate(paths):
        lp = sharp_logprobs(p, K)
        log_probs[i, :len(p)] = lp                # rest is uniform padding (masked out)
    targets = torch.zeros(B, M_max, dtype=torch.long)   # pad with 0 (harmless)
    for i, t in enumerate(targets_list):
        targets[i, :len(t)] = torch.tensor(t)

    batched = transition_kd_loss_batched(log_probs, targets, input_lengths, target_lengths)

    print("\nper-sequence loss:")
    for i in range(B):
        print(f"  seq{i}: single={ref[i]:.6f}  batched={batched[i].item():.6f}  "
              f"match={abs(ref[i]-batched[i].item())<1e-4}")

    max_diff = max(abs(ref[i] - batched[i].item()) for i in range(B))
    print(f"\n[batch==single]   max diff = {max_diff:.2e}  -> {max_diff < 1e-4}")

    # differentiability on the batched path (with padding present)
    log_probs.requires_grad_(True)
    loss = transition_kd_loss_batched(log_probs, targets, input_lengths, target_lengths).mean()
    loss.backward()
    g = log_probs.grad
    print("[differentiable]  grad finite:", torch.isfinite(g).all().item(),
          "| grad nonzero:", (g.abs().sum() > 0).item())

    # check: gradient on PADDED frames must be ~0 (they shouldn't influence loss)
    pad_grad = 0.0
    for i, p in enumerate(paths):
        if len(p) < T_max:
            pad_grad += g[i, len(p):].abs().sum().item()
    print("[pad has no grad] padded-frame grad sum =", f"{pad_grad:.2e}", "->", pad_grad < 1e-6)