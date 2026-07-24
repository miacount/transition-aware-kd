"""Batched CTC forward-backward token occupancy (gamma) on GPU.

Used by dual-occupancy span-KD: the student pools its own posterior with its own
transcript-constrained gamma (detached), instead of the teacher's resampled one.

Semantics match scripts/visualize_ctc_forward_backward.ctc_forward_backward
(token_gamma) exactly; verified against it in tests. Everything runs under
no_grad — gamma is a pooling coordinate, not a gradient path.

Padding handling: for frames t >= enc_len the emission is rewritten so that only
the final blank state emits (log-prob 0, others -inf). Every valid path then
extends uniquely through the padded frames into the final blank state, which
preserves logZ and all state posteriors at valid frames — so a single uniform
DP over T_max is exact for every sequence in the batch.
"""
import torch

NEG = -1e30


@torch.no_grad()
def batched_ctc_token_gamma(log_probs, enc_len, tokens, token_lens, blank_id,
                            blank_penalty=0.0):
    """
    Args:
        log_probs:  (B, T, V) log-softmax over vocab+blank
        enc_len:    (B,) valid frames per sequence
        tokens:     (B, N_max) transcript token ids (no blanks), padded arbitrarily
        token_lens: (B,) valid token counts (must be >= 1)
        blank_id:   blank index
        blank_penalty: nats subtracted from blank log-prob (then renormalized)
                       before the DP — same de-peak as the teacher-side builder.

    Returns:
        gamma: (B, N_max, T) float32; P(state = token u | X, transcript) per frame.
               Zero outside valid frames/tokens.
    """
    B, T, V = log_probs.shape
    device = log_probs.device
    lp = log_probs.float()
    if blank_penalty != 0.0:
        lp = lp.clone()
        lp[:, :, blank_id] -= blank_penalty
        lp = lp - lp.logsumexp(dim=-1, keepdim=True)

    N_max = tokens.shape[1]
    S = 2 * N_max + 1
    tokens = tokens.long().clamp(0, V - 1)
    token_lens = token_lens.long()
    S_len = 2 * token_lens + 1                                   # (B,)

    ext = torch.full((B, S), blank_id, dtype=torch.long, device=device)
    ext[:, 1::2] = tokens                                        # odd states = labels

    # emissions em[b,t,s] = lp[b,t,ext[b,s]]
    em = torch.gather(lp, 2, ext.unsqueeze(1).expand(B, T, S))   # (B, T, S)

    # state validity and padded-frame rewrite
    s_idx = torch.arange(S, device=device).unsqueeze(0)          # (1, S)
    state_valid = s_idx < S_len.unsqueeze(1)                     # (B, S)
    em = em.masked_fill(~state_valid.unsqueeze(1), NEG)
    t_idx = torch.arange(T, device=device).view(1, T, 1)
    pad_frame = t_idx >= enc_len.view(B, 1, 1)                   # (B, T, 1)
    final_blank = (s_idx == (S_len - 1).unsqueeze(1)).unsqueeze(1)  # (B, 1, S)
    em = torch.where(pad_frame & final_blank, torch.zeros_like(em), em)
    em = torch.where(pad_frame & ~final_blank, torch.full_like(em, NEG), em)

    # skip transitions s-2 -> s allowed iff label state and different label
    allow2 = torch.zeros(B, S, dtype=torch.bool, device=device)
    allow2[:, 2:] = (ext[:, 2:] != blank_id) & (ext[:, 2:] != ext[:, :-2])
    allow2 &= state_valid
    allow2_next = torch.zeros_like(allow2)                       # from s over s+2
    allow2_next[:, :-2] = allow2[:, 2:]

    neg_col = torch.full((B, 1), NEG, device=device)
    neg_col2 = torch.full((B, 2), NEG, device=device)

    alphas = torch.empty(B, T, S, device=device)
    a = torch.full((B, S), NEG, device=device)
    a[:, 0] = em[:, 0, 0]
    a[:, 1] = torch.where(S_len > 1, em[:, 0, 1], torch.full_like(em[:, 0, 1], NEG))
    alphas[:, 0] = a
    for t in range(1, T):
        m1 = torch.cat([neg_col, a[:, :-1]], dim=1)
        m2 = torch.cat([neg_col2, a[:, :-2]], dim=1)
        m2 = torch.where(allow2, m2, torch.full_like(m2, NEG))
        a = torch.logaddexp(torch.logaddexp(a, m1), m2) + em[:, t]
        alphas[:, t] = a

    betas = torch.empty(B, T, S, device=device)
    b_ = torch.full((B, S), NEG, device=device)
    b_.scatter_(1, (S_len - 1).unsqueeze(1), 0.0)
    b_.scatter_(1, (S_len - 2).clamp_min(0).unsqueeze(1), 0.0)
    betas[:, T - 1] = b_
    for t in range(T - 2, -1, -1):
        v = b_ + em[:, t + 1]
        n1 = torch.cat([v[:, 1:], neg_col], dim=1)
        n2 = torch.cat([v[:, 2:], neg_col2], dim=1)
        n2 = torch.where(allow2_next, n2, torch.full_like(n2, NEG))
        b_ = torch.logaddexp(torch.logaddexp(v, n1), n2)
        betas[:, t] = b_

    last_a = alphas[:, T - 1]                                    # (B, S)
    log_z = torch.logaddexp(
        last_a.gather(1, (S_len - 1).unsqueeze(1)),
        last_a.gather(1, (S_len - 2).clamp_min(0).unsqueeze(1)),
    ).squeeze(1)                                                 # (B,)

    gamma = (alphas + betas - log_z.view(B, 1, 1)).exp()
    gamma = torch.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
    gamma = gamma[:, :, 1::2].permute(0, 2, 1).contiguous()      # (B, N_max, T)

    frame_valid = (torch.arange(T, device=device).view(1, 1, T)
                   < enc_len.view(B, 1, 1))
    tok_valid = (torch.arange(N_max, device=device).view(1, N_max, 1)
                 < token_lens.view(B, 1, 1))
    return gamma * frame_valid * tok_valid
