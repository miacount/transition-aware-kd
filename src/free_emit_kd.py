"""Pure-PyTorch Free-Emission Span KD utilities."""
import math

import torch


CORRECT = 0
SUBSTITUTION = 1


def curriculum_margin(epoch, margin_start=3, anneal_epochs=70):
    """Integer wide-to-native margin; epoch 0 starts wide, endpoint is native."""
    if margin_start <= 0 or anneal_epochs <= 0:
        return 0
    progress = min(max(float(epoch), 0.0) / float(anneal_epochs), 1.0)
    return int(math.ceil(float(margin_start) * (1.0 - progress) - 1e-12))


def map_interval(start, end, teacher_frames, student_frames, margin=0):
    """Map half-open teacher interval to a covering half-open student interval."""
    if teacher_frames <= 0 or student_frames <= 0 or end <= start:
        return None
    start = max(0, min(int(start), int(teacher_frames)))
    end = max(start, min(int(end), int(teacher_frames)))
    if end <= start:
        return None
    s = int(math.floor(start * student_frames / teacher_frames)) - int(margin)
    e = int(math.ceil(end * student_frames / teacher_frames)) + int(margin)
    s = max(0, min(s, student_frames - 1))
    e = max(s + 1, min(e, student_frames))
    return s, e


def map_intervals_nonoverlap(starts, ends, teacher_frames, student_frames, margin=0):
    """Map ordered intervals, widen them, and clip at neighbour midpoints."""
    native = [map_interval(s, e, teacher_frames, student_frames, margin=0)
              for s, e in zip(starts, ends)]
    valid = [(i, x) for i, x in enumerate(native) if x is not None]
    cells = {}
    for pos, (i, (s, e)) in enumerate(valid):
        left = 0
        right = student_frames
        if pos > 0:
            ps, pe = valid[pos - 1][1]
            left = (pe + s) // 2
        if pos + 1 < len(valid):
            ns, ne = valid[pos + 1][1]
            right = (e + ns) // 2
        # Preserve the native interval and use only the gap for curriculum.
        left = min(left, s)
        right = max(right, e)
        ws = max(left, s - int(margin))
        we = min(right, e + int(margin))
        cells[i] = (ws, max(ws + 1, we))
    return [cells.get(i) for i in range(len(native))]


def free_emit_loss(log_probs, input_lengths, starts, ends, token_ids, target_types,
                   num_targets, teacher_frames, blank_id, sub_weight=0.25,
                   spread_weight=0.05, margin=0):
    """Vectorized mean loss; widening is clipped at neighbour midpoints."""
    B, T, vocab = log_probs.shape
    N = starts.shape[1]
    zero = log_probs.sum() * 0.0
    if N == 0 or T == 0:
        diag = {k: zero.detach() for k in ("correct", "sub", "spread")}
        diag["targets"] = torch.zeros((), device=log_probs.device)
        return zero, diag

    device = log_probs.device
    dtype = log_probs.dtype
    eps = torch.finfo(dtype).eps
    u = torch.arange(N, device=device).view(1, N)
    n = num_targets.to(device=device).clamp(0, N).view(B, 1)
    ts = input_lengths.to(device=device).clamp(0, T).view(B, 1)
    teacher_raw = teacher_frames.to(device=device).view(B, 1)
    tt = teacher_raw.clamp_min(1)
    valid = (u < n) & (ts > 0) & (teacher_raw > 0)

    st = starts.to(device=device).clamp_min(0)
    en = ends.to(device=device).clamp_min(0)
    st = torch.minimum(st, tt)
    en = torch.minimum(torch.maximum(en, st), tt)
    valid = valid & (en > st)

    native_s = torch.div(st * ts, tt, rounding_mode="floor")
    native_e = torch.div(en * ts + tt - 1, tt, rounding_mode="floor")
    native_s = torch.minimum(native_s, (ts - 1).clamp_min(0))
    native_e = torch.minimum(torch.maximum(native_e, native_s + 1), ts)

    prev_e = torch.cat([torch.zeros(B, 1, device=device, dtype=native_e.dtype),
                        native_e[:, :-1]], dim=1)
    next_s = torch.cat([native_s[:, 1:], ts], dim=1)
    has_prev = u > 0
    has_next = (u + 1) < n
    left = torch.where(has_prev, torch.div(prev_e + native_s, 2, rounding_mode="floor"),
                       torch.zeros_like(native_s))
    right = torch.where(has_next, torch.div(native_e + next_s, 2, rounding_mode="floor"),
                        ts.expand_as(native_e))
    left = torch.minimum(left, native_s)
    right = torch.maximum(right, native_e)
    win_s = torch.maximum(left, native_s - int(margin))
    win_e = torch.minimum(right, native_e + int(margin))
    win_e = torch.maximum(win_e, win_s + 1)

    frame = torch.arange(T, device=device).view(1, 1, T)
    interval = valid.unsqueeze(2) & (frame >= win_s.unsqueeze(2)) & (frame < win_e.unsqueeze(2))

    probs = log_probs.exp()
    ids = token_ids.to(device=device).clamp(0, vocab - 1)
    p_token = torch.gather(
        probs.transpose(1, 2), 1, ids.unsqueeze(2).expand(B, N, T))
    neg = torch.full((), torch.finfo(dtype).min, device=device, dtype=dtype)
    token_peak = p_token.masked_fill(~interval, neg).amax(dim=2)
    lengths = interval.sum(dim=2)

    kinds = target_types.to(device=device)
    correct = valid & (kinds == CORRECT)
    substitution = valid & (kinds == SUBSTITUTION)
    active_sub = substitution & (float(sub_weight) > 0.0)
    active = correct | active_sub

    correct_loss = -token_peak.clamp_min(eps).log()
    residual = (p_token * interval.to(dtype)).sum(dim=2) - token_peak
    spread = torch.where(
        lengths > 1,
        residual / (lengths - 1).clamp_min(1).to(dtype),
        torch.zeros_like(residual),
    )

    p_nonblank = 1.0 - probs[:, :, blank_id]
    nonblank_peak = p_nonblank.unsqueeze(1).expand(B, N, T).masked_fill(
        ~interval, neg).amax(dim=2)
    sub_loss = -nonblank_peak.clamp_min(eps).log()

    per_target = torch.where(
        correct,
        correct_loss + float(spread_weight) * spread,
        torch.where(active_sub, float(sub_weight) * sub_loss, torch.zeros_like(sub_loss)),
    )
    denom = active.sum().clamp_min(1).to(dtype)
    loss = per_target.sum() / denom
    diag = {
        "correct": (correct_loss.masked_fill(~correct, 0.0).sum() / denom).detach(),
        "sub": (sub_loss.masked_fill(~active_sub, 0.0).sum() / denom).detach(),
        "spread": (spread.masked_fill(~correct, 0.0).sum() / denom).detach(),
        "targets": active.sum().to(dtype).detach(),
    }
    return loss, diag
