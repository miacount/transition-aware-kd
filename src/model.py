"""NeMo CTC student model with KD modes.

Supported kd_mode:
  none          : CTC only
  trans         : CTC + blank-retained transition KD
  logit         : CTC + frame-wise top-k teacher posterior KD
  token_avg     : CTC + per-token segment averaged KD
  combined      : CTC + logit KD + token_avg KD
  guided        : CTC + guide loss (Kurata & Audhkhasi 2019)
  delayed_logit : CTC + TAB-based logit KD (Li et al. 2025, adapted for non-streaming)
  self_kd       : Self-KD (Kim et al. 2024): intermediate CTC head + frame-level SKD
  cr_ctc        : Consistency-Regularized CTC (Yao et al. 2025, ICLR): two independently
                  strongly-augmented views through the shared encoder, symmetric KL
                  consistency loss between the two posteriors. No teacher needed.

logit_kd_blank_mode (applies to logit, guided, delayed_logit):
  none        : all frames
  elimination : only teacher argmax != blank  (Hilmes et al. 2025 KD-BE)
  symmetric   : elimination + n surrounding frames each side (Hilmes et al. 2025 Sym)
  trim        : all frames between first and last non-blank (Hilmes et al. 2025 Trim)
  threshold   : frames where p(blank) < logit_kd_blank_threshold (Hilmes et al. 2025 Threshold)
  random      : non-blank frames + randomly selected blank frames (Hilmes et al. 2025 Random)
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from nemo.collections.asr.models import EncDecCTCModelBPE

from data import make_dataloader
from transition_kd_batched import transition_kd_loss_batched

_VALID_KD_MODES = (
    "none", "trans", "logit", "token_avg", "combined",
    "guided", "delayed_logit", "self_kd", "aligned_token", "span_kd", "cr_ctc",
)
_NEEDS_FRAME_KD   = ("logit", "combined", "guided", "delayed_logit")
_NEEDS_TOKEN_AVG  = ("token_avg", "aligned_token")
_NEEDS_SPAN_KD    = ("span_kd",)


# ---------------------------------------------------------------------------
# CTC Viterbi forced alignment (module-level, no_grad)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _ctc_viterbi_align(log_probs, tokens, input_lengths, target_lengths, blank_id):
    """
    Batched CTC Viterbi forced alignment.
    Returns per-token segment boundaries via forward DP + back-pointer traceback.

    Args:
        log_probs:      (B, T, K) float  — student log-probs (detached, no grad needed)
        tokens:         (B, L) long      — true transcript tokens (padded, no blanks)
        input_lengths:  (B,) long        — valid T per utterance
        target_lengths: (B,) long        — valid L per utterance
        blank_id:       int

    Returns:
        seg_starts: (B, L) long  — start frame per token (student frame space)
        seg_ends:   (B, L) long  — exclusive end frame per token
    """
    B, T, K = log_probs.shape
    L = tokens.shape[1]
    S = 2 * L + 1
    device = log_probs.device
    dtype = log_probs.dtype
    NEG_INF = torch.finfo(dtype).min / 2

    # Extended label: [blank, t0, blank, t1, ..., tL-1, blank]
    ext = tokens.new_full((B, S), blank_id)
    ext[:, 1::2] = tokens

    # log-prob at each extended position: (B, T, S)
    lp_ext = torch.gather(log_probs, 2, ext.unsqueeze(1).expand(B, T, S))

    # Valid state mask: s < 2*target_lengths+1
    ext_lens = 2 * target_lengths + 1
    s_valid = torch.arange(S, device=device).unsqueeze(0) < ext_lens.unsqueeze(1)  # (B, S)

    # Can-skip: non-blank token at s that differs from s-2
    is_tok = (ext != blank_id)
    diff_sm2 = torch.zeros(B, S, dtype=torch.bool, device=device)
    if S > 2:
        diff_sm2[:, 2:] = ext[:, 2:] != ext[:, :-2]
    can_skip = is_tok & diff_sm2 & s_valid  # (B, S)

    # Forward DP — alpha and back-pointer ptr
    alpha = torch.full((B, T, S), NEG_INF, device=device, dtype=dtype)
    ptr   = torch.zeros(B, T, S, dtype=torch.int8, device=device)  # 0=stay,1=+1,2=+2

    alpha[:, 0, 0] = lp_ext[:, 0, 0]
    if L > 0:
        alpha[:, 0, 1] = lp_ext[:, 0, 1]
    alpha[:, 0] = alpha[:, 0].masked_fill(~s_valid, NEG_INF)

    pad1 = torch.full((B, 1), NEG_INF, device=device, dtype=dtype)
    pad2 = torch.full((B, 2), NEG_INF, device=device, dtype=dtype)
    i8_0 = torch.zeros(B, S, dtype=torch.int8, device=device)
    i8_1 = torch.ones(B, S, dtype=torch.int8, device=device)
    i8_2 = torch.full((B, S), 2, dtype=torch.int8, device=device)

    for t in range(1, T):
        prev = alpha[:, t - 1]
        opt0 = prev
        opt1 = torch.cat([pad1, prev[:, :-1]], 1)
        opt2 = torch.where(can_skip,
                           torch.cat([pad2, prev[:, :-2]], 1),
                           torch.full_like(prev, NEG_INF))
        use2 = opt2 > torch.max(opt0, opt1)
        use1 = ~use2 & (opt1 > opt0)
        best = torch.where(use2, opt2, torch.where(use1, opt1, opt0))
        new_a = best + lp_ext[:, t]
        frame_ok = (t < input_lengths).unsqueeze(1)
        alpha[:, t] = torch.where(s_valid & frame_ok, new_a,
                                   torch.full_like(new_a, NEG_INF))
        ptr[:, t] = torch.where(use2, i8_2, torch.where(use1, i8_1, i8_0))

    # Traceback — recover the optimal path
    b_idx = torch.arange(B, device=device)
    last_t = (input_lengths - 1).clamp(0, T - 1)            # (B,)
    last_s = (ext_lens - 1).clamp(0, S - 1)                 # (B,)
    alt_s  = (ext_lens - 2).clamp(0, S - 1)                 # (B,)
    a_last = alpha[b_idx, last_t, last_s]
    a_alt  = alpha[b_idx, last_t, alt_s]
    cur_s  = torch.where(a_last >= a_alt, last_s, alt_s)    # (B,)

    # frame_assign[b, t] = ext position assigned to frame t
    frame_assign = torch.zeros(B, T, dtype=torch.long, device=device)
    frame_assign[b_idx, last_t] = cur_s

    for t in range(T - 2, -1, -1):
        back   = ptr[b_idx, t + 1, cur_s].long()            # 0, 1, or 2
        new_s  = (cur_s - back).clamp(0)
        t1_ok  = (t + 1) < input_lengths                    # (B,) bool
        cur_s  = torch.where(t1_ok, new_s, cur_s)
        frame_ok = t < input_lengths
        frame_assign[b_idx, t] = torch.where(frame_ok, cur_s,
                                              frame_assign[b_idx, t])

    # Extract segment boundaries: frames where frame_assign == 2k+1 for token k
    seg_starts = torch.zeros(B, L, dtype=torch.long, device=device)
    seg_ends   = torch.ones(B, L, dtype=torch.long, device=device)

    for b in range(B):
        vT = int(input_lengths[b].item())
        vL = int(target_lengths[b].item())
        assign_b = frame_assign[b, :vT]

        for k in range(vL):
            pos    = 2 * k + 1
            frames = (assign_b == pos).nonzero(as_tuple=True)[0]
            if frames.numel() > 0:
                seg_starts[b, k] = frames[0]
                seg_ends[b, k]   = frames[-1] + 1
            else:
                peak = int(alpha[b, :vT, pos].argmax().item())
                seg_starts[b, k] = peak
                seg_ends[b, k]   = peak + 1

    return seg_starts, seg_ends


class TransitionKDModel(EncDecCTCModelBPE):
    def __init__(self, cfg, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.kd_mode = cfg.get("kd_mode", "none")
        self.kd_weight = float(cfg.get("kd_weight", 0.0))
        self.token_avg_kd_weight = float(cfg.get("token_avg_kd_weight", 0.0))
        self.logit_kd_temperature = float(cfg.get("logit_kd_temperature", 2.0))
        self.logit_kd_blank_mode = cfg.get("logit_kd_blank_mode", "none")
        self.logit_kd_blank_n = int(cfg.get("logit_kd_blank_n", 1))
        self.logit_kd_blank_threshold = float(cfg.get("logit_kd_blank_threshold", 0.9))
        self.logit_kd_blank_beta = float(cfg.get("logit_kd_blank_beta", 0.5))
        self.logit_kd_tab_size = int(cfg.get("logit_kd_tab_size", 2))
        self.span_kd_gate_threshold = float(cfg.get("span_kd_gate_threshold", 0.3))
        self.kd_start_step = int(cfg.get("kd_start_step", 0))
        self.transition_kd_weight = float(cfg.get("transition_kd_weight", 0.0))
        self.logit_kd_occ_weight = float(cfg.get("logit_kd_occ_weight", 0.0))
        self.logit_kd_res_weight = float(cfg.get("logit_kd_res_weight", 0.0))
        self.sr_ctc_weight = float(cfg.get("sr_ctc_weight", 0.0))  # SR-CTC smooth reg (Yao 2025)
        self.blank_id = self.decoder.num_classes_with_blank - 1
        # Paper-style lambda: L = (1-λ)*L_CTC + λ*L_KD  (Hilmes et al. 2025)
        # λ=1.0 → pure KD (no CTC loss).  None → legacy formula L_CTC + kd_weight*L_KD.
        _lam = cfg.get("kd_lambda", None)
        self.kd_lambda = float(_lam) if _lam is not None else None

        if self.kd_mode not in _VALID_KD_MODES:
            raise ValueError(f"unsupported kd_mode: {self.kd_mode}")
        if self.kd_mode == "combined":
            if self.kd_weight == 0.0 and self.token_avg_kd_weight == 0.0:
                raise ValueError("combined mode: both kd_weight and token_avg_kd_weight are 0.0")
        elif self.kd_mode not in ("none", "self_kd", "cr_ctc"):
            if self.kd_lambda is None and self.kd_weight == 0.0:
                raise ValueError(
                    f"kd_mode='{self.kd_mode}' but kd_weight=0.0 and kd_lambda not set — KD loss will be zero."
                )

        # Self-KD: intermediate CTC head + forward hook (Kim et al. 2024)
        if self.kd_mode == "self_kd":
            self.skd_split_layer = int(cfg.get("skd_split_layer", 4))
            self.skd_alpha = float(cfg.get("skd_alpha", 0.5))
            enc_dim = int(cfg.encoder.d_model)
            vocab_size = self.decoder.num_classes_with_blank
            self.skd_inter_head = nn.Linear(enc_dim, vocab_size)
            self._skd_inter_hidden = None
            # Hook captures output of encoder layer skd_split_layer (0-indexed: layer N-1)
            self.encoder.layers[self.skd_split_layer - 1].register_forward_hook(
                self._capture_skd_hidden
            )

        # CR-CTC (Yao et al. 2025): dual augmented views + symmetric KL consistency.
        # Paper default: alpha=0.2, total time-masking "volume" x2.5 vs. baseline SpecAugment.
        # Masked area scales as time_masks * time_width (roughly, for non-saturating regions),
        # so each factor is scaled by sqrt(volume_factor) — scaling both by the full factor
        # would compound to factor^2 area and destroy most of the spectrogram.
        if self.kd_mode == "cr_ctc":
            self.cr_ctc_weight = float(cfg.get("cr_ctc_weight", 0.2))
            # icefall default: linearly ramp cr_loss weight over 2000 batches
            self.cr_ctc_warm_step = int(cfg.get("cr_ctc_warm_step", 2000))
            volume_factor = float(cfg.get("cr_ctc_time_mask_factor", 2.5))
            dim_factor = volume_factor ** 0.5
            if getattr(self._cfg, "spec_augment", None) is None:
                raise ValueError("kd_mode='cr_ctc' requires model.spec_augment to be configured")
            cr_spec_cfg = copy.deepcopy(self._cfg.spec_augment)
            cr_spec_cfg["time_masks"] = int(round(int(cr_spec_cfg.get("time_masks", 10)) * dim_factor))
            base_time_width = cr_spec_cfg.get("time_width", 0.05)
            new_time_width = float(base_time_width) * dim_factor
            if isinstance(base_time_width, float):
                # SpecAugment requires float time_width in [0, 1] (fraction of sequence length)
                new_time_width = min(new_time_width, 1.0)
            cr_spec_cfg["time_width"] = new_time_width
            self.spec_augmentation_cr = EncDecCTCModelBPE.from_config_dict(cr_spec_cfg)

        self._wer_accum = {}

    def _capture_skd_hidden(self, module, inp, out):
        # ConformerLayer returns x tensor (B, T, d_model) when not streaming
        self._skd_inter_hidden = out if not isinstance(out, tuple) else out[0]

    def _kd_mode(self):
        return getattr(self, "kd_mode", self._cfg.get("kd_mode", "none"))

    def _make_dataloader_from_cfg(
        self,
        cfg,
        shuffle=False,
        require_transition=False,
        require_frame_kd=False,
        require_token_avg_kd=False,
        require_span_kd=False,
        require_combined_kd=False,
    ):
        return make_dataloader(
            cfg.manifest_filepath,
            self.tokenizer,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            sample_rate=cfg.sample_rate,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=cfg.get("pin_memory", True),
            require_transition=require_transition,
            require_frame_kd=require_frame_kd,
            require_token_avg_kd=require_token_avg_kd,
            require_span_kd=require_span_kd,
            require_combined_kd=require_combined_kd,
        )

    def setup_training_data(self, cfg):
        kd_mode = self._kd_mode()
        self._train_dl = self._make_dataloader_from_cfg(
            cfg,
            shuffle=cfg.shuffle,
            require_transition=(kd_mode == "trans"),
            require_frame_kd=(kd_mode in _NEEDS_FRAME_KD),
            require_token_avg_kd=(kd_mode in _NEEDS_TOKEN_AVG),
            require_span_kd=(kd_mode in _NEEDS_SPAN_KD),
            require_combined_kd=(kd_mode == "combined"),
        )

    def setup_validation_data(self, cfg):
        self._validation_dl = self._make_dataloader_from_cfg(cfg, shuffle=False)

    def setup_test_data(self, cfg):
        if cfg is None:
            return
        test_cfgs = [cfg] if "manifest_filepath" in cfg else list(cfg)
        self.test_names = [c.get("name", f"split_{i}") for i, c in enumerate(test_cfgs)]
        self._test_names = self.test_names
        self.test_wer_metrics = torch.nn.ModuleList([copy.deepcopy(self.wer) for _ in test_cfgs])
        test_dls = [self._make_dataloader_from_cfg(c, shuffle=False) for c in test_cfgs]
        self._test_dl = test_dls[0] if len(test_dls) == 1 else test_dls

    # ------------------------------------------------------------------
    # Blank frame selection mask (shared by logit, guided, delayed_logit)
    # ------------------------------------------------------------------
    def _blank_active_mask(self, ids, probs, valid):
        """Return active frame mask based on logit_kd_blank_mode.

        ids   : (B, T, top_k) - teacher top-k token ids (student frame space)
        probs : (B, T, top_k) - teacher top-k probs
        valid : (B, T) - length-based validity mask
        """
        mode = self.logit_kd_blank_mode
        if mode == "none":
            return valid

        non_blank = (ids[:, :, 0] != self.blank_id)  # (B, T): teacher top-1 != blank

        if mode == "elimination":
            return valid & non_blank

        if mode == "symmetric":
            k = 2 * self.logit_kd_blank_n + 1
            expanded = F.max_pool1d(
                non_blank.float().unsqueeze(1), kernel_size=k, stride=1, padding=self.logit_kd_blank_n
            ).squeeze(1).bool()
            return valid & expanded

        if mode == "trim":
            # Keep frames between first and last non-blank (inclusive)
            nb_in_valid = non_blank & valid
            from_left = nb_in_valid.float().cummax(dim=1).values.bool()
            from_right = nb_in_valid.float().flip(1).cummax(dim=1).values.flip(1).bool()
            return valid & from_left & from_right

        if mode == "threshold":
            # Include if top-1 != blank, OR if blank is top-1 but its prob < threshold
            is_blank_top1 = (ids[:, :, 0] == self.blank_id)
            blank_prob = torch.where(is_blank_top1, probs[:, :, 0], torch.zeros_like(probs[:, :, 0]))
            below_thresh = (~is_blank_top1) | (blank_prob < self.logit_kd_blank_threshold)
            return valid & below_thresh

        if mode == "random":
            blank_frames = valid & (~non_blank)
            random_gate = torch.bernoulli(
                torch.full_like(blank_frames.float(), self.logit_kd_blank_beta)
            ).bool()
            return valid & (non_blank | (blank_frames & random_gate))

        raise ValueError(f"unknown logit_kd_blank_mode: {mode}")

    # ------------------------------------------------------------------
    # KL decomposition at d=1 blank frames
    #   KL(p_T‖p_S) = KL([b_T,1−b_T]‖[b_S,1−b_S]) + (1−b_T)·KL(q_T‖q_S)
    #   L_occ : binary blank/non-blank KL          (occupancy term)
    #   L_res : blank-normalised conditional KL    (residual term)
    # Applied only at d=1 blank frames (adjacent to teacher non-blank spike,
    # but teacher top-1 == blank).
    # ------------------------------------------------------------------
    def _frame_occ_res_loss(self, log_probs, enc_len, fkd_ids, fkd_probs, fkd_lens):
        batch, frames, _ = log_probs.shape
        teacher_frames = fkd_ids.shape[1]
        topk = fkd_ids.shape[2]
        T = self.logit_kd_temperature
        eps = 1e-7

        # Length alignment (same approach as _frame_logit_kd_loss)
        if teacher_frames == frames:
            ids   = fkd_ids.to(log_probs.device)
            probs = fkd_probs.to(log_probs.device, dtype=log_probs.dtype)
            valid_len = torch.minimum(enc_len, fkd_lens.to(log_probs.device))
            valid = (torch.arange(frames, device=log_probs.device).view(1, frames)
                     < valid_len.view(batch, 1))
        else:
            t_pos = (torch.arange(frames, device=log_probs.device, dtype=torch.float32) + 0.5).view(1, frames)
            src_len = enc_len.to(log_probs.device).float().view(batch, 1).clamp_min(1)
            idx = torch.floor(t_pos / src_len * fkd_lens.to(log_probs.device).float().view(batch, 1)).long()
            idx = torch.minimum(idx, (fkd_lens.to(log_probs.device) - 1).view(batch, 1)).clamp_min(0)
            gather_idx = idx.view(batch, frames, 1).expand(batch, frames, topk)
            ids   = torch.gather(fkd_ids.to(log_probs.device), 1, gather_idx)
            probs = torch.gather(fkd_probs.to(log_probs.device, dtype=log_probs.dtype), 1, gather_idx)
            valid = (torch.arange(frames, device=log_probs.device).view(1, frames)
                     < enc_len.view(batch, 1))

        # d=1 blank mask: symmetric n=1 expansion of non-blank, then keep only blank frames
        non_blank = (ids[:, :, 0] != self.blank_id)
        expanded = F.max_pool1d(
            non_blank.float().unsqueeze(1), kernel_size=3, stride=1, padding=1
        ).squeeze(1).bool()
        d1_blank = valid & expanded & ~non_blank  # (B, T)

        # Temperature-scaled student log-probs
        student_lp = torch.log_softmax(log_probs / T, dim=-1)  # (B, T, V)

        # Teacher blank prob (top-1 is blank for all d=1 blank frames)
        b_T = probs[:, :, 0].clamp(eps, 1 - eps)              # (B, T)
        # Student blank prob (temperature-scaled)
        b_S = student_lp[:, :, self.blank_id].exp().clamp(eps, 1 - eps)  # (B, T)

        # --- L_occ : binary KL([b_T, 1-b_T] ‖ [b_S, 1-b_S]) ---
        occ = (b_T * (b_T.log() - b_S.log()) +
               (1 - b_T) * ((1 - b_T).log() - (1 - b_S).log()))  # (B, T)

        # --- L_res : KL(q_T ‖ q_S) over non-blank tokens in top-k ---
        is_nb = (ids != self.blank_id)                         # (B, T, topk)

        # q_T(k) = p_T(k) / (1-b_T), renormalised within top-k
        one_minus_b_T = (1 - b_T).unsqueeze(-1).clamp(min=eps)
        q_T = probs * is_nb.float() / one_minus_b_T           # (B, T, topk)
        q_T = q_T / q_T.sum(dim=-1, keepdim=True).clamp_min(eps)

        # log q_S(k) = student_lp[k] - log(1-b_S)
        log_one_minus_b_S = torch.log1p(-b_S.clamp(max=1 - eps)).unsqueeze(-1)
        selected_lp = torch.gather(student_lp, 2, ids)        # (B, T, topk)
        log_q_S = selected_lp - log_one_minus_b_S             # (B, T, topk)

        res = (q_T * (q_T.clamp(min=eps).log() - log_q_S) * is_nb.float()).sum(dim=-1)  # (B, T)

        mask_f   = d1_blank.float()
        n_active = mask_f.sum().clamp_min(1)
        return (occ * mask_f).sum() / n_active, (res * mask_f).sum() / n_active

    # ------------------------------------------------------------------
    # Logit KD loss (Hilmes et al. 2025 family)
    # ------------------------------------------------------------------
    def _frame_logit_kd_loss(self, log_probs, enc_len, fkd_ids, fkd_probs, fkd_lens):
        batch, frames, _ = log_probs.shape
        teacher_frames = fkd_ids.shape[1]
        topk = fkd_ids.shape[2]
        student_lp = torch.log_softmax(log_probs / self.logit_kd_temperature, dim=-1)

        if teacher_frames == frames:
            ids = fkd_ids.to(log_probs.device)
            probs = fkd_probs.to(log_probs.device, dtype=log_probs.dtype)
            valid_len = torch.minimum(enc_len, fkd_lens.to(log_probs.device))
            valid = torch.arange(frames, device=log_probs.device).view(1, frames) < valid_len.view(batch, 1)
        else:
            t_pos = (torch.arange(frames, device=log_probs.device, dtype=torch.float32) + 0.5).view(1, frames)
            src_len = enc_len.to(log_probs.device).float().view(batch, 1).clamp_min(1)
            idx = torch.floor(t_pos / src_len * fkd_lens.to(log_probs.device).float().view(batch, 1)).long()
            idx = torch.minimum(idx, (fkd_lens.to(log_probs.device) - 1).view(batch, 1)).clamp_min(0)
            gather_idx = idx.view(batch, frames, 1).expand(batch, frames, topk)
            ids = torch.gather(fkd_ids.to(log_probs.device), 1, gather_idx)
            probs = torch.gather(fkd_probs.to(log_probs.device, dtype=log_probs.dtype), 1, gather_idx)
            valid = torch.arange(frames, device=log_probs.device).view(1, frames) < enc_len.view(batch, 1)

        selected_lp = torch.gather(student_lp, 2, ids)
        ce = -(probs * selected_lp).sum(dim=-1)  # (B, T)

        active = self._blank_active_mask(ids, probs, valid)
        return (ce * active.to(ce.dtype)).sum() / active.float().sum().clamp_min(1)

    # ------------------------------------------------------------------
    # Guided KD loss (Kurata & Audhkhasi 2019)
    # Hard cross-entropy at teacher non-blank frames only.
    # Different from KD-BE (soft top-k KL) — uses only teacher argmax (hard label).
    # ------------------------------------------------------------------
    def _guided_kd_loss(self, log_probs, enc_len, fkd_ids, fkd_probs, fkd_lens):
        batch, frames, K = log_probs.shape
        teacher_frames = fkd_ids.shape[1]
        topk = fkd_ids.shape[2]
        student_lp = torch.log_softmax(log_probs, dim=-1)  # no temperature for hard labels

        if teacher_frames == frames:
            ids = fkd_ids.to(log_probs.device)
            probs = fkd_probs.to(log_probs.device, dtype=log_probs.dtype)
            valid_len = torch.minimum(enc_len, fkd_lens.to(log_probs.device))
            valid = torch.arange(frames, device=log_probs.device).view(1, frames) < valid_len.view(batch, 1)
        else:
            t_pos = (torch.arange(frames, device=log_probs.device, dtype=torch.float32) + 0.5).view(1, frames)
            src_len = enc_len.float().view(batch, 1).clamp_min(1)
            idx = torch.floor(t_pos / src_len * fkd_lens.to(log_probs.device).float().view(batch, 1)).long()
            idx = torch.minimum(idx, (fkd_lens.to(log_probs.device) - 1).view(batch, 1)).clamp_min(0)
            gather_idx = idx.view(batch, frames, 1).expand(batch, frames, topk)
            ids = torch.gather(fkd_ids.to(log_probs.device), 1, gather_idx)
            probs = torch.gather(fkd_probs.to(log_probs.device, dtype=log_probs.dtype), 1, gather_idx)
            valid = torch.arange(frames, device=log_probs.device).view(1, frames) < enc_len.view(batch, 1)

        # Hard CE: only use teacher argmax (top-1) token as label
        teacher_argmax = ids[:, :, 0].clamp(0, K - 1)  # (B, T)
        hard_ce = -torch.gather(student_lp, 2, teacher_argmax.unsqueeze(2)).squeeze(2)  # (B, T)

        # Non-blank mask: only supervise at teacher non-blank frames
        non_blank = (ids[:, :, 0] != self.blank_id)
        active = valid & non_blank
        return (hard_ce * active.to(hard_ce.dtype)).sum() / active.float().sum().clamp_min(1)

    # ------------------------------------------------------------------
    # Delayed logit KD loss (Li et al. 2025, TAB adapted for non-streaming)
    # For each teacher frame, search student frames in [s_t, s_t+tab_size] and pick min KL.
    # ------------------------------------------------------------------
    def _delayed_logit_kd_loss(self, log_probs, enc_len, fkd_ids, fkd_probs, fkd_lens):
        B, T_s, K = log_probs.shape
        T_t = fkd_ids.shape[1]
        top_k = fkd_ids.shape[2]
        student_lp = torch.log_softmax(log_probs / self.logit_kd_temperature, dim=-1)  # (B, T_s, K)

        ids = fkd_ids.to(log_probs.device)       # (B, T_t, top_k)
        probs = fkd_probs.to(log_probs.device, dtype=log_probs.dtype)
        fkd_len_dev = fkd_lens.to(log_probs.device)

        # Teacher frame t → base student frame: s = floor((t+0.5) * T_s / T_t)
        t_pos = (torch.arange(T_t, device=log_probs.device, dtype=torch.float32) + 0.5).view(1, T_t)
        base_idx = torch.floor(
            t_pos / fkd_len_dev.float().view(B, 1).clamp_min(1) * enc_len.float().view(B, 1)
        ).long().clamp(0, T_s - 1)  # (B, T_t)

        # Compute CE for each TAB offset τ ∈ {0, …, tab_size}
        kl_per_tau = []
        for tau in range(self.logit_kd_tab_size + 1):
            idx = (base_idx + tau).clamp(0, T_s - 1)  # (B, T_t)
            expand_idx = idx.unsqueeze(2).expand(B, T_t, top_k)
            # Gather student log_probs at shifted student frame
            sel_lp = torch.gather(student_lp, 1, expand_idx)  # (B, T_t, top_k)
            # Gather at teacher top-k token positions
            sel_lp = torch.gather(sel_lp, 2, ids)             # (B, T_t, top_k)
            ce = -(probs * sel_lp).sum(dim=-1)                # (B, T_t)
            kl_per_tau.append(ce)

        # Pick student frame with minimum KL divergence per teacher frame
        min_kl, _ = torch.stack(kl_per_tau, dim=-1).min(dim=-1)  # (B, T_t)

        # Validity: teacher frame within fkd_lens, teacher idx within enc_len range
        t_idx = torch.arange(T_t, device=log_probs.device).view(1, T_t)
        valid = (t_idx < fkd_len_dev.view(B, 1)) & (base_idx < enc_len.view(B, 1))

        # Apply blank masking using teacher-frame-space ids/probs
        active = self._blank_active_mask(ids, probs, valid)
        return (min_kl * active.float()).sum() / active.float().sum().clamp_min(1)

    # ------------------------------------------------------------------
    # Token-avg KD loss
    # ------------------------------------------------------------------
    def _token_avg_kd_loss(self, log_probs, enc_len, seg_starts, seg_ends,
                           avg_ids, avg_probs, num_segs, teacher_frames):
        B, T_max, K = log_probs.shape
        N_max = seg_starts.shape[1]
        device = log_probs.device
        dtype = log_probs.dtype

        scale = enc_len.float() / teacher_frames.float().to(device)
        s_starts = (seg_starts.float() * scale.unsqueeze(1)).long()
        s_ends   = (seg_ends.float()   * scale.unsqueeze(1)).long()
        s_ends   = torch.max(s_ends, s_starts + 1)
        s_ends   = torch.min(s_ends, enc_len.view(B, 1).expand_as(s_ends))

        seg_valid = (torch.arange(N_max, device=device).unsqueeze(0) < num_segs.unsqueeze(1)) \
                    & (s_starts < enc_len.view(B, 1))

        t_idx    = torch.arange(T_max, device=device).view(1, 1, T_max)
        seg_mask = (t_idx >= s_starts.unsqueeze(2)) & (t_idx < s_ends.unsqueeze(2))
        seg_mask = seg_mask & seg_valid.unsqueeze(2)

        seg_len = seg_mask.float().sum(dim=2).clamp_min(1.0)

        student_prob = log_probs.exp().to(dtype)
        seg_prob_sum = torch.bmm(seg_mask.to(dtype), student_prob)
        student_seg_avg = seg_prob_sum / seg_len.unsqueeze(2)

        top_k = avg_ids.shape[2]
        ids_e = avg_ids.to(device).clamp(0, K - 1)
        s_selected = torch.gather(student_seg_avg, 2, ids_e).clamp_min(1e-9)

        ce = -(avg_probs.to(device, dtype=dtype) * s_selected.log()).sum(dim=2)
        ce = ce * seg_valid.to(dtype)
        return ce.sum() / seg_valid.float().sum().clamp_min(1.0)

    # ------------------------------------------------------------------
    # Aligned-token KD loss (ours): dual Viterbi segment matching
    # Teacher segment k ↔ Student forced-alignment segment k (1:1 by token order).
    # ------------------------------------------------------------------
    def _aligned_token_kd_loss(self, log_probs, enc_len,
                                t_seg_starts, t_seg_ends,
                                avg_ids, avg_probs, num_segs_t, teacher_frames,
                                tokens, token_lens):
        B, T_s, K = log_probs.shape
        device = log_probs.device
        dtype = log_probs.dtype

        L_max = int(token_lens.max().item())
        tok = tokens[:, :L_max].clamp(0, K - 1)  # (B, L_max)

        # Student Viterbi forced alignment (no grad)
        s_starts, s_ends = _ctc_viterbi_align(
            log_probs.detach(), tok, enc_len, token_lens, self.blank_id
        )

        # Match teacher segment k with student segment k (by token order)
        N_match = min(t_seg_starts.shape[1], L_max)
        seg_valid = (
            (torch.arange(N_match, device=device).unsqueeze(0) < num_segs_t.unsqueeze(1))
            & (torch.arange(N_match, device=device).unsqueeze(0) < token_lens.unsqueeze(1))
        )  # (B, N_match)

        # Build frame→segment mask using student segment boundaries
        t_idx = torch.arange(T_s, device=device).view(1, 1, T_s)
        s_starts_m = s_starts[:, :N_match]  # (B, N_match)
        s_ends_m   = s_ends[:, :N_match]    # (B, N_match)
        seg_mask = (t_idx >= s_starts_m.unsqueeze(2)) & (t_idx < s_ends_m.unsqueeze(2))
        seg_mask = seg_mask & seg_valid.unsqueeze(2)  # (B, N_match, T_s)

        seg_len = seg_mask.float().sum(dim=2).clamp_min(1.0)
        student_prob = log_probs.exp().to(dtype)
        seg_prob_sum = torch.bmm(seg_mask.to(dtype), student_prob)
        student_seg_avg = seg_prob_sum / seg_len.unsqueeze(2)

        top_k = avg_ids.shape[2]
        ids_e = avg_ids[:, :N_match].to(device).clamp(0, K - 1)
        s_selected = torch.gather(student_seg_avg, 2, ids_e).clamp_min(1e-9)

        ce = -(avg_probs[:, :N_match].to(device, dtype=dtype) * s_selected.log()).sum(dim=2)
        ce = ce * seg_valid.to(dtype)
        return ce.sum() / seg_valid.float().sum().clamp_min(1.0)


    # ------------------------------------------------------------------
    # Span KD loss: teacher semantic q_T,u vs student evidence aggregated
    # over phoneme-derived temporal support a(t,u). Teacher/student are not
    # compared at the same frame.
    # ------------------------------------------------------------------
    def _span_kd_loss(self, log_probs, enc_len, support, avg_ids, avg_probs,
                      gates, num_tokens, teacher_frames):
        B, T_s, K = log_probs.shape
        device = log_probs.device
        dtype = log_probs.dtype
        N_max = support.shape[1]
        top_k = avg_ids.shape[2]
        student_prob = log_probs.exp().to(dtype)
        total = torch.zeros((), device=device, dtype=dtype)
        total_w = torch.zeros((), device=device, dtype=dtype)

        for b in range(B):
            n = int(num_tokens[b].item())
            ts = int(enc_len[b].item())
            tt = int(teacher_frames[b].item())
            if n <= 0 or ts <= 0 or tt <= 0:
                continue
            n = min(n, N_max)
            ts = min(ts, T_s)
            tt = min(tt, support.shape[2])

            sup_t = support[b, :n, :tt].to(device=device, dtype=dtype)  # (N, T_t)
            # Student frame j -> nearest teacher support frame.
            pos = (torch.arange(ts, device=device, dtype=torch.float32) + 0.5) / max(ts, 1) * tt
            idx = torch.floor(pos).long().clamp(0, tt - 1)
            sup_s = sup_t[:, idx]  # (N, T_s_valid)
            sup_s = sup_s / sup_s.sum(dim=1, keepdim=True).clamp_min(1e-8)

            s_avg = torch.matmul(sup_s, student_prob[b, :ts])  # (N, K)
            ids = avg_ids[b, :n].to(device).clamp(0, K - 1)
            probs = avg_probs[b, :n].to(device=device, dtype=dtype)
            selected = torch.gather(s_avg.clamp_min(1e-9), 1, ids)
            ce = -(probs * selected.log()).sum(dim=1)  # (N,)

            w = gates[b, :n].to(device=device, dtype=dtype)
            active = w >= self.span_kd_gate_threshold
            if active.any():
                w_active = w[active]
                total = total + (ce[active] * w_active).sum()
                total_w = total_w + w_active.sum()

        return total / total_w.clamp_min(1.0)

    # ------------------------------------------------------------------
    # Transition frame KD loss (ours): blank frames flanking non-blank segments
    # For each token segment, match teacher's blank-frame distribution just
    # before (left) and after (right) with the corresponding student frame.
    # ------------------------------------------------------------------
    def _transition_frame_kd_loss(self, log_probs, enc_len, num_segs, teacher_frames,
                                   seg_starts, seg_ends,
                                   trans_left_ids, trans_left_probs, trans_left_valid,
                                   trans_right_ids, trans_right_probs, trans_right_valid):
        B, T_max, K = log_probs.shape
        N_max = seg_starts.shape[1]
        device = log_probs.device
        dtype = log_probs.dtype

        scale = enc_len.float() / teacher_frames.float().to(device)  # (B,)
        seg_valid = (torch.arange(N_max, device=device).unsqueeze(0) < num_segs.unsqueeze(1))

        # Map teacher transition frame indices to student frame space
        s_left  = ((seg_starts.float() - 1) * scale.unsqueeze(1)).long().clamp(0, T_max - 1)
        s_right = (seg_ends.float()          * scale.unsqueeze(1)).long().clamp(0, T_max - 1)

        left_valid  = seg_valid & trans_left_valid.to(device)
        right_valid = seg_valid & trans_right_valid.to(device)

        student_lp = torch.log_softmax(log_probs, dim=-1)  # (B, T_max, K)

        def _frame_ce(s_frames, t_ids, t_probs, valid_mask):
            top_k = t_ids.shape[2]
            ids = t_ids.to(device).clamp(0, K - 1)               # (B, N_max, top_k)
            frame_idx = s_frames.unsqueeze(2).expand(B, N_max, top_k)  # (B, N_max, top_k)
            b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N_max, top_k)
            s_sel = student_lp[b_idx, frame_idx, ids].clamp(min=-30.0)  # (B, N_max, top_k)
            tp = t_probs.to(device, dtype=dtype).clamp_min(1e-9)
            return -(tp * s_sel).sum(dim=2) * valid_mask.float()

        ce_left  = _frame_ce(s_left,  trans_left_ids,  trans_left_probs,  left_valid)
        ce_right = _frame_ce(s_right, trans_right_ids, trans_right_probs, right_valid)
        n_valid = left_valid.float().sum() + right_valid.float().sum()
        return (ce_left.sum() + ce_right.sum()) / n_valid.clamp_min(1.0)

    # ------------------------------------------------------------------
    # Self-KD loss (Kim et al. 2024)
    # Returns (inter_log_probs, loss_skd) — CTC on inter is computed in training_step.
    # ------------------------------------------------------------------
    def _self_kd_loss(self, log_probs, enc_len, inter_hidden):
        """
        inter_hidden: (B, T, d_model) captured by hook at encoder layer skd_split_layer.
        log_probs:    (B, T, vocab) from full encoder — treated as self-teacher (detached).
        Returns skd_loss and inter_log_probs for use in CTC loss.
        """
        B, T, _ = inter_hidden.shape
        inter_logits = self.skd_inter_head(inter_hidden)             # (B, T, vocab)
        inter_log_probs = torch.log_softmax(inter_logits, dim=-1)    # (B, T, vocab)

        # Soft targets from full model (detach = no gradient to full encoder from SKD path)
        temp = self.logit_kd_temperature
        soft_teacher = torch.softmax(log_probs.detach() / temp, dim=-1)  # (B, T, vocab)
        inter_lp_temp = torch.log_softmax(inter_logits / temp, dim=-1)
        skd_per_frame = -(soft_teacher * inter_lp_temp).sum(dim=-1)     # (B, T)

        valid = torch.arange(T, device=log_probs.device).view(1, T) < enc_len.view(B, 1)
        loss_skd = (skd_per_frame * valid.float()).sum() / valid.float().sum().clamp_min(1)
        return inter_log_probs, loss_skd

    # ------------------------------------------------------------------
    # CR-CTC (Yao et al. 2025): dual-view consistency regularization
    # ------------------------------------------------------------------
    def _cr_ctc_forward_view(self, input_signal, input_signal_length):
        """One augmented view: calls the real self.forward() (same code path as every
        other kd_mode / eval), with self.spec_augmentation swapped for the strengthened
        self.spec_augmentation_cr for the duration of the call."""
        original_aug = self.spec_augmentation
        self.spec_augmentation = self.spec_augmentation_cr
        try:
            log_probs, encoded_len, _ = self.forward(
                input_signal=input_signal, input_signal_length=input_signal_length
            )
        finally:
            self.spec_augmentation = original_aug
        return log_probs, encoded_len

    def _cr_ctc_consistency_loss(self, log_probs_a, log_probs_b, enc_len):
        """Symmetric per-frame KL with stop-gradient on the target side of each term
        (paper Eq. 4): L_CR = 1/2 * sum_t [ KL(sg(p_b) || p_a) + KL(sg(p_a) || p_b) ].
        This is a raw SUM over frames per utterance (not divided by T), then averaged
        over the batch — matching the paper exactly, not a per-frame mean.
        """
        B, T, _ = log_probs_a.shape
        kl_ab = F.kl_div(log_probs_a, log_probs_b.detach(), log_target=True, reduction="none").sum(-1)
        kl_ba = F.kl_div(log_probs_b, log_probs_a.detach(), log_target=True, reduction="none").sum(-1)
        per_frame = 0.5 * (kl_ab + kl_ba)  # (B, T)
        valid = torch.arange(T, device=log_probs_a.device).view(1, T) < enc_len.view(B, 1)
        per_utt = (per_frame * valid.float()).sum(dim=1)  # (B,) sum over T, per paper Eq. 4
        return per_utt.mean()  # mean over batch

    def _cr_ctc_training_step(self, batch):
        # Matches the official icefall/Zipformer CR-CTC recipe: dual-view + strengthened
        # SpecAugment is active from step 0, but the consistency loss WEIGHT is ramped
        # linearly from 0 to cr_ctc_weight over cr_ctc_warm_step steps. Without this ramp
        # the model can trivially satisfy both CTC and consistency loss by collapsing to
        # all-blank early on, before it has learned any real acoustic-to-token alignment.
        input_signal = batch["wavs"]
        input_signal_length = batch["wav_lens"]
        log_probs_a, enc_len_a = self._cr_ctc_forward_view(input_signal, input_signal_length)
        log_probs_b, enc_len_b = self._cr_ctc_forward_view(input_signal, input_signal_length)

        loss_ctc_a = self.loss(
            log_probs=log_probs_a, targets=batch["tokens"],
            input_lengths=enc_len_a, target_lengths=batch["token_lens"],
        )
        loss_ctc_b = self.loss(
            log_probs=log_probs_b, targets=batch["tokens"],
            input_lengths=enc_len_b, target_lengths=batch["token_lens"],
        )
        loss_ctc = 0.5 * (loss_ctc_a + loss_ctc_b)
        loss_cr = self._cr_ctc_consistency_loss(log_probs_a, log_probs_b, enc_len_a)
        cr_scale = min(float(self.global_step) / max(self.cr_ctc_warm_step, 1), 1.0) * self.cr_ctc_weight
        loss = loss_ctc + cr_scale * loss_cr

        self.log("train/loss",          loss,     prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/ctc_loss",      loss_ctc, on_step=True, on_epoch=True)
        self.log("train/cr_loss",       loss_cr,  prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/cr_loss_scale", cr_scale, on_step=True, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _mix(self, loss_ctc, loss_kd):
        """Combine CTC and KD losses.
        kd_lambda set → paper formula: (1-λ)*L_CTC + λ*L_KD  (λ=1.0 = pure KD)
        kd_lambda None → legacy:        L_CTC + kd_weight*L_KD
        """
        if self.kd_lambda is not None:
            return (1.0 - self.kd_lambda) * loss_ctc + self.kd_lambda * loss_kd
        return loss_ctc + self.kd_weight * loss_kd

    def training_step(self, batch, batch_idx):
        if self.kd_mode == "cr_ctc":
            return self._cr_ctc_training_step(batch)

        log_probs, enc_len, _ = self.forward(
            input_signal=batch["wavs"], input_signal_length=batch["wav_lens"]
        )
        loss_ctc = self.loss(
            log_probs=log_probs,
            targets=batch["tokens"],
            input_lengths=enc_len,
            target_lengths=batch["token_lens"],
        )

        zero = torch.zeros((), device=log_probs.device, dtype=loss_ctc.dtype)

        if self.kd_mode == "none":
            loss = loss_ctc

        elif self.kd_mode == "trans":
            loss_kd = transition_kd_loss_batched(
                log_probs,
                batch["ttargets"],
                input_lengths=enc_len,
                target_lengths=batch["ttarget_lens"],
            ).mean()
            loss = self._mix(loss_ctc, loss_kd)

        elif self.kd_mode == "logit":
            loss_kd = self._frame_logit_kd_loss(
                log_probs, enc_len,
                batch["fkd_ids"], batch["fkd_probs"], batch["fkd_lens"],
            )
            loss = self._mix(loss_ctc, loss_kd)
            if self.logit_kd_occ_weight > 0.0 or self.logit_kd_res_weight > 0.0:
                loss_occ, loss_res = self._frame_occ_res_loss(
                    log_probs, enc_len,
                    batch["fkd_ids"], batch["fkd_probs"], batch["fkd_lens"],
                )
                if self.logit_kd_occ_weight > 0.0:
                    loss = loss + self.logit_kd_occ_weight * loss_occ
                    self.log("train/occ_loss", loss_occ, on_step=True, on_epoch=True)
                if self.logit_kd_res_weight > 0.0:
                    loss = loss + self.logit_kd_res_weight * loss_res
                    self.log("train/res_loss", loss_res, on_step=True, on_epoch=True)

        elif self.kd_mode == "guided":
            loss_kd = self._guided_kd_loss(
                log_probs, enc_len,
                batch["fkd_ids"], batch["fkd_probs"], batch["fkd_lens"],
            )
            loss = self._mix(loss_ctc, loss_kd)

        elif self.kd_mode == "delayed_logit":
            loss_kd = self._delayed_logit_kd_loss(
                log_probs, enc_len,
                batch["fkd_ids"], batch["fkd_probs"], batch["fkd_lens"],
            )
            loss = self._mix(loss_ctc, loss_kd)

        elif self.kd_mode == "token_avg":
            loss_kd = self._token_avg_kd_loss(
                log_probs, enc_len,
                batch["tavg_seg_starts"], batch["tavg_seg_ends"],
                batch["tavg_avg_ids"], batch["tavg_avg_probs"],
                batch["tavg_num_segs"], batch["tavg_teacher_frames"],
            )
            loss_trans = zero
            if self.transition_kd_weight > 0.0 and "tavg_trans_left_ids" in batch:
                loss_trans = self._transition_frame_kd_loss(
                    log_probs, enc_len,
                    batch["tavg_num_segs"], batch["tavg_teacher_frames"],
                    batch["tavg_seg_starts"], batch["tavg_seg_ends"],
                    batch["tavg_trans_left_ids"],  batch["tavg_trans_left_probs"],  batch["tavg_trans_left_valid"],
                    batch["tavg_trans_right_ids"], batch["tavg_trans_right_probs"], batch["tavg_trans_right_valid"],
                )
                self.log("train/trans_kd_loss", loss_trans, on_step=True, on_epoch=True)
            loss = loss_ctc + self.kd_weight * loss_kd + self.transition_kd_weight * loss_trans

        elif self.kd_mode == "aligned_token":
            loss_kd = self._aligned_token_kd_loss(
                log_probs, enc_len,
                batch["tavg_seg_starts"], batch["tavg_seg_ends"],
                batch["tavg_avg_ids"], batch["tavg_avg_probs"],
                batch["tavg_num_segs"], batch["tavg_teacher_frames"],
                batch["tokens"], batch["token_lens"],
            )
            loss = loss_ctc + self.kd_weight * loss_kd

        elif self.kd_mode == "span_kd":
            loss_kd = self._span_kd_loss(
                log_probs, enc_len,
                batch["span_support"], batch["span_avg_ids"], batch["span_avg_probs"],
                batch["span_gates"], batch["span_num_tokens"], batch["span_teacher_frames"],
            )
            loss = self._mix(loss_ctc, loss_kd)

        elif self.kd_mode == "combined":
            loss_logit = self._frame_logit_kd_loss(
                log_probs, enc_len,
                batch["fkd_ids"], batch["fkd_probs"], batch["fkd_lens"],
            ) if self.kd_weight > 0.0 else zero
            loss_tavg = self._token_avg_kd_loss(
                log_probs, enc_len,
                batch["tavg_seg_starts"], batch["tavg_seg_ends"],
                batch["tavg_avg_ids"], batch["tavg_avg_probs"],
                batch["tavg_num_segs"], batch["tavg_teacher_frames"],
            ) if self.token_avg_kd_weight > 0.0 else zero
            loss = loss_ctc + self.kd_weight * loss_logit + self.token_avg_kd_weight * loss_tavg
            self.log("train/logit_kd_loss", loss_logit, on_step=True, on_epoch=True)
            self.log("train/tavg_kd_loss",  loss_tavg,  on_step=True, on_epoch=True)

        elif self.kd_mode == "self_kd":
            # Hook captured inter_hidden during self.forward() above
            inter_hidden = self._skd_inter_hidden  # (B, T, d_model)
            inter_log_probs, loss_skd = self._self_kd_loss(log_probs, enc_len, inter_hidden)
            loss_ctc_inter = self.loss(
                log_probs=inter_log_probs,
                targets=batch["tokens"],
                input_lengths=enc_len,
                target_lengths=batch["token_lens"],
            )
            # (1-α)*CTC_full + α*(CTC_inter + SKD)
            loss = (1 - self.skd_alpha) * loss_ctc + self.skd_alpha * (loss_ctc_inter + loss_skd)
            self.log("train/ctc_inter_loss", loss_ctc_inter, on_step=True, on_epoch=True)
            self.log("train/skd_loss",       loss_skd,       on_step=True, on_epoch=True)

        # SR-CTC smooth regularization (Yao et al. 2025, Appendix A.1): pull the
        # student's per-frame distribution toward its own time-smoothed version,
        # suppressing over-confident peaks (vocab-axis softening). Auxiliary term
        # addable to any teacher-KD mode; needs no extra targets.
        if self.sr_ctc_weight > 0.0:
            loss_sr = self._sr_ctc_loss(log_probs, enc_len)
            loss = loss + self.sr_ctc_weight * loss_sr
            self.log("train/sr_loss", loss_sr, on_step=True, on_epoch=True)

        self.log("train/loss",     loss,      prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/ctc_loss", loss_ctc,  on_step=True, on_epoch=True)
        if self.kd_mode not in ("none", "combined", "self_kd"):
            self.log("train/kd_loss", loss_kd, prog_bar=True, on_step=True, on_epoch=True)  # type: ignore[possibly-undefined]
        return loss

    def _sr_ctc_loss(self, log_probs, enc_len):
        """SR-CTC (Yao et al. 2025, Eq. 5): L = sum_t KL( sg(smooth(z)) || z ),
        where smooth() is a depthwise time convolution with K=(0.25,0.5,0.25).
        Pulls each frame's distribution toward the average of its time-neighbors,
        so peaks are suppressed. Averaged over valid frames."""
        B, T, V = log_probs.shape
        z = log_probs.exp()                                   # (B, T, V) per-frame probs
        # time smoothing: z_s[t] = 0.25 z[t-1] + 0.5 z[t] + 0.25 z[t+1] (edge-replicated)
        left  = torch.cat([z[:, :1], z[:, :-1]], dim=1)
        right = torch.cat([z[:, 1:], z[:, -1:]], dim=1)
        z_s = (0.25 * left + 0.5 * z + 0.25 * right).detach()  # target, stop-gradient
        # KL(z_s || z) per frame = sum_v z_s (log z_s - log z)
        kl = (z_s * (z_s.clamp_min(1e-9).log() - log_probs)).sum(-1)  # (B, T)
        valid = torch.arange(T, device=log_probs.device).view(1, T) < enc_len.view(B, 1)
        return (kl * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _eval_step(self, batch, metric, loss_name):
        log_probs, enc_len, _ = self.forward(
            input_signal=batch["wavs"], input_signal_length=batch["wav_lens"]
        )
        loss = self.loss(
            log_probs=log_probs,
            targets=batch["tokens"],
            input_lengths=enc_len,
            target_lengths=batch["token_lens"],
        )
        metric.update(
            predictions=log_probs,
            predictions_lengths=enc_len,
            targets=batch["tokens"],
            targets_lengths=batch["token_lens"],
        )
        _, wer_num, wer_denom = metric.compute()
        metric.reset()
        acc = self._wer_accum.setdefault(id(metric), [0, 0])
        acc[0] += int(wer_num.item())
        acc[1] += int(wer_denom.item())
        self.log(loss_name, loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch["wavs"].size(0))
        return loss

    def _wer_from_accum(self, metric):
        acc = self._wer_accum.pop(id(metric), [0, 0])
        return acc[0] / max(acc[1], 1)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, self.wer, "val/loss")

    def on_validation_epoch_end(self):
        wer = self._wer_from_accum(self.wer)
        self.log("val/wer", wer, prog_bar=True, sync_dist=True)
        self.log("val_wer", wer, prog_bar=False, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = getattr(self, "test_wer_metrics", None)
        metric = metrics[dataloader_idx] if metrics is not None else self.wer
        test_names = getattr(self, "test_names", None) or ["test"]
        name = test_names[dataloader_idx] if dataloader_idx < len(test_names) else f"split_{dataloader_idx}"
        return self._eval_step(batch, metric, f"test/{name}_loss")

    def on_test_epoch_end(self):
        metrics = getattr(self, "test_wer_metrics", None)
        if metrics is None:
            wer = self._wer_from_accum(self.wer)
            self.log("test/wer", wer, sync_dist=True)
            return
        for name, metric in zip(self.test_names, metrics):
            wer = self._wer_from_accum(metric)
            self.log(f"test/{name}_wer", wer, sync_dist=True)
