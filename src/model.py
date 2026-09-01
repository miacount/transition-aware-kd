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

from ctc_fb import batched_ctc_token_gamma
from data import make_dataloader
from transition_kd_batched import transition_kd_loss_batched
from free_emit_kd import curriculum_margin, free_emit_loss

_VALID_KD_MODES = (
    "none", "trans", "logit", "token_avg", "combined",
    "guided", "delayed_logit", "self_kd", "aligned_token", "span_kd", "cr_ctc", "sctc",
    "boundary_kd", "free_emit", "frame_dkd", "fpkd_dfkd", "fpkd_frkd", "fpkd_pkd",
    "carl_feature", "carl",
)
_NEEDS_FRAME_KD   = (
    "logit", "combined", "guided", "delayed_logit", "frame_dkd",
    "fpkd_dfkd", "fpkd_frkd", "fpkd_pkd", "carl",
)
_NEEDS_TOKEN_AVG  = ("token_avg", "aligned_token")
_NEEDS_SPAN_KD    = ("span_kd",)
_NEEDS_SCTC       = ("sctc",)
_NEEDS_BOUNDARY_KD = ("boundary_kd",)
_NEEDS_FREE_EMIT = ("free_emit",)
_NEEDS_CARL_FEATURE = ("carl_feature", "carl", "fpkd_frkd")


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
        self.ctc_blank_init_bias = float(cfg.get("ctc_blank_init_bias", 0.0))
        self.kd_mode = cfg.get("kd_mode", "none")
        self.kd_weight = float(cfg.get("kd_weight", 0.0))
        self.token_avg_kd_weight = float(cfg.get("token_avg_kd_weight", 0.0))
        self.logit_kd_temperature = float(cfg.get("logit_kd_temperature", 2.0))
        self.logit_kd_blank_mode = cfg.get("logit_kd_blank_mode", "none")
        self.logit_kd_reduction = str(cfg.get("logit_kd_reduction", "frame_mean"))
        if self.logit_kd_reduction not in ("frame_mean", "utterance_sum"):
            raise ValueError("logit_kd_reduction must be frame_mean or utterance_sum")

        self.logit_kd_blank_n = int(cfg.get("logit_kd_blank_n", 1))
        self.logit_kd_blank_threshold = float(cfg.get("logit_kd_blank_threshold", 0.9))
        self.logit_kd_blank_beta = float(cfg.get("logit_kd_blank_beta", 0.5))
        self.logit_kd_tab_size = int(cfg.get("logit_kd_tab_size", 2))
        # cross-rate pooling for frame KD (logit mode): when teacher/student frame
        # rates differ, average the targets of ALL teacher frames that map to a
        # student frame (ratio 2 -> frames 2j & 2j+1) instead of nearest-picking
        # one — a teacher spike can no longer vanish on the dropped parity.
        self.logit_kd_cross_pool = bool(cfg.get("logit_kd_cross_pool", False))
        self.span_kd_gate_threshold = float(cfg.get("span_kd_gate_threshold", 0.3))
        # dual-occupancy span-KD: pool the student with its OWN transcript-constrained
        # de-peaked gamma (on-the-fly, detached) instead of the teacher's resampled one.
        self.span_kd_dual = bool(cfg.get("span_kd_dual", False))
        self.span_kd_student_delta = float(cfg.get("span_kd_student_delta", 6.0))
        # Content/emission decomposition of the (un-normalized) span CE:
        #   CE(q, s_u) = H(q) + KL(q || s_bar_u) + beta * (-log m_u)
        # where m_u = sum of student mass on the teacher top-k set, s_bar_u the
        # student conditional over that set. beta=1 reproduces the current loss
        # EXACTLY; beta=0 = content-only (dark knowledge, no emission pressure).
        self.span_kd_emit_beta = float(cfg.get("span_kd_emit_beta", 1.0))
        # Decoupled raw non-target dark knowledge. This is added outside the
        # legacy kd_weight-scaled Span-KD anchor so its scale is independently
        # controlled: L = L_CTC + kd_weight*L_span + weight*L_NTDK.
        self.span_ntdk_weight = float(cfg.get("span_ntdk_weight", 0.0))
        # Alignment-only branch. Mass3/NTDK stay anchored to the delta=0
        # occurrence core; this branch sees only [gamma_delta-gamma_0]+ and
        # asks for occurrence-GT emission there. Separate normalization keeps
        # a wider support from diluting the semantic core.
        self.span_shoulder_weight = float(cfg.get("span_shoulder_weight", 0.0))
        if self.span_shoulder_weight < 0.0:
            raise ValueError("span_shoulder_weight must be non-negative")
        self.span_ntdk_reliability = bool(cfg.get("span_ntdk_reliability", False))
        # Exact hierarchy: multiply conditional NTDK by absolute teacher NT mass
        # while retaining the ordinary token/gate denominator.
        self.span_ntdk_mass_weighted = bool(cfg.get("span_ntdk_mass_weighted", False))
        if self.span_ntdk_reliability and self.span_ntdk_mass_weighted:
            raise ValueError("span_ntdk_reliability and span_ntdk_mass_weighted are mutually exclusive")
        self.span_ntdk_rel_min = float(cfg.get("span_ntdk_rel_min", 1e-4))
        self.span_ntdk_rel_max = float(cfg.get("span_ntdk_rel_max", 5e-2))
        self.span_ntdk_rel_gt_min = float(cfg.get("span_ntdk_rel_gt_min", 0.5))
        self.span_ntdk_rel_power = float(cfg.get("span_ntdk_rel_power", 0.5))
        if self.span_ntdk_rel_min <= 0 or self.span_ntdk_rel_max < self.span_ntdk_rel_min:
            raise ValueError("invalid NTDK reliability mass bounds")
        # Primary span supervision paired with NTDK:
        #   legacy  : cached blank-removed/top-k Span-KD target (baseline)
        #   hard_gt : occupancy-pooled occurrence GT CE; no teacher class target
        #   mass3   : KL on exact pooled [blank, occurrence-GT, non-target] mass
        #   gt_nt   : KL on [GT, non-target] conditioned on nonblank; no explicit blank target
        # `kd_weight` scales whichever primary mode is selected.  NTDK remains
        # independently controlled by span_ntdk_weight.
        self.span_primary_mode = str(cfg.get("span_primary_mode", "legacy"))
        if self.span_primary_mode not in ("legacy", "hard_gt", "mass3", "gt_nt"):
            raise ValueError(
                f"unknown span_primary_mode={self.span_primary_mode!r}; "
                "expected legacy | hard_gt | mass3 | gt_nt")
        # Flatten the teacher target to uniform over its top-k SET (kills relative
        # confusability weights, keeps the candidate set + emission). Isolates
        # "dark knowledge = relative weights" without the K-count confound.
        self.span_kd_uniform_target = bool(cfg.get("span_kd_uniform_target", False))
        # Time-axis counterpart of span_kd_uniform_target: strip the SHAPE of the
        # teacher occupancy while holding its support (or its width) fixed.
        # See _transform_support for the semantics of each mode.
        self.span_kd_support_mode = str(cfg.get("span_kd_support_mode", "gamma"))
        self.span_kd_support_eps = float(cfg.get("span_kd_support_eps", 0.01))
        self.span_kd_support_width = int(cfg.get("span_kd_support_width", 0))
        # Counterfactual Boundary-KD cell (see scripts/build_boundary_kd_targets.py):
        #   1 soft p_T^delta | 2 hard {blank,y_u} m_delta (proposal) |
        #   3 hard m_zero | 4 hard m_delta, within-token uniform over dist=1 shoulders.
        self.boundary_kd_cell = int(cfg.get("boundary_kd_cell", 2))
        self.boundary_kd_train_min = float(cfg.get("boundary_kd_train_min", 0.0))
        self.free_emit_sub_weight = float(cfg.get("free_emit_sub_weight", 0.25))
        self.free_emit_spread_weight = float(cfg.get("free_emit_spread_weight", 0.05))
        self.free_emit_margin_start = int(cfg.get("free_emit_margin_start", 0))
        self.free_emit_anneal_epochs = int(cfg.get("free_emit_anneal_epochs", 70))
        # Which occupancy mass weights the boundary CE:
        #   residual    : [gamma^delta - gamma^0]_+  (shoulders only; cells 1-4)
        #   gamma_delta : the FULL gamma^delta, i.e. core spike AND shoulders.
        # The residual-only sweep landed 1.2 WER behind span-KD, implying the core
        # frames carry most of the gain; gamma_delta re-includes them while keeping
        # the hard target, full-softmax CE and batch normalisation.
        # gamma_delta reads span-KD's stored `support` (== gamma^delta), so it needs
        # a manifest carrying BOTH teacher_span_kd_path and boundary_kd_path.
        self.boundary_kd_weight_source = str(cfg.get("boundary_kd_weight_source", "residual"))
        # Confidence-Need Weighting (CNW): per-span KD weight
        #   r_u = teacher decisiveness on its top-1 (top1 - top2 prob)
        #   d_u = student need = 1 - student pooled prob on that token (detached)
        #   w_u = clip(r_u^alpha * d_u^gamma, wmin, wmax), gate-weighted-normalized
        #         to mean 1 per utterance so the overall lambda_KD scale is preserved.
        # Concentrates KD on spans the teacher is sure of but the student still misses.
        # Uses ONLY the semantic margin (not occupancy entropy/width, which would
        # re-introduce a preference for peaky teacher timing).
        self.span_kd_cnw = bool(cfg.get("span_kd_cnw", False))
        self.span_kd_cnw_alpha = float(cfg.get("span_kd_cnw_alpha", 1.0))
        self.span_kd_cnw_gamma = float(cfg.get("span_kd_cnw_gamma", 1.0))
        self.span_kd_cnw_wmin = float(cfg.get("span_kd_cnw_wmin", 0.2))
        self.span_kd_cnw_wmax = float(cfg.get("span_kd_cnw_wmax", 5.0))
        self.kd_start_step = int(cfg.get("kd_start_step", 0))
        self.transition_kd_weight = float(cfg.get("transition_kd_weight", 0.0))
        self.logit_kd_occ_weight = float(cfg.get("logit_kd_occ_weight", 0.0))
        self.logit_kd_res_weight = float(cfg.get("logit_kd_res_weight", 0.0))
        self.sr_ctc_weight = float(cfg.get("sr_ctc_weight", 0.0))  # SR-CTC smooth reg (Yao 2025)
        # Label-prior CTC regularization (Huang et al. 2024): normalize the student
        # per-frame posterior by an EMA label prior before the CTC loss, discouraging
        # the peaky blank-dominated solution. Applied to the CTC term only; KD terms
        # keep the raw student posterior. alpha=0 -> off.
        self.label_prior_ctc_alpha = float(cfg.get("label_prior_ctc_alpha", 0.0))
        self.label_prior_momentum = float(cfg.get("label_prior_momentum", 0.99))
        if self.label_prior_ctc_alpha > 0.0:
            self.register_buffer(
                "_label_prior",
                torch.ones(self.decoder.num_classes_with_blank) / self.decoder.num_classes_with_blank,
            )
        self.blank_id = self.decoder.num_classes_with_blank - 1
        if self.ctc_blank_init_bias != 0.0:
            with torch.no_grad():
                self.decoder.decoder_layers[-1].bias[self.blank_id].fill_(self.ctc_blank_init_bias)
        # Paper-style lambda: L = (1-λ)*L_CTC + λ*L_KD  (Hilmes et al. 2025)
        # λ=1.0 → pure KD (no CTC loss).  None → legacy formula L_CTC + kd_weight*L_KD.
        _lam = cfg.get("kd_lambda", None)
        self.kd_lambda = float(_lam) if _lam is not None else None

        # Paper baselines added for the TED2 reproduction suite.
        # Frame-DKD uses the official CVPR'22 DKD defaults. Its target is the
        # teacher top-1 token because CTC has no observed frame label.
        self.frame_dkd_alpha = float(cfg.get("frame_dkd_alpha", 1.0))
        self.frame_dkd_beta = float(cfg.get("frame_dkd_beta", 8.0))
        self.frame_dkd_temperature = float(cfg.get("frame_dkd_temperature", 4.0))
        self.frame_dkd_warmup_epochs = int(cfg.get("frame_dkd_warmup_epochs", 20))
        # FPKD: DFKD (binary blank), FRKD (balanced feature), PKD (CTC+FKL).
        self.fpkd_temperature = float(cfg.get("fpkd_temperature", 1.0))
        self.fpkd_bkl_weight = float(cfg.get("fpkd_bkl_weight", 1.0))
        self.fpkd_nbf_weight = float(cfg.get("fpkd_nbf_weight", 1.0))
        # CARL-CTC adaptation. The source paper's hybrid attention CE is not
        # available in this CTC-only architecture; every CTC-side CARL term is.
        self.carl_temperature = float(cfg.get("carl_temperature", 1.0))
        self.carl_alpha = float(cfg.get("carl_alpha", 1.0))
        self.carl_gamma = float(cfg.get("carl_gamma", 1.0))
        self.carl_lambda = float(cfg.get("carl_lambda", 1.0))
        self.carl_teacher_dim = int(cfg.get("carl_teacher_dim", 176))
        self.carl_classifier_path = str(cfg.get("carl_classifier_path", ""))

        if self.kd_mode not in _VALID_KD_MODES:
            raise ValueError(f"unsupported kd_mode: {self.kd_mode}")
        if self.logit_kd_cross_pool and (self.logit_kd_occ_weight > 0.0 or self.logit_kd_res_weight > 0.0):
            raise ValueError("logit_kd_cross_pool has no pooled path in the OCC/RES aux losses; "
                             "combine them only after implementing pooling there")
        if self.kd_mode == "combined":
            if self.kd_weight == 0.0 and self.token_avg_kd_weight == 0.0:
                raise ValueError("combined mode: both kd_weight and token_avg_kd_weight are 0.0")
        elif self.kd_mode not in (
            "none", "self_kd", "cr_ctc", "frame_dkd", "fpkd_dfkd", "fpkd_frkd",
            "fpkd_pkd", "carl_feature", "carl",
        ):
            active_aux = self.kd_mode == "span_kd" and self.span_ntdk_weight > 0.0
            if self.kd_lambda is None and self.kd_weight == 0.0 and not active_aux:
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
        # Paper default: alpha=0.2; multiply both the number of time masks and
        # their maximum width/fraction by 2.5. Legacy volume scaling remains
        # available only for old checkpoints/runs.
        if self.kd_mode == "cr_ctc":
            self.cr_ctc_weight = float(cfg.get("cr_ctc_weight", 0.2))
            # icefall default: linearly ramp cr_loss weight over 2000 batches
            self.cr_ctc_warm_step = int(cfg.get("cr_ctc_warm_step", 2000))
            volume_factor = float(cfg.get("cr_ctc_time_mask_factor", 2.5))
            masks_scale_cfg = cfg.get("cr_ctc_time_masks_scale", None)
            width_scale_cfg = cfg.get("cr_ctc_time_width_scale", None)
            if masks_scale_cfg is None and width_scale_cfg is None:
                dim_factor = volume_factor ** 0.5
                masks_scale = dim_factor
                width_scale = dim_factor
            elif masks_scale_cfg is None or width_scale_cfg is None:
                raise ValueError("set both cr_ctc_time_masks_scale and cr_ctc_time_width_scale")
            else:
                masks_scale = float(masks_scale_cfg)
                width_scale = float(width_scale_cfg)
            if getattr(self._cfg, "spec_augment", None) is None:
                raise ValueError("kd_mode='cr_ctc' requires model.spec_augment to be configured")
            cr_spec_cfg = copy.deepcopy(self._cfg.spec_augment)
            cr_spec_cfg["time_masks"] = int(round(int(cr_spec_cfg.get("time_masks", 10)) * masks_scale))
            base_time_width = cr_spec_cfg.get("time_width", 0.05)
            new_time_width = float(base_time_width) * width_scale
            if isinstance(base_time_width, float):
                # SpecAugment requires float time_width in [0, 1] (fraction of sequence length)
                new_time_width = min(new_time_width, 1.0)
            cr_spec_cfg["time_width"] = new_time_width
            self.spec_augmentation_cr = EncDecCTCModelBPE.from_config_dict(cr_spec_cfg)

        self._student_encoder_features = None
        if self.kd_mode in _NEEDS_CARL_FEATURE:
            # Conformer exposes d_model; Citrinet uses the decoder input width.
            student_dim = int(cfg.encoder.get("d_model", cfg.decoder.feat_in))
            self.carl_projection = nn.Conv1d(student_dim, self.carl_teacher_dim, kernel_size=1)
            # ConvASRDecoder receives encoder output as (B, D, T).
            self.decoder.decoder_layers.register_forward_pre_hook(self._capture_student_encoder)

        if self.kd_mode == "carl":
            if not self.carl_classifier_path:
                raise ValueError("kd_mode='carl' requires carl_classifier_path")
            artifact = torch.load(self.carl_classifier_path, map_location="cpu", weights_only=False)
            weight = artifact["weight"].float()
            bias = artifact["bias"].float()
            if weight.ndim == 2:
                weight = weight.unsqueeze(-1)
            if weight.shape[1] != self.carl_teacher_dim:
                raise ValueError(
                    f"CARL classifier dim {weight.shape[1]} != carl_teacher_dim {self.carl_teacher_dim}")
            if weight.shape[0] != self.decoder.num_classes_with_blank:
                raise ValueError("CARL teacher/student vocabulary mismatch")
            self.register_buffer("carl_classifier_weight", weight)
            self.register_buffer("carl_classifier_bias", bias)

        self._wer_accum = {}

    def _capture_student_encoder(self, module, inp):
        self._student_encoder_features = inp[0]

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
        require_sctc=False,
        require_combined_kd=False,
        require_boundary_kd=False,
        require_free_emit=False,
        require_carl_feature=False,
    ):
        return make_dataloader(
            cfg.manifest_filepath,
            self.tokenizer,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            sample_rate=cfg.sample_rate,
            max_duration=cfg.get("max_duration", None),
            num_workers=cfg.get("num_workers", 4),
            pin_memory=cfg.get("pin_memory", True),
            require_transition=require_transition,
            require_frame_kd=require_frame_kd,
            require_token_avg_kd=require_token_avg_kd,
            require_span_kd=require_span_kd,
            require_sctc=require_sctc,
            require_combined_kd=require_combined_kd,
            require_boundary_kd=require_boundary_kd,
            require_free_emit=require_free_emit,
            require_carl_feature=require_carl_feature,
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
            require_sctc=(kd_mode in _NEEDS_SCTC),
            require_combined_kd=(kd_mode == "combined"),
            require_boundary_kd=(kd_mode in _NEEDS_BOUNDARY_KD),
            require_free_emit=(kd_mode in _NEEDS_FREE_EMIT),
            require_carl_feature=(kd_mode in _NEEDS_CARL_FEATURE),
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

    def _reduce_frame_kd(self, per_frame, active):
        """Reduce masked frame losses locally or as the paper's sequence sum."""
        mask = active.to(per_frame.dtype)
        if self.logit_kd_reduction == "utterance_sum":
            # Hilmes et al. Eq. (2)/(3): sum over frames/classes per sequence.
            return (per_frame * mask).sum(dim=1).mean()
        return (per_frame * mask).sum() / mask.sum().clamp_min(1.0)


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

        active_override = None
        if teacher_frames == frames:
            ids = fkd_ids.to(log_probs.device)
            probs = fkd_probs.to(log_probs.device, dtype=log_probs.dtype)
            valid_len = torch.minimum(enc_len, fkd_lens.to(log_probs.device))
            valid = torch.arange(frames, device=log_probs.device).view(1, frames) < valid_len.view(batch, 1)
        elif self.logit_kd_cross_pool:
            # cross-rate pooled targets: average the two teacher frames covering
            # student frame j (positions j+0.25 and j+0.75 on the teacher grid).
            # CE against the average target == mean of the two frames' CE terms,
            # so no top-k merging/renormalization is needed.
            fl = fkd_lens.to(log_probs.device)
            src_len = enc_len.to(log_probs.device).float().view(batch, 1).clamp_min(1)

            def _gather(frac):
                pos = (torch.arange(frames, device=log_probs.device, dtype=torch.float32) + frac).view(1, frames)
                idx = torch.floor(pos / src_len * fl.float().view(batch, 1)).long()
                idx = torch.minimum(idx, (fl - 1).view(batch, 1)).clamp_min(0)
                g = idx.view(batch, frames, 1).expand(batch, frames, topk)
                return (torch.gather(fkd_ids.to(log_probs.device), 1, g),
                        torch.gather(fkd_probs.to(log_probs.device, dtype=log_probs.dtype), 1, g))

            ids_a, probs_a = _gather(0.25)
            ids_b, probs_b = _gather(0.75)
            ids = torch.cat([ids_a, ids_b], dim=2)
            probs = 0.5 * torch.cat([probs_a, probs_b], dim=2)
            valid = torch.arange(frames, device=log_probs.device).view(1, frames) < enc_len.view(batch, 1)
            pooled_nb = (ids_a[:, :, 0] != self.blank_id) | (ids_b[:, :, 0] != self.blank_id)
            mode = self.logit_kd_blank_mode
            if mode == "none":
                active_override = valid
            elif mode == "elimination":
                active_override = valid & pooled_nb
            elif mode == "symmetric":
                k = 2 * self.logit_kd_blank_n + 1
                expanded = F.max_pool1d(
                    pooled_nb.float().unsqueeze(1), kernel_size=k, stride=1,
                    padding=self.logit_kd_blank_n).squeeze(1).bool()
                active_override = valid & expanded
            else:
                raise ValueError(f"logit_kd_cross_pool supports blank modes "
                                 f"none/elimination/symmetric, got {mode}")
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

        active = active_override if active_override is not None else self._blank_active_mask(ids, probs, valid)
        return self._reduce_frame_kd(ce, active)

    def _frame_dense_kd_loss(self, log_probs, enc_len, teacher_probs, teacher_lens):
        """Full-vocabulary frame KD without top-k truncation or renormalization."""
        B, T, V = log_probs.shape
        T_t = teacher_probs.shape[1]
        teacher_probs = teacher_probs.to(log_probs.device, dtype=log_probs.dtype)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        teacher_lens = teacher_lens.to(log_probs.device)
        if teacher_probs.shape[2] != V:
            raise ValueError(f"teacher/student vocabulary mismatch: {teacher_probs.shape[2]} != {V}")
        if T_t == T:
            probs = teacher_probs
            valid_len = torch.minimum(enc_len, teacher_lens)
            valid = torch.arange(T, device=log_probs.device).view(1, T) < valid_len.view(B, 1)
        else:
            pos = (torch.arange(T, device=log_probs.device, dtype=torch.float32) + 0.5).view(1, T)
            idx = torch.floor(pos / enc_len.float().view(B, 1).clamp_min(1) * teacher_lens.float().view(B, 1)).long()
            idx = torch.minimum(idx, (teacher_lens - 1).view(B, 1)).clamp_min(0)
            probs = torch.gather(teacher_probs, 1, idx.unsqueeze(-1).expand(B, T, V))
            valid = torch.arange(T, device=log_probs.device).view(1, T) < enc_len.view(B, 1)
        student_lp = torch.log_softmax(log_probs / self.logit_kd_temperature, dim=-1)
        ce = -(probs * student_lp).sum(dim=-1)
        top1_ids = probs.argmax(dim=-1, keepdim=True)
        top1_probs = torch.gather(probs, 2, top1_ids)
        active = self._blank_active_mask(top1_ids, top1_probs, valid)
        return self._reduce_frame_kd(ce, active)

    # ------------------------------------------------------------------
    # Guided KD loss (Kurata & Audhkhasi 2019)
    # Maximize the selected posterior at teacher non-blank argmax frames.
    # This is -p(student), not hard cross-entropy -log p(student).
    # ------------------------------------------------------------------
    def _guided_kd_loss(self, log_probs, enc_len, fkd_ids, fkd_probs, fkd_lens):
        batch, frames, K = log_probs.shape
        teacher_frames = fkd_ids.shape[1]
        topk = fkd_ids.shape[2]

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

        teacher_argmax = ids[:, :, 0].clamp(0, K - 1)  # (B, T)
        selected = torch.gather(log_probs.exp(), 2, teacher_argmax.unsqueeze(2)).squeeze(2)
        active = valid & (teacher_argmax != self.blank_id)
        # Paper L_G is a raw sum over selected positions for each sequence.
        return (-(selected * active.to(selected.dtype)).sum(dim=1)).mean()

    def _guided_dense_loss(self, log_probs, enc_len, teacher_probs, teacher_lens):
        """Guided CTC paper objective using a dense teacher cache."""
        B, T, V = log_probs.shape
        T_t = teacher_probs.shape[1]
        teacher_probs = teacher_probs.to(log_probs.device, dtype=log_probs.dtype)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        teacher_lens = teacher_lens.to(log_probs.device)
        if T_t == T:
            probs = teacher_probs
            valid_len = torch.minimum(enc_len, teacher_lens)
            valid = torch.arange(T, device=log_probs.device).view(1, T) < valid_len.view(B, 1)
        else:
            pos = (torch.arange(T, device=log_probs.device, dtype=torch.float32) + 0.5).view(1, T)
            idx = torch.floor(pos / enc_len.float().view(B, 1).clamp_min(1) * teacher_lens.float().view(B, 1)).long()
            idx = torch.minimum(idx, (teacher_lens - 1).view(B, 1)).clamp_min(0)
            probs = torch.gather(teacher_probs, 1, idx.unsqueeze(-1).expand(B, T, V))
            valid = torch.arange(T, device=log_probs.device).view(1, T) < enc_len.view(B, 1)
        teacher_argmax = probs.argmax(dim=-1)
        active = valid & (teacher_argmax != self.blank_id)
        selected = torch.gather(log_probs.exp(), 2, teacher_argmax.unsqueeze(-1)).squeeze(-1)
        return (-(selected * active.to(selected.dtype)).sum(dim=1)).mean()

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
    def _transform_support(self, sup_s):
        """Time-axis ablation of the occupancy SHAPE (`span_kd_support_mode`).

        The span loss -log sum_t g(t,u) p_S(t,y_u) is minimised by p_S = 1 over
        the whole support, so any g' with the same support has the SAME argmin —
        the shape only steers the optimisation path. These modes strip the shape
        while holding the support (or its width and location) fixed, to measure
        whether the shape carries anything at all.

          gamma   : teacher occupancy as-is (default, = current method)
          uniform : flat over {t : g(t,u) > eps * max_t g(.,u)} — identical
                    frames, no shape
          rect    : flat over a fixed-width window centred on argmax_t g. Width
                    comes from `span_kd_support_width` frames, or from
                    n_eff = 1 / sum_t g~^2 (scripts/span_neff.py) when that is 0.
                    Keeps only the token's LOCATION — not the shape, not the
                    frame set, not the width. width=1 is the single-frame floor:
                    the span collapses to the teacher's spike, so whatever gain
                    survives it is not coming from span width at all.

        Args/returns: (N, T_s) un-normalised support, already on the STUDENT
        frame grid; normalisation happens in the caller, so every mode is on the
        same footing.
        """
        mode = self.span_kd_support_mode
        if mode == "gamma":
            return sup_s
        T = sup_s.shape[1]
        peak = sup_s.max(dim=1, keepdim=True).values.clamp_min(1e-12)
        if mode == "uniform":
            return (sup_s > self.span_kd_support_eps * peak).to(sup_s.dtype)
        if mode == "rect":
            if self.span_kd_support_width > 0:
                half = torch.full((sup_s.shape[0],), (self.span_kd_support_width - 1) / 2.0,
                                  device=sup_s.device, dtype=torch.float32)
            else:
                a = sup_s.float() / sup_s.float().sum(dim=1, keepdim=True).clamp_min(1e-8)
                n_eff = 1.0 / a.pow(2).sum(dim=1).clamp_min(1e-12)      # (N,)
                half = (n_eff.clamp(1.0, float(T)) - 1.0) / 2.0         # (N,)
            centre = sup_s.argmax(dim=1).float()                        # (N,)
            t = torch.arange(T, device=sup_s.device, dtype=torch.float32)
            keep = (t.unsqueeze(0) - centre.unsqueeze(1)).abs() <= half.unsqueeze(1) + 1e-6
            return keep.to(sup_s.dtype)
        raise ValueError(f"unknown span_kd_support_mode: {mode!r} "
                         "(expected gamma | uniform | rect)")

    def _span_kd_loss(self, log_probs, enc_len, support, avg_ids, avg_probs,
                      gates, num_tokens, teacher_frames, student_support=None):
        """student_support: optional (B, N, T_s) detached gamma on the STUDENT frame
        grid (dual-occupancy mode). When given, it replaces the resampled teacher
        support as the student-side pooling weights; the teacher target
        (avg_ids/avg_probs) is unchanged."""
        B, T_s, K = log_probs.shape
        device = log_probs.device
        dtype = log_probs.dtype
        N_max = support.shape[1]
        top_k = avg_ids.shape[2]
        student_prob = log_probs.exp().to(dtype)
        total = torch.zeros((), device=device, dtype=dtype)
        total_w = torch.zeros((), device=device, dtype=dtype)
        # diagnostics (gate-weighted sums), surfaced to the caller for logging
        d_content = torch.zeros((), device=device, dtype=dtype)
        d_emit = torch.zeros((), device=device, dtype=dtype)
        d_m = torch.zeros((), device=device, dtype=dtype)
        d_active = torch.zeros((), device=device, dtype=dtype)
        d_ntok = torch.zeros((), device=device, dtype=dtype)

        for b in range(B):
            n = int(num_tokens[b].item())
            ts = int(enc_len[b].item())
            tt = int(teacher_frames[b].item())
            if n <= 0 or ts <= 0 or tt <= 0:
                continue
            n = min(n, N_max)
            ts = min(ts, T_s)
            tt = min(tt, support.shape[2])

            if student_support is not None:
                n = min(n, student_support.shape[1])
                sup_s = student_support[b, :n, :ts].to(device=device, dtype=dtype)
            else:
                sup_t = support[b, :n, :tt].to(device=device, dtype=dtype)  # (N, T_t)
                # Student frame j covers teacher positions [j, j+1) * tt/ts: average
                # the support there (correct downsampling of a measure). At equal
                # rates both indices are j — identical to the old nearest lookup.
                # Point-sampling instead would drop the even/odd parity's mass and
                # leave ~10% of tokens with fp16-noise weights at 2x ratios.
                base = torch.arange(ts, device=device, dtype=torch.float32) / max(ts, 1) * tt
                idx_a = torch.floor(base + 0.25 * tt / max(ts, 1)).long().clamp(0, tt - 1)
                idx_b = torch.floor(base + 0.75 * tt / max(ts, 1)).long().clamp(0, tt - 1)
                sup_s = 0.5 * (sup_t[:, idx_a] + sup_t[:, idx_b])  # (N, T_s_valid)
            sup_s = self._transform_support(sup_s)
            sup_sum = sup_s.sum(dim=1)
            sup_s = sup_s / sup_sum.clamp_min(1e-8).unsqueeze(1)

            s_avg = torch.matmul(sup_s, student_prob[b, :ts])  # (N, K)
            ids = avg_ids[b, :n].to(device).clamp(0, K - 1)
            probs = avg_probs[b, :n].to(device=device, dtype=dtype)
            if self.span_kd_uniform_target:
                # kill teacher's relative weights; keep only the top-k SET.
                probs = torch.full_like(probs, 1.0 / probs.shape[1])
            selected = torch.gather(s_avg, 1, ids).clamp_min(1e-9)  # s_u(v), un-norm

            # CE(q, s_u) decomposed: content KL(q||s_bar) + beta * emission(-log m_u).
            # beta=1 => l_content + l_emit == -sum q log s_u == original CE (Σq=1).
            m = selected.sum(dim=1)                                 # m_u (top-k mass)
            s_bar = selected / m.clamp_min(1e-9).unsqueeze(1)       # conditional
            l_content = -(probs * s_bar.clamp_min(1e-9).log()).sum(dim=1)
            l_emit = -m.clamp_min(1e-9).log()
            ce = l_content + self.span_kd_emit_beta * l_emit        # (N,)

            # Confidence-Need Weighting (detached; a per-span reweight of the CE).
            if self.span_kd_cnw and selected.shape[1] > 1:
                r_u = (probs[:, 0] - probs[:, 1]).clamp_min(0.0)        # teacher decisiveness
                d_u = (1.0 - selected[:, 0]).clamp(0.0, 1.0)           # student need on top-1
                w_cnw = torch.clamp(
                    r_u.clamp_min(1e-6) ** self.span_kd_cnw_alpha
                    * d_u.clamp_min(1e-6) ** self.span_kd_cnw_gamma,
                    self.span_kd_cnw_wmin, self.span_kd_cnw_wmax).detach()
            else:
                w_cnw = torch.ones_like(ce)

            w = gates[b, :n].to(device=device, dtype=dtype)
            # skip tokens whose pooling support is empty (e.g. student gamma not
            # yet formed in dual mode): their CE is a meaningless constant.
            active = (w >= self.span_kd_gate_threshold) & (sup_sum > 1e-6)
            if active.any():
                w_active = w[active]
                wc = w_cnw[active]
                # gate-weighted-normalize CNW to mean 1 so lambda_KD scale is kept
                wc = wc / ((w_active * wc).sum() / w_active.sum().clamp_min(1e-8)).clamp_min(1e-8)
                eff = w_active * wc
                total = total + (ce[active] * eff).sum()
                total_w = total_w + w_active.sum()
                d_content = d_content + (l_content[active] * w_active).sum()
                d_emit = d_emit + (l_emit[active] * w_active).sum()
                d_m = d_m + (m[active] * w_active).sum()
                d_active = d_active + active.sum().to(dtype)
                d_ntok = d_ntok + torch.tensor(float(n), device=device, dtype=dtype)

        denom = total_w.clamp_min(1.0)
        self._span_diag = {
            "content": (d_content / denom).detach(),   # gate-weighted mean KL(q||s_bar)
            "emit": (d_emit / denom).detach(),          # gate-weighted mean -log m_u
            "m_mean": (d_m / denom).detach(),           # gate-weighted mean top-k mass m_u
            "gate_pass_frac": (d_active / d_ntok.clamp_min(1.0)).detach(),
        }
        return total / denom

    def _span_coarse_loss(self, log_probs, enc_len, support, gates,
                          num_tokens, teacher_frames, tokens, token_lens,
                          teacher_mass3=None, student_support=None):
        """Primary span loss without class-level non-target overlap.

        hard_gt pools the student with the token occupancy and applies CE only
        to the occurrence GT.  mass3 instead matches the teacher's exact coarse
        [blank, occurrence-GT, all non-target] distribution.  The latter leaves
        relations *inside* the non-target bucket exclusively to NTDK.
        """
        B, T_s, vocab = log_probs.shape
        device, dtype = log_probs.device, log_probs.dtype
        student_prob = log_probs.exp().to(dtype)
        total = torch.zeros((), device=device, dtype=dtype)
        total_w = torch.zeros((), device=device, dtype=dtype)
        d_student = torch.zeros(3, device=device, dtype=dtype)
        d_teacher = torch.zeros(3, device=device, dtype=dtype)
        d_student_content = torch.zeros(2, device=device, dtype=dtype)
        d_teacher_content = torch.zeros(2, device=device, dtype=dtype)

        for b in range(B):
            n = min(
                int(num_tokens[b].item()), int(token_lens[b].item()),
                support.shape[1])
            ts = min(int(enc_len[b].item()), T_s)
            tt = min(int(teacher_frames[b].item()), support.shape[2])
            if teacher_mass3 is not None:
                n = min(n, teacher_mass3.shape[1])
            if n <= 0 or ts <= 0 or tt <= 0:
                continue

            if student_support is not None:
                n = min(n, student_support.shape[1])
                sup_s = student_support[b, :n, :ts].to(device=device, dtype=dtype)
            else:
                sup_t = support[b, :n, :tt].to(device=device, dtype=dtype)
                base = torch.arange(ts, device=device, dtype=torch.float32) / max(ts, 1) * tt
                idx_a = torch.floor(base + 0.25 * tt / max(ts, 1)).long().clamp(0, tt - 1)
                idx_b = torch.floor(base + 0.75 * tt / max(ts, 1)).long().clamp(0, tt - 1)
                sup_s = 0.5 * (sup_t[:, idx_a] + sup_t[:, idx_b])
            sup_s = self._transform_support(sup_s)
            sup_sum = sup_s.sum(dim=1)
            sup_s = sup_s / sup_sum.clamp_min(1e-8).unsqueeze(1)
            s_avg = torch.matmul(sup_s, student_prob[b, :ts])

            gt = tokens[b, :n].to(device).clamp(0, vocab - 1)
            row = torch.arange(n, device=device)
            s_blank = s_avg[:, self.blank_id]
            s_gt = s_avg[row, gt]
            s_nt = (1.0 - s_blank - s_gt).clamp_min(0.0)
            s_mass = torch.stack([s_blank, s_gt, s_nt], dim=1)
            s_mass = s_mass / s_mass.sum(dim=1, keepdim=True).clamp_min(1e-9)

            if self.span_primary_mode == "hard_gt":
                per_token = -s_gt.clamp_min(1e-9).log()
                t_mass = torch.zeros_like(s_mass)
                t_mass[:, 1] = 1.0
            elif self.span_primary_mode == "mass3":
                if teacher_mass3 is None:
                    raise ValueError("span_primary_mode=mass3 requires teacher mass3 targets")
                t_mass = teacher_mass3[b, :n].to(device=device, dtype=dtype)
                t_mass = t_mass / t_mass.sum(dim=1, keepdim=True).clamp_min(1e-9)
                positive = t_mass > 0
                terms = torch.where(
                    positive,
                    t_mass * (t_mass.clamp_min(1e-9).log()
                              - s_mass.clamp_min(1e-9).log()),
                    torch.zeros_like(t_mass))
                per_token = terms.sum(dim=1)
            elif self.span_primary_mode == "gt_nt":
                if teacher_mass3 is None:
                    raise ValueError("span_primary_mode=gt_nt requires teacher mass3 targets")
                t_mass = teacher_mass3[b, :n].to(device=device, dtype=dtype)
                t_mass = t_mass / t_mass.sum(dim=1, keepdim=True).clamp_min(1e-9)
                # Remove the explicit pooled blank-mass target. We deliberately do not
                # reweight by teacher nonblank mass: CTC is the only direct
                # supervisor of blank/emission allocation in this ablation.
                t_content = t_mass[:, 1:]
                t_content = t_content / t_content.sum(dim=1, keepdim=True).clamp_min(1e-9)
                s_content = s_mass[:, 1:]
                s_content = s_content / s_content.sum(dim=1, keepdim=True).clamp_min(1e-9)
                positive = t_content > 0
                terms = torch.where(
                    positive,
                    t_content * (t_content.clamp_min(1e-9).log()
                                 - s_content.clamp_min(1e-9).log()),
                    torch.zeros_like(t_content))
                per_token = terms.sum(dim=1)
            else:
                raise RuntimeError(f"coarse loss called for mode {self.span_primary_mode!r}")

            w = gates[b, :n].to(device=device, dtype=dtype)
            active = (w >= self.span_kd_gate_threshold) & (sup_sum > 1e-6)
            if active.any():
                wa = w[active]
                total = total + (per_token[active] * wa).sum()
                total_w = total_w + wa.sum()
                d_student = d_student + (s_mass[active] * wa.unsqueeze(1)).sum(dim=0)
                d_teacher = d_teacher + (t_mass[active] * wa.unsqueeze(1)).sum(dim=0)
                s_diag_content = s_mass[:, 1:]
                s_diag_content = s_diag_content / s_diag_content.sum(dim=1, keepdim=True).clamp_min(1e-9)
                t_diag_content = t_mass[:, 1:]
                t_diag_content = t_diag_content / t_diag_content.sum(dim=1, keepdim=True).clamp_min(1e-9)
                d_student_content = d_student_content + (s_diag_content[active] * wa.unsqueeze(1)).sum(dim=0)
                d_teacher_content = d_teacher_content + (t_diag_content[active] * wa.unsqueeze(1)).sum(dim=0)

        denom = total_w.clamp_min(1.0)
        self._span_primary_diag = {
            "loss": (total / denom).detach(),
            "teacher_blank": (d_teacher[0] / denom).detach(),
            "teacher_gt": (d_teacher[1] / denom).detach(),
            "teacher_nt": (d_teacher[2] / denom).detach(),
            "student_blank": (d_student[0] / denom).detach(),
            "student_gt": (d_student[1] / denom).detach(),
            "student_nt": (d_student[2] / denom).detach(),
            "teacher_gt_given_nb": (d_teacher_content[0] / denom).detach(),
            "teacher_nt_given_nb": (d_teacher_content[1] / denom).detach(),
            "student_gt_given_nb": (d_student_content[0] / denom).detach(),
            "student_nt_given_nb": (d_student_content[1] / denom).detach(),
        }
        return total / denom

    def _span_ntdk_loss(self, log_probs, enc_len, support, dark_ids, dark_probs,
                        dark_tail_prob, gates, num_tokens, teacher_frames,
                        tokens, token_lens, teacher_mass3=None):
        """KL over raw teacher non-target relations using top-M + tail.

        Teacher targets have blank and occurrence GT removed and are already
        normalized over all remaining classes. Student probabilities are pooled
        with the same teacher occupancy, then factorized identically. The tail
        bucket preserves all non-selected nonblank/non-GT probability rather
        than introducing top-k renormalization bias.
        """
        B, T_s, vocab = log_probs.shape
        device, dtype = log_probs.device, log_probs.dtype
        student_prob = log_probs.exp().to(dtype)
        total = torch.zeros((), device=device, dtype=dtype)
        total_w = torch.zeros((), device=device, dtype=dtype)
        d_teacher_tail = torch.zeros((), device=device, dtype=dtype)
        d_student_tail = torch.zeros((), device=device, dtype=dtype)
        d_teacher_entropy = torch.zeros((), device=device, dtype=dtype)
        d_base_w = torch.zeros((), device=device, dtype=dtype)
        d_rel_w = torch.zeros((), device=device, dtype=dtype)
        d_rel_pass = torch.zeros((), device=device, dtype=dtype)

        for b in range(B):
            n = min(
                int(num_tokens[b].item()), int(token_lens[b].item()),
                support.shape[1], dark_ids.shape[1])
            ts = min(int(enc_len[b].item()), T_s)
            tt = min(int(teacher_frames[b].item()), support.shape[2])
            if self.span_ntdk_reliability or self.span_ntdk_mass_weighted:
                if teacher_mass3 is None:
                    raise ValueError("mass-weighted NTDK requires mass3 targets")
                n = min(n, teacher_mass3.shape[1])
            if n <= 0 or ts <= 0 or tt <= 0:
                continue

            sup_t = support[b, :n, :tt].to(device=device, dtype=dtype)
            base = torch.arange(ts, device=device, dtype=torch.float32) / max(ts, 1) * tt
            idx_a = torch.floor(base + 0.25 * tt / max(ts, 1)).long().clamp(0, tt - 1)
            idx_b = torch.floor(base + 0.75 * tt / max(ts, 1)).long().clamp(0, tt - 1)
            sup_s = 0.5 * (sup_t[:, idx_a] + sup_t[:, idx_b])
            sup_s = self._transform_support(sup_s)
            sup_sum = sup_s.sum(dim=1)
            sup_s = sup_s / sup_sum.clamp_min(1e-8).unsqueeze(1)
            s_avg = torch.matmul(sup_s, student_prob[b, :ts])

            ids = dark_ids[b, :n].to(device).clamp(0, vocab - 1)
            t_selected = dark_probs[b, :n].to(device=device, dtype=dtype)
            t_tail = dark_tail_prob[b, :n].to(device=device, dtype=dtype)
            t_nt = torch.cat([t_selected, t_tail.unsqueeze(1)], dim=1)
            t_nt = t_nt / t_nt.sum(dim=1, keepdim=True).clamp_min(1e-9)

            gt = tokens[b, :n].to(device).clamp(0, vocab - 1)
            row = torch.arange(n, device=device)
            nt_mass = (
                1.0 - s_avg[:, self.blank_id] - s_avg[row, gt]
            ).clamp_min(1e-9)
            s_selected = torch.gather(s_avg, 1, ids)
            s_tail = (nt_mass - s_selected.sum(dim=1)).clamp_min(0.0)
            s_nt = torch.cat([s_selected, s_tail.unsqueeze(1)], dim=1)
            s_nt = s_nt / s_nt.sum(dim=1, keepdim=True).clamp_min(1e-9)

            positive = t_nt > 0
            kl_terms = torch.where(
                positive,
                t_nt * (t_nt.clamp_min(1e-9).log() - s_nt.clamp_min(1e-9).log()),
                torch.zeros_like(t_nt))
            kl = kl_terms.sum(dim=1)

            w = gates[b, :n].to(device=device, dtype=dtype)
            active = (w >= self.span_kd_gate_threshold) & (sup_sum > 1e-6)
            reliability = torch.ones(n, device=device, dtype=dtype)
            if self.span_ntdk_reliability or self.span_ntdk_mass_weighted:
                tm = teacher_mass3[b, :n].to(device=device, dtype=dtype)
                tm = tm / tm.sum(dim=1, keepdim=True).clamp_min(1e-9)
                t_gt_abs, t_nt_abs = tm[:, 1], tm[:, 2]
                if self.span_ntdk_mass_weighted:
                    reliability = t_nt_abs
                else:
                    t_gt_given_nb = t_gt_abs / (t_gt_abs + t_nt_abs).clamp_min(1e-9)
                    reliability = t_nt_abs.clamp(
                        min=self.span_ntdk_rel_min, max=self.span_ntdk_rel_max
                    ).pow(self.span_ntdk_rel_power)
                    reliability = reliability * (t_gt_given_nb >= self.span_ntdk_rel_gt_min).to(dtype)
            if active.any():
                base_wa = w[active]
                wa = base_wa * reliability[active]
                total = total + (kl[active] * wa).sum()
                # Keep the token denominator for exact mean p_T(NT) * KL_cond.
                # Dividing by sum(p_T(NT)) would cancel the mass attenuation.
                if self.span_ntdk_mass_weighted:
                    total_w = total_w + base_wa.sum()
                else:
                    total_w = total_w + wa.sum()
                d_base_w = d_base_w + base_wa.sum()
                d_rel_w = d_rel_w + wa.sum()
                d_rel_pass = d_rel_pass + (
                    base_wa * (reliability[active] > 0).to(dtype)
                ).sum()
                d_teacher_tail = d_teacher_tail + (t_nt[active, -1] * wa).sum()
                d_student_tail = d_student_tail + (s_nt[active, -1] * wa).sum()
                entropy = -(t_nt[active] * t_nt[active].clamp_min(1e-9).log()).sum(dim=1)
                d_teacher_entropy = d_teacher_entropy + (entropy * wa).sum()

        denom = total_w.clamp_min(1e-8)
        self._span_ntdk_diag = {
            "loss": (total / denom).detach(),
            "teacher_tail": (d_teacher_tail / denom).detach(),
            "student_tail": (d_student_tail / denom).detach(),
            "teacher_entropy": (d_teacher_entropy / denom).detach(),
            "reliability_mean": (d_rel_w / d_base_w.clamp_min(1e-8)).detach(),
            "reliability_pass_frac": (d_rel_pass / d_base_w.clamp_min(1e-8)).detach(),
        }
        return total / denom

    def _span_shoulder_loss(self, log_probs, enc_len, core_support,
                            expanded_support, gates, num_tokens,
                            teacher_frames, tokens, token_lens):
        """Occurrence-GT emission on the positive delta-induced FB residual.

        The residual is formed on the teacher grid before resampling. It is
        normalized independently per occurrence, so the delta=0 core retains
        its original Mass3/NTDK scale. Tokens with no positive residual are
        skipped (in particular, delta=0 gives an exact zero loss).
        """
        B, T_s, vocab = log_probs.shape
        device, dtype = log_probs.device, log_probs.dtype
        student_prob = log_probs.exp().to(dtype)
        total = log_probs.sum() * 0.0
        total_w = torch.zeros((), device=device, dtype=dtype)
        d_active = torch.zeros((), device=device, dtype=dtype)
        d_tokens = torch.zeros((), device=device, dtype=dtype)
        d_gt = torch.zeros((), device=device, dtype=dtype)

        for b in range(B):
            n = min(
                int(num_tokens[b].item()), int(token_lens[b].item()),
                core_support.shape[1], expanded_support.shape[1])
            ts = min(int(enc_len[b].item()), T_s)
            tt = min(
                int(teacher_frames[b].item()), core_support.shape[2],
                expanded_support.shape[2])
            if n <= 0 or ts <= 0 or tt <= 0:
                continue

            core_t = core_support[b, :n, :tt].to(device=device, dtype=dtype)
            expanded_t = expanded_support[b, :n, :tt].to(device=device, dtype=dtype)
            residual_t = (expanded_t - core_t).clamp_min(0.0)

            base = torch.arange(ts, device=device, dtype=torch.float32) / max(ts, 1) * tt
            idx_a = torch.floor(base + 0.25 * tt / max(ts, 1)).long().clamp(0, tt - 1)
            idx_b = torch.floor(base + 0.75 * tt / max(ts, 1)).long().clamp(0, tt - 1)
            residual_s = 0.5 * (residual_t[:, idx_a] + residual_t[:, idx_b])
            residual_sum = residual_s.sum(dim=1)
            residual_s = residual_s / residual_sum.clamp_min(1e-8).unsqueeze(1)
            s_avg = torch.matmul(residual_s, student_prob[b, :ts])

            gt = tokens[b, :n].to(device).clamp(0, vocab - 1)
            row = torch.arange(n, device=device)
            s_gt = s_avg[row, gt]
            per_token = -s_gt.clamp_min(1e-9).log()
            w = gates[b, :n].to(device=device, dtype=dtype)
            active = (w >= self.span_kd_gate_threshold) & (residual_sum > 1e-6)
            d_tokens = d_tokens + torch.tensor(float(n), device=device, dtype=dtype)
            if active.any():
                wa = w[active]
                total = total + (per_token[active] * wa).sum()
                total_w = total_w + wa.sum()
                d_active = d_active + active.sum().to(dtype)
                d_gt = d_gt + (s_gt[active] * wa).sum()

        denom = total_w.clamp_min(1.0)
        self._span_shoulder_diag = {
            "loss": (total / denom).detach(),
            "active_frac": (d_active / d_tokens.clamp_min(1.0)).detach(),
            "student_gt": (d_gt / denom).detach(),
        }
        return total / denom

    def _sctc_loss(self, log_probs, enc_len, sctc_gamma, sctc_bpe_ids,
                   num_tokens, teacher_frames):
        """Sequence-level KD (Huang et al. 2018, Eq. 10).

        Per-frame cross-entropy of the student log-posterior against the
        teacher's transcript-constrained forward-backward occupancy:
            L = -sum_t [ blank_occ(t)*log y_blank(t)
                         + sum_u gamma(t,u)*log y_{id(u)}(t) ]
        averaged over valid frames. gamma(t,u) is the teacher's soft alignment
        of transcript token u onto frame t; the remaining mass (1 - sum_u gamma)
        is the blank occupancy. Teacher frames are mapped onto the student frame
        grid by nearest-frame resampling (same as span_kd)."""
        B, T_s, V = log_probs.shape
        device = log_probs.device
        dtype = log_probs.dtype
        total = torch.zeros((), device=device, dtype=dtype)
        total_w = torch.zeros((), device=device, dtype=dtype)

        for b in range(B):
            n = int(num_tokens[b].item())
            ts = int(enc_len[b].item())
            tt = int(teacher_frames[b].item())
            if n <= 0 or ts <= 0 or tt <= 0:
                continue
            n = min(n, sctc_gamma.shape[1])
            ts = min(ts, T_s)
            tt = min(tt, sctc_gamma.shape[2])

            gamma_t = sctc_gamma[b, :n, :tt].to(device=device, dtype=dtype)   # (N, T_t)
            # map each student frame to its teacher support frame
            # average the gamma over the teacher frames covered by each student
            # frame (measure downsampling; identical to nearest at equal rates —
            # point-sampling at 2x would drop the skipped parity's spikes entirely)
            base = torch.arange(ts, device=device, dtype=torch.float32) / max(ts, 1) * tt
            idx_a = torch.floor(base + 0.25 * tt / max(ts, 1)).long().clamp(0, tt - 1)
            idx_b = torch.floor(base + 0.75 * tt / max(ts, 1)).long().clamp(0, tt - 1)
            gamma_s = 0.5 * (gamma_t[:, idx_a] + gamma_t[:, idx_b])            # (N, T_s_valid)
            blank_occ = (1.0 - gamma_s.sum(dim=0)).clamp_min(0.0)             # (T_s_valid,)

            lp = log_probs[b, :ts]                                            # (T_s_valid, V)
            ids = sctc_bpe_ids[b, :n].to(device).clamp(0, V - 1)             # (N,)
            # token CE: sum_u gamma(t,u) * log y_{id(u)}(t)
            tok_lp = lp[:, ids].transpose(0, 1)                               # (N, T_s_valid)
            ce = -(gamma_s * tok_lp).sum()
            ce = ce - (blank_occ * lp[:, self.blank_id]).sum()
            total = total + ce
            total_w = total_w + float(ts)

        return total / total_w.clamp_min(1.0)

    # ------------------------------------------------------------------
    # Counterfactual Boundary-KD: distil ONLY blank-suppression's own gain
    #   r_{t,u} = [gamma^delta - gamma^0]_+   (shoulder frames delta newly added)
    # at those frames, in full-softmax with NO {blank,y_u} renormalisation, so
    # neither a blank shortcut nor a third-token shortcut can dodge the loss.
    # Weight is the RAW residual mass (batch-normalised only); per-token
    # normalisation is deliberately avoided so residual-poor tokens stay quiet.
    # ------------------------------------------------------------------
    def _boundary_kd_loss(self, log_probs, enc_len, batch):
        B, T_s, V = log_probs.shape
        device = log_probs.device
        dtype = log_probs.dtype
        cell = self.boundary_kd_cell
        eps = 1e-8

        res_t = batch["bd_res_t"]; res_u = batch["bd_res_u"]
        res_r = batch["bd_res_r"]; res_valid = batch["bd_res_valid"]
        m_delta = batch["bd_m_delta"]; m_zero = batch["bd_m_zero"]
        spike = batch["bd_spike"]; y = batch["bd_y"]
        num_tokens = batch["bd_num_tokens"]; teacher_frames = batch["bd_teacher_frames"]
        thr = self.boundary_kd_train_min

        total = torch.zeros((), device=device, dtype=dtype)
        total_w = torch.zeros((), device=device, dtype=dtype)
        diag_frames = torch.zeros((), device=device, dtype=dtype)

        for b in range(B):
            ts = int(enc_len[b].item()); tt = int(teacher_frames[b].item())
            n = int(num_tokens[b].item())
            if ts <= 0 or tt <= 0 or n <= 0:
                continue
            ts = min(ts, T_s)
            lp = log_probs[b, :ts]                                   # (ts, V)
            m_d = m_delta[b].to(device=device, dtype=dtype)
            m_z = m_zero[b].to(device=device, dtype=dtype)
            yb = y[b, :n].to(device).clamp(0, V - 1)

            # ---- build (teacher_frame, token, weight) entries for this cell ----
            if self.boundary_kd_weight_source == "gamma_delta":
                # full gamma^delta (core + shoulders) from span-KD's stored support
                sup = batch["span_support"][b, :n, :tt].to(device=device, dtype=dtype)
                nz = sup > max(thr, 1e-6)
                if not nz.any():
                    continue
                tok, fr_t = torch.nonzero(nz, as_tuple=True)     # support is (N, T)
                w = sup[tok, fr_t]
            elif cell == 4:
                # within-token uniform over dist=1 shoulders, mass A_u preserved.
                v = res_valid[b]
                ru = res_u[b][v]; rr = res_r[b][v]
                if ru.numel() == 0:
                    continue
                A = torch.zeros(n, device=device, dtype=dtype)
                A.scatter_add_(0, ru.clamp(0, n - 1), rr.to(dtype))
                sp = spike[b, :n].to(device)
                left = (sp - 1); right = (sp + 1)
                # valid shoulder frames inside [0, tt)
                lv = (left >= 0) & (left < tt)
                rv = (right >= 0) & (right < tt)
                ndir = lv.to(dtype) + rv.to(dtype)                  # |D_u| in {0,1,2}
                has = (A > 0) & (ndir > 0)
                per = torch.where(has, A / ndir.clamp_min(1.0), torch.zeros_like(A))
                fr_list, tok_list, w_list = [], [], []
                for side, valid_side in ((left, lv), (right, rv)):
                    sel = has & valid_side
                    if sel.any():
                        fr_list.append(side[sel])
                        tok_list.append(torch.nonzero(sel, as_tuple=False).squeeze(1))
                        w_list.append(per[sel])
                if not fr_list:
                    continue
                fr_t = torch.cat(fr_list); tok = torch.cat(tok_list); w = torch.cat(w_list)
            else:
                v = res_valid[b] & (res_r[b] > thr)
                if not v.any():
                    continue
                fr_t = res_t[b][v].clamp(0, tt - 1)
                tok = res_u[b][v].clamp(0, n - 1)
                w = res_r[b][v].to(dtype)

            # ---- map teacher frame -> student frame (nearest; identity at 4x/4x) ----
            fs = torch.floor((fr_t.float() + 0.5) / max(tt, 1) * ts).long().clamp(0, ts - 1)
            lp_e = lp[fs]                                            # (E, V)
            yk = yb[tok]                                            # (E,) target token per entry

            if cell == 1:
                # soft: -sum_v p_T^delta(v) log p_S(v), full softmax
                soft_t = batch["bd_soft_t"][b]                      # (F,)
                soft_p = batch["bd_soft_p"][b].to(device=device, dtype=dtype)  # (F, V)
                # every residual frame is present in soft_t (unique residual frames)
                pos = torch.searchsorted(soft_t, fr_t)
                pos = pos.clamp(0, soft_t.numel() - 1)
                p_soft = soft_p[pos]                                # (E, V)
                ce = -(p_soft * lp_e).sum(dim=1)                   # (E,)
            else:
                m = (m_z if cell == 3 else m_d)[fr_t]              # (E,) emission mass
                lp_blank = lp_e[:, self.blank_id]
                lp_y = lp_e.gather(1, yk.unsqueeze(1)).squeeze(1)
                ce = -((1.0 - m) * lp_blank + m * lp_y)           # (E,) hard, full-softmax

            total = total + (w * ce).sum()
            total_w = total_w + w.sum()
            diag_frames = diag_frames + torch.tensor(float(w.numel()), device=device, dtype=dtype)

        self._boundary_diag = {
            "res_frames_per_batch": (diag_frames / max(B, 1)).detach(),
            "weight_sum": total_w.detach(),
        }
        return total / total_w.clamp_min(eps)

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
    # FPKD / Frame-DKD / CARL paper baselines
    # ------------------------------------------------------------------
    def _align_dense_teacher(self, log_probs, enc_len, teacher_probs, teacher_lens):
        """Nearest-centre teacher-to-student frame mapping with validity mask."""
        B, T, V = log_probs.shape
        # Keep the cached bfloat16 posterior in float32. Casting it to an
        # autocast/student dtype here can underflow the small probabilities that
        # become relevant again for high-temperature KD (notably DKD at T=4).
        teacher_probs = teacher_probs.to(log_probs.device, dtype=torch.float32)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        teacher_lens = teacher_lens.to(log_probs.device)
        if teacher_probs.shape[-1] != V:
            raise ValueError("teacher/student vocabulary mismatch")
        pos = (torch.arange(T, device=log_probs.device, dtype=torch.float32) + 0.5).view(1, T)
        idx = torch.floor(
            pos / enc_len.float().view(B, 1).clamp_min(1)
            * teacher_lens.float().view(B, 1)
        ).long()
        idx = torch.minimum(idx, (teacher_lens - 1).view(B, 1)).clamp_min(0)
        probs = torch.gather(teacher_probs, 1, idx.unsqueeze(-1).expand(B, T, V))
        valid = torch.arange(T, device=log_probs.device).view(1, T) < enc_len.view(B, 1)
        return probs, valid

    @staticmethod
    def _teacher_probs_at_temperature(probs, temperature):
        """Recover a temperature-scaled distribution from cached T=1 probs.

        For p=softmax(z), softmax(z/T) is exactly p**(1/T), renormalized.
        Computing that identity directly avoids imposing a large log floor on
        the long CTC vocabulary tail. Exact cached zeros remain zero rather
        than being assigned artificial probability mass.
        """
        temp = float(temperature)
        if temp <= 0.0:
            raise ValueError(f"temperature must be positive, got {temp}")
        weights = probs.float().clamp_min(0.0).pow(1.0 / temp)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny)

    @staticmethod
    def _masked_mean(values, mask):
        weight = mask.to(values.dtype)
        return (values * weight).sum() / weight.sum().clamp_min(1.0)

    @staticmethod
    def _binary_bernoulli_kl(teacher_positive, student_positive):
        """Numerically stable KL between Bernoulli distributions.

        Dense CTC targets are cached at reduced precision and therefore often
        contain exact zeros and ones. Preserve those exact teacher targets,
        but keep the student logarithm away from zero. ``xlogy`` implements
        the limiting value 0 * log(0) = 0 without producing NaNs.
        """
        teacher_positive = teacher_positive.float().clamp(0.0, 1.0)
        student_positive = student_positive.float()
        eps = 1e-6  # representable below 1.0 in float32 (unlike 1e-8)
        student_positive = student_positive.clamp(eps, 1.0 - eps)
        teacher = torch.stack(
            [teacher_positive, 1.0 - teacher_positive], dim=-1)
        student = torch.stack(
            [student_positive, 1.0 - student_positive], dim=-1)
        return (
            torch.special.xlogy(teacher, teacher)
            - torch.special.xlogy(teacher, student)
        ).sum(dim=-1)

    def _binary_blank_kl(self, log_probs, enc_len, teacher_probs, teacher_lens, temperature):
        probs, valid = self._align_dense_teacher(log_probs, enc_len, teacher_probs, teacher_lens)
        temp = float(temperature)
        t_prob = self._teacher_probs_at_temperature(probs, temp)
        s_prob = torch.softmax(log_probs / temp, dim=-1)
        tb = t_prob[..., self.blank_id]
        sb = s_prob[..., self.blank_id]
        kl = self._binary_bernoulli_kl(tb, sb) * (temp ** 2)
        return self._masked_mean(kl, valid)

    def _fpkd_fkl_loss(self, log_probs, enc_len, teacher_probs, teacher_lens):
        """Factorized KL: BKL on all frames + reweighted NBFKL on teacher NB frames.

        The BFKL term (conditional nonblank distribution at teacher blank frames)
        is deliberately eliminated, as prescribed by Tian et al. 2024.
        """
        probs, valid = self._align_dense_teacher(log_probs, enc_len, teacher_probs, teacher_lens)
        temp = self.fpkd_temperature
        t_prob = self._teacher_probs_at_temperature(probs, temp)
        s_lp = torch.log_softmax(log_probs / temp, dim=-1)
        s_prob = s_lp.exp()

        tb = t_prob[..., self.blank_id]
        sb = s_prob[..., self.blank_id]
        bkl_frame = self._binary_bernoulli_kl(tb, sb) * (temp ** 2)
        bkl = self._masked_mean(bkl_frame, valid)

        nb_mask = valid & (probs.argmax(-1) != self.blank_id)
        t_nb = t_prob.clone()
        t_nb[..., self.blank_id] = 0.0
        t_nb = t_nb / t_nb.sum(-1, keepdim=True).clamp_min(1e-12)
        s_nb_logits = log_probs / temp
        s_nb_logits = s_nb_logits.clone()
        s_nb_logits[..., self.blank_id] = -1e4
        s_nb_lp = torch.log_softmax(s_nb_logits, dim=-1)
        nbf_terms = torch.where(
            t_nb > 0, t_nb * (t_nb.clamp_min(1e-12).log() - s_nb_lp),
            torch.zeros_like(t_nb))
        # Exact KL factorization carries the teacher nonblank mass outside
        # the conditional nonblank-token KL.
        nbf_frame = (1.0 - tb) * nbf_terms.sum(-1) * (temp ** 2)
        nbfkl = self._masked_mean(nbf_frame, nb_mask)
        return self.fpkd_bkl_weight * bkl + self.fpkd_nbf_weight * nbfkl, bkl, nbfkl

    def _feature_kd_loss(self, enc_len, teacher_features, teacher_lens, balanced=False):
        student = self._student_encoder_features
        if student is None:
            raise RuntimeError("student encoder feature hook did not run")
        projected = self.carl_projection(student)  # (B, D_teacher, T_student)
        total_nb = projected.new_zeros(())
        total_b = projected.new_zeros(())
        count_nb = projected.new_zeros(())
        count_b = projected.new_zeros(())
        for b in range(projected.shape[0]):
            ts = min(int(enc_len[b].item()), projected.shape[-1])
            tt = min(int(teacher_lens[b].item()), teacher_features.shape[1])
            if ts <= 0 or tt <= 0:
                continue
            tf = teacher_features[b, :tt].to(projected.device, projected.dtype).transpose(0, 1)
            tf = F.interpolate(tf.unsqueeze(0), size=ts, mode="linear", align_corners=False).squeeze(0)
            per = 1.0 - F.cosine_similarity(projected[b, :, :ts], tf, dim=0)
            if balanced:
                # FRKD blank/nonblank masks are supplied by the caller.
                raise RuntimeError("balanced feature loss requires _fpkd_frkd_loss")
            total_nb = total_nb + per.sum()
            count_nb = count_nb + ts
        return total_nb / count_nb.clamp_min(1.0)

    def _fpkd_frkd_loss(self, enc_len, teacher_features, teacher_feature_lens,
                        teacher_probs, teacher_prob_lens):
        student = self._student_encoder_features
        if student is None:
            raise RuntimeError("student encoder feature hook did not run")
        projected = self.carl_projection(student)
        # Align the teacher posterior only to obtain the paper's blank/NB split.
        dummy_lp = projected.new_zeros(projected.shape[0], projected.shape[-1],
                                       teacher_probs.shape[-1])
        probs, _ = self._align_dense_teacher(dummy_lp, enc_len, teacher_probs, teacher_prob_lens)
        top1 = probs.argmax(-1)
        nb_total = projected.new_zeros(())
        b_total = projected.new_zeros(())
        nb_count = projected.new_zeros(())
        b_count = projected.new_zeros(())
        for b in range(projected.shape[0]):
            ts = min(int(enc_len[b].item()), projected.shape[-1])
            tt = min(int(teacher_feature_lens[b].item()), teacher_features.shape[1])
            if ts <= 0 or tt <= 0:
                continue
            tf = teacher_features[b, :tt].to(projected.device, projected.dtype).transpose(0, 1)
            tf = F.interpolate(tf.unsqueeze(0), size=ts, mode="linear", align_corners=False).squeeze(0)
            per = 1.0 - F.cosine_similarity(projected[b, :, :ts], tf, dim=0)
            nb = top1[b, :ts] != self.blank_id
            blank = ~nb
            nb_total = nb_total + (per * nb.to(per.dtype)).sum()
            b_total = b_total + (per * blank.to(per.dtype)).sum()
            nb_count = nb_count + nb.sum()
            b_count = b_count + blank.sum()
        # Equal group weight removes the extreme CTC blank-frame imbalance.
        return 0.5 * (nb_total / nb_count.clamp_min(1.0) + b_total / b_count.clamp_min(1.0))

    def _frame_dkd_loss(self, log_probs, enc_len, teacher_probs, teacher_lens):
        probs, valid = self._align_dense_teacher(log_probs, enc_len, teacher_probs, teacher_lens)
        temp = self.frame_dkd_temperature
        t_prob = self._teacher_probs_at_temperature(probs, temp)
        s_prob = torch.softmax(log_probs / temp, dim=-1)
        target = probs.argmax(-1)
        tp = torch.gather(t_prob, -1, target.unsqueeze(-1)).squeeze(-1).clamp(1e-8, 1 - 1e-8)
        sp = torch.gather(s_prob, -1, target.unsqueeze(-1)).squeeze(-1).clamp(1e-8, 1 - 1e-8)
        t_bin = torch.stack([tp, 1 - tp], -1)
        s_bin = torch.stack([sp, 1 - sp], -1)
        tckd = (t_bin * (t_bin.log() - s_bin.log())).sum(-1) * (temp ** 2)

        target_mask = F.one_hot(target, num_classes=log_probs.shape[-1]).bool()
        t_nt = t_prob.masked_fill(target_mask, 0.0)
        s_nt = s_prob.masked_fill(target_mask, 0.0)
        t_nt = t_nt / t_nt.sum(-1, keepdim=True).clamp_min(1e-12)
        s_nt = s_nt / s_nt.sum(-1, keepdim=True).clamp_min(1e-12)
        nckd = (t_nt * (t_nt.clamp_min(1e-12).log() - s_nt.clamp_min(1e-12).log())).sum(-1)
        nckd = nckd * (temp ** 2)
        return (self.frame_dkd_alpha * self._masked_mean(tckd, valid)
                + self.frame_dkd_beta * self._masked_mean(nckd, valid))

    def _carl_be_loss(self, student_lp, enc_len, teacher_probs, teacher_lens):
        probs, valid = self._align_dense_teacher(student_lp, enc_len, teacher_probs, teacher_lens)
        temp = self.carl_temperature
        t_prob = self._teacher_probs_at_temperature(probs, temp)
        s_lp = torch.log_softmax(student_lp / temp, dim=-1)
        kl = F.kl_div(s_lp, t_prob, reduction="none").sum(-1)
        active = valid & (probs.argmax(-1) != self.blank_id)
        return self._masked_mean(kl, active)

    def _carl_projected_log_probs(self):
        student = self._student_encoder_features
        if student is None:
            raise RuntimeError("student encoder feature hook did not run")
        projected = self.carl_projection(student)
        logits = F.conv1d(projected, self.carl_classifier_weight, self.carl_classifier_bias)
        return torch.log_softmax(logits.transpose(1, 2), dim=-1)

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
        # CTC term: optionally normalized by an EMA label prior (Huang et al. 2024).
        # KD terms below always use the raw `log_probs`.
        ctc_log_probs = log_probs
        if self.label_prior_ctc_alpha > 0.0:
            with torch.no_grad():
                B, T, V = log_probs.shape
                p = log_probs.exp()
                valid = (torch.arange(T, device=log_probs.device).view(1, T) < enc_len.view(B, 1)).to(p.dtype)
                batch_prior = (p * valid.unsqueeze(-1)).sum(dim=(0, 1)) / valid.sum().clamp_min(1.0)
                m = self.label_prior_momentum
                self._label_prior.mul_(m).add_(batch_prior.to(self._label_prior.dtype), alpha=1.0 - m)
                self._label_prior.div_(self._label_prior.sum().clamp_min(1e-8))
            log_prior = self._label_prior.clamp_min(1e-8).log().to(log_probs.dtype)
            ctc_log_probs = torch.log_softmax(log_probs - self.label_prior_ctc_alpha * log_prior.view(1, 1, V), dim=-1)
        loss_ctc = self.loss(
            log_probs=ctc_log_probs,
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
            if "fkd_dense_probs" in batch:
                loss_kd = self._frame_dense_kd_loss(
                    log_probs, enc_len, batch["fkd_dense_probs"], batch["fkd_lens"])
            else:
                loss_kd = self._frame_logit_kd_loss(
                    log_probs, enc_len,
                    batch["fkd_ids"], batch["fkd_probs"], batch["fkd_lens"],
                )
            loss = self._mix(loss_ctc, loss_kd)
            if self.logit_kd_occ_weight > 0.0 or self.logit_kd_res_weight > 0.0:
                if "fkd_dense_probs" in batch:
                    raise ValueError("OCC/RES auxiliary losses are unsupported for dense frame targets")
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
            if "fkd_dense_probs" in batch:
                loss_kd = self._guided_dense_loss(
                    log_probs, enc_len, batch["fkd_dense_probs"], batch["fkd_lens"])
            else:
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
            if self.kd_start_step and self.global_step < self.kd_start_step:
                # CTC-only warm-up (needed in dual mode: early student gamma is noise)
                loss_kd = torch.zeros((), device=log_probs.device, dtype=loss_ctc.dtype)
                loss = loss_ctc
            else:
                student_sup = None
                if self.span_kd_dual:
                    student_sup = batched_ctc_token_gamma(
                        log_probs.detach(), enc_len,
                        batch["tokens"], batch["token_lens"],
                        self.blank_id, blank_penalty=self.span_kd_student_delta,
                    )
                if self.span_primary_mode == "legacy":
                    loss_kd = self._span_kd_loss(
                        log_probs, enc_len,
                        batch["span_support"], batch["span_avg_ids"], batch["span_avg_probs"],
                        batch["span_gates"], batch["span_num_tokens"], batch["span_teacher_frames"],
                        student_support=student_sup,
                    )
                else:
                    teacher_mass3 = None
                    if self.span_primary_mode in ("mass3", "gt_nt"):
                        if "span_mass3_probs" not in batch:
                            raise ValueError(
                                f"span_primary_mode={self.span_primary_mode} but batch lacks span_mass3_probs; "
                                "build targets with --with-mass3")
                        teacher_mass3 = batch["span_mass3_probs"]
                    loss_kd = self._span_coarse_loss(
                        log_probs, enc_len, batch["span_support"], batch["span_gates"],
                        batch["span_num_tokens"], batch["span_teacher_frames"],
                        batch["tokens"], batch["token_lens"],
                        teacher_mass3=teacher_mass3, student_support=student_sup)
                loss = self._mix(loss_ctc, loss_kd)
                if self.span_ntdk_weight > 0.0:
                    required = ("span_dark_ids", "span_dark_probs", "span_dark_tail_prob")
                    missing = [key for key in required if key not in batch]
                    if missing:
                        raise ValueError(
                            f"span_ntdk_weight={self.span_ntdk_weight} but batch lacks {missing}; "
                            "build targets with --dark-top-m")
                    loss_ntdk = self._span_ntdk_loss(
                        log_probs, enc_len, batch["span_support"],
                        batch["span_dark_ids"], batch["span_dark_probs"],
                        batch["span_dark_tail_prob"], batch["span_gates"],
                        batch["span_num_tokens"], batch["span_teacher_frames"],
                        batch["tokens"], batch["token_lens"], teacher_mass3=batch.get("span_mass3_probs"))
                    loss = loss + self.span_ntdk_weight * loss_ntdk
                    for k, v in self._span_ntdk_diag.items():
                        self.log(f"train/span_ntdk_{k}", v, on_step=True, on_epoch=True)
                if self.span_shoulder_weight > 0.0:
                    if "span_expanded_support" not in batch:
                        raise ValueError(
                            "span_shoulder_weight>0 but batch lacks span_expanded_support; "
                            "use a core+shoulder manifest")
                    loss_shoulder = self._span_shoulder_loss(
                        log_probs, enc_len, batch["span_support"],
                        batch["span_expanded_support"], batch["span_gates"],
                        batch["span_num_tokens"], batch["span_teacher_frames"],
                        batch["tokens"], batch["token_lens"])
                    loss = loss + self.span_shoulder_weight * loss_shoulder
                    for k, v in self._span_shoulder_diag.items():
                        self.log(f"train/span_shoulder_{k}", v, on_step=True, on_epoch=True)
                diag = getattr(self, "_span_diag", None)
                if diag is not None:
                    for k, v in diag.items():
                        self.log(f"train/span_{k}", v, on_step=True, on_epoch=True)
                primary_diag = getattr(self, "_span_primary_diag", None)
                if primary_diag is not None:
                    for k, v in primary_diag.items():
                        self.log(f"train/span_primary_{k}", v, on_step=True, on_epoch=True)

        elif self.kd_mode == "frame_dkd":
            loss_kd = self._frame_dkd_loss(
                log_probs, enc_len, batch["fkd_dense_probs"], batch["fkd_lens"])
            warm = min(float(self.current_epoch + 1) / max(self.frame_dkd_warmup_epochs, 1), 1.0)
            loss = loss_ctc + warm * loss_kd
            self.log("train/frame_dkd_warm", warm, on_step=False, on_epoch=True)

        elif self.kd_mode == "fpkd_dfkd":
            loss_kd = self._binary_blank_kl(
                log_probs, enc_len, batch["fkd_dense_probs"], batch["fkd_lens"],
                self.fpkd_temperature)
            loss = loss_kd

        elif self.kd_mode == "fpkd_frkd":
            loss_kd = self._fpkd_frkd_loss(
                enc_len, batch["carl_features"], batch["carl_teacher_frames"],
                batch["fkd_dense_probs"], batch["fkd_lens"])
            loss = loss_kd

        elif self.kd_mode == "fpkd_pkd":
            loss_kd, loss_bkl, loss_nbfkl = self._fpkd_fkl_loss(
                log_probs, enc_len, batch["fkd_dense_probs"], batch["fkd_lens"])
            loss = loss_ctc + loss_kd
            self.log("train/fpkd_bkl", loss_bkl, on_step=True, on_epoch=True)
            self.log("train/fpkd_nbfkl", loss_nbfkl, on_step=True, on_epoch=True)

        elif self.kd_mode == "carl_feature":
            loss_kd = self._feature_kd_loss(
                enc_len, batch["carl_features"], batch["carl_teacher_frames"])
            loss = loss_kd

        elif self.kd_mode == "carl":
            loss_be = self._carl_be_loss(
                log_probs, enc_len, batch["fkd_dense_probs"], batch["fkd_lens"])
            carl_lp = self._carl_projected_log_probs()
            loss_carl_ctc = self.loss(
                log_probs=carl_lp, targets=batch["tokens"], input_lengths=enc_len,
                target_lengths=batch["token_lens"])
            loss_carl_be = self._carl_be_loss(
                carl_lp, enc_len, batch["fkd_dense_probs"], batch["fkd_lens"])
            loss_kd = (self.carl_alpha * loss_be
                       + self.carl_gamma * loss_carl_ctc
                       + self.carl_lambda * loss_carl_be)
            loss = loss_ctc + loss_kd
            self.log("train/carl_be", loss_be, on_step=True, on_epoch=True)
            self.log("train/carl_ctc", loss_carl_ctc, on_step=True, on_epoch=True)
            self.log("train/carl_aux_be", loss_carl_be, on_step=True, on_epoch=True)

        elif self.kd_mode == "sctc":
            loss_kd = self._sctc_loss(
                log_probs, enc_len,
                batch["sctc_gamma"], batch["sctc_bpe_ids"],
                batch["sctc_num_tokens"], batch["sctc_teacher_frames"],
            )
            loss = self._mix(loss_ctc, loss_kd)

        elif self.kd_mode == "boundary_kd":
            loss_kd = self._boundary_kd_loss(log_probs, enc_len, batch)
            loss = self._mix(loss_ctc, loss_kd)
            diag = getattr(self, "_boundary_diag", None)
            if diag is not None:
                for k, v in diag.items():
                    self.log(f"train/bd_{k}", v, on_step=True, on_epoch=True)

        elif self.kd_mode == "free_emit":
            margin = curriculum_margin(
                int(self.current_epoch), self.free_emit_margin_start, self.free_emit_anneal_epochs)
            loss_kd, diag = free_emit_loss(
                log_probs, enc_len, batch["fe_starts"], batch["fe_ends"],
                batch["fe_token_ids"], batch["fe_types"], batch["fe_num_targets"],
                batch["fe_teacher_frames"], self.blank_id,
                sub_weight=self.free_emit_sub_weight,
                spread_weight=self.free_emit_spread_weight, margin=margin)
            loss = self._mix(loss_ctc, loss_kd)
            self.log("train/free_emit_margin", float(margin), on_step=False, on_epoch=True)
            for k, v in diag.items():
                self.log(f"train/free_emit_{k}", v, on_step=True, on_epoch=True)

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
