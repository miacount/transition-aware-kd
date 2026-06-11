"""NeMo CTC student model with clean KD modes.

Supported modes:
  none  : CTC only
  trans : CTC + blank-retained transition KD
  logit : CTC + frame-wise top-k teacher posterior KD
"""
import copy

import torch
from nemo.collections.asr.models import EncDecCTCModelBPE

from data import make_dataloader
from transition_kd_batched import transition_kd_loss_batched


class TransitionKDModel(EncDecCTCModelBPE):
    def __init__(self, cfg, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.kd_mode = cfg.get("kd_mode", "none")
        self.kd_weight = float(cfg.get("kd_weight", 0.0))
        self.logit_kd_temperature = float(cfg.get("logit_kd_temperature", 2.0))
        self.blank_id = self.decoder.num_classes_with_blank - 1
        if self.kd_mode not in ("none", "trans", "logit"):
            raise ValueError(f"unsupported kd_mode: {self.kd_mode}")
        if self.kd_mode != "none" and self.kd_weight == 0.0:
            raise ValueError(
                f"kd_mode='{self.kd_mode}' but kd_weight=0.0 — KD loss will be zero. "
                "Set kd_weight > 0 or set kd_mode='none'."
            )
        self._wer_accum = {}

    def _kd_mode(self):
        return getattr(self, "kd_mode", self._cfg.get("kd_mode", "none"))

    def _make_dataloader_from_cfg(
        self,
        cfg,
        shuffle=False,
        require_transition=False,
        require_frame_kd=False,
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
        )

    def setup_training_data(self, cfg):
        kd_mode = self._kd_mode()
        self._train_dl = self._make_dataloader_from_cfg(
            cfg,
            shuffle=cfg.shuffle,
            require_transition=(kd_mode == "trans"),
            require_frame_kd=(kd_mode == "logit"),
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
        ce = -(probs * selected_lp).sum(dim=-1)
        ce = ce * valid.to(ce.dtype)
        return ce.sum() / valid.sum().clamp_min(1).to(ce.dtype)

    def training_step(self, batch, batch_idx):
        log_probs, enc_len, _ = self.forward(
            input_signal=batch["wavs"], input_signal_length=batch["wav_lens"]
        )
        loss_ctc = self.loss(
            log_probs=log_probs,
            targets=batch["tokens"],
            input_lengths=enc_len,
            target_lengths=batch["token_lens"],
        )

        if self.kd_mode == "none":
            loss_kd = torch.zeros((), device=log_probs.device, dtype=loss_ctc.dtype)
        elif self.kd_mode == "trans":
            loss_kd = transition_kd_loss_batched(
                log_probs,
                batch["ttargets"],
                input_lengths=enc_len,
                target_lengths=batch["ttarget_lens"],
            ).mean()
        elif self.kd_mode == "logit":
            loss_kd = self._frame_logit_kd_loss(
                log_probs,
                enc_len,
                batch["fkd_ids"],
                batch["fkd_probs"],
                batch["fkd_lens"],
            )

        loss = loss_ctc if self.kd_mode == "none" else loss_ctc + self.kd_weight * loss_kd
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/ctc_loss", loss_ctc, on_step=True, on_epoch=True)
        if self.kd_mode != "none":
            self.log("train/kd_loss", loss_kd, prog_bar=True, on_step=True, on_epoch=True)
        return loss

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
        # NeMo WER.update() overwrites state (not accumulates), so compute+reset per batch
        # and accumulate numerator/denominator manually.
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
