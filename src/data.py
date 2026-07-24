"""Dataset/collate utilities for CTC KD experiments."""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


def read_manifest(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class ASRKDDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        tokenizer,
        sample_rate=16000,
        require_transition=False,
        require_frame_kd=False,
        require_token_avg_kd=False,
        require_span_kd=False,
        require_sctc=False,
        require_combined_kd=False,
        require_boundary_kd=False,
    ):
        self.manifest_path = Path(manifest_path)
        self.rows = read_manifest(manifest_path)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.require_transition = require_transition
        self.require_frame_kd = require_frame_kd or require_combined_kd
        self.require_token_avg_kd = require_token_avg_kd or require_combined_kd
        self.require_span_kd = require_span_kd
        self.require_sctc = require_sctc
        self.require_boundary_kd = require_boundary_kd
        if require_transition:
            self.rows = [r for r in self.rows if r.get("teacher_target")]
        if self.require_frame_kd:
            self.rows = [r for r in self.rows if r.get("teacher_frame_kd_path")]
        if self.require_token_avg_kd:
            self.rows = [r for r in self.rows if r.get("teacher_token_avg_path")]
        if self.require_span_kd:
            self.rows = [r for r in self.rows if r.get("teacher_span_kd_path")]
        if self.require_sctc:
            self.rows = [r for r in self.rows if r.get("teacher_sctc_path")]
        if self.require_boundary_kd:
            self.rows = [r for r in self.rows if r.get("boundary_kd_path")]
        if not self.rows:
            raise ValueError(f"no usable rows in manifest: {manifest_path}")

    def __len__(self):
        return len(self.rows)

    def _load_audio(self, path):
        import soundfile as sf

        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        return torch.from_numpy(wav)

    def __getitem__(self, idx):
        row = self.rows[idx]
        wav = self._load_audio(row["audio_filepath"])
        tokens = torch.tensor(self.tokenizer.text_to_ids(row["text"]), dtype=torch.long)
        sample = {"wav": wav, "tokens": tokens}

        if row.get("teacher_target"):
            sample["ttarget"] = torch.tensor(row["teacher_target"], dtype=torch.long)
        elif self.require_transition:
            raise ValueError(f"missing teacher_target: {row['audio_filepath']}")

        if row.get("teacher_frame_kd_path"):
            kd_path = Path(row["teacher_frame_kd_path"])
            if not kd_path.is_absolute():
                kd_path = self.manifest_path.parent / kd_path
            kd = torch.load(kd_path, map_location="cpu", weights_only=False)
            sample["fkd_ids"] = kd["ids"].long()
            sample["fkd_probs"] = kd["probs"].float()
        elif self.require_frame_kd:
            raise ValueError(f"missing teacher_frame_kd_path: {row['audio_filepath']}")

        if row.get("teacher_token_avg_path"):
            ta_path = Path(row["teacher_token_avg_path"])
            if not ta_path.is_absolute():
                ta_path = self.manifest_path.parent / ta_path
            ta = torch.load(ta_path, map_location="cpu", weights_only=False)
            sample["tavg_seg_starts"]     = ta["seg_starts"].long()
            sample["tavg_seg_ends"]       = ta["seg_ends"].long()
            sample["tavg_avg_ids"]        = ta["avg_ids"].long()
            sample["tavg_avg_probs"]      = ta["avg_probs"].float()
            sample["tavg_teacher_frames"] = int(ta["teacher_frames"])
            if "trans_left_ids" in ta:
                sample["tavg_trans_left_ids"]   = ta["trans_left_ids"].long()
                sample["tavg_trans_left_probs"] = ta["trans_left_probs"].float()
                sample["tavg_trans_left_valid"] = ta["trans_left_valid"]
                sample["tavg_trans_right_ids"]   = ta["trans_right_ids"].long()
                sample["tavg_trans_right_probs"] = ta["trans_right_probs"].float()
                sample["tavg_trans_right_valid"] = ta["trans_right_valid"]
        elif self.require_token_avg_kd:
            raise ValueError(f"missing teacher_token_avg_path: {row['audio_filepath']}")

        if row.get("teacher_span_kd_path"):
            sp_path = Path(row["teacher_span_kd_path"])
            if not sp_path.is_absolute():
                sp_path = self.manifest_path.parent / sp_path
            sp = torch.load(sp_path, map_location="cpu", weights_only=False)
            sample["span_support"] = sp["support"].float()
            sample["span_avg_ids"] = sp["avg_ids"].long()
            sample["span_avg_probs"] = sp["avg_probs"].float()
            sample["span_gates"] = sp["gates"].float()
            sample["span_teacher_frames"] = int(sp["teacher_frames"])
        elif self.require_span_kd:
            raise ValueError(f"missing teacher_span_kd_path: {row['audio_filepath']}")

        if row.get("teacher_sctc_path"):
            sc_path = Path(row["teacher_sctc_path"])
            if not sc_path.is_absolute():
                sc_path = self.manifest_path.parent / sc_path
            sc = torch.load(sc_path, map_location="cpu", weights_only=False)
            sample["sctc_gamma"] = sc["token_gamma"].float()      # (N, T)
            sample["sctc_bpe_ids"] = sc["bpe_ids"].long()         # (N,)
            sample["sctc_teacher_frames"] = int(sc["teacher_frames"])
        elif self.require_sctc:
            raise ValueError(f"missing teacher_sctc_path: {row['audio_filepath']}")

        if row.get("boundary_kd_path"):
            bd_path = Path(row["boundary_kd_path"])
            if not bd_path.is_absolute():
                bd_path = self.manifest_path.parent / bd_path
            bd = torch.load(bd_path, map_location="cpu", weights_only=False)
            sample["bd_res_t"] = bd["res_t"].long()      # (R,)
            sample["bd_res_u"] = bd["res_u"].long()      # (R,)
            sample["bd_res_r"] = bd["res_r"].float()     # (R,)
            sample["bd_m_delta"] = bd["m_delta"].float()  # (T,)
            sample["bd_m_zero"] = bd["m_zero"].float()    # (T,)
            sample["bd_spike"] = bd["spike"].long()       # (N,)
            sample["bd_y"] = bd["y"].long()               # (N,)
            sample["bd_teacher_frames"] = int(bd["teacher_frames"])
            sample["bd_num_tokens"] = int(bd["num_tokens"])
            soft_rel = row.get("boundary_kd_soft_path")
            if soft_rel:  # cell 1 only
                sp = Path(soft_rel)
                if not sp.is_absolute():
                    sp = self.manifest_path.parent / sp
                sd = torch.load(sp, map_location="cpu", weights_only=False)
                sample["bd_soft_t"] = sd["soft_t"].long()     # (F,)
                sample["bd_soft_p"] = sd["soft_p"].float()    # (F,V)
        elif self.require_boundary_kd:
            raise ValueError(f"missing boundary_kd_path: {row['audio_filepath']}")

        return sample


def collate_fn(batch):
    if not batch:
        raise ValueError("collate_fn received an empty batch")
    batch_size = len(batch)
    wav_max = max(x["wav"].numel() for x in batch)
    tok_max = max(x["tokens"].numel() for x in batch)
    has_transition = all("ttarget" in x for x in batch)
    has_frame_kd = all("fkd_ids" in x and "fkd_probs" in x for x in batch)
    has_token_avg = all("tavg_seg_starts" in x for x in batch)
    has_span_kd = all("span_support" in x for x in batch)
    has_sctc = all("sctc_gamma" in x for x in batch)
    has_boundary_kd = all("bd_res_t" in x for x in batch)
    has_boundary_soft = has_boundary_kd and all("bd_soft_p" in x for x in batch)

    wavs = torch.zeros(batch_size, wav_max, dtype=torch.float32)
    wav_lens = torch.zeros(batch_size, dtype=torch.long)
    tokens = torch.zeros(batch_size, tok_max, dtype=torch.long)
    token_lens = torch.zeros(batch_size, dtype=torch.long)

    out = {
        "wavs": wavs,
        "wav_lens": wav_lens,
        "tokens": tokens,
        "token_lens": token_lens,
    }

    if has_transition:
        tgt_max = max(x["ttarget"].numel() for x in batch)
        out["ttargets"] = torch.zeros(batch_size, tgt_max, dtype=torch.long)
        out["ttarget_lens"] = torch.zeros(batch_size, dtype=torch.long)

    if has_frame_kd:
        frame_max = max(x["fkd_ids"].shape[0] for x in batch)
        topk_max = max(x["fkd_ids"].shape[1] for x in batch)
        out["fkd_ids"] = torch.zeros(batch_size, frame_max, topk_max, dtype=torch.long)
        out["fkd_probs"] = torch.zeros(batch_size, frame_max, topk_max, dtype=torch.float32)
        out["fkd_lens"] = torch.zeros(batch_size, dtype=torch.long)

    has_trans_kd = has_token_avg and all("tavg_trans_left_ids" in x for x in batch)
    if has_token_avg:
        seg_max = max(x["tavg_seg_starts"].numel() for x in batch)
        topk_max = max(x["tavg_avg_ids"].shape[1] for x in batch)
        out["tavg_seg_starts"]     = torch.zeros(batch_size, seg_max, dtype=torch.long)
        out["tavg_seg_ends"]       = torch.zeros(batch_size, seg_max, dtype=torch.long)
        out["tavg_avg_ids"]        = torch.zeros(batch_size, seg_max, topk_max, dtype=torch.long)
        out["tavg_avg_probs"]      = torch.zeros(batch_size, seg_max, topk_max, dtype=torch.float32)
        out["tavg_num_segs"]       = torch.zeros(batch_size, dtype=torch.long)
        out["tavg_teacher_frames"] = torch.zeros(batch_size, dtype=torch.long)
    if has_trans_kd:
        out["tavg_trans_left_ids"]   = torch.zeros(batch_size, seg_max, topk_max, dtype=torch.long)
        out["tavg_trans_left_probs"] = torch.zeros(batch_size, seg_max, topk_max, dtype=torch.float32)
        out["tavg_trans_left_valid"] = torch.zeros(batch_size, seg_max, dtype=torch.bool)
        out["tavg_trans_right_ids"]   = torch.zeros(batch_size, seg_max, topk_max, dtype=torch.long)
        out["tavg_trans_right_probs"] = torch.zeros(batch_size, seg_max, topk_max, dtype=torch.float32)
        out["tavg_trans_right_valid"] = torch.zeros(batch_size, seg_max, dtype=torch.bool)

    if has_span_kd:
        span_n_max = max(x["span_support"].shape[0] for x in batch)
        span_t_max = max(x["span_support"].shape[1] for x in batch)
        span_topk_max = max(x["span_avg_ids"].shape[1] for x in batch)
        out["span_support"] = torch.zeros(batch_size, span_n_max, span_t_max, dtype=torch.float32)
        out["span_avg_ids"] = torch.zeros(batch_size, span_n_max, span_topk_max, dtype=torch.long)
        out["span_avg_probs"] = torch.zeros(batch_size, span_n_max, span_topk_max, dtype=torch.float32)
        out["span_gates"] = torch.zeros(batch_size, span_n_max, dtype=torch.float32)
        out["span_num_tokens"] = torch.zeros(batch_size, dtype=torch.long)
        out["span_teacher_frames"] = torch.zeros(batch_size, dtype=torch.long)

    if has_sctc:
        sctc_n_max = max(x["sctc_gamma"].shape[0] for x in batch)
        sctc_t_max = max(x["sctc_gamma"].shape[1] for x in batch)
        out["sctc_gamma"] = torch.zeros(batch_size, sctc_n_max, sctc_t_max, dtype=torch.float32)
        out["sctc_bpe_ids"] = torch.zeros(batch_size, sctc_n_max, dtype=torch.long)
        out["sctc_num_tokens"] = torch.zeros(batch_size, dtype=torch.long)
        out["sctc_teacher_frames"] = torch.zeros(batch_size, dtype=torch.long)

    if has_boundary_kd:
        bd_r_max = max(x["bd_res_t"].numel() for x in batch)
        bd_n_max = max(x["bd_num_tokens"] for x in batch)
        bd_t_max = max(x["bd_teacher_frames"] for x in batch)
        out["bd_res_t"] = torch.zeros(batch_size, bd_r_max, dtype=torch.long)
        out["bd_res_u"] = torch.zeros(batch_size, bd_r_max, dtype=torch.long)
        out["bd_res_r"] = torch.zeros(batch_size, bd_r_max, dtype=torch.float32)
        out["bd_res_valid"] = torch.zeros(batch_size, bd_r_max, dtype=torch.bool)
        out["bd_m_delta"] = torch.zeros(batch_size, bd_t_max, dtype=torch.float32)
        out["bd_m_zero"] = torch.zeros(batch_size, bd_t_max, dtype=torch.float32)
        out["bd_spike"] = torch.zeros(batch_size, bd_n_max, dtype=torch.long)
        out["bd_y"] = torch.zeros(batch_size, bd_n_max, dtype=torch.long)
        out["bd_num_tokens"] = torch.zeros(batch_size, dtype=torch.long)
        out["bd_teacher_frames"] = torch.zeros(batch_size, dtype=torch.long)
        if has_boundary_soft:
            V = batch[0]["bd_soft_p"].shape[1]
            bd_f_max = max(x["bd_soft_t"].numel() for x in batch)
            out["bd_soft_t"] = torch.full((batch_size, bd_f_max), -1, dtype=torch.long)
            out["bd_soft_p"] = torch.zeros(batch_size, bd_f_max, V, dtype=torch.float32)

    for i, sample in enumerate(batch):
        wav = sample["wav"]
        tok = sample["tokens"]
        wavs[i, : wav.numel()] = wav
        wav_lens[i] = wav.numel()
        tokens[i, : tok.numel()] = tok
        token_lens[i] = tok.numel()

        if has_transition:
            tgt = sample["ttarget"]
            out["ttargets"][i, : tgt.numel()] = tgt
            out["ttarget_lens"][i] = tgt.numel()

        if has_frame_kd:
            ids = sample["fkd_ids"]
            probs = sample["fkd_probs"]
            frames, topk = ids.shape
            out["fkd_ids"][i, :frames, :topk] = ids
            out["fkd_probs"][i, :frames, :topk] = probs
            out["fkd_lens"][i] = frames

        if has_token_avg:
            n = sample["tavg_seg_starts"].numel()
            tk = sample["tavg_avg_ids"].shape[1]
            out["tavg_seg_starts"][i, :n]      = sample["tavg_seg_starts"]
            out["tavg_seg_ends"][i, :n]        = sample["tavg_seg_ends"]
            out["tavg_avg_ids"][i, :n, :tk]    = sample["tavg_avg_ids"]
            out["tavg_avg_probs"][i, :n, :tk]  = sample["tavg_avg_probs"]
            out["tavg_num_segs"][i]            = n
            out["tavg_teacher_frames"][i]      = sample["tavg_teacher_frames"]
        if has_trans_kd:
            out["tavg_trans_left_ids"][i, :n, :tk]   = sample["tavg_trans_left_ids"]
            out["tavg_trans_left_probs"][i, :n, :tk] = sample["tavg_trans_left_probs"]
            out["tavg_trans_left_valid"][i, :n]      = sample["tavg_trans_left_valid"]
            out["tavg_trans_right_ids"][i, :n, :tk]   = sample["tavg_trans_right_ids"]
            out["tavg_trans_right_probs"][i, :n, :tk] = sample["tavg_trans_right_probs"]
            out["tavg_trans_right_valid"][i, :n]      = sample["tavg_trans_right_valid"]

        if has_span_kd:
            sn, sf = sample["span_support"].shape
            stk = sample["span_avg_ids"].shape[1]
            out["span_support"][i, :sn, :sf] = sample["span_support"]
            out["span_avg_ids"][i, :sn, :stk] = sample["span_avg_ids"]
            out["span_avg_probs"][i, :sn, :stk] = sample["span_avg_probs"]
            out["span_gates"][i, :sn] = sample["span_gates"]
            out["span_num_tokens"][i] = sn
            out["span_teacher_frames"][i] = sample["span_teacher_frames"]

        if has_sctc:
            gn, gt = sample["sctc_gamma"].shape
            out["sctc_gamma"][i, :gn, :gt] = sample["sctc_gamma"]
            out["sctc_bpe_ids"][i, :gn] = sample["sctc_bpe_ids"]
            out["sctc_num_tokens"][i] = gn
            out["sctc_teacher_frames"][i] = sample["sctc_teacher_frames"]

        if has_boundary_kd:
            rn = sample["bd_res_t"].numel()
            tt = sample["bd_teacher_frames"]
            nn = sample["bd_num_tokens"]
            out["bd_res_t"][i, :rn] = sample["bd_res_t"]
            out["bd_res_u"][i, :rn] = sample["bd_res_u"]
            out["bd_res_r"][i, :rn] = sample["bd_res_r"]
            out["bd_res_valid"][i, :rn] = True
            out["bd_m_delta"][i, :tt] = sample["bd_m_delta"]
            out["bd_m_zero"][i, :tt] = sample["bd_m_zero"]
            out["bd_spike"][i, :nn] = sample["bd_spike"]
            out["bd_y"][i, :nn] = sample["bd_y"]
            out["bd_num_tokens"][i] = nn
            out["bd_teacher_frames"][i] = tt
            if has_boundary_soft:
                fn = sample["bd_soft_t"].numel()
                out["bd_soft_t"][i, :fn] = sample["bd_soft_t"]
                out["bd_soft_p"][i, :fn] = sample["bd_soft_p"]

    return out


def make_dataloader(
    manifest_path,
    tokenizer,
    batch_size,
    shuffle,
    sample_rate=16000,
    num_workers=4,
    pin_memory=True,
    require_transition=False,
    require_frame_kd=False,
    require_token_avg_kd=False,
    require_span_kd=False,
    require_sctc=False,
    require_combined_kd=False,
    require_boundary_kd=False,
):
    dataset = ASRKDDataset(
        manifest_path,
        tokenizer,
        sample_rate=sample_rate,
        require_transition=require_transition,
        require_frame_kd=require_frame_kd,
        require_token_avg_kd=require_token_avg_kd,
        require_span_kd=require_span_kd,
        require_sctc=require_sctc,
        require_combined_kd=require_combined_kd,
        require_boundary_kd=require_boundary_kd,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
