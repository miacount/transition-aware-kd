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
    ):
        self.manifest_path = Path(manifest_path)
        self.rows = read_manifest(manifest_path)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.require_transition = require_transition
        self.require_frame_kd = require_frame_kd
        if require_transition:
            self.rows = [r for r in self.rows if r.get("teacher_target")]
        if require_frame_kd:
            self.rows = [r for r in self.rows if r.get("teacher_frame_kd_path")]
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

        return sample


def collate_fn(batch):
    if not batch:
        raise ValueError("collate_fn received an empty batch")
    batch_size = len(batch)
    wav_max = max(x["wav"].numel() for x in batch)
    tok_max = max(x["tokens"].numel() for x in batch)
    has_transition = all("ttarget" in x for x in batch)
    has_frame_kd = all("fkd_ids" in x and "fkd_probs" in x for x in batch)

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
):
    dataset = ASRKDDataset(
        manifest_path,
        tokenizer,
        sample_rate=sample_rate,
        require_transition=require_transition,
        require_frame_kd=require_frame_kd,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
