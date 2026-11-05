"""Encode every clip once with the frozen Wan VAE and cache the latents.

Training then never touches pixels, which is what makes a 2-week schedule
feasible on one GPU.

Output: <out-dir>/<split>/shard_XXXXX.pt, each a dict with
  video_ids   list[str]
  captions    list[list[str]]
  latents     fp16 [N, 16, 1+(F-1)/4, R/8, R/8]   (video clip, F frames at `fps`)
  images      fp16 [N, K, 16, 1, R/8, R/8]         (K single frames, for stage-A image training)
and <out-dir>/meta.json.

Usage:
  python -m T2V.data.prepare_latents --manifest T2V/data/manifest.parquet --out-dir /data/t2v/latents
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from T2V.model.encoders import WAN_REPO, VideoVAE
from T2V.utils.video_io import read_frames, sample_clip, uniform_frames


def resize_crop(frames: np.ndarray, res: int) -> torch.Tensor:
    """uint8 [T,H,W,3] -> float [3,T,res,res] in [-1,1] (short side resize + centre crop)."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # T,3,H,W
    h, w = x.shape[-2:]
    s = res / min(h, w)
    nh, nw = max(res, round(h * s)), max(res, round(w * s))
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)
    top, left = (nh - res) // 2, (nw - res) // 2
    x = x[..., top : top + res, left : left + res]
    return (x / 127.5 - 1).clamp(-1, 1).permute(1, 0, 2, 3).contiguous()


class ClipDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, num_frames: int, fps: float, res: int, image_frames: int):
        self.rows = rows.reset_index(drop=True)
        self.num_frames, self.fps, self.res, self.k = num_frames, fps, res, image_frames

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        try:
            frames, times = read_frames(r["path"])
            if len(frames) > 1 and times[-1] > 0:
                src_fps = (len(frames) - 1) / times[-1]
                if src_fps < 0.9 * self.fps:  # sampling faster than the source repeats frames -> choppy motion
                    print(f"[prepare] warning: {r['video_id']} is {src_fps:.1f} fps, below --fps {self.fps}; "
                          f"frames will repeat")
            clip = resize_crop(sample_clip(frames, times, self.num_frames, self.fps), self.res)
            imgs = resize_crop(uniform_frames(frames, self.k), self.res)  # [3,K,R,R]
        except Exception as e:  # corrupt or unreadable video: skip it
            print(f"[prepare] skip {r['video_id']}: {e}")
            return None
        return r["video_id"], list(r["captions"]), clip, imgs


def _collate(batch):
    return [b for b in batch if b is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--num-frames", type=int, default=17, help="must be 4k+1 for the Wan VAE")
    ap.add_argument("--fps", type=float, default=3.0,
                    help="sampling rate; the team's MSR-VTT/MSVD copies (VLM2Vec) are stored at 3 fps")
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--image-frames", type=int, default=4)
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="max clips per split (debugging)")
    ap.add_argument("--vae", default=WAN_REPO)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    assert (a.num_frames - 1) % 4 == 0, "--num-frames must be 4k+1"

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(a.manifest)
    vae = VideoVAE(a.vae, device=a.device)

    meta = {
        "vae": a.vae,
        "num_frames": a.num_frames,
        "fps": a.fps,
        "resolution": a.resolution,
        "image_frames": a.image_frames,
        "latent_shape": [16, 1 + (a.num_frames - 1) // 4, a.resolution // 8, a.resolution // 8],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    for split in a.splits:
        rows = manifest[manifest["split"] == split]
        if a.limit:
            rows = rows.head(a.limit)
        split_dir = out / split
        split_dir.mkdir(exist_ok=True)
        n_shards = (len(rows) + a.shard_size - 1) // a.shard_size
        print(f"[prepare] {split}: {len(rows)} clips -> {n_shards} shards")

        for s in range(n_shards):
            path = split_dir / f"shard_{s:05d}.pt"
            if path.exists():  # resumable
                continue
            part = rows.iloc[s * a.shard_size : (s + 1) * a.shard_size]
            ds = ClipDataset(part, a.num_frames, a.fps, a.resolution, a.image_frames)
            dl = DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers, collate_fn=_collate)
            ids, caps, lat, img = [], [], [], []
            for batch in dl:
                if not batch:
                    continue
                vid, cap, clip, imgs = zip(*batch)
                clip = torch.stack(clip)  # B,3,F,R,R
                imgs = torch.stack(imgs)  # B,3,K,R,R
                b, _, k, r, _ = imgs.shape
                lat.append(vae.encode(clip).half().cpu())
                flat = imgs.permute(0, 2, 1, 3, 4).reshape(b * k, 3, 1, r, r)
                img.append(vae.encode(flat).half().cpu().view(b, k, 16, 1, r // 8, r // 8))
                ids += vid
                caps += cap
            torch.save(
                {"video_ids": ids, "captions": caps, "latents": torch.cat(lat), "images": torch.cat(img)}, path
            )
            print(f"[prepare] wrote {path} ({len(ids)} clips)")


if __name__ == "__main__":
    main()
