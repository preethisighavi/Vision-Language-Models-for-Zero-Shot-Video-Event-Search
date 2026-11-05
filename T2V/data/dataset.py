"""In-memory dataset over cached latents (the whole MSR-VTT+MSVD cache is < 1 GB)."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    """stage="image": one random single-frame latent [C,1,h,w] per item (stage A).
    stage="video":  the clip latent [C,T,h,w] (stage B).
    Each item also returns one caption picked at random from the clip's captions."""

    def __init__(self, latent_dir: str | Path, split: str = "train", stage: str = "video"):
        shards = sorted((Path(latent_dir) / split).glob("shard_*.pt"))
        if not shards:
            raise FileNotFoundError(f"no shards in {Path(latent_dir) / split}; run T2V.data.prepare_latents")
        self.stage = stage
        self.ids, self.captions, lat, img = [], [], [], []
        for p in shards:
            d = torch.load(p, map_location="cpu")
            self.ids += d["video_ids"]
            self.captions += d["captions"]
            lat.append(d["latents"])
            img.append(d["images"])
        self.latents = torch.cat(lat)
        self.images = torch.cat(img)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        cap = random.choice(self.captions[i])
        if self.stage == "image":
            k = random.randrange(self.images.shape[1])
            return self.images[i, k].float(), cap
        return self.latents[i].float(), cap


def collate(batch):
    x, caps = zip(*batch)
    return torch.stack(x), list(caps)


class SyntheticLatents(Dataset):
    """Random latents + fixed captions, for CPU smoke tests without data or downloads."""

    def __init__(self, n: int = 64, shape=(16, 5, 16, 16), stage: str = "video"):
        g = torch.Generator().manual_seed(0)
        if stage == "image":
            shape = (shape[0], 1, *shape[2:])
        c, t, h, w = shape
        # smooth (low-frequency) latents, closer to real VAE latents than white noise
        coarse = torch.randn(n, c, t, max(1, h // 4), max(1, w // 4), generator=g)
        x = torch.nn.functional.interpolate(coarse, size=(t, h, w), mode="trilinear", align_corners=False)
        self.x = (x - x.mean()) / x.std()
        self.caps = [f"synthetic clip number {i}" for i in range(n)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.caps[i]
