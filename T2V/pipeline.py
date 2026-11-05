"""SpartanGen inference: prompt -> frames. Shared by training previews, sample.py, serve.py and eval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from T2V.model.dit import SpartanDiT, build_model
from T2V.model.encoders import TextEncoder, VideoVAE
from T2V.model.flow import sample
from T2V.utils.video_io import to_uint8


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SpartanPipeline:
    def __init__(self, model: SpartanDiT, text_encoder: TextEncoder, vae: VideoVAE | None, cfg: dict, device: str):
        self.model, self.text_encoder, self.vae = model.eval(), text_encoder, vae
        self.cfg, self.device = cfg, device
        self._uncond = None

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path, device: str | None = None, use_ema: bool = True):
        device = device or default_device()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        te = TextEncoder(cfg["text_encoder"], device=device)
        model = build_model(cfg, te.dim)
        model.load_state_dict(ckpt["ema" if use_ema and "ema" in ckpt else "model"])
        model.to(device)
        vae = VideoVAE(cfg["vae"], device=device) if cfg.get("vae") else None
        return cls(model, te, vae, cfg, device)

    def _cond(self, prompts: list[str]) -> dict:
        text, mask, pooled = self.text_encoder(prompts)
        return {"text": text, "text_mask": mask, "pooled": pooled}

    @torch.no_grad()
    def latents(self, prompts: list[str], seed: int = 0, steps=None, cfg=None, shift=None, frames: int | None = None):
        s = self.cfg["sample"]
        c, t, h, w = self.cfg["data"]["latent_shape"]
        if frames is not None:
            t = frames
        cond = self._cond(prompts)
        if self._uncond is None:
            self._uncond = self._cond([""])
        uncond = {k: v.expand(len(prompts), *v.shape[1:]) for k, v in self._uncond.items()}
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        return sample(
            self.model,
            (len(prompts), c, t, h, w),
            cond,
            uncond,
            steps=steps or s["steps"],
            cfg=s["cfg"] if cfg is None else cfg,
            shift=shift or s.get("shift", 1.0),
            generator=torch.Generator().manual_seed(seed),
            device=self.device,
            dtype=dtype,
        )

    @torch.no_grad()
    def generate(self, prompts: list[str], seed: int = 0, **kw) -> list[np.ndarray]:
        """Returns one uint8 [T,H,W,3] array per prompt."""
        if self.vae is None:
            raise RuntimeError("config has no VAE; cannot decode to pixels")
        z = self.latents(prompts, seed=seed, **kw)
        out = []
        for zi in z:  # one clip at a time: decoding a whole batch runs out of memory on laptops (MPS)
            video = self.vae.decode(zi[None]).float().cpu()[0]  # 3,T,H,W in [-1,1]
            out.append(to_uint8(video.permute(1, 2, 3, 0).numpy()))
            if self.device == "mps":
                torch.mps.empty_cache()
        return out

    @property
    def fps(self) -> float:
        return float(self.cfg["data"].get("fps", 3))


def tile(videos: list[np.ndarray], cols: int = 2) -> np.ndarray:
    """Arrange equally sized [T,H,W,3] videos in a grid."""
    rows = (len(videos) + cols - 1) // cols
    blank = np.zeros_like(videos[0])
    padded = videos + [blank] * (rows * cols - len(videos))
    grid_rows = [np.concatenate(padded[r * cols : (r + 1) * cols], axis=2) for r in range(rows)]
    return np.concatenate(grid_rows, axis=1)
