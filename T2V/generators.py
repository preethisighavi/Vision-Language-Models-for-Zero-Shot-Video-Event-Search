"""Common interface over the two generators so serve.py and eval treat them the same.

    gen = get_generator("spartangen-tiny")   # or "wan-1.3b"
    gen.generate("a dog runs on the beach", "out.mp4", seed=0)
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from T2V.utils.video_io import write_mp4

WAN_NEGATIVE = (
    "blurry, low quality, static, still image, worst quality, jpeg artifacts, deformed, "
    "disfigured, extra fingers, watermark, subtitles, text"
)


class Generator:
    id: str
    label: str
    fps: float

    def frames(self, prompt: str, seed: int = 0, **kw) -> np.ndarray:  # uint8 [T,H,W,3]
        raise NotImplementedError

    def generate(self, prompt: str, out_path: str | Path, seed: int = 0, **kw) -> dict:
        f = self.frames(prompt, seed=seed, **kw)
        write_mp4(f, out_path, fps=self.fps)
        return {"model": self.id, "prompt": prompt, "seed": seed, "frames": int(f.shape[0]),
                "height": int(f.shape[1]), "width": int(f.shape[2]), "fps": self.fps}


class SpartanGenerator(Generator):
    id, label = "spartangen-tiny", "SpartanGen-Tiny (ours, 128px)"

    def __init__(self, ckpt: str | None = None, device: str | None = None):
        from T2V.pipeline import SpartanPipeline

        ckpt = ckpt or os.environ.get("SPARTANGEN_CKPT", "T2V/runs/tiny_video/last.pt")
        self.pipe = SpartanPipeline.from_checkpoint(ckpt, device=device)
        self.fps = self.pipe.fps

    def frames(self, prompt, seed=0, steps=None, cfg=None, **_):
        return self.pipe.generate([prompt], seed=seed, steps=steps, cfg=cfg)[0]


class WanGenerator(Generator):
    """Wan 2.1 T2V 1.3B through diffusers (pretrained baseline)."""

    id, label = "wan-1.3b", "Wan 2.1 T2V 1.3B (pretrained baseline)"
    fps = 16.0

    def __init__(self, repo: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", device: str = "cuda",
                 height: int = 480, width: int = 832, num_frames: int = 33):
        import torch
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline

        vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(repo, vae=vae, torch_dtype=torch.bfloat16)
        # flow_shift 3.0 is the recommended value for 480p
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
        self.pipe = pipe.to(device)
        self.device = device
        self.size = (height, width, num_frames)

    def frames(self, prompt, seed=0, steps=None, cfg=None, num_frames=None, height=None, width=None, **_):
        import torch

        h, w, n = height or self.size[0], width or self.size[1], num_frames or self.size[2]
        out = self.pipe(
            prompt=prompt, negative_prompt=WAN_NEGATIVE, height=h, width=w, num_frames=n,
            num_inference_steps=steps or 30, guidance_scale=5.0 if cfg is None else cfg,
            generator=torch.Generator(device=self.device).manual_seed(seed), output_type="np",
        ).frames[0]
        return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


REGISTRY = {SpartanGenerator.id: SpartanGenerator, WanGenerator.id: WanGenerator}
_cache: dict[str, Generator] = {}
_lock = threading.Lock()


def get_generator(model_id: str) -> Generator:
    if model_id not in REGISTRY:
        raise KeyError(f"unknown model {model_id!r}; choose from {sorted(REGISTRY)}")
    with _lock:
        if model_id not in _cache:
            _cache[model_id] = REGISTRY[model_id]()
        return _cache[model_id]
