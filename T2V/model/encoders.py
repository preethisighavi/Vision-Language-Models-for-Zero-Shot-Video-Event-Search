"""Frozen, pretrained encoders: Wan 2.1 video VAE and CLIP text encoder.

Only the denoiser (model/dit.py) is trained. A "dummy" backend for each lets the
training loop run on CPU without downloading weights (used by the smoke test).
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


class VideoVAE(nn.Module):
    """Wraps AutoencoderKLWan. Latents are normalised per channel to ~N(0,1)."""

    def __init__(self, repo: str = WAN_REPO, device: str = "cuda", dtype=torch.float32):
        super().__init__()
        from diffusers import AutoencoderKLWan

        self.vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=dtype).to(device).eval()
        self.vae.requires_grad_(False)
        cfg = self.vae.config
        self.register_buffer("mean", torch.tensor(cfg.latents_mean, device=device).view(1, -1, 1, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.latents_std, device=device).view(1, -1, 1, 1, 1))
        self.dtype = dtype
        self.device = device

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """video [B,3,T,H,W] in [-1,1] -> normalised latent [B,16,1+(T-1)/4,H/8,W/8]."""
        z = self.vae.encode(video.to(self.device, self.dtype)).latent_dist.mode()
        return (z - self.mean) / self.std

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """normalised latent -> video [B,3,T,H,W] in [-1,1]."""
        z = z.to(self.device, self.dtype) * self.std + self.mean
        return self.vae.decode(z).sample.clamp(-1, 1)


class TextEncoder(nn.Module):
    """Returns (tokens [B,L,D], mask [B,L] bool, pooled [B,D])."""

    def __init__(self, name: str = "openai/clip-vit-large-patch14", device: str = "cuda", max_length: int = 77):
        super().__init__()
        self.name = name
        self.device = device
        self.max_length = max_length
        if name == "dummy":
            self.dim = 768
            return
        from transformers import CLIPTextModel, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        self.model = CLIPTextModel.from_pretrained(name).to(device).eval()
        self.model.requires_grad_(False)
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, captions: list[str]):
        if self.name == "dummy":
            return self._dummy(captions)
        tok = self.tokenizer(
            captions, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        out = self.model(**tok)
        return out.last_hidden_state.float(), tok.attention_mask.bool(), out.pooler_output.float()

    def _dummy(self, captions: list[str]):
        """Deterministic pseudo-embeddings per caption (no downloads)."""
        L, D = 16, self.dim
        toks = []
        for c in captions:
            seed = int(hashlib.md5(c.encode()).hexdigest()[:8], 16)
            g = torch.Generator().manual_seed(seed)
            toks.append(torch.randn(L, D, generator=g))
        tokens = torch.stack(toks).to(self.device)
        mask = torch.ones(len(captions), L, dtype=torch.bool, device=self.device)
        return tokens, mask, tokens.mean(1)
