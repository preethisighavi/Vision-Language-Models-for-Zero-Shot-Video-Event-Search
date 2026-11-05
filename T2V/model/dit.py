"""SpartanGen DiT: a small video diffusion transformer trained from scratch.

Design (see T2V/README.md for paper references):
- 1x2x2 patchify of the VAE latent -> tokens laid out as [B, T, S=h*w, D]
- per block: spatial self-attn (2D RoPE) -> temporal self-attn (1D RoPE, tanh gate
  initialised to 0, Flamingo-style, so an image-pretrained model starts unchanged)
  -> text cross-attn -> MLP
- adaLN-Zero conditioning on (timestep + pooled text); QK RMSNorm for stability
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ── rotary position embeddings ───────────────────────────────────────────────

def _rope_angles(pos: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device, dtype=torch.float32) / dim))
    return pos.float()[:, None] * inv[None, :]  # [L, dim/2]


def rope_1d(length: int, head_dim: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    ang = _rope_angles(torch.arange(length, device=device), head_dim)
    return ang.cos(), ang.sin()


def rope_2d(h: int, w: int, head_dim: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """First half of the rotary pairs encodes the row, second half the column."""
    ah = _rope_angles(torch.arange(h, device=device), head_dim // 2)  # [h, hd/4]
    aw = _rope_angles(torch.arange(w, device=device), head_dim // 2)  # [w, hd/4]
    ang = torch.cat([ah[:, None, :].expand(h, w, -1), aw[None, :, :].expand(h, w, -1)], dim=-1)
    ang = ang.reshape(h * w, -1)  # [h*w, hd/2]
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [N, heads, L, hd]; cos/sin [L, hd/2]."""
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return out.type_as(x)


# ── building blocks ──────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_dim: int | None = None):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(kv_dim or dim, 2 * dim)
        self.o = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def forward(self, x, ctx=None, mask=None, rope=None):
        n, l, d = x.shape
        ctx = x if ctx is None else ctx
        s = ctx.shape[1]
        q = self.q(x).view(n, l, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(ctx).view(n, s, 2, self.heads, self.head_dim).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(n, l, d))


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn_s = Attention(dim, heads)
        self.norm_t = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn_t = Attention(dim, heads)
        self.temporal_gate = nn.Parameter(torch.zeros(1))
        self.norm_x = nn.LayerNorm(dim, eps=1e-6)
        self.attn_x = Attention(dim, heads)
        self.norm_m = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        # shift/scale/gate (spatial), shift/scale (temporal), shift/scale/gate (mlp)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 8 * dim))

    def forward(self, x, c, ctx, ctx_mask, rope_s, rope_t):
        b, t, s, d = x.shape
        sh_s, sc_s, g_s, sh_t, sc_t, sh_m, sc_m, g_m = self.ada(c)[:, None, None].chunk(8, dim=-1)

        h = modulate(self.norm_s(x), sh_s, sc_s).reshape(b * t, s, d)
        x = x + g_s * self.attn_s(h, rope=rope_s).view(b, t, s, d)

        if t > 1:  # skipped for single-frame (image) training
            h = modulate(self.norm_t(x), sh_t, sc_t).permute(0, 2, 1, 3).reshape(b * s, t, d)
            h = self.attn_t(h, rope=rope_t).view(b, s, t, d).permute(0, 2, 1, 3)
            x = x + torch.tanh(self.temporal_gate) * h

        h = self.norm_x(x).reshape(b, t * s, d)
        x = x + self.attn_x(h, ctx=ctx, mask=ctx_mask[:, None, None, :]).view(b, t, s, d)

        x = x + g_m * self.mlp(modulate(self.norm_m(x), sh_m, sc_m))
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = (t.float() * 1000)[:, None] * freqs[None]
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=-1))


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.linear = nn.Linear(dim, out_dim)

    def forward(self, x, c):
        shift, scale = self.ada(c)[:, None, None].chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


# ── model ────────────────────────────────────────────────────────────────────

class SpartanDiT(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        hidden: int = 768,
        depth: int = 12,
        heads: int = 12,
        text_dim: int = 768,
        mlp_ratio: float = 4.0,
        grad_ckpt: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = tuple(patch_size)
        self.hidden, self.heads = hidden, heads
        self.grad_ckpt = grad_ckpt
        assert (hidden // heads) % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"

        self.patch_embed = nn.Conv3d(in_channels, hidden, kernel_size=self.patch_size, stride=self.patch_size)
        self.t_embed = TimestepEmbedder(hidden)
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, hidden))
        self.pool_proj = nn.Linear(text_dim, hidden)
        self.blocks = nn.ModuleList([Block(hidden, heads, mlp_ratio) for _ in range(depth)])
        pt, ph, pw = self.patch_size
        self.final = FinalLayer(hidden, pt * ph * pw * in_channels)
        self._init_weights()

    def _init_weights(self):
        def basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(basic)
        w = self.patch_embed.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)
        for blk in self.blocks:  # adaLN-Zero
            nn.init.zeros_(blk.ada[-1].weight)
            nn.init.zeros_(blk.ada[-1].bias)
        nn.init.zeros_(self.final.ada[-1].weight)
        nn.init.zeros_(self.final.ada[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def forward(self, x, t, text, text_mask, pooled):
        """x [B,C,T,H,W] noisy latent, t [B] in [0,1], text [B,L,text_dim] -> velocity [B,C,T,H,W]."""
        b, c, T, H, W = x.shape
        pt, ph, pw = self.patch_size
        x = self.patch_embed(x)  # [B,D,T',h,w]
        tp, h, w = x.shape[2:]
        x = x.permute(0, 2, 3, 4, 1).reshape(b, tp, h * w, self.hidden)

        cond = self.t_embed(t) + self.pool_proj(pooled)
        ctx = self.text_proj(text)
        head_dim = self.hidden // self.heads
        rope_s = rope_2d(h, w, head_dim, x.device)
        rope_t = rope_1d(tp, head_dim, x.device)

        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                x = checkpoint(blk, x, cond, ctx, text_mask, rope_s, rope_t, use_reentrant=False)
            else:
                x = blk(x, cond, ctx, text_mask, rope_s, rope_t)

        x = self.final(x, cond)  # [B,T',S,pt*ph*pw*C]
        x = x.view(b, tp, h, w, pt, ph, pw, c).permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(b, c, tp * pt, h * ph, w * pw)


def build_model(cfg: dict, text_dim: int) -> SpartanDiT:
    m = cfg["model"]
    return SpartanDiT(
        in_channels=m["in_channels"],
        patch_size=tuple(m["patch_size"]),
        hidden=m["hidden"],
        depth=m["depth"],
        heads=m["heads"],
        text_dim=text_dim,
        mlp_ratio=m.get("mlp_ratio", 4.0),
        grad_ckpt=m.get("grad_ckpt", False),
    )
