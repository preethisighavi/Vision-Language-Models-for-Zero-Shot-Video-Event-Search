"""Rectified-flow / flow-matching objective and sampler.

Convention: x_t = (1 - t) * x0 + t * noise, t in [0, 1], t = 1 is pure noise.
The model predicts the velocity v = noise - x0. Sampling integrates dx/dt = v
from t = 1 down to t = 0 with Euler steps and classifier-free guidance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_t(n: int, device, mode: str = "logit_normal", mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    if mode == "logit_normal":  # SD3: concentrates training on mid-noise levels
        return torch.sigmoid(torch.randn(n, device=device) * std + mean)
    return torch.rand(n, device=device)


def shift_t(t: torch.Tensor, shift: float) -> torch.Tensor:
    """Resolution-dependent timestep shift (SD3); shift > 1 spends more steps at high noise."""
    if shift == 1.0:
        return t
    return shift * t / (1 + (shift - 1) * t)


def flow_loss(model, x0: torch.Tensor, cond: dict, t_mode: str = "logit_normal", shift: float = 1.0) -> torch.Tensor:
    b = x0.shape[0]
    t = shift_t(sample_t(b, x0.device, t_mode), shift)
    noise = torch.randn_like(x0)
    tt = t.view(b, *([1] * (x0.ndim - 1)))
    xt = (1 - tt) * x0 + tt * noise
    v = model(xt, t, **cond)
    return F.mse_loss(v.float(), (noise - x0).float())


def _cat_cond(a: dict, b: dict) -> dict:
    return {k: torch.cat([a[k], b[k]]) for k in a}


@torch.no_grad()
def sample(
    model,
    shape: tuple[int, ...],
    cond: dict,
    uncond: dict | None,
    steps: int = 50,
    cfg: float = 6.0,
    shift: float = 1.0,
    generator: torch.Generator | None = None,
    device="cuda",
    dtype=torch.bfloat16,
) -> torch.Tensor:
    x = torch.randn(shape, generator=generator, device="cpu").to(device)
    ts = shift_t(torch.linspace(1.0, 0.0, steps + 1, device=device), shift)
    use_cfg = uncond is not None and cfg != 1.0
    both = _cat_cond(cond, uncond) if use_cfg else cond
    autocast = torch.autocast(device_type=torch.device(device).type, dtype=dtype, enabled=dtype != torch.float32)
    for i in range(steps):
        t, t_next = ts[i], ts[i + 1]
        with autocast:
            if use_cfg:
                xin = torch.cat([x, x])
                v_c, v_u = model(xin, t.expand(xin.shape[0]), **both).float().chunk(2)
                v = v_u + cfg * (v_c - v_u)
            else:
                v = model(x, t.expand(x.shape[0]), **both).float()
        x = x + (t_next - t) * v
    return x
