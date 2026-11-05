"""Turn a training checkpoint into a slim, inference-only file you can run anywhere.

A training checkpoint (last.pt, ~3 GB) holds the raw weights, the EMA weights, optimiser
and scheduler state. Generation only needs the EMA weights + config: in fp16 that is ~400 MB.

  python -m T2V.export --ckpt T2V/runs/tiny_video/last.pt --out spartangen_tiny.pt
  # optional: also download the frozen Wan VAE + CLIP text encoder into the HF cache now,
  # so the machine that runs the model later does not need internet on first use
  python -m T2V.export --ckpt ... --out ... --prefetch-encoders

The exported file loads exactly like a training checkpoint:
  python -m T2V.sample --ckpt spartangen_tiny.pt --prompt "a dog runs on the beach"
  SPARTANGEN_CKPT=spartangen_tiny.pt uvicorn T2V.serve:app --port 8100
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import torch

FORMAT = "spartangen-inference-v1"


def export(ckpt_path: str | Path, out_path: str | Path, dtype: str = "fp16", use_ema: bool = True) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key = "ema" if use_ema and "ema" in ckpt else "model"
    cast = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    weights = {k: (v.to(cast) if v.is_floating_point() else v) for k, v in ckpt[key].items()}
    meta = {
        "format": FORMAT,
        "source": str(ckpt_path),
        "weights": key,
        "dtype": dtype,
        "step": ckpt.get("step"),
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "params": sum(v.numel() for v in weights.values()),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # stored under "ema" so SpartanPipeline.from_checkpoint loads it unchanged
    torch.save({"ema": weights, "config": ckpt["config"], "step": ckpt.get("step"), "export": meta}, out_path)
    return meta


def prefetch_encoders(cfg: dict) -> None:
    """Download (or confirm cached) the frozen pretrained parts the checkpoint does not contain."""
    from diffusers import AutoencoderKLWan
    from transformers import CLIPTextModel, CLIPTokenizer

    if cfg.get("vae"):
        AutoencoderKLWan.from_pretrained(cfg["vae"], subfolder="vae")
        print(f"[export] cached VAE {cfg['vae']}")
    if cfg.get("text_encoder") and cfg["text_encoder"] != "dummy":
        CLIPTokenizer.from_pretrained(cfg["text_encoder"])
        CLIPTextModel.from_pretrained(cfg["text_encoder"])
        print(f"[export] cached text encoder {cfg['text_encoder']}")


def main():
    ap = argparse.ArgumentParser(description="Export an inference-only SpartanGen checkpoint")
    ap.add_argument("--ckpt", required=True, help="training checkpoint, e.g. T2V/runs/tiny_video/last.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--raw", action="store_true", help="export the raw (non-EMA) weights instead")
    ap.add_argument("--prefetch-encoders", action="store_true")
    a = ap.parse_args()

    meta = export(a.ckpt, a.out, a.dtype, use_ema=not a.raw)
    src_mb = Path(a.ckpt).stat().st_size / 2**20
    out_mb = Path(a.out).stat().st_size / 2**20
    print(f"[export] {meta['weights']} weights, step {meta['step']}, {meta['params'] / 1e6:.1f}M params, "
          f"{meta['dtype']}: {src_mb:.0f} MB -> {out_mb:.0f} MB at {a.out}")
    if a.prefetch_encoders:
        prefetch_encoders(torch.load(a.out, map_location="cpu", weights_only=False)["config"])


if __name__ == "__main__":
    main()
