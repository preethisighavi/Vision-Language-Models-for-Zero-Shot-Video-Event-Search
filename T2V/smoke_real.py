"""End-to-end check on REAL video, any machine (Colab T4, Mac MPS, a GPU box). ~10-20 min.

  python -m T2V.smoke_real --out T2V/runs/smoke_real

1. downloads 6 MSR-VTT test clips + captions from Hugging Face (VLM2Vec/MSR-VTT, ~2 MB)
2. caches their latents with the real Wan 2.1 VAE (3 fps, 17 frames, 128px)
3. VAE round trip: decode the latents, PSNR vs the input clips      -> PASS if mean >= 22 dB
4. overfits a small SpartanGen (hidden 256, depth 4) on the 6 clips with real CLIP captions
5. generates from the captions alone; each output must be nearest to its own clip and the mean
   PSNR vs the real clips must be >= 18 dB (the Mac reference run gets ~22-28 dB at 3000 steps)
writes <out>/report.json, vae_roundtrip.mp4 and generated.mp4 (left = real, right = ours).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

CLIPS = ["video7021", "video7061", "video7060", "video7146", "video7147", "video7117"]
REPO = "VLM2Vec/MSR-VTT"


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """per-clip PSNR for tensors in [-1, 1] shaped [N, ...]."""
    mse = ((a - b) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(4 / mse)


def main():
    from huggingface_hub import hf_hub_download

    from T2V.data.prepare_latents import ClipDataset
    from T2V.model.encoders import VideoVAE
    from T2V.pipeline import SpartanPipeline, default_device, tile
    from T2V.utils.video_io import to_uint8, write_mp4

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="T2V/runs/smoke_real")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = a.device or default_device()
    report: dict = {"device": device, "torch": torch.__version__}
    if device == "cuda":
        report["gpu"] = torch.cuda.get_device_name()
    t_start = time.time()

    # 1. data
    data = out / "data"
    caps_path = hf_hub_download(REPO, "msrvtt_test_1k.json", repo_type="dataset", local_dir=data)
    captions = {r["video_id"]: r["caption"] for r in json.load(open(caps_path))}
    paths = [hf_hub_download(REPO, f"raw_videos/{v}.mp4", repo_type="dataset", local_dir=data) for v in CLIPS]
    manifest = pd.DataFrame({"video_id": CLIPS, "source": "MSR-VTT", "split": "train",
                             "captions": [[captions[v]] for v in CLIPS], "path": paths})
    manifest.to_parquet(data / "manifest.parquet", index=False)
    print(f"[smoke] downloaded {len(CLIPS)} clips")

    # 2. latents (same code path as full training)
    lat_dir = out / "latents"
    subprocess.run([sys.executable, "-m", "T2V.data.prepare_latents", "--manifest", str(data / "manifest.parquet"),
                    "--out-dir", str(lat_dir), "--splits", "train", "--fps", "3", "--batch-size", "2",
                    "--workers", "0", "--device", device], check=True)
    shard = torch.load(lat_dir / "train" / "shard_00000.pt")
    lat = shard["latents"].float()
    report["latent_mean"], report["latent_std"] = round(lat.mean().item(), 3), round(lat.std().item(), 3)

    # 3. VAE round trip
    vae = VideoVAE(device=device)
    rec = torch.cat([vae.decode(z[None]).float().cpu() for z in lat])
    ds = ClipDataset(manifest, 17, 3, 128, 4)
    orig = torch.stack([ds[i][2] for i in range(len(CLIPS))])
    p = psnr(rec, orig)
    report["vae_psnr_db"] = {v: round(x, 1) for v, x in zip(CLIPS, p.tolist())}
    report["vae_psnr_mean_db"] = round(p.mean().item(), 1)
    to_frames = lambda t: [to_uint8(v.permute(1, 2, 3, 0).numpy()) for v in t]  # noqa: E731
    write_mp4(tile([np.concatenate([o, r], 2) for o, r in zip(to_frames(orig), to_frames(rec))]),
              out / "vae_roundtrip.mp4", fps=3)
    del vae
    print(f"[smoke] VAE round trip: {report['vae_psnr_mean_db']} dB")

    # 4. overfit a small SpartanGen on the 6 clips
    cfg = {
        "text_encoder": "openai/clip-vit-large-patch14", "vae": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "data": {"latent_dir": str(lat_dir), "stage": "video", "latent_shape": [16, 5, 16, 16], "fps": 3},
        "model": {"in_channels": 16, "patch_size": [1, 2, 2], "hidden": 256, "depth": 4, "heads": 4, "mlp_ratio": 4.0},
        "train": {"out_dir": str(out / "run"), "steps": a.steps, "batch_size": 6, "lr": 5e-4, "warmup": 50,
                  "weight_decay": 0.0, "grad_clip": 1.0, "ema_decay": 0.999, "caption_dropout": 0.1,
                  "t_mode": "logit_normal", "num_workers": 0, "log_every": 250, "ckpt_every": 0,
                  "sample_every": 0, "compile": False, "seed": 0},
        "sample": {"steps": 50, "cfg": 1.0, "shift": 1.0, "preview_prompts": [captions[CLIPS[0]]]},
    }
    (out / "overfit.yaml").write_text(yaml.safe_dump(cfg))
    t0 = time.time()
    res = subprocess.run([sys.executable, "-m", "T2V.train", "--config", str(out / "overfit.yaml"),
                          "--overfit-one-batch", "--device", device], check=True, capture_output=True, text=True)
    report["train_seconds"] = round(time.time() - t0)
    overfit_lines = [l for l in res.stdout.splitlines() if l.startswith("[overfit]")]
    print("\n".join(overfit_lines))
    report["overfit"] = overfit_lines

    # 5. generate from captions alone
    pipe = SpartanPipeline.from_checkpoint(out / "run_overfit" / "last.pt", device=device, use_ema=False)
    caps = [captions[v] for v in CLIPS]
    t0 = time.time()
    gen = pipe.generate(caps, seed=1, cfg=1.0)
    report["generate_seconds_6_clips"] = round(time.time() - t0, 1)
    g = torch.tensor(np.stack(gen)).float().permute(0, 4, 1, 2, 3) / 127.5 - 1  # N,3,T,H,W
    pg = psnr(g, orig)
    dist = ((orig[:, None] - g[None]) ** 2).flatten(2).mean(2)  # [orig, gen]
    nearest = dist.argmin(0).tolist()
    report["gen_psnr_db"] = {c: round(x, 1) for c, x in zip(caps, pg.tolist())}
    report["gen_nearest_is_own_clip"] = nearest == list(range(len(CLIPS)))
    write_mp4(tile([np.concatenate([o, r], 2) for o, r in zip(to_frames(orig), gen)]), out / "generated.mp4", fps=3)

    report["gen_psnr_mean_db"] = round(pg.mean().item(), 1)
    report["vae_pass"] = report["vae_psnr_mean_db"] >= 22
    # nearest-clip alone passes even for blurry early outputs; also require recognisable reconstructions
    report["gen_pass"] = bool(report["gen_nearest_is_own_clip"] and report["gen_psnr_mean_db"] >= 18)
    report["PASS"] = bool(report["vae_pass"] and report["gen_pass"])
    report["total_minutes"] = round((time.time() - t_start) / 60, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[smoke] {'PASS' if report['PASS'] else 'FAIL'} - videos in {out}")


if __name__ == "__main__":
    main()
