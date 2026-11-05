"""Train the SpartanGen DiT with flow matching on cached VAE latents (single GPU).

  # sanity check first: loss must fall towards ~0 and samples must reproduce the batch
  python -m T2V.train --config T2V/configs/tiny_video.yaml --overfit-one-batch --steps 2000
  # stage A (images), then stage B (video) initialised from stage A
  python -m T2V.train --config T2V/configs/tiny_image.yaml
  python -m T2V.train --config T2V/configs/tiny_video.yaml --init-from T2V/runs/tiny_image/last.pt
  # resume an interrupted run
  python -m T2V.train --config T2V/configs/tiny_video.yaml --resume T2V/runs/tiny_video/last.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from T2V.data.dataset import LatentDataset, SyntheticLatents, collate
from T2V.model.dit import build_model
from T2V.model.encoders import TextEncoder, VideoVAE
from T2V.model.flow import flow_loss
from T2V.pipeline import SpartanPipeline, default_device, load_config, tile, use_bf16
from T2V.utils.video_io import write_mp4


def infinite(loader):
    while True:
        yield from loader


@torch.no_grad()
def update_ema(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p.detach(), 1 - decay)


def save_ckpt(path, model, ema, opt, sched, step, cfg):
    raw = getattr(model, "_orig_mod", model)
    torch.save(
        {"model": raw.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
         "sched": sched.state_dict(), "step": step, "config": cfg},
        path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", help="continue a run (model, EMA, optimiser, step)")
    ap.add_argument("--init-from", help="load weights only, e.g. stage-A checkpoint for stage B")
    ap.add_argument("--overfit-one-batch", action="store_true")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    tc = cfg["train"]
    if a.steps:
        tc["steps"] = a.steps
    device = a.device or default_device()
    torch.manual_seed(tc["seed"])
    random.seed(tc["seed"])
    out = Path(tc["out_dir"] + ("_overfit" if a.overfit_one_batch else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    # data
    dc = cfg["data"]
    if dc.get("synthetic"):
        ds = SyntheticLatents(n=64, shape=tuple(dc["latent_shape"]), stage=dc["stage"])
    else:
        ds = LatentDataset(dc["latent_dir"], "train", stage=dc["stage"])
    loader = DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
                        num_workers=tc["num_workers"], collate_fn=collate, pin_memory=device == "cuda")
    batches = infinite(loader)
    print(f"[train] {len(ds)} training items, stage={dc['stage']}, device={device}")

    # model
    text_enc = TextEncoder(cfg["text_encoder"], device=device)
    model = build_model(cfg, text_enc.dim).to(device)
    print(f"[train] DiT parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    if a.init_from:
        sd = torch.load(a.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["ema"])
        print(f"[train] initialised from EMA weights of {a.init_from}")
    ema = copy.deepcopy(model).eval().requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], betas=(0.9, 0.999), weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, tc["warmup"])))
    step = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        print(f"[train] resumed at step {step}")

    train_model = torch.compile(model) if tc.get("compile") else model
    bf16 = use_bf16(device)
    if device == "cuda" and not bf16:
        print("[train] GPU has no native bf16 (e.g. T4/V100): training in fp32, which is slower")

    fixed = next(batches) if a.overfit_one_batch else None
    p_drop = 0.0 if a.overfit_one_batch else tc["caption_dropout"]
    ema_decay = 0.0 if a.overfit_one_batch else tc["ema_decay"]
    # logit-normal rarely visits t≈1, where sampling starts; a short overfit needs uniform t
    t_mode = "uniform" if a.overfit_one_batch else tc["t_mode"]

    vae = None
    log_f = open(out / "metrics.jsonl", "a")
    t0, running = time.time(), 0.0
    model.train()
    while step < tc["steps"]:
        x, caps = fixed if fixed is not None else next(batches)
        x = x.to(device, non_blocking=True)
        caps = ["" if random.random() < p_drop else c for c in caps]
        text, mask, pooled = text_enc(caps)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
            loss = flow_loss(train_model, x, {"text": text, "text_mask": mask, "pooled": pooled}, t_mode=t_mode)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite loss at step {step}; lower lr or check latent normalisation")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        sched.step()
        update_ema(ema, model, ema_decay)
        step += 1
        running += loss.item()

        if step % tc["log_every"] == 0:
            dt = time.time() - t0
            rec = {"step": step, "loss": running / tc["log_every"], "grad_norm": float(gnorm),
                   "lr": sched.get_last_lr()[0], "it_per_s": tc["log_every"] / dt}
            print(json.dumps(rec))
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            t0, running = time.time(), 0.0

        if tc["ckpt_every"] and step % tc["ckpt_every"] == 0:
            save_ckpt(out / "last.pt", model, ema, opt, sched, step, cfg)

        if tc.get("sample_every") and step % tc["sample_every"] == 0 and cfg.get("vae"):
            vae = vae or VideoVAE(cfg["vae"], device=device)
            pipe = SpartanPipeline(ema, text_enc, vae, cfg, device)
            prompts = cfg["sample"]["preview_prompts"]
            if dc["stage"] == "image":
                vids = pipe.generate(prompts, seed=0, frames=1)
            else:
                vids = pipe.generate(prompts, seed=0)
            write_mp4(tile(vids), out / "samples" / f"step_{step:07d}.mp4", fps=pipe.fps)
            ema.eval()
            model.train()

    save_ckpt(out / "last.pt", model, ema, opt, sched, step, cfg)
    print(f"[train] done at step {step}; checkpoint {out / 'last.pt'}")

    if a.overfit_one_batch:
        # the model should now regenerate the memorised batch from its captions
        model.eval()
        x, caps = fixed
        x = x.to(device)
        with torch.no_grad():
            for name, c in [("true", caps), ("shuffled", caps[1:] + caps[:1])]:
                tx, mk, pl = text_enc(c)
                row = []
                for tv in (0.2, 0.5, 0.8, 0.95, 0.99):
                    n = torch.randn_like(x)
                    t = torch.full((x.shape[0],), tv, device=device)
                    v = model((1 - tv) * x + tv * n, t, tx, mk, pl)
                    row.append(f"t={tv}:{((v - (n - x)) ** 2).mean().item():.3f}")
                print(f"[overfit] loss with {name} captions  " + "  ".join(row))
        x = x.cpu()
        pipe = SpartanPipeline(model, text_enc, None, cfg, device)
        z = pipe.latents(caps, seed=1, cfg=1.0, steps=cfg["sample"]["steps"], frames=x.shape[2])
        mse = torch.mean((z.cpu() - x) ** 2).item()
        print(f"[overfit] sample-vs-target latent MSE = {mse:.4f} (target variance {x.var().item():.3f}); "
              f"PASS if well below 0.1")


if __name__ == "__main__":
    main()
