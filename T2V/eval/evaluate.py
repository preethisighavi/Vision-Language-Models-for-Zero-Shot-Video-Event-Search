"""Score generated clips against their prompts and against the real test videos.

  python -m T2V.eval.evaluate --set T2V/eval_runs/msrvtt512 --models spartangen-tiny wan-1.3b real --log

Metrics per model:
  CLIPScore    100 * mean frame-text cosine (CLIP ViT-B/32 by default, same as the data pipeline)
  R@1/5/10     text->video retrieval among the generated set: is the clip made from caption i
               ranked first for caption i? Uses the team's CLIP (pass --clip to use the fine-tuned one)
  FVD          Frechet distance of I3D features, generated vs real test clips (16 frames, 224px)
"real" scores the real test videos themselves: the ceiling for CLIPScore and R@k.
--log appends the numbers to results/model_results.csv (shared team harness).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

from T2V.utils.video_io import read_frames, uniform_frames

I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"  # StyleGAN-V FVD detector
REPO = Path(__file__).resolve().parents[2]


def load_video(path: str, n: int) -> np.ndarray:
    frames, _ = read_frames(path)
    return uniform_frames(frames, n)


# ── CLIP metrics ─────────────────────────────────────────────────────────────

class Clip:
    def __init__(self, name: str, device: str):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(name).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        self.device = device

    @torch.no_grad()
    def frames(self, frames: np.ndarray) -> torch.Tensor:
        px = self.proc(images=list(frames), return_tensors="pt")["pixel_values"].to(self.device)
        return F.normalize(self.model.get_image_features(pixel_values=px), dim=-1)

    @torch.no_grad()
    def texts(self, texts: list[str]) -> torch.Tensor:
        tok = self.proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        return F.normalize(self.model.get_text_features(**tok), dim=-1)


def clip_metrics(clip: Clip, paths: list[str], captions: list[str], n_frames: int = 8) -> dict:
    txt = clip.texts(captions)
    vids, scores = [], []
    for i, p in enumerate(paths):
        f = clip.frames(load_video(p, n_frames))  # [n,D]
        scores.append(100 * torch.clamp(f @ txt[i], min=0).mean().item())
        vids.append(F.normalize(f.mean(0), dim=-1))
    sim = txt @ torch.stack(vids).T  # [N captions, N videos]
    ranks = (sim > sim.diag()[:, None]).sum(1).cpu().numpy()  # 0 = correct video ranked first
    return {
        "CLIPScore": float(np.mean(scores)),
        "R@1": float((ranks < 1).mean()),
        "R@5": float((ranks < 5).mean()),
        "R@10": float((ranks < 10).mean()),
        "MedR": float(np.median(ranks) + 1),
    }


# ── FVD ──────────────────────────────────────────────────────────────────────

def load_i3d(device: str):
    cache = Path.home() / ".cache" / "t2v" / "i3d_torchscript.pt"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(I3D_URL, str(cache))
    return torch.jit.load(str(cache)).eval().to(device)


@torch.no_grad()
def i3d_features(i3d, paths: list[str], device: str, n_frames: int = 16, bs: int = 8) -> np.ndarray:
    feats = []
    for s in range(0, len(paths), bs):
        batch = []
        for p in paths[s : s + bs]:
            x = torch.from_numpy(load_video(p, n_frames)).permute(0, 3, 1, 2).float() / 255  # T,3,H,W
            h, w = x.shape[-2:]
            scale = 224 / min(h, w)
            x = F.interpolate(x, size=(max(224, round(h * scale)), max(224, round(w * scale))),
                              mode="bilinear", align_corners=False, antialias=True)
            t, l = (x.shape[-2] - 224) // 2, (x.shape[-1] - 224) // 2
            x = x[..., t : t + 224, l : l + 224]
            batch.append((x * 2 - 1).permute(1, 0, 2, 3))  # 3,T,224,224 in [-1,1]
        feats.append(i3d(torch.stack(batch).to(device), rescale=False, resize=False, return_features=True).cpu())
    return torch.cat(feats).numpy().astype(np.float64)


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="directory written by make_eval_set")
    ap.add_argument("--models", nargs="+", default=["spartangen-tiny", "wan-1.3b", "real"])
    ap.add_argument("--clip", default="openai/clip-vit-base-patch32", help="HF id or path of a fine-tuned CLIP")
    ap.add_argument("--no-fvd", action="store_true")
    ap.add_argument("--log", action="store_true", help="append to results/model_results.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    root = Path(a.set)
    prompts = [json.loads(l) for l in (root / "prompts.jsonl").read_text().splitlines()]
    captions = [p["caption"] for p in prompts]
    real_paths = [p["real_path"] for p in prompts]

    clip = Clip(a.clip, a.device)
    i3d = None if a.no_fvd else load_i3d(a.device)
    real_feats = None if a.no_fvd else i3d_features(i3d, real_paths, a.device)

    results = {}
    for model_id in a.models:
        paths = real_paths if model_id == "real" else [str(root / model_id / f"{p['video_id']}.mp4") for p in prompts]
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            print(f"[eval] {model_id}: {len(missing)} clips missing, skipped (run make_eval_set first)")
            continue
        r = clip_metrics(clip, paths, captions)
        if not a.no_fvd and model_id != "real":
            r["FVD"] = frechet(i3d_features(i3d, paths, a.device), real_feats)
        results[model_id] = r
        print(f"[eval] {model_id}: " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()))

    (root / "metrics.json").write_text(json.dumps(results, indent=2))
    if a.log:
        sys.path.insert(0, str(REPO))
        from results.log_result import log_metrics

        for model_id, r in results.items():
            log_metrics(model="T2V", variant=model_id, dataset=f"MSR-VTT-test-{len(prompts)}",
                        metrics=r, source_file=str(root / "metrics.json"))


if __name__ == "__main__":
    main()
