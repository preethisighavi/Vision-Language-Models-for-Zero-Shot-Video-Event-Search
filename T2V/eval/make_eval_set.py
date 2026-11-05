"""Generate one clip per held-out test caption for each model (resumable).

  python -m T2V.eval.make_eval_set --manifest T2V/data/manifest.parquet \
      --models spartangen-tiny wan-1.3b --n 512 --out T2V/eval_runs/msrvtt512

Writes <out>/<model>/<video_id>.mp4 and <out>/prompts.jsonl
({video_id, caption, real_path}), shared by all models so they see identical prompts.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from T2V.generators import REGISTRY, get_generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--models", nargs="+", default=["spartangen-tiny"], choices=sorted(REGISTRY))
    ap.add_argument("--source", default="MSR-VTT")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts_path = out / "prompts.jsonl"
    if prompts_path.exists():
        prompts = [json.loads(l) for l in prompts_path.read_text().splitlines()]
    else:
        m = pd.read_parquet(a.manifest)
        test = m[(m["split"] == "test") & (m["source"] == a.source)].sample(n=a.n, random_state=a.seed)
        prompts = [
            {"video_id": r.video_id, "caption": list(r.captions)[i % len(r.captions)], "real_path": r.path}
            for i, r in enumerate(test.itertuples())
        ]
        prompts_path.write_text("\n".join(json.dumps(p) for p in prompts) + "\n")

    for model_id in a.models:
        gen = get_generator(model_id)
        d = out / model_id
        d.mkdir(exist_ok=True)
        for i, p in enumerate(prompts):
            path = d / f"{p['video_id']}.mp4"
            if path.exists():
                continue
            gen.generate(p["caption"], path, seed=a.seed + i)
            if i % 25 == 0:
                print(f"[eval-set] {model_id}: {i + 1}/{len(prompts)}")
        print(f"[eval-set] {model_id}: done -> {d}")


if __name__ == "__main__":
    main()
