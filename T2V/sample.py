"""Generate a video from a prompt.

  python -m T2V.sample --model spartangen-tiny --prompt "a man is playing guitar" --out out.mp4
  python -m T2V.sample --model wan-1.3b --prompt "a man is playing guitar" --out wan.mp4
"""

import argparse
import json
import os

from T2V.generators import REGISTRY, get_generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="spartangen-tiny", choices=sorted(REGISTRY))
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="sample.mp4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--cfg", type=float)
    ap.add_argument("--ckpt", help="SpartanGen checkpoint (default: $SPARTANGEN_CKPT or T2V/runs/tiny_video/last.pt)")
    a = ap.parse_args()
    if a.ckpt:
        os.environ["SPARTANGEN_CKPT"] = a.ckpt
    info = get_generator(a.model).generate(a.prompt, a.out, seed=a.seed, steps=a.steps, cfg=a.cfg)
    print(json.dumps({**info, "path": a.out}))


if __name__ == "__main__":
    main()
