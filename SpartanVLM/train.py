import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from data_pipeline import SpartanPreprocessor
from model import SpartanVLM, load_checkpoint_state


def read_annotation_rows(path):
    data = json.loads(path.read_text())
    if isinstance(data, list):
        rows = data
    else:
        captions = {}
        for row in data["sentences"]:
            captions.setdefault(row["video_id"], []).append(row["caption"])
        rows = [{"video_id": row["video_id"], "caption": captions[row["video_id"]]}
                for row in data["videos"] if row["video_id"] in captions]

    unique = {}
    for row in rows:
        video = row.get("video") or row.get("video_path")
        if not video:
            video = f"{row['video_id']}.mp4"
        captions = row.get("caption", row.get("text", []))
        if isinstance(captions, str):
            captions = [captions]
        captions = [caption.strip() for caption in captions if isinstance(caption, str) and caption.strip()]
        if captions:
            unique.setdefault(Path(video).name, {"video": Path(video).name, "captions": []})
            unique[Path(video).name]["captions"].extend(captions)
    return list(unique.values())


def read_webvid_manifest(manifest_path):
    rows = []
    with manifest_path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            video = next((row[key] for key in ("video_path", "video", "path", "filename", "videoid")
                          if row.get(key)), None)
            caption = next((row[key] for key in ("caption", "text", "name", "title") if row.get(key)), None)
            if video and caption:
                if not Path(video).suffix:
                    video = f"{video}.mp4"
                rows.append({"video": video, "captions": [caption.strip()]})
    if not rows:
        raise ValueError(f"No video/caption rows found in WebVid manifest: {manifest_path}")
    return rows


class VideoCaptionDataset(Dataset):
    def __init__(self, name, rows, video_root, transform, num_frames=4, sample_cap=None, seed=42):
        self.name = name
        self.video_root = Path(video_root)
        self.transform = transform
        self.num_frames = num_frames
        self.rows = rows
        if sample_cap and len(self.rows) > sample_cap:
            self.rows = random.Random(seed).sample(self.rows, sample_cap)
        missing = [row["video"] for row in self.rows if not (self.video_root / row["video"]).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{name}: {len(missing)}/{len(self.rows)} videos missing under {self.video_root}; "
                f"examples: {missing[:5]}"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.video_root / row["video"]
        capture = cv2.VideoCapture(str(path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            capture.release()
            raise ValueError(f"Could not decode video: {path}")
        frames = []
        for frame_index in np.linspace(0, frame_count - 1, self.num_frames, dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        capture.release()
        if not frames:
            raise ValueError(f"Could not decode frames: {path}")
        frames.extend([frames[-1]] * (self.num_frames - len(frames)))
        pixels = torch.stack([self.transform(frame) for frame in frames[:self.num_frames]])
        caption = random.choice(row["captions"])
        return pixels, caption, self.name


def make_dataset(name, annotations, video_root, preprocessor, sample_cap, seed):
    rows = annotations if isinstance(annotations, list) else read_annotation_rows(Path(annotations))
    return VideoCaptionDataset(name, rows, video_root, preprocessor.vision_transform,
                               sample_cap=sample_cap, seed=seed)


def collate_batch(batch, preprocessor):
    videos, captions, sources = zip(*batch)
    input_ids, attention_mask = preprocessor.process_text(list(captions))
    return torch.stack(videos), input_ids, attention_mask, list(sources)


def main():
    parser = argparse.ArgumentParser(description="CUDA training for the SpartanVLM SpaceTimeTransformer.")
    parser.add_argument("--msrvtt-annotations", type=Path, default=Path("../data/msrvtt/msrvtt_train_7k.json"))
    parser.add_argument("--msrvtt-videos", type=Path, default=Path("../data/msrvtt/videos"))
    parser.add_argument("--msvd-annotations", type=Path, default=Path("../data/msvd/msvd_train.json"))
    parser.add_argument("--msvd-videos", type=Path, default=Path("../data/msvd/videos"))
    parser.add_argument("--webvid-manifest", type=Path, required=True)
    parser.add_argument("--webvid-videos", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/spartan_vrlm"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-samples-per-source", type=int)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires a CUDA GPU. Run it in a CUDA-enabled runtime.")
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least 2 for contrastive training")
    if args.init_checkpoint and not args.init_checkpoint.is_file():
        raise FileNotFoundError(f"Initialization checkpoint not found: {args.init_checkpoint}")
    for path in (args.msrvtt_annotations, args.msvd_annotations, args.webvid_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Training data file not found: {path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=precision == torch.float16)
    preprocessor = SpartanPreprocessor(max_seq_length=32)

    datasets = [
        make_dataset("MSR-VTT", args.msrvtt_annotations, args.msrvtt_videos,
                     preprocessor, args.max_samples_per_source, args.seed),
        make_dataset("MSVD", args.msvd_annotations, args.msvd_videos,
                     preprocessor, args.max_samples_per_source, args.seed),
        make_dataset("WebVid", read_webvid_manifest(args.webvid_manifest), args.webvid_videos,
                     preprocessor, args.max_samples_per_source, args.seed),
    ]
    if any(len(dataset) == 0 for dataset in datasets):
        raise ValueError("Each configured dataset must contain at least one usable video-caption pair")

    combined = ConcatDataset(datasets)
    weights = torch.cat([
        torch.full((len(dataset),), 1.0 / len(dataset) / len(datasets), dtype=torch.double)
        for dataset in datasets
    ])
    samples_per_epoch = args.samples_per_epoch or sum(len(dataset) for dataset in datasets)
    sampler = WeightedRandomSampler(weights, samples_per_epoch, replacement=True)
    if samples_per_epoch < args.batch_size:
        raise ValueError("samples-per-epoch must be at least one full batch")
    loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda batch: collate_batch(batch, preprocessor),
        persistent_workers=args.workers > 0,
    )

    model = SpartanVLM().to(device)
    init_mode = "random_initialization"
    if args.init_checkpoint:
        model.load_state_dict(load_checkpoint_state(args.init_checkpoint), strict=True)
        init_mode = f"resume_checkpoint:{args.init_checkpoint}"

    learning_rate = args.learning_rate or (3e-5 if args.init_checkpoint else 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    optimizer_steps = max(1, math.ceil(len(loader) / args.gradient_accumulation) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimizer_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model": "SpartanVLM",
        "architecture": "SpaceTimeTransformer + DistilBERT dual encoder",
        "init_mode": init_mode,
        "datasets": {dataset.name: len(dataset) for dataset in datasets},
        "sampling": "equal source probability with replacement",
        "samples_per_epoch": samples_per_epoch,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": learning_rate,
        "weight_decay": args.weight_decay,
        "loss": "symmetric in-batch InfoNCE, temperature 0.07",
        "precision": str(precision),
        "seed": args.seed,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    log_path = args.output_dir / "training_log.jsonl"
    model.train()
    step = 0
    started = time.time()

    with log_path.open("w") as log:
        for epoch in range(args.epochs):
            losses = []
            source_counts = {dataset.name: 0 for dataset in datasets}
            optimizer.zero_grad(set_to_none=True)
            for batch_index, (videos, input_ids, attention_mask, sources) in enumerate(loader):
                videos = videos.to(device, non_blocking=True)
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                for source in sources:
                    source_counts[source] += 1

                with torch.autocast("cuda", dtype=precision):
                    video_embeds = F.normalize(model.encode_video(videos), dim=-1)
                    text_embeds = F.normalize(model.encode_text(input_ids, attention_mask), dim=-1)
                    logits = video_embeds @ text_embeds.T / 0.07
                    labels = torch.arange(logits.shape[0], device=device)
                    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
                    scaled_loss = loss / args.gradient_accumulation

                scaler.scale(scaled_loss).backward()
                is_update = (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(loader)
                if is_update:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                losses.append(float(loss.detach().cpu()))
                step += 1
                if step % 50 == 0:
                    print(f"epoch={epoch + 1} step={step} loss={losses[-1]:.4f}")

            record = {
                "epoch": epoch + 1,
                "mean_train_loss": float(np.mean(losses)),
                "samples_by_source": source_counts,
                "elapsed_seconds": round(time.time() - started, 2),
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            checkpoint = {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "optimizer": optimizer.state_dict(),
                "run_config": run_config,
            }
            torch.save(checkpoint, args.output_dir / f"Spartan_VRLM_epoch_{epoch + 1:03d}.pth.tar")
            print(json.dumps(record))


if __name__ == "__main__":
    main()
