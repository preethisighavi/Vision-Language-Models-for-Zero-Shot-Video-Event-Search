"""Cross-modal contrastive fine-tuning for InternVL."""

import os
import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import onnxruntime as ort
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
DATASETS = ("msvd", "msrvtt")


def load_encoder(vision_path, text_path, tokenizer_dir):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    vision = ort.InferenceSession(str(vision_path), providers=providers)
    text = ort.InferenceSession(str(text_path), providers=providers)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    return vision, text, tokenizer


class Encoder:
    def __init__(self, vision_sess, text_sess, tokenizer):
        self.vision_sess = vision_sess
        self.text_sess = text_sess
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_frames(self, video_path, num_frames=8):
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        idxs = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
        return torch.stack([self.transform(f) for f in frames[:num_frames]]).numpy()

    def encode_video(self, video_path):
        frames = self.extract_frames(video_path, num_frames=8)
        batch = np.expand_dims(frames, axis=0).astype(np.float32)
        out = self.vision_sess.run(["image_embedding"], {"image": batch})[0]
        norm = out / np.linalg.norm(out, axis=-1, keepdims=True)
        return norm[0]

    def encode_texts(self, captions, batch_size=64):
        all_embs = []
        for i in range(0, len(captions), batch_size):
            batch = captions[i:i+batch_size]
            enc = self.tokenizer(batch, padding="max_length", max_length=77, truncation=True, return_tensors="np")
            out = self.text_sess.run(["text_embedding"], {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64)
            })[0]
            norm = out / np.linalg.norm(out, axis=-1, keepdims=True)
            all_embs.append(norm)
        return np.concatenate(all_embs, axis=0)


def read_items(data_root, dataset, split="train"):
    suffix = "_7k" if dataset == "msrvtt" and split == "train" else "_1k" if dataset == "msrvtt" and split == "test" else ""
    ann_file = data_root / dataset / f"{dataset}_{split}{suffix}.json"
    rows = json.loads(ann_file.read_text())
    items = []
    video_dir = data_root / dataset / "videos"
    for r in rows:
        fn = r.get("video") or (r["video_id"] + (".avi" if dataset == "msvd" else ".mp4"))
        vpath = video_dir / Path(fn).name
        if vpath.is_file():
            caption = r["caption"]
            if isinstance(caption, list):
                caption = caption[0]
            items.append({"video": str(vpath), "caption": caption, "video_id": Path(fn).stem})
    return items


def feature_cache(cache_root, dataset, split, items, encoder):
    cache_dir = cache_root / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_video_features.npy"
    if cache_path.exists():
        return np.load(cache_path, allow_pickle=True).item()
    feats = {}
    unique_videos = list({it["video"] for it in items})
    for v in tqdm(unique_videos, desc=f"Caching {dataset} videos"):
        feats[v] = encoder.encode_video(v)
    np.save(cache_path, feats)
    return feats


def caption_features(items, encoder, train=True):
    captions = [it["caption"] for it in items]
    return encoder.encode_texts(captions)


class VideoPairs(Dataset):
    def __init__(self, items, video_features, text_features, seed=42):
        self.items = items
        self.video_features = video_features
        self.text_features = text_features

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        v_feat = torch.from_numpy(self.video_features[item["video"]]).float()
        t_feat = torch.from_numpy(self.text_features[idx]).float()
        return v_feat, t_feat


class Adapter(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + 0.1 * self.net(x)


class RetrievalAdapters(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.video_adapter = Adapter(dim)
        self.text_adapter = Adapter(dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * float(np.log(1 / 0.07)))

    def forward(self, v, t):
        v = F.normalize(self.video_adapter(v), dim=-1)
        t = F.normalize(self.text_adapter(t), dim=-1)
        return v, t


def symmetric_loss(model, video, caption):
    v, t = model(video, caption)
    logit_scale = model.logit_scale.exp()
    logits_per_video = logit_scale * (v @ t.T)
    logits_per_text = logits_per_video.T
    labels = torch.arange(len(logits_per_video), device=video.device)
    loss = (F.cross_entropy(logits_per_video, labels) + F.cross_entropy(logits_per_text, labels)) / 2
    return loss


def main():
    parser = argparse.ArgumentParser(description="Fine-tune InternVL retrieval adapters.")
    parser.add_argument("--data-root", type=Path, default=BASE_DIR / ".." / "data")
    parser.add_argument("--vision-model", type=Path, default=BASE_DIR / "internvl_vision_fp16.onnx")
    parser.add_argument("--text-model", type=Path, default=BASE_DIR / "internvl_text_fp16.onnx")
    parser.add_argument("--tokenizer", type=Path, default=BASE_DIR / "tokenizer")
    parser.add_argument("--cache-dir", type=Path, default=BASE_DIR / "cache")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "checkpoints" / "internvl_finetuned")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")

    vision, text, tokenizer = load_encoder(args.vision_model, args.text_model, args.tokenizer)
    encoder = Encoder(vision, text, tokenizer)

    train_rows, video_features, text_features = {}, {}, {}
    for dataset in DATASETS:
        train_rows[dataset] = read_items(args.data_root, dataset, "train")
        video_features[dataset] = feature_cache(
            args.cache_dir, dataset, "train", train_rows[dataset], encoder
        )
        text_features[dataset] = caption_features(
            train_rows[dataset], encoder, train=True
        )
        print(f"{dataset}: {len(train_rows[dataset])} training caption rows")

    train_sets = {
        dataset: VideoPairs(
            train_rows[dataset],
            video_features[dataset],
            text_features[dataset],
            args.seed,
        )
        for dataset in DATASETS
    }
    loaders = {
        dataset: DataLoader(
            train_sets[dataset],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        for dataset in DATASETS
    }
    if any(len(loader) == 0 for loader in loaders.values()):
        raise RuntimeError("Each dataset needs at least one complete training batch of matched videos")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = RetrievalAdapters().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=2e-5)

    baseline_path = args.output_dir / "video2clip_epoch_00_baseline.pth"
    torch.save({"epoch": 0, "adapter_state_dict": model.state_dict()}, baseline_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        iterators = {dataset: iter(loaders[dataset]) for dataset in DATASETS}
        steps = min(len(loaders[dataset]) for dataset in DATASETS)
        losses = []

        for _ in tqdm(range(steps), desc=f"Epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            domain_losses = []

            for dataset in DATASETS:
                video, caption = next(iterators[dataset])
                domain_losses.append(
                    symmetric_loss(model, video.to(device), caption.to(device))
                )

            loss = torch.stack(domain_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                model.logit_scale.clamp_(max=float(np.log(100)))

            losses.append(loss.item())

        scheduler.step()
        checkpoint_path = args.output_dir / f"video2clip_epoch_{epoch:02d}.pth"
        torch.save(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "adapter_state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
        print(f"epoch={epoch} loss={np.mean(losses):.4f}; saved {checkpoint_path.name}")

    print(f"Training finished. Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
