import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from data_pipeline import SpartanPreprocessor
from model import load_spartan_vlm


def load_pairs(annotation_path):
    data = json.loads(annotation_path.read_text())
    if isinstance(data, list):
        pairs = []
        seen = set()
        for row in data:
            video = Path(row.get("video") or f"{row['video_id']}.mp4").name
            caption = row.get("caption")
            if isinstance(caption, list):
                caption = next((value for value in caption if isinstance(value, str) and value.strip()), None)
            if video not in seen and isinstance(caption, str) and caption.strip():
                pairs.append((video, caption.strip()))
                seen.add(video)
        return pairs

    captions = {}
    for row in data["sentences"]:
        captions.setdefault(row["video_id"], row["caption"])
    return [(f"{row['video_id']}.mp4", captions[row["video_id"]])
            for row in data["videos"] if row["video_id"] in captions]


def extract_frames(video_path, num_frames):
    capture = cv2.VideoCapture(str(video_path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count < 1:
        capture.release()
        raise ValueError(f"Could not read video metadata: {video_path}")

    frames = []
    for index in np.linspace(0, count - 1, num_frames, dtype=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    if not frames:
        raise ValueError(f"Could not decode video: {video_path}")
    frames.extend([frames[-1]] * (num_frames - len(frames)))
    return frames[:num_frames]


def resolve_videos(pairs, videos_dir, archive_path):
    if videos_dir.is_dir():
        missing = [name for name, _ in pairs if not (videos_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} videos, e.g. {missing[:5]}")
        return videos_dir, None

    if not archive_path.is_file():
        raise FileNotFoundError(f"Video directory or archive not found: {videos_dir}, {archive_path}")

    temp_dir = tempfile.TemporaryDirectory(prefix="spartan-videos-")
    wanted = {name for name, _ in pairs}
    with zipfile.ZipFile(archive_path) as archive:
        members = {Path(info.filename).name: info for info in archive.infolist()
                   if not info.is_dir() and Path(info.filename).name in wanted}
        missing = sorted(wanted - members.keys())
        if missing:
            temp_dir.cleanup()
            raise FileNotFoundError(f"Archive is missing {len(missing)} videos, e.g. {missing[:5]}")
        for name, info in members.items():
            with archive.open(info) as source, (Path(temp_dir.name) / name).open("wb") as destination:
                destination.write(source.read())
    return Path(temp_dir.name), temp_dir


def retrieval_metrics(similarity):
    order = np.argsort(-similarity, axis=1, kind="stable")
    ranks = (order == np.arange(len(order))[:, None]).argmax(axis=1) + 1
    return {
        "R@1": float(100 * np.mean(ranks <= 1)),
        "R@5": float(100 * np.mean(ranks <= 5)),
        "R@10": float(100 * np.mean(ranks <= 10)),
        "MRR": float(100 * np.mean(1 / ranks)),
        "Median Rank": float(np.median(ranks)),
        "Mean Rank": float(np.mean(ranks)),
        "Num Queries": int(len(ranks)),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Evaluate the SpartanVLM SpaceTimeTransformer checkpoint.")
    parser.add_argument("--annotations", type=Path, default=Path("../data/msrvtt/msrvtt_test_1k.json"))
    parser.add_argument("--videos", type=Path, default=Path("../data/msrvtt/videos"))
    parser.add_argument("--archive", type=Path, default=Path("../data/msrvtt/test_videos.zip"))
    parser.add_argument("--checkpoint", type=Path, default=Path("Spartan_VRLM.pth.tar"))
    parser.add_argument("--output", type=Path, default=Path("results/spartan_vlm_msrvtt.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if not args.annotations.is_file():
        raise FileNotFoundError(f"Annotations not found: {args.annotations}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"SpartanVLM checkpoint not found: {args.checkpoint}")

    pairs = load_pairs(args.annotations)
    video_dir, temp_dir = resolve_videos(pairs, args.videos, args.archive)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    preprocessor = SpartanPreprocessor(max_seq_length=32)
    model = load_spartan_vlm(args.checkpoint, device)

    video_embeds, text_embeds = [], []
    try:
        with torch.inference_mode():
            for start in tqdm(range(0, len(pairs), args.batch_size), desc=f"Encoding {args.annotations.parent.name}"):
                batch = pairs[start:start + args.batch_size]
                pixel_values = torch.stack([
                    torch.stack([preprocessor.vision_transform(frame) for frame in
                                 extract_frames(video_dir / filename, model.num_frames)])
                    for filename, _ in batch
                ]).to(device)
                input_ids, attention_mask = preprocessor.process_text([caption for _, caption in batch])
                video_embeds.append(model.encode_video(pixel_values).cpu())
                text_embeds.append(model.encode_text(input_ids.to(device), attention_mask.to(device)).cpu())
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    videos = F.normalize(torch.cat(video_embeds), dim=1)
    texts = F.normalize(torch.cat(text_embeds), dim=1)
    result = {
        "model": "SpartanVLM",
        "architecture": "SpartanVLM SpaceTimeTransformer + DistilBERT dual encoder",
        "dataset": args.annotations.parent.name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_provenance": "SpartanVLM trained checkpoint (4-frame SpaceTimeTransformer + DistilBERT at epoch 6)",
        "protocol": "text-to-video retrieval; one caption per video",
        "caption_selection": "first non-empty caption per video for datasets with multiple captions",
        "frames_per_video": model.num_frames,
        "input_resolution": 224,
        "metrics": retrieval_metrics((texts @ videos.T).numpy()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
