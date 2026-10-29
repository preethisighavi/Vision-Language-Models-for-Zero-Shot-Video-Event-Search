"""Evaluate the fine-tuned InternVideo2-CLIP checkpoint on retrieval splits."""

import os
import json
import hashlib
import argparse
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import onnxruntime as ort
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent

EXPECTED_ENCODER_HASHES = {
    "iv2clip_vision_fp16.onnx": "141e726ae5dcf0bf7c3f21054e186ce8021410c403f73574e0d3dbb430be38f5",
    "iv2clip_text_fp16.onnx": "6e41fa1c7fb94e6b438375048603f818dc6c6b5ee35f3d7259905905a3760f0b",
}


class InternVideo2CLIP:
    def __init__(self, vision_model_path, text_model_path, tokenizer_dir, check_hashes=False):
        vision_model_path = Path(vision_model_path)
        text_model_path = Path(text_model_path)
        tokenizer_dir = Path(tokenizer_dir)

        if not vision_model_path.is_file():
            raise FileNotFoundError(f"Vision model not found at {vision_model_path}")
        if not text_model_path.is_file():
            raise FileNotFoundError(f"Text model not found at {text_model_path}")
        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(f"Tokenizer directory not found at {tokenizer_dir}")

        if check_hashes:
            for path in (vision_model_path, text_model_path):
                expected = EXPECTED_ENCODER_HASHES.get(path.name)
                if expected:
                    digest = hashlib.sha256()
                    with open(path, "rb") as model_file:
                        for chunk in iter(lambda: model_file.read(8 * 1024 * 1024), b""):
                            digest.update(chunk)
                    actual = digest.hexdigest()
                    if actual != expected:
                        print(f"Warning: SHA256 checksum mismatch for {path.name} (got {actual}, expected {expected})")

        print("Loading InternVideo2-CLIP ONNX inference sessions...")
        providers = ['CoreMLExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.text_sess = ort.InferenceSession(str(text_model_path), providers=providers)
            self.vis_sess = ort.InferenceSession(str(vision_model_path), providers=providers)
        except Exception as exc:
            print(f"Hardware-accelerated provider initialization fallback to CPU: {exc}")
            self.text_sess = ort.InferenceSession(str(text_model_path), providers=['CPUExecutionProvider'])
            self.vis_sess = ort.InferenceSession(str(vision_model_path), providers=['CPUExecutionProvider'])
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_frames(self, video_path: str, num_frames: int = 8) -> np.ndarray:
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

        tensor_frames = torch.stack([self.transform(f) for f in frames[:num_frames]])
        return tensor_frames.numpy()

    def encode_video(self, video_path: str) -> np.ndarray:
        frames = self.extract_frames(video_path, num_frames=8)
        batch = np.expand_dims(frames, axis=0).astype(np.float32)
        out = self.vis_sess.run(['image_embedding'], {'image': batch})[0]
        norm = out / np.linalg.norm(out, axis=-1, keepdims=True)
        return norm

    def encode_texts(self, captions: list, batch_size: int = 64) -> np.ndarray:
        all_embs = []
        for i in range(0, len(captions), batch_size):
            batch = captions[i:i+batch_size]
            enc = self.tokenizer(batch, padding='max_length', max_length=77, truncation=True, return_tensors='np')
            out = self.text_sess.run(['text_embedding'], {
                'input_ids': enc['input_ids'].astype(np.int64),
                'attention_mask': enc['attention_mask'].astype(np.int64)
            })[0]
            norm = out / np.linalg.norm(out, axis=-1, keepdims=True)
            all_embs.append(norm)
        return np.concatenate(all_embs, axis=0)


def compute_metrics(sim_matrix: np.ndarray):
    n = sim_matrix.shape[0]
    ranks = np.empty(n)
    for i in range(n):
        scores = sim_matrix[i]
        sorted_idx = np.argsort(-scores).tolist()
        ranks[i] = sorted_idx.index(i) + 1

    r1 = 100.0 * (ranks <= 1).mean()
    r5 = 100.0 * (ranks <= 5).mean()
    r10 = 100.0 * (ranks <= 10).mean()
    mrr = 100.0 * (1.0 / ranks).mean()
    medr = float(np.median(ranks))
    meanr = float(np.mean(ranks))

    return {
        "R@1": float(r1),
        "R@5": float(r5),
        "R@10": float(r10),
        "MRR": float(mrr),
        "Median Rank": medr,
        "Mean Rank": meanr,
        "Num Queries": int(n)
    }


def evaluate_dataset(model: InternVideo2CLIP, name: str, vids_dir: str, anno_file: str, output_dir: str, batch_size: int = 64):
    print(f"\n{'='*55}")
    print(f" Starting InternVideo2-CLIP Evaluation on: {name}")
    print(f"{'='*55}")

    if not os.path.isfile(anno_file):
        raise FileNotFoundError(f"Annotation file not found: {anno_file}")

    with open(anno_file) as f:
        data = json.load(f)

    valid_items = []
    for item in data:
        vid_file = item.get("video") or (item["video_id"] + (".avi" if "msvd" in name.lower() else ".mp4"))
        vid_path = os.path.join(vids_dir, vid_file)
        if os.path.exists(vid_path) and os.path.getsize(vid_path) > 0:
            caption = item["caption"]
            if isinstance(caption, list):
                caption = caption[0]
            valid_items.append({"path": vid_path, "caption": caption, "video": vid_file})

    n = len(valid_items)
    if n == 0:
        raise RuntimeError(f"No valid video-caption pairs found for {name} under {vids_dir}")

    print(f"Loaded {n} verified test video-caption pairs for {name}.")

    # Step 1: Encode all videos
    print(f"Encoding {n} videos...")
    v_embs = []
    for it in tqdm(valid_items, desc=f"{name} Videos"):
        v_embs.append(model.encode_video(it["path"]))
    V = np.concatenate(v_embs, axis=0) # (N, 512)

    # Step 2: Encode all captions
    print(f"Encoding {n} captions...")
    captions = [it["caption"] for it in valid_items]
    T = model.encode_texts(captions, batch_size=batch_size) # (N, 512)

    # Step 3: Compute cosine similarity matrix
    print("Computing similarity matrix...")
    sim_matrix = T @ V.T # (N, N)

    # Step 4: Compute metrics
    metrics = compute_metrics(sim_matrix)

    print("\n" + "-"*48)
    print(f"  InternVideo2-CLIP Results: {name}")
    print("-"*48)
    print(f"  Queries Evaluated : {metrics['Num Queries']}")
    print(f"  Recall@1 (R@1)    : {metrics['R@1']:.2f}%")
    print(f"  Recall@5 (R@5)    : {metrics['R@5']:.2f}%")
    print(f"  Recall@10 (R@10)  : {metrics['R@10']:.2f}%")
    print(f"  MRR               : {metrics['MRR']:.2f}%")
    print(f"  Median Rank (MedR): {metrics['Median Rank']:.1f}")
    print(f"  Mean Rank (MeanR) : {metrics['Mean Rank']:.1f}")
    print("-"*48)

    metrics.update({
        "display_name": "InternVideo2-CLIP fine-tuned",
        "model": "InternVideo2-CLIP",
        "protocol": "text-to-video retrieval; one caption per video",
    })
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"internvideo2_clip_finetuned_{name.lower().replace('-', '_')}.json")
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved results to {out_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned InternVideo2-CLIP model on retrieval benchmarks.")
    parser.add_argument("--data-root", type=Path, default=BASE_DIR / ".." / "data", help="Root directory containing msrvtt and msvd data")
    parser.add_argument("--vision-model", type=Path, default=BASE_DIR / "iv2clip_vision_fp16.onnx", help="Path to vision ONNX model")
    parser.add_argument("--text-model", type=Path, default=BASE_DIR / "iv2clip_text_fp16.onnx", help="Path to text ONNX model")
    parser.add_argument("--tokenizer", type=Path, default=BASE_DIR / "tokenizer", help="Path to tokenizer directory")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "results", help="Directory to save evaluation results")
    parser.add_argument("--dataset", choices=["all", "msrvtt", "msvd"], default="all", help="Dataset(s) to evaluate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for text encoding")
    parser.add_argument("--check-hashes", action="store_true", help="Verify SHA256 hashes of ONNX model files")
    args = parser.parse_args()

    model = InternVideo2CLIP(
        vision_model_path=args.vision_model,
        text_model_path=args.text_model,
        tokenizer_dir=args.tokenizer,
        check_hashes=args.check_hashes,
    )

    msrvtt_vids = args.data_root / "msrvtt" / "videos"
    msrvtt_anno = args.data_root / "msrvtt" / "msrvtt_test_1k.json"

    msvd_vids = args.data_root / "msvd" / "videos"
    msvd_anno = args.data_root / "msvd" / "msvd_test.json"

    if args.dataset in ("all", "msrvtt"):
        evaluate_dataset(model, "MSR-VTT", str(msrvtt_vids), str(msrvtt_anno), str(args.output_dir), args.batch_size)

    if args.dataset in ("all", "msvd"):
        evaluate_dataset(model, "MSVD", str(msvd_vids), str(msvd_anno), str(args.output_dir), args.batch_size)

    print("\n" + "="*55)
    print(" INTERNVIDEO2-CLIP BENCHMARK RUN COMPLETED!")
    print("="*55)


if __name__ == "__main__":
    main()
