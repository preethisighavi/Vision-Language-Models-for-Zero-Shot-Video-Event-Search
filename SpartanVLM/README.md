# SpartanVLM

## Overview

SpartanVLM is a lightweight, efficient video-language dual-encoder model designed for clip-level text-to-video retrieval and search. It encodes videos and natural-language text queries into a shared embedding space, ranking candidate videos via cosine similarity.

---

## Architecture

SpartanVLM implements an end-to-end dual-encoder architecture:

1. **Video Encoder (SpaceTimeTransformer):**
   - Processes 4 uniformly sampled video frames resized to 224×224 resolution.
   - Divides each frame into 16×16 spatial patches (196 patches per frame).
   - 12 transformer layers, 768 hidden dimensions, and 12 attention heads with divided space-time self-attention to capture spatial appearance and temporal dynamics.
   - Extracts a global 768-dimensional visual representation from the class token.

2. **Text Encoder (DistilBERT):**
   - 6-layer transformer with 768 hidden dimensions.
   - Processes input captions tokenized up to 32 tokens.
   - Extracts a 768-dimensional sentence embedding from the `[CLS]` token.

3. **Multimodal Projections:**
   - Visual projection head (`vid_proj`) maps the 768-dimensional video features to a 256-dimensional embedding.
   - Text projection head (`txt_proj`) maps the 768-dimensional text features to a 256-dimensional embedding.
   - Embeddings are L2-normalized; video retrieval is performed via cosine similarity scoring.

---

## Checkpoint and Benchmark Results

The trained model checkpoint is `Spartan_VRLM.pth.tar` (SHA-256: `39a7ca84c1a7976d805d5eebd59e56326003d6d8098cff20c4bc92b7dad146da`).

The model was evaluated on standard zero-shot video-text retrieval benchmarks under the full test protocol (1 caption per video, ranking across the entire test gallery):

| Dataset | Split | R@1 | R@5 | R@10 | MRR | Median Rank | Mean Rank | Queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **MSR-VTT** | Test 1k | **21.60%** | **42.10%** | **54.40%** | **32.12%** | **8.0** | 51.69 | 1,000 |
| **MSVD** | Test | **38.66%** | **69.40%** | **79.85%** | **52.55%** | **2.0** | 12.55 | 670 |

Evaluation outputs and configuration metrics are stored in machine-readable JSON format under `results/`.

---

## Training Details

SpartanVLM is trained from scratch using the full pipeline provided in `train.py`:
- **Training Objective:** Symmetric in-batch cross-entropy contrastive loss (InfoNCE) with temperature $\tau = 0.07$.
- **Optimizer:** AdamW with weight decay $0.01$.
- **Schedule:** Cosine annealing learning-rate schedule.
- **Precision:** Mixed precision training (BFloat16 / Float16 with gradient scaling).
- **Supported Datasets:** Multi-source training supporting MSR-VTT, MSVD, and WebVid with source-balanced sampling.

Detailed training configurations and options are documented in [TRAINING.md](TRAINING.md).

---

## File Structure

- `model.py`: SpartanVLM architecture definition (`SpaceTimeTransformer`, `DistilBERT`, projection heads).
- `data_pipeline.py`: Frame extraction, video transforms, and text tokenization.
- `train.py`: End-to-end training implementation.
- `evaluate.py`: Retrieval evaluation engine for MSR-VTT and MSVD.
- `results/`: Evaluation benchmark results (`spartan_vlm_msrvtt.json`, `spartan_vlm_msvd.json`).
- `tokenizer/`: DistilBERT tokenizer vocabulary and configuration files.

---

## Evaluation

To evaluate a SpartanVLM checkpoint on MSR-VTT or MSVD:

```bash
python evaluate.py \
  --annotations ../data/msrvtt/msrvtt_test_1k.json \
  --videos ../data/msrvtt/videos \
  --checkpoint Spartan_VRLM.pth.tar \
  --output results/spartan_vlm_msrvtt.json
```
