# Vision-Language Models for Zero-Shot Video & Event Search

DATA 298A/298B — San José State University, Department of Applied Data Science
Team 1: Abhinav Vummidichetty, Ariel Hsieh, Joseph Chang, Preethi Sighavi, Vaheedur Rehman Mahmed

## Problem

Searching for a specific moment inside a video is slow and manual. Most systems that make video content searchable need expensive, task-specific labeled training data. This project builds an end-to-end **zero-shot** video and event search system: given a natural-language query, it retrieves the most relevant video segment and an approximate timestamp, without training on labeled examples for that specific content.

## Models

Five vision-language approaches are compared under a shared data pipeline and evaluation framework:

| Model | Folder | Description |
|---|---|---|
| **CLIP** | `CLIP/` | Baseline + fine-tuned zero-shot retrieval (ViT-B/32 frame embeddings + cosine similarity) |
| **BLIP** | `BLIP/` | Fine-tuned image-captioning model (Query Transformer + frozen LLM) |
| **Flamingo-inspired** | `Flamingo/` | Perceiver Resampler (frozen CLIP + frozen TinyLlama-1.1B) trained with contrastive loss |
| **SpartanVLM** | `SpartanVLM/` | Custom SpaceTimeTransformer + DistilBERT dual-encoder, trained from scratch |
| **InternVideo2-CLIP** | `InternVideo2_CLIP/` | High-capacity 24-layer ViT dual-encoder, fine-tuned with LoRA + contrastive learning |

## Benchmark Results

All models evaluated on the standard 1-caption-per-video text-to-video retrieval protocol (full test gallery). Metrics: Recall@K, Mean Reciprocal Rank (MRR), Median Rank.

### MSR-VTT Test (1k split — 1,000 queries)

| Model | R@1 | R@5 | R@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| CLIP (zero-shot) | ~25% | — | — | — | — |
| CLIP (fine-tuned) | — | — | — | — | — |
| BLIP | — | — | — | — | — |
| Flamingo-inspired | — | — | — | — | — |
| **SpartanVLM** | **21.60%** | **42.10%** | **54.40%** | **32.12%** | **8.0** |
| **InternVideo2-CLIP (fine-tuned)** | **42.00%** | **65.80%** | **75.30%** | **52.95%** | **2.0** |

### MSVD Test (670 queries)

| Model | R@1 | R@5 | R@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| **SpartanVLM** | **38.66%** | **69.40%** | **79.85%** | **52.55%** | **2.0** |
| **InternVideo2-CLIP (fine-tuned)** | **65.82%** | **91.04%** | **95.67%** | **76.69%** | **1.0** |

Full per-model results are in each model's `results/` folder (JSON + IEEE technical reports).

## Datasets

- **MSR-VTT** — ~10,000 video clips, 20 categories, multiple human captions per video
- **MSVD** — ~1,970 short video clips with official train/val/test splits
- **WebVid** — Large-scale web video dataset used for SpartanVLM and InternVideo2-CLIP training augmentation

See `DATA/` for the processing pipeline (Airflow-based ingestion, deduplication, CLIP-quality filtering, keyframe extraction, train/val/test split construction).

## Repository Structure

```
.
├── pipeline/              # Core feature-extraction pipeline (CLIP embeddings, Pinecone indexing)
├── DATA/                  # Datasets, EDA notebooks, and the Airflow data-engineering pipeline
├── CLIP/                  # CLIP zero-shot & fine-tuned retrieval notebooks and results
├── BLIP/                  # BLIP fine-tuning, evaluation notebooks and results
├── Flamingo/              # Flamingo-inspired model: architecture, training, evaluation notebooks
├── SpartanVLM/            # SpartanVLM: custom SpaceTimeTransformer, training, evaluation, results
├── InternVideo2_CLIP/     # InternVideo2-CLIP: LoRA fine-tuning, evaluation, Colab notebook, results
├── Prototype/             # FastAPI + ChromaDB web app for interactive video search
├── EDA/                   # Exploratory data analysis dashboard and report
├── Docs/                  # Project workbooks and documentation
├── requirements.txt       # Dependencies for pipeline/, Flamingo/, and top-level notebooks
```

Model weights, raw video files, ONNX model files, and large intermediate data are intentionally excluded from this repository (see `.gitignore`) — they're too large for Git and are managed separately.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For SpartanVLM specifically:
```bash
cd SpartanVLM
pip install -r requirements.txt
python train.py --msrvtt-annotations ../data/msrvtt/msrvtt_train_7k.json \
                --msrvtt-videos ../data/msrvtt/videos \
                --msvd-annotations ../data/msvd/msvd_train.json \
                --msvd-videos ../data/msvd/videos \
                --webvid-manifest ../data/webvid/manifest.csv \
                --webvid-videos ../data/webvid/videos
python evaluate.py --annotations ../data/msrvtt/msrvtt_test_1k.json \
                   --videos ../data/msrvtt/videos \
                   --checkpoint Spartan_VRLM.pth.tar
```

For InternVideo2-CLIP specifically:
```bash
cd InternVideo2_CLIP
pip install -r requirements.txt
python train.py --epochs 10 --batch-size 64
python evaluate_internvideo2_clip_finetuned.py
```

For the Prototype web app specifically:
```bash
cd Prototype
pip install -r requirements.txt
uvicorn main:app --reload
```

For the Airflow data pipeline specifically, see `DATA/vlm_pipeline/requirements.txt` and its `docker-compose.yml`.
