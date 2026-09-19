# Vision-Language Models for Zero-Shot Video & Event Search

DATA 298A/298B — San José State University, Department of Applied Data Science
Team 1: Abhinav Vummidichetty, Ariel Hsieh, Joseph Chang, Preethi Sighavi, Vaheedur Rehman Mahmed

## Problem

Searching for a specific moment inside a video is slow and manual. Most systems that make video content searchable need expensive, task-specific labeled training data. This project builds an end-to-end **zero-shot** video and event search system: given a natural-language query, it retrieves the most relevant video segment and an approximate timestamp, without training on labeled examples for that specific content.

## Models

Four vision-language approaches are compared under a shared data pipeline and evaluation framework:

| Model | Description |
|---|---|
| **CLIP** | Baseline + fine-tuned zero-shot retrieval (frame embeddings + cosine similarity) |
| **BLIP** | Fine-tuned image-captioning model (Query Transformer + frozen LLM) |
| **Flamingo-inspired** | Perceiver Resampler (frozen CLIP + frozen TinyLlama-1.1B) trained with contrastive loss |
| **Spartan VLM** | Custom architecture — layer-wise cross-modal fusion + joint retrieval/temporal-span prediction |

## Datasets

- **MSR-VTT** — ~10,000 video clips, 20 categories, multiple human captions per video
- **MSVD** — ~1,970 short video clips with official train/val/test splits

See `DATA/` for the processing pipeline (Airflow-based ingestion, deduplication, CLIP-quality filtering, keyframe extraction, train/val/test split construction).

## Repository structure

```
.
├── pipeline/              # Core feature-extraction pipeline (CLIP embeddings, Pinecone indexing)
├── DATA/                  # Datasets, EDA notebooks, and the Airflow data-engineering pipeline
├── Flamingo/              # Flamingo-inspired model: architecture, training, evaluation notebooks
├── Prototype/             # FastAPI + ChromaDB web app for interactive video search
├── EDA/                   # Exploratory data analysis dashboard and report
├── *.ipynb                # CLIP/BLIP fine-tuning, evaluation, and inference notebooks
├── requirements.txt       # Dependencies for pipeline/, Flamingo/, and top-level notebooks
```

Model weights, raw video files, and large intermediate data are intentionally excluded from this repository (see `.gitignore`) — they're too large for Git and are managed separately.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For the Prototype web app specifically:
```bash
cd Prototype
pip install -r requirements.txt
uvicorn main:app --reload
```

For the Airflow data pipeline specifically, see `DATA/vlm_pipeline/requirements.txt` and its `docker-compose.yml`.

## Status

See [`PROJECT_STATUS_REFERENCE.md`](PROJECT_STATUS_REFERENCE.md) and [`298B_MASTER_PLAN.md`](298B_MASTER_PLAN.md) for current progress, architecture details, and the remaining work for this semester.
