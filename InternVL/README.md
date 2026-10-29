# InternVL (Fine-Tuned)

## Overview

This repository contains the fine-tuned **InternVL** model for zero-shot text-to-video retrieval and search, along with its evaluation pipeline, benchmark results on MSR-VTT and MSVD, tokenizer assets, and fine-tuning artifacts.

---

## Model Architecture and Fine-Tuning

InternVL is a high-capacity vision-language retrieval model:
- **Vision Tower:** 24-layer Vision Transformer (ViT) with 14×14 spatial patch projection, hidden width of 1,024, and 16 attention heads. Processes 8 uniformly sampled video frames at 224×224 resolution.
- **Text Tower:** Multimodal text transformer mapping natural-language queries into a shared 512-dimensional embedding space.
- **Fine-Tuning Methodology:** Fine-tuned using cross-modal contrastive learning (Symmetric InfoNCE) with parameter-efficient Low-Rank Adaptation (LoRA) and representation preservation regularization, optimizing feature alignment while preserving generalization.

---

## Evaluation Benchmark Results

The fine-tuned model was evaluated on standard zero-shot video-text retrieval benchmarks under the full official evaluation protocol (1 query caption per video, ranked across the entire test gallery):

| Dataset | Split | R@1 | R@5 | R@10 | MRR | Median Rank | Mean Rank | Total Queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **MSR-VTT** | Test 1k | **42.00%** | **65.80%** | **75.30%** | **52.95%** | **2.0** | 22.30 | 1,000 |
| **MSVD** | Test | **65.82%** | **91.04%** | **95.67%** | **76.69%** | **1.0** | 3.02 | 670 |

Evaluation outputs and metric artifacts are saved under `results/`.

---

## Repository Files

- `train.py`: Command-line training script for cross-modal contrastive fine-tuning.
- `InternVideo2_Finetune_Colab.ipynb`: Complete fine-tuning notebook for Google Colab and cloud GPU environments.
- `evaluate.py`: Unified retrieval evaluation engine supporting baseline and fine-tuned modes on MSR-VTT and MSVD.
- `tokenizer/`: Tokenizer vocabulary and configuration files.
- `results/`: Machine-readable evaluation result JSONs and technical summary report.
- `checkpoints/`: Model configuration and checkpoint specifications.

---

## Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run fine-tuning:
```bash
python train.py --epochs 10 --batch-size 64
```

3. Run retrieval evaluation:

- **Baseline zero-shot evaluation:**
```bash
python evaluate.py --mode baseline
```

- **Fine-tuned adapter evaluation:**
```bash
python evaluate.py --mode finetuned --checkpoint checkpoints/best_internvideo2_adapter.pth
```
