# InternVL Fine-Tuned Checkpoint

## Overview

This directory documents the fine-tuned **InternVL** model checkpoint (`InternVL_Finetuned.safetensors`).

The model is adapted for zero-shot text-to-video retrieval, mapping video clips and natural-language queries into a shared 512-dimensional embedding space.

---

## Checkpoint Details

- **Model Name:** InternVL Fine-Tuned
- **Checkpoint File:** `InternVL_Finetuned.safetensors`
- **SHA-256 Checksum:** `d3f3401d5da979587382c29705b6eb260f41b452708b7da2b705bf8f5970c7ea`
- **Output Embedding Dimension:** 512 (L2-normalized)

---

## Architecture & Adaptation

- **Vision Encoder:** 24-block Vision Transformer (ViT-L/14) with 1,024 hidden width and 16 attention heads. Input frames: 8 uniformly sampled frames at 224×224 resolution.
- **Text Encoder:** Multimodal sequence encoder with max sequence length 77 tokens.
- **Adaptation Strategy:** Parameter-efficient Low-Rank Adaptation (LoRA) targeting the projection layers (`qkv`, `proj`) in the last 12 vision blocks with rank $r=8$, scaling factor $\alpha=16$, and dropout $0.05$.
- **Objective:** Symmetric cross-modal InfoNCE contrastive learning with teacher-student feature preservation regularization.

---

## Benchmark Performance

| Benchmark | Split | R@1 | R@5 | R@10 | MRR | Median Rank |
|---|---|---:|---:|---:|---:|---:|
| **MSR-VTT** | 1,000 queries | **42.00%** | **65.80%** | **75.30%** | **52.95%** | **2.0** |
| **MSVD** | 670 queries | **65.82%** | **91.04%** | **95.67%** | **76.69%** | **1.0** |
