# InternVL Fine-Tuned: MSR-VTT and MSVD Evaluation

**Technical Report**

## Abstract

This report presents the architecture, fine-tuning methodology, and benchmark evaluation results for the fine-tuned **InternVL** model on two standard video-text retrieval benchmarks: MSR-VTT (1,000 test video-caption queries) and MSVD (670 test queries). Under the full text-to-video retrieval protocol, the fine-tuned model achieves a Recall@1 of **42.00%**, Recall@5 of **65.80%**, and MRR of **52.95%** on MSR-VTT, alongside a Recall@1 of **65.82%**, Recall@5 of **91.04%**, and MRR of **76.69%** on MSVD.

**Index Terms—** InternVL, video-text retrieval, contrastive learning, MSVD, MSR-VTT, LoRA.

---

## I. Model Architecture

The model builds upon the InternVL vision-language dual-encoder architecture:
1. **Vision Transformer (ViT):** The visual backbone processes 8 uniformly sampled video frames at 224×224 resolution using 14×14 patch projection. It consists of 24 transformer blocks with hidden width 1,024 and 16 attention heads.
2. **Text Tower:** Encodes natural-language queries up to 77 tokens, mapped into a joint 512-dimensional embedding space.
3. **Cross-Modal Alignment:** Both video and text embeddings are L2-normalized; cosine similarity scores measure cross-modal alignment between queries and videos.

---

## II. Fine-Tuning Methodology

Fine-tuning is conducted using symmetric cross-modal contrastive learning (InfoNCE) combined with parameter-efficient fine-tuning (PEFT):
- **Parameter-Efficient Adaptation:** Low-Rank Adaptation (LoRA) targets query, key, value (`qkv`), and output projection (`proj`) weights in the vision transformer blocks, utilizing rank $r = 8$, scaling factor $\alpha = 16$, and dropout $0.05$.
- **Regularization:** Teacher-student feature preservation regularization is applied to prevent catastrophic feature distortion, incorporating embedding mean-squared error, cosine preservation, and cross-modal similarity matching.
- **Optimization:** Trained using the AdamW optimizer with cosine learning-rate decay, warm-up scheduling, and mixed precision.

---

## III. Datasets and Evaluation Protocol

The model was evaluated on two benchmark datasets:
- **MSR-VTT (1k Test Split):** 1,000 video-caption test pairs covering diverse open-domain activities and scenarios.
- **MSVD Test Split:** 670 video-caption test pairs of open-domain YouTube video clips.

### Retrieval Protocol
Videos are sampled uniformly across 8 frames at 224×224 resolution with bicubic interpolation and standard channel normalization. For each test caption query, cosine similarities are computed across the full test gallery, and candidate videos are ranked in descending order. Standard retrieval metrics are computed: Recall@1 (R@1), Recall@5 (R@5), Recall@10 (R@10), Mean Reciprocal Rank (MRR), Median Rank, and Mean Rank.

---

## IV. Evaluation Results

The evaluation results on MSR-VTT and MSVD are summarized below:

| Dataset | Evaluation State | R@1 | R@5 | R@10 | MRR | Median Rank | Mean Rank | Queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **MSR-VTT** | Baseline | 41.90% | 65.70% | 75.90% | 52.87% | 2.0 | — | 1,000 |
| **MSR-VTT** | **InternVL Fine-Tuned** | **42.00%** | **65.80%** | **75.30%** | **52.95%** | **2.0** | 22.30 | 1,000 |
| **MSVD** | Baseline | 65.82% | 91.04% | 95.52% | 76.72% | 1.0 | — | 670 |
| **MSVD** | **InternVL Fine-Tuned** | **65.82%** | **91.04%** | **95.67%** | **76.69%** | **1.0** | 3.02 | 670 |

---

## References

[1] Y. Wang *et al*., “InternVideo2: Scaling Foundation Models for Multimodal Video Understanding,” in *Proc. European Conf. Comput. Vis. (ECCV)*, 2024.

[2] A. Radford *et al*., “Learning Transferable Visual Models From Natural Language Supervision,” in *Proc. Int. Conf. Mach. Learn. (ICML)*, vol. 139, pp. 8748–8763, 2021.

[3] E. J. Hu *et al*., “LoRA: Low-Rank Adaptation of Large Language Models,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2022.

[4] D. L. Chen and W. B. Dolan, “Collecting Highly Parallel Data for Paraphrase Evaluation,” in *Proc. 49th Annu. Meeting Assoc. Comput. Linguistics: Human Language Technologies*, pp. 190–200, 2011.

[5] J. Xu, T. Mei, T. Yao, and Y. Rui, “MSR-VTT: A Large Video Description Dataset for Bridging Video and Language,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, pp. 5288–5296, 2016.
