# SpartanVLM: Architecture, Training from Scratch, and Evaluation

**Technical Report**

## Abstract

This report documents the architectural design, end-to-end training protocol, and empirical evaluation results for **SpartanVLM**, a lightweight video-language dual-encoder model developed for clip-level zero-shot text-to-video retrieval. SpartanVLM pairs a 12-block Space-Time Transformer with a 6-layer DistilBERT text encoder, projecting features into a shared 256-dimensional multimodal embedding space. Trained from scratch using symmetric in-batch contrastive learning (InfoNCE) across MSR-VTT, MSVD, and WebVid datasets, SpartanVLM achieves a Recall@1 of **21.60%**, Recall@5 of **42.10%**, and MRR of **32.12%** on the full 1,000-query MSR-VTT test benchmark, alongside a Recall@1 of **38.66%**, Recall@5 of **69.40%**, and MRR of **52.55%** on the 670-query MSVD test benchmark.

**Index Terms—** SpartanVLM, SpaceTimeTransformer, DistilBERT, video-text retrieval, contrastive learning, MSR-VTT, MSVD.

---

## I. Architecture Overview

SpartanVLM adopts a decoupled dual-encoder structure designed for efficient inference:
1. **Visual Backbone (SpaceTimeTransformer):**
   - Uniformly samples 4 video frames at 224×224 resolution.
   - Applies a 2D patch projection with 16×16 kernel and stride, yielding 196 spatial tokens per frame (784 tokens per clip).
   - Incorporates learned spatial positional embeddings and zero-initialized temporal embeddings.
   - Contains 12 transformer blocks with divided space-time attention (12 heads, hidden width 768, MLP width 3,072) to capture fine-grained motion and semantics.
   - A linear projection head maps the class token to a 256-dimensional embedding.
2. **Text Backbone (DistilBERT):**
   - Tokenizes input queries with a vocabulary of 30,522 tokens, padded or truncated to 32 tokens.
   - Employs 6 transformer layers (12 heads, hidden width 768).
   - The first token representation passes through a ReLU activation and linear projection to 256 dimensions.
3. **Similarity & Metric Space:**
   - Both visual and text embeddings are L2-normalized.
   - Cosine similarity with a fixed temperature $\tau = 0.07$ governs scoring and retrieval ranking.

---

## II. End-to-End Training Protocol

- **Initialization:** Trained from scratch using random parameter initialization.
- **Objective:** Symmetric in-batch InfoNCE cross-entropy loss aligning matched video-caption pairs while treating in-batch cross-pairs as negatives:
  $$\mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{v \rightarrow t} + \mathcal{L}_{t \rightarrow v} \right)$$
- **Optimization:** AdamW optimizer with cosine learning-rate decay, initial learning rate $1 \times 10^{-4}$, weight decay $0.01$, and global gradient norm clipping of $1.0$.
- **Numerical Precision:** Automatic mixed precision (CUDA bfloat16 / float16 with gradient scaling).

---

## III. Benchmark Evaluation Results

Evaluated under the standard 1-caption-per-video text-to-video retrieval protocol:

| Dataset | Split | R@1 | R@5 | R@10 | MRR | Median Rank | Mean Rank | Queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **MSR-VTT** | Test 1k | **21.60%** | **42.10%** | **54.40%** | **32.12%** | **8.0** | 51.69 | 1,000 |
| **MSVD** | Test | **38.66%** | **69.40%** | **79.85%** | **52.55%** | **2.0** | 12.55 | 670 |

---

## References

[1] V. Sanh *et al*., “DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,” *arXiv:1910.01108*, 2019.

[2] J. Xu *et al*., “MSR-VTT: A Large Video Description Dataset for Bridging Video and Language,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016.

[3] D. Chen and W. B. Dolan, “Collecting Highly Parallel Data for Paraphrase Evaluation,” in *Proc. 49th Annu. Meeting Assoc. Comput. Linguistics (ACL)*, 2011.

[4] I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2019.
