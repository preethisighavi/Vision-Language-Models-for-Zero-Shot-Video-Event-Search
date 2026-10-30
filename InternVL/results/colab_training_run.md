# InternVL Training Run Record

This record documents the 10-epoch cross-modal contrastive fine-tuning progression of the InternVL model across the MSVD and MSR-VTT training splits:

| Epoch | Training Loss |
|---:|---:|
| 1 | 2.4232 |
| 2 | 2.3998 |
| 3 | 2.4071 |
| 4 | 2.3818 |
| 5 | 2.3811 |
| 6 | 2.3676 |
| 7 | 2.3653 |
| 8 | 2.3656 |
| 9 | 2.3458 |
| 10 | 2.3339 |

The training objective shows consistent convergence across epochs, reducing contrastive InfoNCE loss from 2.4232 down to 2.3339, yielding the final fine-tuned model checkpoint for zero-shot retrieval.
