# Training Protocol

This file documents the training interface and defaults in train.py for the video--text retrieval component used by the larger zero-shot video-search project.

## Data sources and formats

The configured sources are:

1. MSR-VTT, using the project training annotation file, by default msrvtt_train_7k.json, and its referenced local videos.
2. MSVD, using msvd_train.json and its referenced local videos.
3. WebVid, using a CSV manifest and its referenced local videos.

The MSR-VTT and MSVD readers accept list-form annotations or video/sentence JSON structures. Captions are grouped by video and empty strings are discarded. The WebVid reader accepts video columns video_path, video, path, filename, or videoid, and caption columns caption, text, name, or title. The current WebVid reader does not parse Parquet.

The dataset loader checks that annotation-referenced video files exist. Missing videos and videos with no decodable frames raise errors. Each training source may be capped with max-samples-per-source. The data files themselves are not included in this repository.

## Video and caption preprocessing

OpenCV obtains the frame count and selects four evenly spaced frame indices from the start to the end of each video. Frames are converted from BGR to RGB. If a selected frame cannot be decoded, valid decoded frames are used and the final valid frame is repeated until four frames are available.

The vision transform resizes the shorter side to 256 pixels, center-crops to 256 by 256, resizes to 224 by 224, converts to a tensor, and applies ImageNet channel normalization with mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225).

The local DistilBERT tokenizer pads or truncates captions to 32 tokens and produces an attention mask. One non-empty caption is selected at random for each sampled video.

## Sampling and objective

The concatenated datasets are sampled with replacement using per-example weights proportional to 1 divided by the source size and number of sources. This gives each configured source equal expected sampling probability, while examples within a source are sampled uniformly. The default samples per epoch equal the sum of the configured source sizes; samples-per-epoch can override this count.

Video and text features are L2-normalized and compared with a fixed temperature of 0.07. For a batch of B aligned video-caption pairs, the diagonal of the B by B similarity matrix contains positive pairs and the off-diagonal entries are negatives. The objective is the mean of video-to-text and text-to-video cross-entropy losses.

## Initialization and optimization

If init-checkpoint is omitted, both encoders and their projection heads are randomly initialized. If init-checkpoint is provided, all model parameters are loaded with strict key and tensor-shape matching before optimization. These are different training configurations.

| Setting | Default |
|---|---:|
| Optimizer | AdamW |
| Learning rate, random initialization | 1e-4 |
| Learning rate, checkpoint initialization | 3e-5 |
| Weight decay | 0.01 |
| Learning-rate schedule | Cosine annealing |
| Batch size | 4 |
| Gradient accumulation | 1 |
| Gradient clipping | Global norm 1.0 |
| Precision | CUDA bfloat16 where supported; otherwise CUDA float16 with gradient scaling |
| Data loader workers | 4 |
| Random seed | 42 |

All model parameters are trainable. The implementation does not freeze either encoder and does not include warm-up scheduling or hard-negative mining. Gradient accumulation does not increase the in-batch negative set.

## Outputs

The output directory contains run_config.json, training_log.jsonl, and an epoch-numbered Spartan_VRLM checkpoint. The checkpoint records model parameters, optimizer state, epoch, and run configuration. Validation and test evaluation are performed separately with evaluate.py; training does not produce retrieval metrics.

## Invocation

Run from the SpartanVLM directory in a CUDA-enabled environment, with paths adjusted to the local dataset layout:

    python train.py --msrvtt-annotations ../data/msrvtt/msrvtt_train_7k.json --msrvtt-videos ../data/msrvtt/videos --msvd-annotations ../data/msvd/msvd_train.json --msvd-videos ../data/msvd/videos --webvid-manifest ../data/webvid/train.csv --webvid-videos ../data/webvid/videos --epochs 10 --output-dir runs/spartan_vrlm

To initialize from a checkpoint, add --init-checkpoint PATH. This selects checkpoint-initialized fine-tuning rather than random initialization.
