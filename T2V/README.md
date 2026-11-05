# T2V — text-to-video generation

Two generators, compared under one evaluation, and served to the Spartan Search website:

| Model | What it is | Output |
|---|---|---|
| **SpartanGen-Tiny** (ours) | ~200M-param video diffusion transformer, **denoiser trained from scratch** on MSR-VTT + MSVD with flow matching; frozen Wan 2.1 VAE + frozen CLIP ViT-L/14 text encoder | 17 frames, 128×128, 3 fps (~5.3 s) |
| **Wan 2.1 T2V 1.3B** | open pretrained model (diffusers `WanPipeline`), the quality baseline | 33 frames, 480×832, 16 fps |

A generated clip is ingested into Spartan Search's `generated` collection, so the team's retrieval models
(and the Search page) can find it. The same retrieval models score generation quality (`eval/evaluate.py`).

## How SpartanGen works

```
prompt ──CLIP text encoder (frozen)──► tokens ─┐
                                              ▼ cross-attention
noise [16×5×16×16] ──► SpartanDiT × 50 Euler steps (CFG) ──► latent ──Wan VAE decoder (frozen)──► video
```

- **Data rate.** The team's MSR-VTT/MSVD copies (Hugging Face `VLM2Vec/*`, mirrored on the shared Drive) are
  298×224 at **3 fps**, so clips are sampled at 3 fps: 17 frames ≈ 5.3 s. Each latent frame then spans ~1.3 s, so
  fast motion blurs (VAE round trip: 22.9 dB on a dance clip vs 29 dB on slow scenes). `prepare_latents` warns if a
  source is slower than `--fps`. Using original 30 fps MSR-VTT would allow `--fps 8` for smoother motion.
- **Latents.** Wan 2.1 3D causal VAE: 8× space, 4× time, 16 channels. A 17-frame 128px clip becomes 16×5×16×16,
  normalised per channel. Everything is encoded once (`data/prepare_latents.py`), so training never touches pixels.
- **Denoiser** (`model/dit.py`). 1×2×2 patches → 320 tokens. 12 blocks of
  spatial self-attn (2D RoPE) → temporal self-attn (1D RoPE, tanh gate starting at 0) → text cross-attn → MLP,
  with adaLN-Zero conditioning on timestep + pooled text, and QK-RMSNorm.
- **Objective** (`model/flow.py`). Rectified flow: `x_t = (1−t)·x0 + t·ε`, predict `v = ε − x0`, logit-normal `t`.
  10% of captions are dropped (replaced by `""`) so classifier-free guidance works at sampling time.
- **Two stages.** A: single frames (T=1, temporal layers skipped) to learn appearance from 4 frames per clip.
  B: 17-frame clips, initialised from A; the temporal gates start at 0, so B begins as the image model.

Paper basis: DiT (2212.09748), Latte (2401.03048), Video LDM (2304.08818), Flow Matching (2210.02747),
SD3 (2403.03206), PixArt-α (2310.00426), CogVideoX (2408.06072), Wan (2503.20314), Open-Sora 2.0 (2503.09642).

## Run it (GPU machine, from the repo root)

```bash
pip install -r T2V/requirements.txt

# 0. sanity check on CPU, no downloads (≈2 min): must print PASS-level MSE (≈0.01)
python -m T2V.train --config T2V/configs/smoke.yaml --overfit-one-batch --steps 4000 --device cpu

# 1. clip manifest (reuses DATA/vlm_pipeline annotation loaders; video_id = file stem under the dataset dirs)
python -m T2V.data.manifest --msrvtt-dir /data/MSRVTT --msvd-dir /data/MSVD --out T2V/data/manifest.parquet

# 2. cache latents (once; resumable)
python -m T2V.data.prepare_latents --manifest T2V/data/manifest.parquet --out-dir /data/t2v/latents

# 3. overfit one real batch: loss → ~0 and samples reproduce the clips
python -m T2V.train --config T2V/configs/tiny_video.yaml --overfit-one-batch --steps 3000

# 4. stage A (images) then stage B (video); preview mp4s land in T2V/runs/*/samples/
python -m T2V.train --config T2V/configs/tiny_image.yaml
python -m T2V.train --config T2V/configs/tiny_video.yaml --init-from T2V/runs/tiny_image/last.pt
#    resume after an interruption: --resume T2V/runs/tiny_video/last.pt

# 5. sample
python -m T2V.sample --model spartangen-tiny --prompt "a man is playing a guitar" --out ours.mp4
python -m T2V.sample --model wan-1.3b        --prompt "a man is playing a guitar" --out wan.mp4

# 6. evaluate on 512 held-out MSR-VTT test captions; --log appends to results/model_results.csv
python -m T2V.eval.make_eval_set --manifest T2V/data/manifest.parquet --models spartangen-tiny wan-1.3b --n 512 --out T2V/eval_runs/msrvtt512
python -m T2V.eval.evaluate --set T2V/eval_runs/msrvtt512 --models spartangen-tiny wan-1.3b real --log
#    --clip <path> scores retrieval with the team's fine-tuned CLIP instead of openai/clip-vit-base-patch32
```

`overfit` checks print the loss at several noise levels with true and shuffled captions. Shuffled must be much
worse (proves text conditioning works); loss at t≈0.99 must also be low (that is where sampling starts).

## Export the trained model (run it anywhere)

Training checkpoints (`last.pt`, ~3 GB) include optimiser state for resuming. To run the model elsewhere
(your laptop, the website's server), export just the EMA weights in fp16 (~400 MB for the full model):

```bash
python -m T2V.export --ckpt T2V/runs/tiny_video/last.pt --out spartangen_tiny.pt --prefetch-encoders
python -m T2V.sample --model spartangen-tiny --ckpt spartangen_tiny.pt --prompt "a dog runs on the beach" --out dog.mp4
```

The frozen Wan VAE and CLIP text encoder are not in the file; they download from Hugging Face on first use
(~1.5 GB, cached afterwards). `--prefetch-encoders` caches them now, so the target machine can run offline if
you copy `~/.cache/huggingface` with the weights. The model picks CUDA, then Apple MPS, then CPU automatically.

## Serve to the website

```bash
# GPU machine
T2V_MODELS=spartangen-tiny,wan-1.3b SPARTANGEN_CKPT=T2V/runs/tiny_video/last.pt T2V_API_KEY=<secret> T2V_PRELOAD=1 \
  uvicorn T2V.serve:app --host 0.0.0.0 --port 8100

# Spartan Search backend (.env)
GENERATOR_URL=http://<gpu-host>:8100
GENERATOR_API_KEY=<secret>
```

The website's **Generate** page (`/generate`) lists the models from `/health`, queues a job, and shows
Generating → Decoded → Embedding → Indexed. When the GPU server is down the page says so and still shows past
generations. Backend: `app/api/generate.py`, `app/services/generation_service.py`, tests in `tests/test_generate.py`.

## Metrics

| Metric | Meaning | Better |
|---|---|---|
| CLIPScore | 100 × mean frame–caption cosine (CLIP B/32) | higher |
| R@1/5/10, MedR | retrieve the clip generated from caption *i* among all generated clips | higher / lower |
| FVD | Fréchet distance of I3D features vs. the real test clips (16 frames, 224px; StyleGAN-V detector) | lower |

`real` rows score the real test videos: the ceiling for CLIPScore and R@k. FVD on 512 clips is noisy and only
comparable within this protocol, not with numbers in papers (which use other sample counts and resolutions).

## Verified on real data (Apple M3, MPS)

6 real MSR-VTT test clips, real Wan VAE + CLIP, small DiT (hidden 256, depth 4), 3,000 overfit steps (11 min):
VAE round trip 26.7 dB mean PSNR; generations from captions alone reproduce their own clip (22–28 dB, each
nearest to the right original), motion within ~3% of the original; shuffled captions give 40–100× higher loss;
6 clips generated + decoded in 14 s. Data: `huggingface-cli download VLM2Vec/MSR-VTT --repo-type dataset`
(individual files, so a subset can be fetched with `--include`).

## Expectations and fallbacks

128px clips from ~12k training videos will be blurry but should match the prompt's scene, colours and rough motion.
If training diverges: check latent normalisation first, then drop lr to 5e-5 or the model to `hidden: 512, depth: 8`.
If it overfits (val samples copy training clips): add an OpenVid-1M subset or raise `caption_dropout`.
