"""
model.py — Flamingo-inspired video-language model
Perceiver Resampler + dual encoder contrastive retrieval.

Architecture:
  Video: CLIP frame embeddings → PerceiverResampler → mean pool → video_proj → L2 norm
  Text:  tokens → frozen TinyLlama → mean pool → text_proj → L2 norm
  Loss:  symmetric InfoNCE contrastive

FROZEN:  TinyLlama LLM
TRAINED: PerceiverResampler, video_proj, text_proj, logit_scale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── 1. Perceiver Layer ────────────────────────────────────────────────────────

class PerceiverLayer(nn.Module):
    """Single cross-attention + feed-forward block inside the Perceiver."""

    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.xattn   = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, latents, context, key_padding_mask=None):
        attn_out, _ = self.xattn(
            query=self.norm_q(latents),
            key=self.norm_kv(context),
            value=self.norm_kv(context),
            key_padding_mask=key_padding_mask,
        )
        latents = latents + attn_out
        latents = latents + self.ff(latents)
        return latents


# ── 2. Perceiver Resampler ────────────────────────────────────────────────────

class PerceiverResampler(nn.Module):
    """
    Compresses (B, T, clip_dim) → (B, num_latents, clip_dim).

    Core Flamingo innovation: variable-length frame embeddings compressed
    into a fixed number of latent tokens via learned cross-attention.
    TRAINED during training.
    """

    def __init__(self, dim=768, num_latents=64, depth=6, heads=8, dim_head=64):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.layers  = nn.ModuleList(
            [PerceiverLayer(dim, heads, dim_head) for _ in range(depth)])
        self.norm    = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        """
        x    : (B, T, dim) — CLIP frame embeddings
        mask : (B, T)      — 1 for real frames, 0 for padding
        returns (B, num_latents, dim)
        """
        B = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        kpm = (~mask.bool()) if mask is not None else None
        for layer in self.layers:
            latents = layer(latents, x, key_padding_mask=kpm)
        return self.norm(latents)


# ── 3. Full Model ─────────────────────────────────────────────────────────────

class FlamingoVideoModel(nn.Module):
    """
    Flamingo-inspired video-language model for contrastive retrieval.

    FROZEN:  llm (TinyLlama) — forward pass only, weights never updated
    TRAINED: perceiver, video_proj, text_proj, logit_scale
    """

    def __init__(self, llm, llm_dim, num_llm_layers,
                 perceiver_latents=64, perceiver_depth=6,
                 xattn_heads=8, xattn_every_n=2,
                 clip_dim=768, proj_dim=512):
        super().__init__()

        self.llm     = llm
        self.llm_dim = llm_dim

        # Perceiver Resampler — trained
        self.perceiver = PerceiverResampler(
            dim=clip_dim,
            num_latents=perceiver_latents,
            depth=perceiver_depth,
            heads=xattn_heads,
        )

        # Projection heads — trained
        self.video_proj  = nn.Linear(clip_dim, proj_dim)
        self.text_proj   = nn.Linear(llm_dim,  proj_dim)

        # Learnable temperature — same init as CLIP (log(1/0.07) ≈ 2.66)
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    def encode_video(self, visual_embeds, frame_masks=None):
        """
        visual_embeds : (B, T, clip_dim) — pre-extracted CLIP frame features
        frame_masks   : (B, T)           — 1=real frame, 0=padding
        returns       : (B, proj_dim) L2-normalised video embedding
        """
        vis_tokens = self.perceiver(visual_embeds, frame_masks)  # (B, N, clip_dim)
        video_rep  = vis_tokens.mean(dim=1)                      # (B, clip_dim)
        video_emb  = self.video_proj(video_rep)                  # (B, proj_dim)
        return F.normalize(video_emb, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        """
        Runs text through the frozen LLM (forward pass only, no gradient).
        returns : (B, proj_dim) L2-normalised text embedding
        """
        with torch.no_grad():
            out = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden   = out.hidden_states[-1]                              # (B, S, llm_dim)
        mask     = attention_mask.unsqueeze(-1).float()
        text_rep = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1) # (B, llm_dim)
        text_emb = self.text_proj(text_rep.float())                   # (B, proj_dim)
        return F.normalize(text_emb, dim=-1)

    def forward(self, visual_embeds, frame_masks, input_ids, attention_mask):
        """Symmetric InfoNCE contrastive loss."""
        vid_emb = self.encode_video(visual_embeds, frame_masks)
        txt_emb = self.encode_text(input_ids, attention_mask)

        scale  = self.logit_scale.exp().clamp(max=100)
        logits = scale * vid_emb @ txt_emb.T
        labels = torch.arange(len(logits), device=logits.device)

        loss = (F.cross_entropy(logits,   labels)
              + F.cross_entropy(logits.T, labels)) / 2
        return loss, vid_emb, txt_emb


# ── 4. Config and builder ─────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    'llm_name'          : 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'perceiver_latents' : 64,
    'perceiver_depth'   : 6,
    'xattn_heads'       : 8,   # used by Perceiver internally
    'xattn_every_n'     : 2,   # kept for config compatibility, not used in model
    'clip_dim'          : 768,
    'proj_dim'          : 512,
}


def build_model(config=None, device=None):
    """
    Loads TinyLlama (frozen) and wraps in FlamingoVideoModel.
    Returns (model, tokenizer).
    """
    if config is None:
        config = DEFAULT_CONFIG
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(config['llm_name'])
    tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        config['llm_name'],
        torch_dtype=torch.float16,
        device_map='auto',
    )
    for p in llm.parameters():
        p.requires_grad = False

    llm_dim        = llm.config.hidden_size
    num_llm_layers = llm.config.num_hidden_layers

    model = FlamingoVideoModel(
        llm=llm,
        llm_dim=llm_dim,
        num_llm_layers=num_llm_layers,
        perceiver_latents=config['perceiver_latents'],
        perceiver_depth=config['perceiver_depth'],
        xattn_heads=config['xattn_heads'],
        xattn_every_n=config['xattn_every_n'],
        clip_dim=config['clip_dim'],
        proj_dim=config['proj_dim'],
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'LLM: {config["llm_name"]} | hidden={llm_dim} | layers={num_llm_layers}')
    print(f'Parameters — total: {total:,} | trainable: {trainable:,} | frozen: {total-trainable:,}')

    return model, tokenizer
