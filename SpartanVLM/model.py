import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.layers import DropPath, trunc_normal_
from transformers import DistilBertConfig, DistilBertModel


def _attention(q, k, v):
    return torch.matmul(q, k.transpose(-2, -1)).softmax(dim=-1) @ v


class VideoPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=4):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2 * num_frames
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        batch, frames, channels, height, width = x.shape
        if frames > self.num_frames:
            raise ValueError(f"Expected at most {self.num_frames} frames, received {frames}")
        return self.proj(x.reshape(batch * frames, channels, height, width))


class VarAttention(nn.Module):
    def __init__(self, dim=768, num_heads=12, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0,
                 initialize="random"):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if initialize == "zeros":
            nn.init.zeros_(self.qkv.weight)
            nn.init.zeros_(self.qkv.bias)
            nn.init.ones_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mode, frames, patches):
        batch, _, dim = x.shape
        head_dim = dim // self.num_heads
        q, k, v = self.qkv(x).reshape(batch, -1, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q = q * self.scale
        cls_out = _attention(q[:, :, :1], k, v)
        cls_k, cls_v = k[:, :, :1], v[:, :, :1]
        q, k, v = (t[:, :, 1:] for t in (q, k, v))

        if mode == "space":
            q, k, v = (rearrange(t, "b h (f n) d -> (b f h) n d", f=frames) for t in (q, k, v))
            repeats = frames
        elif mode == "time":
            q, k, v = (rearrange(t, "b h (f n) d -> (b n h) f d", n=patches) for t in (q, k, v))
            repeats = patches
        else:
            raise ValueError(f"Unknown attention mode: {mode}")

        cls_k = repeat(cls_k, "b h one d -> (b r h) one d", r=repeats)
        cls_v = repeat(cls_v, "b h one d -> (b r h) one d", r=repeats)
        k = torch.cat((cls_k, k), dim=1)
        v = torch.cat((cls_v, v), dim=1)
        out = _attention(q, k, v)

        if mode == "space":
            out = rearrange(out, "(b f h) n d -> b (f n) (h d)", b=batch, f=frames, h=self.num_heads)
        else:
            out = rearrange(out, "(b n h) f d -> b (f n) (h d)", b=batch, n=patches, h=self.num_heads)
        cls_out = rearrange(cls_out, "b h one d -> b one (h d)")
        out = torch.cat((cls_out, out), dim=1)
        return self.proj_drop(self.proj(out))


class Mlp(nn.Module):
    def __init__(self, in_features=768, hidden_features=3072, out_features=768, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SpaceTimeBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True, drop_path=0.0,
                 norm_layer=None, time_init="zeros", attention_style="space-time"):
        super().__init__()
        norm_layer = norm_layer or (lambda size: nn.LayerNorm(size, eps=1e-6))
        self.norm1 = norm_layer(dim)
        self.attn = VarAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.timeattn = VarAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, initialize=time_init)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim)
        self.norm3 = norm_layer(dim)
        self.attention_style = attention_style

    def forward(self, x, frames, patches):
        residual_input = x
        time_output = self.timeattn(self.norm3(x), "time", frames, patches)
        time_residual = residual_input + time_output
        space = self.attn(self.norm1(time_residual), "space", frames, patches)
        if self.attention_style != "space-time":
            raise ValueError(f"Unsupported attention style: {self.attention_style}")
        residual = residual_input + self.drop_path(space)
        return residual + self.drop_path(self.mlp(self.norm2(residual)))


class SpaceTimeTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_frames=4, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, qkv_bias=True, drop_path_rate=0.0, time_init="zeros",
                 attention_style="space-time"):
        super().__init__()
        self.num_classes = 0
        self.num_features = self.embed_dim = embed_dim
        self.num_frames = num_frames
        norm_layer = lambda size: nn.LayerNorm(size, eps=1e-6)
        self.patch_embed = VideoPatchEmbed(img_size, patch_size, 3, embed_dim, num_frames)
        self.patches_per_frame = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patches_per_frame + 1, embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, rates[i], norm_layer, time_init,
                           attention_style)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.pre_logits = nn.Identity()
        self.head = nn.Identity()
        self.fc = nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch, frames = x.shape[:2]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x.reshape(batch, -1, self.embed_dim)
        x = torch.cat((self.cls_token.expand(batch, -1, -1), x), dim=1)
        cls_pos = self.pos_embed[:, :1]
        spatial_pos = self.pos_embed[:, 1:].repeat(1, self.num_frames, 1)
        temporal_pos = self.temporal_embed.repeat_interleave(self.patches_per_frame, dim=1)
        positions = torch.cat((cls_pos, spatial_pos + temporal_pos), dim=1)
        x = self.pos_drop(x + positions[:, :x.shape[1]])
        for block in self.blocks:
            x = block(x, frames, self.patches_per_frame)
        return self.head(self.pre_logits(self.norm(x)[:, 0]))


class SpartanVLM(nn.Module):
    """SpartanVLM SpaceTimeTransformer and DistilBERT dual-encoder model for video-text retrieval."""

    def __init__(self, num_frames=4):
        super().__init__()
        self.num_frames = num_frames
        text_config = DistilBertConfig.from_pretrained(
            Path(__file__).with_name("tokenizer"), local_files_only=True
        )
        self.text_model = DistilBertModel(text_config)
        self.video_model = SpaceTimeTransformer(num_frames=num_frames)
        self.txt_proj = nn.Sequential(nn.ReLU(), nn.Linear(768, 256))
        self.vid_proj = nn.Sequential(nn.Linear(768, 256))

    def encode_video(self, video):
        return self.vid_proj(self.video_model(video))

    def encode_text(self, input_ids, attention_mask):
        features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.txt_proj(features)


def load_checkpoint_state(path):
    from pathlib import PosixPath

    prior_module = sys.modules.get("parse_config")
    module = types.ModuleType("parse_config")
    config_class = type("ConfigParser", (), {})
    config_class.__module__ = "parse_config"
    module.ConfigParser = config_class
    sys.modules["parse_config"] = module

    try:
        with torch.serialization.safe_globals([config_class, PosixPath]):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    finally:
        if prior_module is None:
            sys.modules.pop("parse_config", None)
        else:
            sys.modules["parse_config"] = prior_module

    state = checkpoint.get("state_dict", checkpoint)
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_spartan_vlm(checkpoint_path, device="cpu"):
    model = SpartanVLM()
    model.load_state_dict(load_checkpoint_state(Path(checkpoint_path)), strict=True)
    return model.to(device).eval()
