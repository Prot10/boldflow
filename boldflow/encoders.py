"""EEG encoders: REVE temporal backbone + Multi-Scale Spectral (MSS) branch."""
from __future__ import annotations

import itertools
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from linear_attention_transformer import LinearAttentionTransformer


# 10-20 electrode positions on the unit sphere (nose +x, left ear +y, vertex +z).
# Covers the 26 NeuroBOLT channels and the extra 8 used by OpenNeuroSleep.
# Lookup is case-insensitive (see ``_get_coord``).
EEG_CHANNEL_COORDS: Dict[str, Tuple[float, float, float]] = {
    # NeuroBOLT 26 channels
    "FP1": (0.950, 0.309, -0.035), "FP2": (0.950, -0.309, -0.035),
    "F3":  (0.673, 0.545, 0.500),  "F4":  (0.673, -0.545, 0.500),
    "C3":  (0.000, 0.707, 0.707),  "C4":  (0.000, -0.707, 0.707),
    "P3":  (-0.545, 0.673, 0.500), "P4":  (-0.545, -0.673, 0.500),
    "O1":  (-0.950, 0.309, -0.035),"O2":  (-0.950, -0.309, -0.035),
    "F7":  (0.587, 0.809, -0.035), "F8":  (0.587, -0.809, -0.035),
    "T7":  (0.000, 1.000, 0.000),  "T8":  (0.000, -1.000, 0.000),
    "P7":  (-0.587, 0.809, -0.035),"P8":  (-0.587, -0.809, -0.035),
    "FPZ": (1.000, 0.000, 0.000),  "FZ":  (0.707, 0.000, 0.707),
    "CZ":  (0.000, 0.000, 1.000),  "PZ":  (-0.707, 0.000, 0.707),
    "POZ": (-0.809, 0.000, 0.309), "OZ":  (-1.000, 0.000, 0.000),
    "FT9": (0.309, 0.950, -0.035), "FT10":(0.309, -0.950, -0.035),
    "TP9": (-0.309, 0.950, -0.035),"TP10":(-0.309, -0.950, -0.035),
    # Extra channels for the 30-channel sleep montage
    "FC1": (0.354, 0.354, 0.866),  "FC2": (0.354, -0.354, 0.866),
    "CP1": (-0.354, 0.354, 0.866), "CP2": (-0.354, -0.354, 0.866),
    "FC5": (0.354, 0.809, 0.469),  "FC6": (0.354, -0.809, 0.469),
    "CP5": (-0.354, 0.809, 0.469), "CP6": (-0.354, -0.809, 0.469),
}

# Default 26-channel order (NeuroBOLT layout).
DEFAULT_CHANNEL_ORDER = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "FPZ", "FZ", "CZ", "PZ",
    "POZ", "OZ", "FT9", "FT10", "TP9", "TP10",
)

# 30-channel order used by OpenNeuroSleep.
SLEEP_CHANNEL_ORDER = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "FZ", "CZ", "PZ", "OZ",
    "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6", "TP9", "TP10",
)


def _get_coord(name: str) -> Tuple[float, float, float]:
    """Case-insensitive lookup in ``EEG_CHANNEL_COORDS``."""
    upper = name.upper()
    if upper in EEG_CHANNEL_COORDS:
        return EEG_CHANNEL_COORDS[upper]
    raise KeyError(
        f"channel {name!r} not in EEG_CHANNEL_COORDS; add its 10-20 coordinate "
        f"to boldflow.encoders.EEG_CHANNEL_COORDS or pass a custom channel_order."
    )


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm; computed in float32 then cast back."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class GEGLU(nn.Module):
    """Gated linear unit: ``GELU(gate) * x``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


class _REVEAttention(nn.Module):
    """MHSA with RMSNorm pre-norm, no projection bias."""

    def __init__(self, dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.to_qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)


class _REVEFeedForward(nn.Module):
    """RMSNorm -> Linear -> GEGLU -> Linear, no bias."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # net.{0..3} key order matches the released REVE checkpoint.
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, hidden_dim * 2, bias=False),
            GEGLU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierEmb4D(nn.Module):
    """Frozen 4D Fourier features over (x, y, z, time) -> R^{dimension}.

    Enumerates (k_x, k_y, k_z, k_t) tuples, truncates to ``dimension // 2``,
    concatenates ``[cos, sin]`` so the output has exactly ``dimension`` channels.
    """

    def __init__(
        self,
        n_freqs: int = 4,
        dimension: int = 512,
        margin: float = 0.4,
        time_increment: float = 0.1,
    ):
        super().__init__()
        self.time_increment = time_increment
        freq_width = 1 + 2 * margin
        freqs_1d = [2 * math.pi * k / freq_width for k in range(n_freqs)]
        grid = list(itertools.product(freqs_1d, repeat=4))
        grid_t = torch.tensor(grid, dtype=torch.float32)
        n_keep = min(len(grid), dimension // 2)
        if len(grid) > n_keep:
            grid_t = grid_t[:n_keep]
        self.register_buffer("freq_grid", grid_t)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        pos = positions.clone()
        pos[..., 3] *= self.time_increment
        phase = torch.einsum("bnd,kd->bnk", pos, self.freq_grid)
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

    @staticmethod
    def add_time_patches(ch_pos: torch.Tensor, n_patches: int) -> torch.Tensor:
        """(B, C, 3) -> (B, C*H, 4) by appending a time index."""
        b, c, _ = ch_pos.shape
        t = torch.arange(n_patches, device=ch_pos.device, dtype=ch_pos.dtype)
        pos = ch_pos.unsqueeze(2).expand(b, c, n_patches, 3)
        t_exp = t.view(1, 1, n_patches, 1).expand(b, c, n_patches, 1)
        return torch.cat([pos, t_exp], dim=-1).reshape(b, c * n_patches, 4)


class _AttentionPooling(nn.Module):
    """Pool tokens to a single vector via cross-attention with one learned query."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        q = self.query.expand(b, -1, -1)
        out, _ = self.cross_attn(q, x, x)
        return self.norm(out.squeeze(1))


class REVEEncoder(nn.Module):
    """REVE EEG encoder. ``(B, C, T)`` z-scored EEG -> ``(B, embed_dim)``."""

    def __init__(
        self,
        embed_dim: int = 512,
        depth: int = 22,
        heads: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 2.66,
        patch_size: int = 200,
        patch_stride: int = 180,
        n_fourier_freqs: int = 4,
        channel_order: tuple[str, ...] = DEFAULT_CHANNEL_ORDER,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        hidden_dim = int(embed_dim * mlp_ratio)

        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_size, embed_dim))
        self.fourier4d = FourierEmb4D(n_fourier_freqs, embed_dim)
        self.mlp4d = nn.Sequential(
            nn.Linear(4, embed_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.ln = nn.LayerNorm(embed_dim)

        # Layout matches the released REVE checkpoint: transformer.layers.{i}.{0,1}.
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList([
            nn.ModuleList([
                _REVEAttention(embed_dim, heads, head_dim),
                _REVEFeedForward(embed_dim, hidden_dim),
            ])
            for _ in range(depth)
        ])

        self.attn_pool = _AttentionPooling(embed_dim, heads)
        coords = torch.tensor(
            [_get_coord(ch) for ch in channel_order], dtype=torch.float32,
        )
        self.register_buffer("channel_coords", coords)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        # The slice ``self.channel_coords[:c]`` would silently truncate or
        # mis-align if c != len(channel_coords); fail loudly instead.
        assert c == self.channel_coords.shape[0], (
            f"input has {c} channels but encoder was built with "
            f"{self.channel_coords.shape[0]} channel coordinates; pass the "
            f"matching channel_order or fix the data."
        )
        patches = x.unfold(2, self.patch_size, self.patch_stride)
        h = patches.shape[2]
        patches = patches.reshape(b, c * h, self.patch_size)
        tokens = self.to_patch_embedding(patches)

        ch_pos = self.channel_coords.unsqueeze(0).expand(b, -1, -1)
        pos_4d = FourierEmb4D.add_time_patches(ch_pos, h)
        tokens = tokens + self.ln(self.fourier4d(pos_4d) + self.mlp4d(pos_4d))

        for attn, ff in self.transformer.layers:
            tokens = tokens + attn(tokens)
            tokens = tokens + ff(tokens)
        return self.attn_pool(tokens)


class _Projection(nn.Module):
    """Linear wrapper; the inner name matches checkpoint key ``projection.*``."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class _PosEncoding(nn.Module):
    """Sinusoidal PE; key ``positional_encoding.pe`` matches the checkpoint."""

    def __init__(self, dim: int, max_len: int = 5000, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(1, max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class MSSEncoder(nn.Module):
    """Multi-Scale Spectral encoder. ``(B, C, T)`` -> ``(B, embed_dim)``."""

    def __init__(
        self,
        embed_dim: int = 512,
        input_length: int = 6400,
        n_channels: int = 26,
        scales: tuple[int, ...] = (100, 200, 400, 800),
        depth: int = 4,
        heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.scales = tuple(scales)
        self.input_length = input_length
        self.n_channels = n_channels
        base_time_out = input_length // self.scales[0]

        self.patch_freq_embeddings = nn.ModuleList(
            [_Projection(s // 2 + 1, embed_dim) for s in self.scales]
        )
        self.patch_time_embeddings = nn.ModuleList(
            [_Projection(input_length // s, base_time_out) for s in self.scales]
        )
        self.channel_tokens = nn.Embedding(n_channels, embed_dim)
        self.register_buffer("channel_indices", torch.arange(n_channels))
        self.positional_encoding = _PosEncoding(embed_dim, dropout=dropout)
        self.transformer = LinearAttentionTransformer(
            dim=embed_dim, heads=heads, depth=depth, max_seq_len=2048,
            attn_layer_dropout=dropout, attn_dropout=dropout,
        )
        self.base_time_out = base_time_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        assert c == self.n_channels, f"expected {self.n_channels} channels, got {c}"
        channel_embs = []
        for ch in range(c):
            ch_data = x[:, ch, :]
            scale_embs = []
            for si, scale in enumerate(self.scales):
                # Non-overlapping rectangular STFT matches NeuroBOLT's MSS;
                # changing it hurts r catastrophically (paper appendix).
                window = torch.ones(scale, device=x.device)
                spec = torch.stft(
                    ch_data, n_fft=scale, hop_length=scale, window=window,
                    center=False, onesided=True, return_complex=True,
                )
                mag = torch.abs(spec)
                freq_emb = self.patch_freq_embeddings[si](mag.permute(0, 2, 1))
                time_emb = self.patch_time_embeddings[si](
                    freq_emb.permute(0, 2, 1)
                ).permute(0, 2, 1)
                scale_embs.append(time_emb)

            ch_emb = torch.stack(scale_embs, dim=0).sum(dim=0)
            ch_tok = self.channel_tokens(self.channel_indices[ch : ch + 1])
            ch_emb = ch_emb + ch_tok.unsqueeze(0).expand(b, self.base_time_out, -1)
            ch_emb = self.positional_encoding(ch_emb)
            channel_embs.append(ch_emb)

        x = torch.cat(channel_embs, dim=1)
        return self.transformer(x).mean(dim=1)
