"""The B0 PPI predictor (motif-graph `V3_1`, config b0_v31), inference only.

Vendored from motif-graph-prompt-tuning-for-inductive-protein-protein-interaction-prediction @ bb01ee1:
src/model/egostitch/classifier/{b0_v31.py (V3_1), layers.py}, src/score_universe.py (_load_checkpoint).
Only what b0_v31 exercises is kept: mixing "none" (the pair trunk has no cross-attention layers),
pair_context_gated readout, abba_max order aggregation, no spectral norm, no topo_gen/kd heads.
Dropout and stochastic depth are eval no-ops and are omitted; module names match the checkpoint so
`load_state_dict(strict=True)` takes `best.pt` as is. Arithmetic is copied verbatim (including the
encoder's `x + (y - x)` residual), since bf16 logits change with any reordering.
"""

from pathlib import Path

import torch
import yaml
from torch import nn


def padding_mask(lengths, max_len):
    """True = pad, shape (batch, max_len)."""
    return torch.arange(max_len, device=lengths.device).expand(lengths.size(0), max_len) >= lengths.unsqueeze(1)


class SiameseEncoder(nn.Module):
    """Shared per-protein transformer over per-residue tokens (BOS/EOS included)."""

    def __init__(self, input_dim, d_model, n_layers, n_heads):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, embeddings, lengths):
        projected = self.input_projection(embeddings)
        mask = padding_mask(lengths, projected.size(1))
        for layer in self.layers:
            residual_in = projected
            y = layer(projected, src_key_padding_mask=mask)
            projected = residual_in + (y - residual_in)  # the source's stochastic-depth residual, kept bit-exact
        return self.output_norm(projected)


def inner_token_mask(x, pad):
    """Valid tokens strictly between BOS (position 0) and EOS (last valid position)."""
    valid = ~pad
    positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
    mask = valid & (positions > 0) & (positions < valid.sum(dim=1, keepdim=True) - 1)
    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError("Pair readout requires at least one inner_token token between BOS and EOS")
    return mask


def masked_mean(x, mask):
    mask_3d = mask.float().unsqueeze(-1)
    return (x * mask_3d).sum(dim=1) / mask_3d.sum(dim=1).clamp_min(1.0)


def masked_max(x, mask):
    return x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min).max(dim=1).values


def _block(d_in, d_out):
    """LayerNorm -> Linear -> GELU -> Dropout, the source's Sequential layout (indices 0, 1 hold weights)."""
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(0.0))


class PairContextGatedReadout(nn.Module):
    """Inner-token mean/max/pair-conditioned-attention readout of one directed pair."""

    def __init__(self, d_model):
        super().__init__()
        self.context_proj = _block(d_model * 2, d_model)
        self.attn_scorer = nn.Sequential(*_block(d_model * 2, d_model), nn.Linear(d_model, 1))
        self.branch_gate = nn.Linear(d_model * 12, 3)
        self.branch_proj = _block(d_model * 4, d_model)
        self.final_proj = _block(d_model * 2, d_model)

    def _attention_pool(self, x, mask, other_context):
        context = other_context.unsqueeze(1).expand(-1, x.size(1), -1)
        scores = self.attn_scorer(torch.cat([x, context], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)

    @staticmethod
    def _pair_features(a, b):
        return torch.cat([a, b, torch.abs(a - b), a * b], dim=-1)

    def forward(self, h_a, h_b, cls_vec, pad_a, pad_b):
        inner_a = inner_token_mask(h_a, pad_a)
        inner_b = inner_token_mask(h_b, pad_b)
        mean_a, mean_b = masked_mean(h_a, inner_a), masked_mean(h_b, inner_b)
        max_a, max_b = masked_max(h_a, inner_a), masked_max(h_b, inner_b)
        context_a = self.context_proj(torch.cat([mean_a, max_a], dim=-1))
        context_b = self.context_proj(torch.cat([mean_b, max_b], dim=-1))
        attn_a = self._attention_pool(h_a, inner_a, context_b)
        attn_b = self._attention_pool(h_b, inner_b, context_a)
        branches = torch.stack(
            [
                self._pair_features(mean_a, mean_b),
                self._pair_features(max_a, max_b),
                self._pair_features(attn_a, attn_b),
            ],
            dim=1,
        )
        gate_weights = torch.softmax(self.branch_gate(branches.flatten(start_dim=1)), dim=1).unsqueeze(-1)
        projected = self.branch_proj((branches * gate_weights).sum(dim=1))
        return self.final_proj(torch.cat([cls_vec, projected], dim=-1))


class PairTrunk(nn.Module):
    """`PairCrossAttention` with mixing "none": a learned cls vector plus the gated readout."""

    def __init__(self, d_model):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.pair_context_readout = PairContextGatedReadout(d_model)

    def forward(self, h_a, h_b, lengths_a, lengths_b):
        cls_vec = self.cls_token.repeat(h_a.size(0), 1, 1).squeeze(1)
        pad_a, pad_b = padding_mask(lengths_a, h_a.size(1)), padding_mask(lengths_b, h_b.size(1))
        return self.pair_context_readout(h_a, h_b, cls_vec, pad_a, pad_b)


class MLPHead(nn.Module):
    """(Linear -> LayerNorm -> GELU -> Dropout) per hidden dim, then Linear to one logit."""

    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers, dim = [], input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.0)]
            dim = hidden
        self.layers = nn.Sequential(*layers, nn.Linear(dim, 1))

    def forward(self, x):
        return self.layers(x)


class B0(nn.Module):
    """Encode each protein once (`encoder`), then score encoded pairs (`logits_from_encoded`)."""

    def __init__(self, input_dim, d_model, encoder_layers, n_heads, mlp_hidden_dims):
        super().__init__()
        self.d_model = d_model
        self.encoder = SiameseEncoder(input_dim, d_model, encoder_layers, n_heads)
        self.cross_attention = PairTrunk(d_model)
        self.output_head = MLPHead(d_model, mlp_hidden_dims)

    def logits_from_encoded(self, encoded_a, encoded_b, lengths_a, lengths_b):
        """abba_max: feature-wise max over both endpoint orders, then the head. Returns (B, 1)."""
        feature_ab = self.cross_attention(encoded_a, encoded_b, lengths_a, lengths_b)
        feature_ba = self.cross_attention(encoded_b, encoded_a, lengths_b, lengths_a)
        return self.output_head(torch.max(torch.stack([feature_ab, feature_ba], dim=-1), dim=-1).values)


def build_model(config):
    """Build B0 from a `model.config` mapping; reject any option this vendored copy does not implement."""
    mlp, readout = config["mlp_head"], config.get("pair_readout") or {}
    unsupported = {
        "mixing.mode": (config.get("mixing") or {}).get("mode", "bidirectional_cross") != "none",
        "pair_readout.mode": readout.get("mode", "rich_pooling") != "pair_context_gated",
        "pair_readout.order_aggregation": readout.get("order_aggregation", "single") != "abba_max",
        "pair_readout.spectral_norm": bool(readout.get("spectral_norm", False)),
        "mlp_head.spectral_norm": bool(mlp.get("spectral_norm", False)),
        "mlp_head.activation": mlp.get("activation", "gelu") != "gelu",
        "mlp_head.norm": mlp.get("norm", "layernorm") != "layernorm",
        "topo_gen": config.get("topo_gen") is not None,
        "kd_rep_dim": int(config.get("kd_rep_dim", 0)) != 0,
        "kd_struct_dim": int(config.get("kd_struct_dim", 0)) != 0,
    }
    bad = [key for key, flag in unsupported.items() if flag]
    if bad:
        raise ValueError(f"model config uses options the vendored B0 does not implement: {bad}")
    return B0(
        input_dim=int(config["input_dim"]),
        d_model=int(config["d_model"]),
        encoder_layers=int(config["encoder_layers"]),
        n_heads=int(config["n_heads"]),
        mlp_hidden_dims=[int(d) for d in mlp["hidden_dims"]],
    )


def load_config(path):
    """`model.config` of a motif training YAML (family must be v3_1)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise ValueError(f"model config {path} must contain a model mapping")
    if raw["model"].get("family") != "v3_1":
        raise ValueError(f"model config {path} family {raw['model'].get('family')!r} is not 'v3_1'")
    return raw["model"]["config"]


def load_model(checkpoint, config_path):
    """Rebuild the frozen model on CPU in eval mode; returns (model, model_state)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = load_config(config_path)
    if all(key in ckpt for key in ("model_state", "model_family", "model_config")):
        if ckpt["model_family"] != "v3_1":
            raise ValueError(f"checkpoint {checkpoint}: model_family {ckpt['model_family']!r} is not 'v3_1'")
        if ckpt["model_config"] != config:
            raise ValueError(f"checkpoint {checkpoint}: embedded model_config differs from {config_path}")
        state = ckpt["model_state"]
    else:
        state = ckpt  # bare legacy state dict, as the source accepts with an explicit config
    model = build_model(config)
    model.load_state_dict(state, strict=True)
    return model.eval(), state
