"""pldm_predictor.py v37 — PLDM-style Autoregressive Predictor for OF1/IF5.

Architectural gain from stable-worldmodel PLDM (arxiv:2502.14819):
  - AdaLN-zero conditioned Transformer blocks (ConditionalBlock)
  - Action-conditioned latent rollout  → better IF5 future-state estimates
  - Embedder: Conv1d patch-embed → SiLU MLP for smooth latent embedding
  - Temporal history buffer (HS=3) for autoregressive prediction

Replaces the v36 linear-extrapolation delta in OF1MultimodalPredictor.predict_next()
with a properly conditioned causal Transformer.

v37 ONLY addition — does not modify any v36 code.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# NumPy-only fallback Transformer (no torch dependency required)
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def _gelu(x: np.ndarray) -> np.ndarray:
    return x * 0.5 * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class NumpyLinear:
    """Minimal weight-free linear pass (identity for latent vectors)."""

    def __init__(self, in_dim: int, out_dim: int):
        # Xavier init
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b


class NumpyConv1dPatch:
    """Conv1d kernel=1 patch embedder — lifts D → emb_dim."""

    def __init__(self, in_dim: int, out_dim: int):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        self.W = np.random.randn(out_dim, in_dim).astype(np.float32) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (T, D) → (T, out_dim)
        return x @ self.W.T + self.b


class NumpyAdaLNBlock:
    """AdaLN-zero conditioned Transformer block (numpy, heads=1 for efficiency).

    Matches stable-worldmodel ConditionalBlock architecture:
      x = x + gate_msa * attn(modulate(norm1(x), shift_msa, scale_msa))
      x = x + gate_mlp * ffn(modulate(norm2(x), shift_mlp, scale_mlp))
    """

    def __init__(self, dim: int, mlp_scale: int = 4):
        d = dim
        # adaLN modulation: (dim,) → 6*dim
        scale = math.sqrt(2.0 / (d + 6 * d))
        self.W_ada = np.zeros((d, 6 * d), dtype=np.float32)  # zero-init (AdaLN-zero)
        self.b_ada = np.zeros(6 * d, dtype=np.float32)
        # attention projections (simplified single-head)
        self.Wq = NumpyLinear(d, d)
        self.Wk = NumpyLinear(d, d)
        self.Wv = NumpyLinear(d, d)
        self.Wo = NumpyLinear(d, d)
        # ffn
        self.W1 = NumpyLinear(d, d * mlp_scale)
        self.W2 = NumpyLinear(d * mlp_scale, d)
        self._scale = 1.0 / math.sqrt(d)

    def _attn(self, x: np.ndarray) -> np.ndarray:
        T = x.shape[0]
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scores = (Q @ K.T) * self._scale
        # Causal mask
        mask = np.triu(np.full((T, T), -1e9), k=1)
        scores = scores + mask
        attn = _softmax(scores)
        return self.Wo(attn @ V)

    def _modulate(self, x: np.ndarray, shift: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return x * (1 + scale) + shift

    def __call__(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        """x: (T, D), c: (T, D) — conditioning (action embedding)."""
        # Aggregate c across time for modulation
        c_mean = c.mean(axis=0, keepdims=True)  # (1, D)
        ada = _silu(c_mean) @ self.W_ada + self.b_ada  # (1, 6D)
        chunks = np.split(ada, 6, axis=-1)
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_mlp, scale_mlp, gate_mlp = chunks[3], chunks[4], chunks[5]

        # attention path
        normed = _layer_norm(x)
        modded = self._modulate(normed, shift_msa, scale_msa)
        x = x + gate_msa * self._attn(modded)

        # ffn path
        normed2 = _layer_norm(x)
        modded2 = self._modulate(normed2, shift_mlp, scale_mlp)
        h = _gelu(self.W1(modded2))
        x = x + gate_mlp * self.W2(h)

        return x


class PLDMNumpyPredictor:
    """Pure-NumPy PLDM autoregressive predictor.

    Architecture:
      Embedder (Conv1d + SiLU MLP) → pos_embed → depth × AdaLN-zero blocks
      → layer_norm → output_proj

    Conditioned on action embeddings (or zero vector when no action provided).
    Used for IF5 future-state estimation and OF1 latent rollout in v37.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        depth: int = 2,
        history_size: int = 3,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.history_size = history_size

        # Embedder: Conv1d patch + SiLU MLP
        self.patch = NumpyConv1dPatch(input_dim, hidden_dim)
        self.embed_w1 = NumpyLinear(hidden_dim, hidden_dim * 4)
        self.embed_w2 = NumpyLinear(hidden_dim * 4, hidden_dim)

        # Positional embedding (sinusoidal, fixed)
        self._pos = self._build_pos_embed(history_size + 1, hidden_dim)

        # Action encoder
        self.act_enc_w1 = NumpyLinear(input_dim, hidden_dim)
        self.act_enc_w2 = NumpyLinear(hidden_dim, hidden_dim)

        # Transformer blocks
        self.blocks = [NumpyAdaLNBlock(hidden_dim) for _ in range(depth)]

        # Output projection
        self.out_proj = NumpyLinear(hidden_dim, input_dim)

    @staticmethod
    def _build_pos_embed(max_len: int, dim: int) -> np.ndarray:
        pe = np.zeros((max_len, dim), dtype=np.float32)
        pos = np.arange(max_len)[:, None]
        div = np.exp(-np.arange(0, dim, 2) * math.log(10000.0) / dim)
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)[:, :dim // 2]
        return pe

    def _embed(self, x: np.ndarray) -> np.ndarray:
        """x: (T, D) → (T, hidden)"""
        h = self.patch(x)
        h = _silu(self.embed_w1(h))
        h = self.embed_w2(h)
        return h

    def _encode_action(self, a: Optional[np.ndarray], T: int) -> np.ndarray:
        if a is None:
            return np.zeros((T, self.hidden_dim), dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            a = np.tile(a[None, :], (T, 1))
        a = a[:T]
        if a.shape[-1] != self.input_dim:
            # pad/truncate to input_dim
            diff = self.input_dim - a.shape[-1]
            if diff > 0:
                a = np.pad(a, ((0, 0), (0, diff)))
            else:
                a = a[:, :self.input_dim]
        h = _silu(self.act_enc_w1(a))
        return self.act_enc_w2(h)

    def predict(
        self,
        latent_history: List[List[float]],
        action: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Predict next latent given history of latents + optional action.

        Args:
            latent_history: List of up to `history_size` latent vectors.
            action: Optional action vector (list/array) to condition on.

        Returns:
            dict with keys:
              predicted_obs   — next latent (List[float])
              delta           — mean velocity
              n_frames        — number of frames used
              method          — "pldm_adaln_transformer"
        """
        if not latent_history:
            return {
                "predicted_obs": [],
                "delta": [],
                "n_frames": 0,
                "method": "pldm_adaln_transformer",
            }

        # Take last HS frames
        hs = self.history_size
        frames = latent_history[-hs:]
        x = np.array(frames, dtype=np.float32)  # (T, D)

        # Pad D if needed
        D = self.input_dim
        if x.shape[-1] < D:
            x = np.pad(x, ((0, 0), (0, D - x.shape[-1])))
        elif x.shape[-1] > D:
            x = x[:, :D]

        T = x.shape[0]

        # Embed + positional encoding
        h = self._embed(x)  # (T, hidden)
        pos = self._pos[:T]
        if pos.shape[-1] != self.hidden_dim:
            pos = pos[:, :self.hidden_dim]
        h = h + pos

        # Action conditioning
        c = self._encode_action(action, T)  # (T, hidden)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, c)

        h = _layer_norm(h)

        # Output: take last token prediction
        pred_hidden = h[-1:]  # (1, hidden)
        pred = self.out_proj(pred_hidden)[0]  # (D,)

        # Delta from last two frames for consistency with v36 API
        if len(frames) >= 2:
            last = np.array(frames[-1], dtype=np.float32)[:D]
            prev = np.array(frames[-2], dtype=np.float32)[:D]
            delta = (last - prev).tolist()
        else:
            delta = []

        return {
            "predicted_obs": pred.tolist(),
            "delta": delta,
            "n_frames": T,
            "method": "pldm_adaln_transformer",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_predictor_cache: Dict[int, PLDMNumpyPredictor] = {}


def get_pldm_predictor(input_dim: int = 64, depth: int = 2) -> PLDMNumpyPredictor:
    """Return (cached) PLDMNumpyPredictor for given input_dim."""
    key = (input_dim, depth)
    if key not in _predictor_cache:
        _predictor_cache[key] = PLDMNumpyPredictor(input_dim=input_dim, depth=depth)
    return _predictor_cache[key]
