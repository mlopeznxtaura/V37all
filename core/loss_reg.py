"""loss_reg.py v37 — Regularization losses from stable-worldmodel.

Architectural gains:
  - VCReg  (Variance-Covariance Regularizer)  — prevents latent space collapse
  - SIGReg (Sketch Isotropic Gaussian Reg)    — arxiv:2511.08544
  - PLDMLoss (VCReg + temporal alignment + IDM) — arxiv:2502.14819
  - TemporalStraighteningLoss                 — arxiv:2603.12231

All implemented in pure NumPy for v37 (torch version optional if available).
Plugs into GoliasLoss via L_reg term added in v37.

v37 ONLY addition.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# VCReg
# ─────────────────────────────────────────────────────────────────────────────

class VCReg:
    """Variance-Covariance Regularizer (NumPy).

    Penalizes:
      - Low variance (collapse): std_loss = mean(ReLU(1 - std_per_dim))
      - Off-diagonal covariance (entanglement): cov_loss = off-diag Frobenius²/D

    Input: z (B, T, D) or (B, D) latent matrix
    """

    def __init__(self, eps: float = 1e-4):
        self.eps = eps

    def _std_loss(self, z: np.ndarray) -> float:
        # z: (T, B, D) after transpose
        z = np.transpose(z, (1, 0, 2))  # (T, B, D)
        std = np.sqrt(z.var(axis=1) + self.eps)  # (T, D)
        loss = np.mean(np.maximum(0.0, 1.0 - std))
        return float(loss)

    def _cov_loss(self, z: np.ndarray) -> float:
        B, T, D = z.shape
        zt = np.transpose(z, (1, 0, 2))  # (T, B, D)
        cov = np.einsum("tbi,tbj->tij", zt, zt) / max(B - 1, 1)  # (T, D, D)
        # Off-diagonal Frobenius²
        diag_sq = np.einsum("tii->ti", cov) ** 2  # (T, D)
        total = (cov ** 2).sum(axis=(-2, -1))  # (T,)
        off_diag = (total - diag_sq.sum(axis=-1)) / max(D ** 2 - D, 1)
        return float(off_diag.mean())

    def __call__(self, z: np.ndarray) -> Dict[str, float]:
        """z: (B, T, D) or (B, D)"""
        if z.ndim == 2:
            z = z[:, None, :]  # (B, 1, D)
        z = z - z.mean(axis=0, keepdims=True)
        return {
            "std_loss": self._std_loss(z),
            "cov_loss": self._cov_loss(z),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SIGReg
# ─────────────────────────────────────────────────────────────────────────────

class SIGReg:
    """Sketch Isotropic Gaussian Regularizer (NumPy, arxiv:2511.08544).

    Matches latent distribution to N(0,I) via random projections + Epps-Pulley.
    """

    def __init__(self, knots: int = 17, num_proj: int = 128, seed: int = 0):
        self.num_proj = num_proj
        t = np.linspace(0, 3, knots, dtype=np.float32)
        dt = 3.0 / (knots - 1)
        weights = np.full(knots, 2 * dt, dtype=np.float32)
        weights[[0, -1]] = dt
        window = np.exp(-(t ** 2) / 2.0)
        self.t = t                         # (knots,)
        self.phi = window                  # (knots,)
        self.weights = weights * window    # (knots,)
        self._rng = np.random.default_rng(seed)

    def __call__(self, z: np.ndarray) -> float:
        """z: (T, B, D) or (B, D)"""
        if z.ndim == 2:
            z = z[None, :, :]  # (1, B, D)
        T, B, D = z.shape
        # Random projections
        A = self._rng.standard_normal((D, self.num_proj)).astype(np.float32)
        A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-8)
        # z @ A → (T, B, num_proj)
        proj = (z @ A)  # (T, B, P)
        # Epps-Pulley: x_t = proj * t, compute cosine/sine means
        # proj: (T, B, P), t: (knots,)
        x_t = proj[:, :, :, None] * self.t[None, None, None, :]  # (T, B, P, K)
        cos_mean = x_t.cos_approx = np.cos(x_t).mean(axis=1)  # (T, P, K) mean over B
        sin_mean = np.sin(x_t).mean(axis=1)
        err = (cos_mean - self.phi[None, None, :]) ** 2 + sin_mean ** 2
        statistic = (err * self.weights[None, None, :]).sum(axis=-1) * B
        return float(statistic.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Straightening Loss
# ─────────────────────────────────────────────────────────────────────────────

def temporal_straightening_loss(z: np.ndarray) -> float:
    """Mean pairwise negative cosine similarity of latent velocities.

    Encourages the latent trajectory to be a straight line (arxiv:2603.12231).

    z: (B, T, D)  — T ≥ 3 required
    """
    if z.shape[1] < 3:
        return 0.0
    v = z[:, 1:] - z[:, :-1]          # (B, T-1, D)
    v1 = v[:, :-1]                     # (B, T-2, D)
    v2 = v[:, 1:]
    eps = 1e-8
    norm1 = np.linalg.norm(v1, axis=-1, keepdims=True) + eps
    norm2 = np.linalg.norm(v2, axis=-1, keepdims=True) + eps
    cos_sim = (v1 / norm1 * v2 / norm2).sum(axis=-1)  # (B, T-2)
    return float(-cos_sim.mean())


# ─────────────────────────────────────────────────────────────────────────────
# PLDM composite loss
# ─────────────────────────────────────────────────────────────────────────────

class PLDMLoss:
    """VCReg + temporal alignment + IDM + optional SIGReg.

    Mirrors stable-worldmodel PLDMLoss (arxiv:2502.14819).

    z:        (B, T, D) latent trajectory
    a_pred:   (B, T-1, A) predicted actions (from IDM)
    a_target: (B, T-1, A) ground-truth actions
    """

    def __init__(self, use_sigreg: bool = False):
        self.vc_reg = VCReg()
        self.sigreg = SIGReg() if use_sigreg else None

    def __call__(
        self,
        z: np.ndarray,
        a_pred: Optional[np.ndarray] = None,
        a_target: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}

        # IDM loss
        if a_pred is not None and a_target is not None:
            diff = a_pred - a_target
            out["idm_loss"] = float((diff ** 2).mean())

        # Temporal alignment: z[t] ≈ z[t+1]
        if z.shape[1] >= 2:
            out["temp_align_loss"] = float(((z[:, :-1] - z[:, 1:]) ** 2).mean())

        # Temporal straightening
        out["temporal_straight_loss"] = temporal_straightening_loss(z)

        # VCReg
        out.update(self.vc_reg(z))

        # SIGReg (optional)
        if self.sigreg is not None:
            zt = np.transpose(z, (1, 0, 2))  # (T, B, D)
            out["sig_loss"] = self.sigreg(zt)

        # Composite scalar
        weights = {
            "idm_loss": 1.0,
            "temp_align_loss": 0.5,
            "temporal_straight_loss": 0.3,
            "std_loss": 0.5,
            "cov_loss": 0.25,
            "sig_loss": 0.2,
        }
        total = sum(weights.get(k, 0.1) * v for k, v in out.items())
        out["total"] = float(total)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Golias integration helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_latent_reg_loss(
    latent_history: List[List[float]],
    a_pred: Optional[List[List[float]]] = None,
    a_target: Optional[List[List[float]]] = None,
) -> Dict[str, float]:
    """Compute PLDMLoss from Golias latent_history list-of-vectors.

    Returns empty dict (with total=0.0) if history is too short.
    """
    if len(latent_history) < 2:
        return {"total": 0.0}

    z = np.array(latent_history, dtype=np.float32)[None, :, :]  # (1, T, D)

    ap = np.array(a_pred, dtype=np.float32)[None] if a_pred else None
    at = np.array(a_target, dtype=np.float32)[None] if a_target else None

    loss_fn = PLDMLoss(use_sigreg=False)
    return loss_fn(z, ap, at)
