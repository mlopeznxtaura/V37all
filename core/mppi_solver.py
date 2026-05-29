"""mppi_solver.py v37 — MPPI planning solver adapted from stable-worldmodel.

Architectural gain from stable-worldmodel MPPISolver (arxiv: model-based planning):
  - Softmax-weighted mean update over sampled action candidates
  - TopK elite filtering before weight computation
  - Temperature-scaled cost weighting (stabilized via min-cost subtraction)
  - Warm-start from prior mean for smooth plan continuation

In the Golias architecture, MPPILatentSolver replaces the simple
if7.rel_scalar action-proxy with a proper model-predictive planning loop
over the PLDMNumpyPredictor. This increases OF1 confidence and IF5 quality.

v37 ONLY addition — pure NumPy, no torch dependency.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class MPPILatentSolver:
    """NumPy MPPI solver over latent world model.

    Uses PLDMNumpyPredictor as cost model:
      cost(action_seq) = L2(final_latent, goal_latent)

    Args:
        predictor:    PLDMNumpyPredictor instance.
        num_samples:  Action candidates per iteration.
        n_steps:      MPPI optimization steps.
        topk:         Elite samples for weighted update.
        temperature:  Softmax temperature (lower = sharper).
        var_scale:    Initial action noise variance.
        horizon:      Planning horizon (steps).
        action_dim:   Dimension of action vector.
        seed:         RNG seed.
    """

    def __init__(
        self,
        predictor=None,
        num_samples: int = 64,
        n_steps: int = 10,
        topk: int = 16,
        temperature: float = 0.5,
        var_scale: float = 1.0,
        horizon: int = 8,
        action_dim: int = 4,
        seed: int = 42,
    ):
        self.predictor = predictor
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.temperature = temperature
        self.var_scale = var_scale
        self.horizon = horizon
        self.action_dim = action_dim
        self._rng = np.random.default_rng(seed)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _rollout_cost(
        self,
        latent_history: List[List[float]],
        action_candidates: np.ndarray,
        goal_latent: List[float],
    ) -> np.ndarray:
        """Compute costs for N action candidate sequences.

        Args:
            latent_history:    Current latent history (list of vectors).
            action_candidates: (N, H, A) — N seqs of H-step actions.
            goal_latent:       Target latent vector.

        Returns:
            costs: (N,) float32
        """
        N, H, _ = action_candidates.shape
        goal = np.array(goal_latent, dtype=np.float32)
        costs = np.zeros(N, dtype=np.float32)

        if self.predictor is None:
            # Fallback: random cost if no predictor attached
            costs = self._rng.standard_normal(N).astype(np.float32) ** 2
            return costs

        for i in range(N):
            hist = list(latent_history)
            for h in range(H):
                result = self.predictor.predict(hist, action=action_candidates[i, h])
                next_lat = result["predicted_obs"]
                if next_lat:
                    hist.append(next_lat)

            # Cost = L2 distance to goal at final step
            if hist:
                final = np.array(hist[-1], dtype=np.float32)
                d = final[:len(goal)] - goal[:len(final)]
                costs[i] = float(np.sqrt((d ** 2).sum()))

        return costs

    # ── main solve ─────────────────────────────────────────────────────────

    def solve(
        self,
        latent_history: List[List[float]],
        goal_latent: List[float],
        init_actions: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run MPPI optimization.

        Returns:
            dict with:
              actions        — (H, A) optimal action sequence
              costs          — final elite costs
              elapsed_ms     — wall-clock time
              n_steps        — iterations run
        """
        t0 = time.time()
        H, A = self.horizon, self.action_dim

        # Warm-start
        if init_actions is not None and init_actions.shape == (H, A):
            mean = init_actions.copy().astype(np.float32)
        else:
            mean = np.zeros((H, A), dtype=np.float32)

        var = np.full((H, A), self.var_scale, dtype=np.float32)

        for step in range(self.n_steps):
            # Sample candidates: (N, H, A)
            noise = self._rng.standard_normal((self.num_samples, H, A)).astype(np.float32)
            candidates = mean[None] + noise * np.sqrt(var)[None]
            # Force first candidate = mean (zero noise)
            candidates[0] = mean

            costs = self._rollout_cost(latent_history, candidates, goal_latent)

            # TopK elite filter
            k = min(self.topk, self.num_samples)
            topk_idx = np.argpartition(costs, k)[:k]
            elite_costs = costs[topk_idx]
            elite_cands = candidates[topk_idx]

            # MPPI softmax weighting (min-cost stabilized)
            min_c = elite_costs.min()
            scaled = elite_costs - min_c
            weights = np.exp(-scaled / max(self.temperature, 1e-8))
            weights = weights / (weights.sum() + 1e-12)

            # Weighted mean update
            mean = (weights[:, None, None] * elite_cands).sum(axis=0)

        elapsed_ms = (time.time() - t0) * 1000.0

        return {
            "actions": mean,           # (H, A)
            "costs": elite_costs.tolist(),
            "elapsed_ms": elapsed_ms,
            "n_steps": self.n_steps,
            "best_cost": float(elite_costs.min()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level factory
# ─────────────────────────────────────────────────────────────────────────────

_mppi_cache: Optional[MPPILatentSolver] = None


def get_mppi_solver(
    predictor=None,
    num_samples: int = 64,
    n_steps: int = 5,
    topk: int = 16,
    temperature: float = 0.5,
    horizon: int = 4,
    action_dim: int = 4,
) -> MPPILatentSolver:
    """Return a cached MPPILatentSolver."""
    global _mppi_cache
    if _mppi_cache is None:
        _mppi_cache = MPPILatentSolver(
            predictor=predictor,
            num_samples=num_samples,
            n_steps=n_steps,
            topk=topk,
            temperature=temperature,
            horizon=horizon,
            action_dim=action_dim,
        )
    # If predictor has been upgraded, re-attach
    if predictor is not None and _mppi_cache.predictor is None:
        _mppi_cache.predictor = predictor
    return _mppi_cache
