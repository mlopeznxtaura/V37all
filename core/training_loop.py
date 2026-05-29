"""training_loop.py v35 — θ* outer optimization training loop.

Runs golias_sequence over a corpus of IngestRecords with:
  - EWC (λ=5000) catastrophic forgetting suppression (arxiv:1612.00796)
  - Adaptive λ reweighting per-step (worst-loss-component upweighting)
  - Per-domain perplexity evaluation (text path)
  - Flash mini-corpus cycles with configurable batch size
  - Progress callbacks for Stage 2 terminal mirror

This directly implements the v34 theta_optimization agent gap.
"""

from __future__ import annotations

import time
import math
from typing import Callable, Dict, List, Optional

from .golias_types import (
    IngestRecord, TrainingState, OllamaConfig, GoliasLoss,
)
from .golias_engine import golias_sequence, theta_star_update


# ─────────────────────────────────────────────────────────────────────────────
# Training config
# ─────────────────────────────────────────────────────────────────────────────

class TrainingConfig:
    def __init__(
        self,
        epochs: int = 3,
        batch_size: int = 4,
        lr: float = 0.05,
        ewc_lambda: float = 5000.0,
        max_loss_history: int = 100,
        ollama_config: Optional[OllamaConfig] = None,
        agent_gap_dir: str = "agents",
        flash_mini_corpus: bool = True,  # cycle through corpus in small batches
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.ewc_lambda = ewc_lambda
        self.max_loss_history = max_loss_history
        self.ollama_config = ollama_config or OllamaConfig()
        self.agent_gap_dir = agent_gap_dir
        self.flash_mini_corpus = flash_mini_corpus


# ─────────────────────────────────────────────────────────────────────────────
# Per-domain perplexity
# ─────────────────────────────────────────────────────────────────────────────

def domain_perplexity(frame_results: List[dict]) -> Dict[str, float]:
    """Compute per-modality perplexity from coherence loss."""
    by_modality: Dict[str, List[float]] = {}
    for fr in frame_results:
        m = fr.get("of1_modality", "text")
        by_modality.setdefault(m, []).append(fr.get("loss_coherence", 0.0))

    return {
        m: math.exp(min(sum(vals) / max(len(vals), 1), 10.0))
        for m, vals in by_modality.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training run
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    records: List[IngestRecord],
    config: Optional[TrainingConfig] = None,
    initial_state: Optional[TrainingState] = None,
    on_step: Optional[Callable[[dict], None]] = None,
    on_epoch: Optional[Callable[[dict], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> TrainingState:
    """Run multi-epoch θ* training loop.

    Args:
        records: Corpus of IngestRecords to train on.
        config: TrainingConfig.
        initial_state: Resume from existing TrainingState.
        on_step: Callback per batch, receives step_info dict.
        on_epoch: Callback per epoch, receives epoch_info dict.
        stop_flag: Callable returning True to stop training early.

    Returns:
        Final TrainingState.
    """
    cfg = config or TrainingConfig()
    ts = initial_state or TrainingState(ewc_lambda=cfg.ewc_lambda)
    ts.running = True

    if not records:
        ts.status_msg = "no records"
        ts.running = False
        return ts

    n = len(records)

    for epoch in range(cfg.epochs):
        ts.epoch = epoch
        epoch_losses = []
        epoch_start = time.time()

        # Flash mini-corpus: cycle through in batch_size chunks
        if cfg.flash_mini_corpus:
            batches = [
                records[i: i + cfg.batch_size]
                for i in range(0, n, cfg.batch_size)
            ]
        else:
            batches = [records]

        for batch_idx, batch in enumerate(batches):
            if stop_flag and stop_flag():
                ts.status_msg = "stopped by user"
                ts.running = False
                return ts

            batch_start = time.time()
            stream_lines = []

            def _stream_cb(chunk: str):
                stream_lines.append(chunk)

            seq_result = golias_sequence(
                batch,
                training_state=ts,
                ollama_config=cfg.ollama_config,
                agent_gap_dir=cfg.agent_gap_dir,
                stream_callback=_stream_cb,
            )

            # θ* already updated inside golias_sequence; sync back
            ts = seq_result["training_state"]
            ts.total_frames += len(batch)

            batch_elapsed = time.time() - batch_start
            avg_loss = seq_result.get("avg_L_Golias", 0.0)
            epoch_losses.append(avg_loss)

            perp = domain_perplexity(seq_result.get("frames", []))

            # Trim loss history
            if len(ts.loss_history) > cfg.max_loss_history:
                ts.loss_history = ts.loss_history[-cfg.max_loss_history:]

            step_info = {
                "epoch": epoch,
                "batch": batch_idx,
                "n_batches": len(batches),
                "batch_size": len(batch),
                "avg_loss": avg_loss,
                "avg_coherence": seq_result.get("avg_L_coherence", 0.0),
                "avg_stability": seq_result.get("avg_stability_norm", 0.0),
                "avg_confidence": seq_result.get("avg_of1_confidence", 0.0),
                "theta_1_temp": ts.theta_1.get("temperature", 1.0),
                "theta_2_epsilon": ts.theta_2.get("epsilon", 0.6),
                "perplexity": perp,
                "elapsed_ms": batch_elapsed * 1000.0,
                "step": ts.step,
                "lambda_weights": dict(ts.lambda_weights),
                "stream_preview": "".join(stream_lines)[:200],
                "frames": seq_result.get("frames", []),
            }
            ts.status_msg = (
                f"epoch={epoch} batch={batch_idx}/{len(batches)} "
                f"loss={avg_loss:.4f} temp={ts.theta_1.get('temperature', 1.0):.3f} "
                f"ppl={ts.perplexity:.2f}"
            )

            if on_step:
                on_step(step_info)

        epoch_elapsed = time.time() - epoch_start
        epoch_avg = sum(epoch_losses) / max(len(epoch_losses), 1)

        if on_epoch:
            on_epoch({
                "epoch": epoch,
                "epoch_avg_loss": epoch_avg,
                "epoch_elapsed_s": epoch_elapsed,
                "theta_1": dict(ts.theta_1),
                "theta_2": dict(ts.theta_2),
                "lambda_weights": dict(ts.lambda_weights),
                "perplexity": ts.perplexity,
                "step": ts.step,
                "total_frames": ts.total_frames,
            })

    ts.running = False
    ts.status_msg = f"done — {ts.step} steps, {ts.total_frames} frames, ppl={ts.perplexity:.2f}"
    return ts
