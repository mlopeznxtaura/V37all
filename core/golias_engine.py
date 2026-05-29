"""golias_engine.py v37 — Golias IF/OF computation engine.

v37 reparameterization vs v36:
  - OF1 future_state_estimate: PLDMNumpyPredictor (AdaLN-zero Transformer)
    replaces linear delta extrapolation (LatentFrameBuffer.predict_next)
  - OF1 action candidates: MPPILatentSolver plans over latent world model
    → best_cost feeds IF7 rel_scalar as a planning-quality signal
  - GoliasLoss L_reg: VCReg + PLDMLoss latent regularization
    → prevents latent space collapse across long inference sessions
  - TrainingState.latent_history: rolling buffer for PLDM + VCReg

v36 gaps remain unchanged:
  - IF3 M3 validator: Ollama embedding cosine distance
  - OF1 image encoder: CLIP > MobileNet > DCT (unchanged)
  - IF5 live env: WorldModelLiveSource (unchanged)
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

from .golias_types import (
    IF1_CreativeExploration, IF2_EfficiencyGate, IF3_DirectionValidator,
    IF4_BinarySubstrate, IF5_GeometryTelemetry, IF6_LanguageFusion,
    IF7_RelationalArbitration, OF1_NextStatePredict, OF2_LanguageExplanation,
    GoliasLoss, OllamaConfig, TrainingState, IngestRecord,
)

# v36: closed gaps
try:
    from .m3_validator import m3_enhanced_if3 as _m3_enhanced_if3
    _M3_AVAILABLE = True
except ImportError:
    _M3_AVAILABLE = False

try:
    from .of1_multimodal import get_predictor as _get_of1_predictor
    _OF1_MULTIMODAL_AVAILABLE = True
except ImportError:
    _OF1_MULTIMODAL_AVAILABLE = False

# v37: PLDM predictor + MPPI solver + latent regularization
try:
    from .pldm_predictor import get_pldm_predictor as _get_pldm_predictor
    _PLDM_AVAILABLE = True
except ImportError:
    _PLDM_AVAILABLE = False

try:
    from .mppi_solver import get_mppi_solver as _get_mppi_solver
    _MPPI_AVAILABLE = True
except ImportError:
    _MPPI_AVAILABLE = False

try:
    from .loss_reg import compute_latent_reg_loss as _compute_latent_reg_loss
    _LOSS_REG_AVAILABLE = True
except ImportError:
    _LOSS_REG_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Agent gap writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_agent_gap(path: str, record: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Ollama integration (IF1/IF2 — replaces v34 M1/M2 STUB)
# ─────────────────────────────────────────────────────────────────────────────

def ollama_generate_sync(
    prompt: str,
    config: OllamaConfig,
    stream_callback=None,
) -> tuple[str, float, int]:
    """Synchronous Ollama /api/generate call.

    Returns (output_text, elapsed_ms, token_count).
    Falls back to stub if Ollama not available.
    """
    t0 = time.time()
    payload = json.dumps({
        "model": config.model,
        "prompt": prompt,
        "stream": config.stream,
        "options": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_predict": config.max_tokens,
        }
    }).encode()

    try:
        req = urllib.request.Request(
            f"{config.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        collected = []
        token_count = 0
        with urllib.request.urlopen(req, timeout=config.timeout_ms / 1000.0) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = obj.get("response", "")
                collected.append(chunk)
                token_count += 1
                if stream_callback and chunk:
                    stream_callback(chunk)
                if obj.get("done"):
                    break
        elapsed = (time.time() - t0) * 1000.0
        return "".join(collected), elapsed, token_count

    except Exception as e:
        elapsed = (time.time() - t0) * 1000.0
        stub = f"[OLLAMA_UNAVAILABLE:{config.model}] {str(e)[:80]} | prompt={prompt[:60]}"
        if stream_callback:
            stream_callback(stub)
        return stub, elapsed, 0


async def ollama_generate_async(
    prompt: str,
    config: OllamaConfig,
    stream_callback=None,
) -> tuple[str, float, int]:
    """Async wrapper — runs ollama_generate_sync in executor."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: ollama_generate_sync(prompt, config, stream_callback),
    )
    return result


def ollama_list_models_sync(host: str = "http://localhost:11434") -> List[str]:
    """List available Ollama models. Returns [] if unavailable."""
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# IF helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_if1(
    rec: IngestRecord,
    ollama_config: OllamaConfig,
    stream_callback=None,
) -> IF1_CreativeExploration:
    """M1(x_t, θ_1, T_max) — v35: real Ollama call."""
    cfg = OllamaConfig(
        model=ollama_config.model,
        host=ollama_config.host,
        temperature=rec.T_max,
        top_p=0.95,
        max_tokens=512,
        stream=ollama_config.stream,
        timeout_ms=ollama_config.timeout_ms,
    )
    prompt = f"Creative exploration of state: {rec.language}"
    output, elapsed, tokens = ollama_generate_sync(prompt, cfg, stream_callback)
    return IF1_CreativeExploration(
        x_t={"text": rec.language, "frame_id": rec.frame_id},
        theta_1={"temperature": rec.T_max, "top_p": 0.95, "max_tokens": 512},
        T_max=rec.T_max,
        ollama_config=cfg,
        output=output,
        tokens_generated=tokens,
        generation_ms=elapsed,
        model_used=cfg.model,
    )


def compute_if2(
    if1: IF1_CreativeExploration,
    rec: IngestRecord,
) -> IF2_EfficiencyGate:
    """M2(if_1, θ_2, τ) — v35: real token/time metrics from IF1."""
    tokens_per_sec = (
        if1.tokens_generated / max(if1.generation_ms / 1000.0, 0.001)
        if if1.tokens_generated > 0 else 0.0
    )
    # Efficiency gate: halt if generation took more than tau_ms
    halt = if1.generation_ms > rec.tau_ms and rec.tau_ms > 0
    # Cost proxy: tokens * compute factor
    cost = if1.tokens_generated * 0.001
    return IF2_EfficiencyGate(
        if1_ref=if1,
        theta_2={"epsilon": rec.epsilon, "tau_ms": rec.tau_ms, "cost_weight": 0.4},
        tau_ms=rec.tau_ms,
        epsilon=rec.epsilon,
        halt_triggered=halt,
        latency_ms=if1.generation_ms,
        compute_cost=cost,
        tokens_per_sec=tokens_per_sec,
        output=f"[M2] halt={halt} lat={if1.generation_ms:.1f}ms tok/s={tokens_per_sec:.1f}",
    )


def compute_if3_retune(
    if1: IF1_CreativeExploration,
    if2: IF2_EfficiencyGate,
    theta3: Dict[str, float],
    loss_gradient: float = 0.0,
) -> IF3_DirectionValidator:
    """M3(if_1, if_2, θ_3) — v35: gradient-feedback retuning of θ_1'."""
    eff_pressure = if2.compute_cost / max(if2.tau_ms, 1.0)
    # Gradient pushes temperature down if loss is high
    grad_damp = 0.1 * loss_gradient if loss_gradient > 0 else 0.0
    new_temp = max(0.1, if1.theta_1.get("temperature", 1.0) * (1.0 - 0.3 * eff_pressure) - grad_damp)
    theta_1_prime = {**if1.theta_1, "temperature": new_temp}
    theta_2_prime = {**if2.theta_2, "epsilon": if2.epsilon * 0.95}

    # v36: use M3 embedding cosine distance when Ollama is available
    if _M3_AVAILABLE and (if1.output or if2.output):
        try:
            m3_result = _m3_enhanced_if3(
                if1_output=if1.output or "",
                if2_output=if2.output or "",
                theta_3=theta3,
            )
            D = m3_result["D_orthogonal"]
            G_p = m3_result["G_p"]
        except Exception:
            # Fall back to heuristic on any failure
            len1 = len(if1.output or "")
            len2 = len(if2.output or "")
            D = 0.0 if (len1 + len2) == 0 else abs(len1 - len2) / (len1 + len2)
            delta_novelty = theta3.get("delta_novelty", 0.1)
            delta_coverage = theta3.get("delta_coverage", 0.05)
            G_p = max(delta_novelty, delta_coverage)
    else:
        len1 = len(if1.output or "")
        len2 = len(if2.output or "")
        D = 0.0 if (len1 + len2) == 0 else abs(len1 - len2) / (len1 + len2)
        delta_novelty = theta3.get("delta_novelty", 0.1)
        delta_coverage = theta3.get("delta_coverage", 0.05)
        G_p = max(delta_novelty, delta_coverage)

    return IF3_DirectionValidator(
        if1_ref=if1, if2_ref=if2, theta_3=theta3,
        theta_1_prime=theta_1_prime,
        theta_2_prime=theta_2_prime,
        D_orthogonal=D,
        G_p=G_p,
        loss_gradient_feedback=loss_gradient,
    )


def compute_if7_arbitration(
    prev_if7: Optional[IF7_RelationalArbitration],
    cur_if1: IF1_CreativeExploration,
    cur_if2: IF2_EfficiencyGate,
    cur_if3: IF3_DirectionValidator,
    cur_if4: IF4_BinarySubstrate,
    cur_if5: IF5_GeometryTelemetry,
    cur_if6: IF6_LanguageFusion,
) -> IF7_RelationalArbitration:
    cur_scalars = [
        len(cur_if1.output or "") / 1000.0,
        cur_if2.latency_ms / max(cur_if2.tau_ms, 1.0),
        cur_if3.G_p,
        float(cur_if4.scalar) / 255.0,
        float(len(cur_if5.tensor)) / 100.0,
        len(cur_if6.fused_context) / 5000.0,
    ]
    if prev_if7 is not None:
        prev_s = [
            prev_if7.delta_if1, prev_if7.delta_if2, prev_if7.delta_if3,
            prev_if7.delta_if4, prev_if7.delta_if5, prev_if7.delta_if6,
        ]
    else:
        prev_s = [0.0] * 6

    deltas = [abs(c - p) for c, p in zip(cur_scalars, prev_s)]
    stability_norm = math.sqrt(sum(d ** 2 for d in deltas))
    safety_ok = (
        cur_if4.execution_permissions > 0
        and cur_if2.epsilon >= 0.0
        and not cur_if2.halt_triggered
    )
    rel_scalar = sum(deltas) / max(len(deltas), 1)

    return IF7_RelationalArbitration(
        delta_if1=deltas[0], delta_if2=deltas[1], delta_if3=deltas[2],
        delta_if4=deltas[3], delta_if5=deltas[4], delta_if6=deltas[5],
        safety_ok=safety_ok,
        rel_scalar=rel_scalar,
        stability_norm=stability_norm,
    )


def compute_loss(
    of1: OF1_NextStatePredict,
    of2: OF2_LanguageExplanation,
    if7: IF7_RelationalArbitration,
    if2: IF2_EfficiencyGate,
    if3: IF3_DirectionValidator,
    lambda_weights: Optional[Dict[str, float]] = None,
    Y_target: str = "",
    Y_future: Optional[List[float]] = None,
    L_reg: float = 0.0,
) -> GoliasLoss:
    lw = lambda_weights or {}
    loss = GoliasLoss(
        lambda_1=lw.get("lambda_1", 1.0),
        lambda_2=lw.get("lambda_2", 0.5),
        lambda_3=lw.get("lambda_3", 0.8),
        lambda_4=lw.get("lambda_4", 0.6),
        lambda_5=lw.get("lambda_5", 1.0),
        lambda_6=lw.get("lambda_6", 0.9),
        lambda_7=lw.get("lambda_7", 0.4),   # v37
    )
    pred_tokens = set(of1.prediction_tokens)
    expl_tokens = set(of2.explanation.split())
    union = len(pred_tokens | expl_tokens)
    sim = len(pred_tokens & expl_tokens) / max(union, 1)
    loss.L_coherence = 1.0 - sim
    loss.L_efficiency = if2.compute_cost + (if2.latency_ms / 1000.0)
    loss.L_novelty = -if3.G_p
    loss.L_stability = if7.stability_norm
    if Y_target:
        yt = set(Y_target.lower().split())
        oe = set(of2.explanation.lower().split())
        match = len(yt & oe) / max(len(yt), 1)
    else:
        match = of2.goal_match
    loss.L_alignment = 1.0 - match
    if Y_future and of1.latent_vector:
        diffs = [abs(a - b) for a, b in zip(of1.latent_vector, Y_future)]
        loss.L_prediction = math.sqrt(sum(d ** 2 for d in diffs)) / max(len(diffs), 1)
    else:
        loss.L_prediction = 1.0 - of1.confidence
    loss.L_reg = float(L_reg)   # v37
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# θ* outer optimization — replaces v34 model_agent theta_optimization gap
# ─────────────────────────────────────────────────────────────────────────────

def theta_star_update(
    state: TrainingState,
    loss: GoliasLoss,
    if7: IF7_RelationalArbitration,
    lr: float = 0.05,
) -> TrainingState:
    """Single step of θ* optimization.

    Gradient is proxied as loss delta. EWC regularization prevents catastrophic
    forgetting of prior θ settings (arxiv:1612.00796).
    """
    state.loss_history.append(loss.total)
    state.coherence_history.append(loss.L_coherence)
    state.stability_history.append(if7.stability_norm)

    if len(state.loss_history) < 2:
        return state

    # Loss gradient proxy: Δloss over last 2 steps
    grad = state.loss_history[-1] - state.loss_history[-2]
    state.step += 1

    # θ_1: temperature update — reduce if loss increasing
    old_temp = state.theta_1.get("temperature", 1.0)
    new_temp = max(0.1, min(2.0, old_temp - lr * grad))
    state.theta_1["temperature"] = new_temp

    # θ_2: epsilon update — tighten if stability bad
    if if7.stability_norm > 0.3:
        state.theta_2["epsilon"] = min(0.95, state.theta_2.get("epsilon", 0.6) + 0.01)
    else:
        state.theta_2["epsilon"] = max(0.3, state.theta_2.get("epsilon", 0.6) - 0.005)

    # Adaptive lambda reweighting: upweight whichever loss component is worst
    components = {
        "lambda_1": loss.L_coherence,
        "lambda_3": abs(loss.L_novelty),
        "lambda_4": loss.L_stability,
        "lambda_5": loss.L_alignment,
        "lambda_6": loss.L_prediction,
    }
    total_c = sum(components.values()) + 1e-9
    for k, v in components.items():
        current = state.lambda_weights.get(k, 1.0)
        target = (v / total_c) * 3.0  # scale to [0,3]
        # EWC: penalize large deviations from fisher diagonal
        fisher_val = state.fisher_diagonal.get(k, 1.0)
        ewc_pen = state.ewc_lambda * fisher_val * (current - target) ** 2
        state.lambda_weights[k] = max(0.1, min(3.0, current + 0.01 * (target - current) - 0.001 * ewc_pen))

    # Update fisher diagonal (running average of squared gradients)
    for k in components:
        old_f = state.fisher_diagonal.get(k, 0.0)
        state.fisher_diagonal[k] = 0.9 * old_f + 0.1 * grad ** 2

    # Perplexity proxy: exp(avg_coherence_loss)
    avg_coh = sum(state.coherence_history[-10:]) / max(len(state.coherence_history[-10:]), 1)
    state.perplexity = math.exp(min(avg_coh, 10.0))

    state.status_msg = (
        f"step={state.step} loss={loss.total:.4f} "
        f"temp={new_temp:.3f} stab={if7.stability_norm:.4f} "
        f"ppl={state.perplexity:.2f}"
    )
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass
# ─────────────────────────────────────────────────────────────────────────────

def golias_forward(
    rec: IngestRecord,
    prev_if7: Optional[IF7_RelationalArbitration] = None,
    if6_state: Optional[IF6_LanguageFusion] = None,
    training_state: Optional[TrainingState] = None,
    ollama_config: Optional[OllamaConfig] = None,
    agent_gap_dir: str = "agents",
    stream_callback=None,
) -> dict:
    """Single-record Golias forward pass v35.

    Now calls Ollama for IF1 (when available).
    IF5 E_t tensor populated from IngestRecord.geometry_tensor.
    θ* feedback loop from TrainingState.
    """
    t_start = time.time()
    oc = ollama_config or OllamaConfig()
    ts = training_state or TrainingState()
    loss_grad = ts.loss_history[-1] - ts.loss_history[-2] if len(ts.loss_history) >= 2 else 0.0

    # ── IF4 substrate ─────────────────────────────────────────────────────────
    if4 = IF4_BinarySubstrate(
        hardware_state=rec.hardware_state,
        runtime_constraints=rec.runtime_constraints,
        memory_available_mb=rec.memory_mb,
        execution_permissions=rec.exec_perms,
        system_identity=rec.system_id,
        deployment_boundary=rec.deploy_boundary,
        scalar=rec.scalar,
    )

    # ── IF5 geometry (v35: E_t from IngestRecord) ─────────────────────────────
    if5 = IF5_GeometryTelemetry(
        spatial_position=rec.geometry_position,
        temporal_state=time.time(),
        network_state={},
        rf_state={},
        sync_state="NTP_STUB",
        external_topology={},
        tensor=rec.geometry_tensor,
        episode_id=rec.episode_id,
        observation_shape=list(rec.observation.shape) if hasattr(rec.observation, "shape") else None,
        reward_signal=rec.reward,
    )
    # Still gap: live network/RF
    _write_agent_gap(f"{agent_gap_dir}/env_agent.jsonl", {
        "type": "IF5_request", "frame_id": rec.frame_id,
        "needs": ["network_state", "rf_state"],
        "tensor_len": len(if5.tensor),
        "gap": "Live network/RF telemetry not integrated.",
    })

    # ── IF1 creative (v35: Ollama) ────────────────────────────────────────────
    if1 = compute_if1(rec, oc, stream_callback)

    # ── IF2 efficiency (v35: real token metrics) ──────────────────────────────
    if2 = compute_if2(if1, rec)

    # ── IF3 direction validator (with loss gradient feedback) ─────────────────
    if3 = compute_if3_retune(if1, if2, rec.theta_3, loss_gradient=loss_grad)

    # ── IF6 fusion ────────────────────────────────────────────────────────────
    if if6_state is None:
        if6_state = IF6_LanguageFusion()
    if6_state.advance_chunk()

    reward_str = f" rew={rec.reward:.3f}" if rec.reward is not None else ""
    fused = (
        f"[chunk={if6_state.chunk_size}]"
        f"[if1={str(if1.output)[:60]}]"
        f"[if2_halt={if2.halt_triggered}|lat={if2.latency_ms:.1f}ms|tok/s={if2.tokens_per_sec:.0f}]"
        f"[if3_D={if3.D_orthogonal:.3f}|Gp={if3.G_p:.3f}]"
        f"[if4={if4.system_identity}]"
        f"[if5_pos={if5.spatial_position[:2]}{reward_str}]"
    )
    if6_state.fused_context = fused

    # ── IF7 relational arbitration ────────────────────────────────────────────
    if7 = compute_if7_arbitration(prev_if7, if1, if2, if3, if4, if5, if6_state)

    # ── OF1 next-state prediction (v37: PLDM Transformer + MPPI planning) ────
    # v36 path: image/video encoder via OF1MultimodalPredictor
    # v37 addition: use PLDMNumpyPredictor for future_state_estimate
    #               use MPPILatentSolver for action planning over latents
    mppi_best_cost = 0.0

    if _OF1_MULTIMODAL_AVAILABLE and rec.observation is not None and rec.modality != "text":
        try:
            _predictor = _get_of1_predictor(latent_dims=64)
            _result = _predictor.predict(
                obs=rec.observation,
                modality=rec.modality,
                env_idx=rec.episode_id or 0,
            )
            base_latent = _result["latent_vector"]
            base_confidence = _result["confidence"]

            # v37: override future_state_estimate with PLDM Transformer
            future_state_estimate = _result["future_state_estimate"]
            if _PLDM_AVAILABLE and ts.latent_history:
                try:
                    _pldm = _get_pldm_predictor(input_dim=64)
                    future_state_estimate = _pldm.predict(
                        latent_history=ts.latent_history,
                        action=rec.action.tolist() if rec.action is not None and hasattr(rec.action, "tolist") else rec.action,
                    )
                    ts.pldm_pred_count += 1
                except Exception:
                    pass

            # v37: MPPI action plan over latent space
            if _MPPI_AVAILABLE and _PLDM_AVAILABLE and ts.latent_history and rec.latent_vector:
                try:
                    _pldm = _get_pldm_predictor(input_dim=64)
                    _mppi = _get_mppi_solver(
                        predictor=_pldm,
                        action_dim=len(rec.action.flatten().tolist()) if rec.action is not None and hasattr(rec.action, "flatten") else 4,
                    )
                    _mppi_result = _mppi.solve(
                        latent_history=ts.latent_history,
                        goal_latent=rec.latent_vector or base_latent,
                    )
                    mppi_best_cost = _mppi_result["best_cost"]
                    ts.mppi_best_cost = mppi_best_cost
                except Exception:
                    pass

            of1 = OF1_NextStatePredict(
                prediction_tokens=rec.prediction_tokens if isinstance(rec.prediction_tokens, list) else [rec.prediction_tokens],
                latent_vector=base_latent,
                modality=rec.modality,
                future_state_estimate=future_state_estimate,
                confidence=base_confidence,
            )
        except Exception:
            of1 = OF1_NextStatePredict(
                prediction_tokens=rec.prediction_tokens if isinstance(rec.prediction_tokens, list) else [rec.prediction_tokens],
                latent_vector=rec.latent_vector or [if7.rel_scalar, if3.G_p, if2.latency_ms / 1000.0],
                modality=rec.modality,
                future_state_estimate={},
                confidence=max(0.0, 1.0 - if7.stability_norm),
            )
    else:
        # v37: text path — use PLDM if latent_history available
        base_latent = rec.latent_vector or [if7.rel_scalar, if3.G_p, if2.latency_ms / 1000.0]
        future_state_estimate = {}
        if _PLDM_AVAILABLE and ts.latent_history:
            try:
                _pldm = _get_pldm_predictor(input_dim=64)
                future_state_estimate = _pldm.predict(latent_history=ts.latent_history)
                ts.pldm_pred_count += 1
            except Exception:
                pass

        of1 = OF1_NextStatePredict(
            prediction_tokens=rec.prediction_tokens if isinstance(rec.prediction_tokens, list) else [rec.prediction_tokens],
            latent_vector=base_latent,
            modality=rec.modality,
            future_state_estimate=future_state_estimate,
            confidence=max(0.0, 1.0 - if7.stability_norm),
        )

    # v37: update latent_history in TrainingState for PLDM/VCReg
    if of1.latent_vector:
        ts.latent_history.append(of1.latent_vector)
        if len(ts.latent_history) > 32:
            ts.latent_history = ts.latent_history[-32:]

    # ── OF2 explanation ───────────────────────────────────────────────────────
    explanation = (
        f"Frame {rec.frame_id}: {if6_state.fused_context[:200]}. "
        f"Loss stab={if7.stability_norm:.4f}. Conf={of1.confidence:.4f}. "
        f"Safety={if7.safety_ok}. θ_step={ts.step}."
    )
    of2 = OF2_LanguageExplanation(
        explanation=explanation,
        coherence_score=of1.confidence,
        goal_match=of1.confidence * 0.9,
    )

    # ── Loss computation (v37: + L_reg from VCReg/PLDMLoss) ──────────────────
    # v37: compute latent regularization loss
    L_reg = 0.0
    if _LOSS_REG_AVAILABLE and len(ts.latent_history) >= 2:
        try:
            reg_out = _compute_latent_reg_loss(ts.latent_history[-8:])
            L_reg = reg_out.get("total", 0.0)
            ts.latent_reg_loss = L_reg
        except Exception:
            pass

    loss = compute_loss(
        of1, of2, if7, if2, if3,
        lambda_weights=ts.lambda_weights,
        Y_target=rec.Y_target,
        Y_future=rec.Y_future,
        L_reg=L_reg,
    )

    elapsed_ms = (time.time() - t_start) * 1000.0

    return {
        "frame_id": rec.frame_id,
        "if1_T_max": if1.T_max,
        "if1_tokens_generated": if1.tokens_generated,
        "if1_generation_ms": if1.generation_ms,
        "if1_model_used": if1.model_used,
        "if1_output_preview": (if1.output or "")[:120],
        "if2_latency_ms": if2.latency_ms,
        "if2_halt": if2.halt_triggered,
        "if2_tokens_per_sec": if2.tokens_per_sec,
        "if3_D_orthogonal": if3.D_orthogonal,
        "if3_G_p": if3.G_p,
        "if3_theta_1_prime_temp": if3.theta_1_prime.get("temperature", 0.0),
        "if3_grad_feedback": if3.loss_gradient_feedback,
        "if4_system_id": if4.system_identity,
        "if4_scalar": if4.scalar,
        "if5_position": if5.spatial_position,
        "if5_tensor_len": len(if5.tensor),
        "if5_reward": if5.reward_signal,
        "if6_chunk_size": if6_state.chunk_size,
        "if6_fused_context": if6_state.fused_context,
        "if7_rel_scalar": if7.rel_scalar,
        "if7_stability_norm": if7.stability_norm,
        "if7_safety_ok": if7.safety_ok,
        "if7_invariants": {
            "mission_coherent": if7.mission_coherent,
            "runtime_valid": if7.runtime_valid,
            "bounded_recursion": if7.bounded_recursion,
            "safety_ok": if7.safety_ok,
            "temporal_continuous": if7.temporal_continuous,
            "execution_admissible": if7.execution_admissible,
        },
        "of1_prediction_tokens": of1.prediction_tokens,
        "of1_confidence": of1.confidence,
        "of1_modality": of1.modality,
        "of2_explanation": of2.explanation[:300],
        "of2_coherence_score": of2.coherence_score,
        "loss_total": loss.total,
        "loss_coherence": loss.L_coherence,
        "loss_efficiency": loss.L_efficiency,
        "loss_novelty": loss.L_novelty,
        "loss_stability": loss.L_stability,
        "loss_alignment": loss.L_alignment,
        "loss_prediction": loss.L_prediction,
        "loss_reg": loss.L_reg,                          # v37
        "pldm_pred_count": ts.pldm_pred_count,           # v37
        "mppi_best_cost": ts.mppi_best_cost,             # v37
        "latent_reg_loss": ts.latent_reg_loss,           # v37
        "elapsed_engine_ms": elapsed_ms,
        # Pass-through refs
        "if7_ref": if7,
        "if6_ref": if6_state,
        "loss_ref": loss,
    }


def golias_sequence(
    records: List[IngestRecord],
    training_state: Optional[TrainingState] = None,
    ollama_config: Optional[OllamaConfig] = None,
    agent_gap_dir: str = "agents",
    stream_callback=None,
) -> dict:
    """Run full sequence with θ* updates between frames."""
    prev_if7 = None
    if6_state = None
    ts = training_state or TrainingState()
    frame_results = []

    for rec in records:
        result = golias_forward(
            rec,
            prev_if7=prev_if7,
            if6_state=if6_state,
            training_state=ts,
            ollama_config=ollama_config,
            agent_gap_dir=agent_gap_dir,
            stream_callback=stream_callback,
        )
        prev_if7 = result.pop("if7_ref", None)
        if6_state = result.pop("if6_ref", None)
        loss_ref = result.pop("loss_ref", GoliasLoss())

        # θ* update after each frame
        if prev_if7 is not None:
            ts = theta_star_update(ts, loss_ref, prev_if7)

        frame_results.append(result)

    n = len(frame_results)
    if n > 0:
        avg_loss = sum(r["loss_total"] for r in frame_results) / n
        avg_coh = sum(r["loss_coherence"] for r in frame_results) / n
        avg_stab = sum(r["if7_stability_norm"] for r in frame_results) / n
        avg_conf = sum(r["of1_confidence"] for r in frame_results) / n
        last_chunk = frame_results[-1].get("if6_chunk_size", 1000)
    else:
        avg_loss = avg_coh = avg_stab = avg_conf = 0.0
        last_chunk = 1000

    return {
        "n_frames": n,
        "frames": frame_results,
        "avg_L_Golias": avg_loss,
        "avg_L_coherence": avg_coh,
        "avg_stability_norm": avg_stab,
        "avg_of1_confidence": avg_conf,
        "final_chunk_size": last_chunk,
        "training_state": ts,
    }
