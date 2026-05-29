"""golias_types.py v37 — Canonical data types for Golias IF/OF architecture.

v37 additions (stable-worldmodel architectural gains):
  GoliasLoss       → + L_reg term (VCReg + PLDMLoss latent regularization)
  TrainingState    → + latent_history, pldm_pred_count, mppi_best_cost
  IngestRecord     → + system_id updated to "golias_v37"

v35/v36 contract unchanged — all existing fields preserved.

v35 reparameterization changes vs v34:
  IF5_GeometryTelemetry  → now carries full WorldModelRecord (E_t from ReplayBuffer/Dataset)
  IF1_CreativeExploration → now carries OllamaConfig for local model integration
  IF2_EfficiencyGate      → now carries real token/time tracking from Ollama stream
  TrainingState           → new: θ* optimization state (was agent gap in v34)
  IngestRecord            → new: stable-worldmodel-aligned ingest schema (Stage 1)
  InferenceRequest        → new: structured Ollama inference request (Stage 3)

References:
  Transformer         arxiv:1706.03762
  MAML meta-learning  arxiv:1703.01161
  Relational Networks arxiv:1703.04136
  Causal ML           arxiv:1912.01700
  Efficient xformers  arxiv:2202.08906
  Multimodal fusion   arxiv:2205.14135
  DDPM latent pred    arxiv:2006.11239
  Multi-obj loss      arxiv:2401.04088
  EWC                 arxiv:1612.00796
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Ingest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IngestRecord:
    """Unified ingest record bridging JSONL IF-schema + WorldModel dataset schema.

    Reparameterizes the IF5 E_t tensor gap from v34 by accepting either
    stable-worldmodel-style episode dicts or raw Golias IF JSONL records.
    """
    frame_id: Any
    language: str
    geometry_position: List[float] = field(default_factory=lambda: [0., 0., 0.])
    geometry_tensor: List[float] = field(default_factory=list)

    # WorldModel-style columns (mapped from stable-worldmodel Dataset schema)
    observation: Optional[Any] = None      # pixels or obs vector
    action: Optional[Any] = None           # action vector
    reward: Optional[float] = None
    done: Optional[bool] = None
    episode_id: Optional[int] = None

    # Substrate
    hardware_state: int = 0b11110000
    runtime_constraints: int = 0b00001111
    memory_mb: float = 4096.0
    exec_perms: int = 0xFF
    system_id: str = "golias_v37"
    deploy_boundary: str = "sandbox"
    scalar: int = 0xAB

    # IF params
    T_max: float = 1.2
    tau_ms: float = 200.0
    epsilon: float = 0.6
    compute_cost: float = 0.12
    latency_ms: float = 45.0
    theta_3: Dict[str, float] = field(default_factory=lambda: {"delta_novelty": 0.12, "delta_coverage": 0.08})
    prediction_tokens: List[str] = field(default_factory=list)
    latent_vector: List[float] = field(default_factory=list)
    confidence: float = 0.9
    Y_target: str = ""
    Y_future: Optional[List[float]] = None
    modality: str = "text"

    @classmethod
    def from_jsonl_dict(cls, d: dict) -> "IngestRecord":
        geo = d.get("geometry", {})
        sub = d.get("substrate", {})
        return cls(
            frame_id=d.get("frame_id"),
            language=d.get("language", d.get("context", "")),
            geometry_position=geo.get("position", [0., 0., 0.]),
            geometry_tensor=geo.get("tensor", []),
            hardware_state=sub.get("hardware_state", 0b11110000),
            runtime_constraints=sub.get("runtime_constraints", 0b00001111),
            memory_mb=sub.get("memory_mb", 4096.0),
            exec_perms=sub.get("exec_perms", 0xFF),
            system_id=sub.get("system_id", "golias_v35"),
            deploy_boundary=sub.get("deploy_boundary", "sandbox"),
            scalar=sub.get("scalar", 0xAB),
            T_max=d.get("T_max", 1.2),
            tau_ms=d.get("tau_ms", 200.0),
            epsilon=d.get("epsilon", 0.6),
            compute_cost=d.get("compute_cost", 0.12),
            latency_ms=d.get("latency_ms", 45.0),
            theta_3=d.get("theta_3", {"delta_novelty": 0.12, "delta_coverage": 0.08}),
            prediction_tokens=d.get("next_frame_prediction_tokens", d.get("prediction_tokens", [])),
            latent_vector=d.get("latent_vector", []),
            confidence=d.get("confidence", 0.9),
            Y_target=d.get("Y_target", ""),
            Y_future=d.get("Y_future"),
            modality=d.get("modality", "text"),
        )

    @classmethod
    def from_worldmodel_episode(cls, ep_dict: dict, frame_idx: int = 0) -> "IngestRecord":
        """Map a stable-worldmodel episode dict slice into an IngestRecord."""
        obs = ep_dict.get("observation", ep_dict.get("obs"))
        act = ep_dict.get("action")
        rew = ep_dict.get("reward")
        done = ep_dict.get("done", ep_dict.get("terminated", False))

        # Build E_t tensor from obs if available
        tensor: List[float] = []
        if obs is not None:
            try:
                import numpy as np
                arr = np.asarray(obs).flatten()
                tensor = arr[:16].tolist()  # cap at 16 for IF6 fusion
            except Exception:
                pass

        return cls(
            frame_id=f"ep{ep_dict.get('episode_id', 0)}_f{frame_idx}",
            language=str(ep_dict.get("language", ep_dict.get("info", {}).get("description", ""))),
            geometry_tensor=tensor,
            observation=obs,
            action=act,
            reward=float(rew) if rew is not None else None,
            done=bool(done),
            episode_id=ep_dict.get("episode_id"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# IF structs (v35 — reparameterized for local model + real telemetry)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OllamaConfig:
    """Local Ollama model config — replaces the M1/M2/M3 STUB gaps from v34."""
    model: str = "llama3.2"
    host: str = "http://localhost:11434"
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 256
    stream: bool = True
    timeout_ms: float = 5000.0


@dataclass
class IF1_CreativeExploration:
    """if_1 = M1(x_t, θ_1, T_max) — v35: backed by Ollama local inference."""
    x_t: Dict[str, Any]
    theta_1: Dict[str, float]
    T_max: float
    ollama_config: OllamaConfig = field(default_factory=OllamaConfig)
    output: Optional[str] = None
    tokens_generated: int = 0
    generation_ms: float = 0.0
    model_used: str = ""


@dataclass
class IF2_EfficiencyGate:
    """if_2 = M2(if_1, θ_2, τ) — v35: real token/time tracking from stream."""
    if1_ref: IF1_CreativeExploration = field(default_factory=lambda: IF1_CreativeExploration({}, {}, 1.0))
    theta_2: Dict[str, float] = field(default_factory=dict)
    tau_ms: float = 200.0
    epsilon: float = 0.6
    halt_triggered: bool = False
    latency_ms: float = 0.0
    compute_cost: float = 0.0
    tokens_per_sec: float = 0.0
    output: Optional[str] = None


@dataclass
class IF3_DirectionValidator:
    """if_3 = M3(if_1, if_2, θ_3) — v35: theta* adaptive retune with loss gradient feedback."""
    if1_ref: IF1_CreativeExploration = field(default_factory=lambda: IF1_CreativeExploration({}, {}, 1.0))
    if2_ref: IF2_EfficiencyGate = field(default_factory=IF2_EfficiencyGate)
    theta_3: Dict[str, float] = field(default_factory=dict)
    theta_1_prime: Dict[str, float] = field(default_factory=dict)
    theta_2_prime: Dict[str, float] = field(default_factory=dict)
    D_orthogonal: float = 0.0
    G_p: float = 0.0
    loss_gradient_feedback: float = 0.0   # v35: from θ* loop
    output: Optional[str] = None


@dataclass
class IF4_BinarySubstrate:
    """if_4 = B(S_t) — unchanged from v34."""
    hardware_state: int = 0b11110000
    runtime_constraints: int = 0b00001111
    memory_available_mb: float = 4096.0
    execution_permissions: int = 0xFF
    system_identity: str = "golias_v35"
    deployment_boundary: str = "sandbox"
    scalar: int = 0xAB


@dataclass
class IF5_GeometryTelemetry:
    """if_5 = G(E_t) — v35: E_t tensor now populated from IngestRecord (WorldModel bridge)."""
    spatial_position: List[float] = field(default_factory=lambda: [0., 0., 0.])
    temporal_state: float = 0.0
    network_state: Dict[str, Any] = field(default_factory=dict)
    rf_state: Dict[str, Any] = field(default_factory=dict)
    sync_state: str = "NTP_STUB"
    external_topology: Dict[str, Any] = field(default_factory=dict)
    tensor: List[float] = field(default_factory=list)
    # v35 additions
    episode_id: Optional[int] = None
    observation_shape: Optional[List[int]] = None
    reward_signal: Optional[float] = None


@dataclass
class IF6_LanguageFusion:
    """if_6 = Φ(if_1 ⊕ … ⊕ if_5) — rotating chunk scheduler unchanged."""
    CHUNK_SCHEDULE = [800, 900, 1000, 1100, 1200]
    chunk_index: int = 0
    fused_context: str = ""
    chunk_size: int = 1000

    def advance_chunk(self):
        self.chunk_index = (self.chunk_index + 1) % len(self.CHUNK_SCHEDULE)
        self.chunk_size = self.CHUNK_SCHEDULE[self.chunk_index]


@dataclass
class IF7_RelationalArbitration:
    """if_7 = Rel(Δif_1…Δif_6) — v35: stability_norm feeds back into θ* loop."""
    delta_if1: float = 0.0
    delta_if2: float = 0.0
    delta_if3: float = 0.0
    delta_if4: float = 0.0
    delta_if5: float = 0.0
    delta_if6: float = 0.0
    mission_coherent: bool = True
    runtime_valid: bool = True
    bounded_recursion: bool = True
    safety_ok: bool = True
    temporal_continuous: bool = True
    execution_admissible: bool = True
    rel_scalar: float = 0.0
    stability_norm: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# OF structs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OF1_NextStatePredict:
    """of_1 = F_pred(if_1…if_7)."""
    prediction_tokens: List[str] = field(default_factory=list)
    latent_vector: List[float] = field(default_factory=list)
    modality: str = "text"
    future_state_estimate: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class OF2_LanguageExplanation:
    """of_2 = F_explain(of_1, if_1…if_7)."""
    explanation: str = ""
    coherence_score: float = 0.0
    goal_match: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoliasLoss:
    """L_Golias = λ1·L_coh + λ2·L_eff + λ3·L_nov + λ4·L_sta + λ5·L_aln + λ6·L_pred"""
    lambda_1: float = 1.0
    lambda_2: float = 0.5
    lambda_3: float = 0.8
    lambda_4: float = 0.6
    lambda_5: float = 1.0
    lambda_6: float = 0.9
    L_coherence: float = 0.0
    L_efficiency: float = 0.0
    L_novelty: float = 0.0
    L_stability: float = 0.0
    L_alignment: float = 0.0
    L_prediction: float = 0.0

    # v37: latent regularization (VCReg + PLDMLoss)
    lambda_7: float = 0.4
    L_reg: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.lambda_1 * self.L_coherence
            + self.lambda_2 * self.L_efficiency
            + self.lambda_3 * self.L_novelty
            + self.lambda_4 * self.L_stability
            + self.lambda_5 * self.L_alignment
            + self.lambda_6 * self.L_prediction
            + self.lambda_7 * self.L_reg
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Training State (θ* loop — was agent gap in v34)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingState:
    """θ* optimization state. Tracks the outer loop over Golias loss.

    v35 reparameterizes the v34 model_agent theta_optimization gap:
    - Per-step loss gradient recorded here
    - Adaptive λ reweighting based on running loss stats
    - EWC penalty term for catastrophic forgetting prevention (arxiv:1612.00796)
    """
    step: int = 0
    epoch: int = 0
    total_frames: int = 0
    # Current theta params (what v34 called θ*)
    theta_1: Dict[str, float] = field(default_factory=lambda: {
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 256
    })
    theta_2: Dict[str, float] = field(default_factory=lambda: {
        "epsilon": 0.6, "tau_ms": 200.0, "cost_weight": 0.4
    })
    theta_3: Dict[str, float] = field(default_factory=lambda: {
        "delta_novelty": 0.12, "delta_coverage": 0.08
    })
    # Loss history for gradient estimation
    loss_history: List[float] = field(default_factory=list)
    coherence_history: List[float] = field(default_factory=list)
    stability_history: List[float] = field(default_factory=list)
    # EWC
    ewc_lambda: float = 5000.0
    fisher_diagonal: Dict[str, float] = field(default_factory=dict)
    # Adaptive lambda weights
    lambda_weights: Dict[str, float] = field(default_factory=lambda: {
        "lambda_1": 1.0, "lambda_2": 0.5, "lambda_3": 0.8,
        "lambda_4": 0.6, "lambda_5": 1.0, "lambda_6": 0.9,
        "lambda_7": 0.4,  # v37: latent regularization
    })
    # Status
    running: bool = False
    status_msg: str = "idle"
    perplexity: float = 0.0
    # v37: PLDM latent history + MPPI stats
    latent_history: List[List[float]] = field(default_factory=list)
    pldm_pred_count: int = 0
    mppi_best_cost: float = 0.0
    latent_reg_loss: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Inference
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceRequest:
    """Structured inference request for Stage 3 (Ollama / local model)."""
    prompt: str = ""
    system_prompt: str = ""
    ollama_config: OllamaConfig = field(default_factory=OllamaConfig)
    if_context: Optional[str] = None   # IF6 fused context to inject
    session_id: str = ""
    # Output
    response: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    done: bool = False
    error: Optional[str] = None
