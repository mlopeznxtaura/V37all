# v37all — Golias Architecture Pipeline v37

## Deliverable

```
v37all/
├── v35_gui.py              — Main entry: NiceGUI three-stage terminal mirror
├── core/
│   ├── __init__.py
│   ├── golias_types.py     — v37: + L_reg in GoliasLoss, latent_history in TrainingState
│   ├── golias_engine.py    — v37: PLDM predictor + MPPI planner + L_reg wired in
│   ├── ingest_bridge.py    — Stage 1 (unchanged from v36)
│   ├── training_loop.py    — Stage 2 (unchanged from v36)
│   ├── inference_engine.py — Stage 3 (unchanged from v36)
│   ├── worldmodel_live.py  — v36: Live World → IngestRecord (unchanged)
│   ├── m3_validator.py     — v36: M3 Ollama embedding validator (unchanged)
│   ├── of1_multimodal.py   — v36: OF1 CLIP/MobileNet/DCT encoder (unchanged)
│   ├── pldm_predictor.py   — v37 NEW: AdaLN-zero Transformer latent predictor
│   ├── mppi_solver.py      — v37 NEW: MPPI model-predictive planning over latents
│   └── loss_reg.py         — v37 NEW: VCReg + SIGReg + PLDMLoss + TemporalStraightening
├── data/
│   └── golias_default.jsonl
└── agents/
    ├── model_agent.jsonl
    └── env_agent.jsonl
```

## Run

```bash
pip install nicegui
pip install stable-worldmodel
pip install torch torchvision   # optional — DCT fallback always works
pip install clip-by-openai      # optional
python v35_gui.py
# → http://localhost:8766
```

## v37 vs v36

| Component | v36 | v37 |
|---|---|---|
| OF1 future_state_estimate | Linear delta extrapolation (LatentFrameBuffer) | **PLDMNumpyPredictor — AdaLN-zero Transformer, action-conditioned** |
| OF1 action planning | None (rel_scalar proxy only) | **MPPILatentSolver — TopK elite MPPI over latent rollouts** |
| GoliasLoss | 6 components (λ1–λ6) | **+ L_reg (λ7=0.4): VCReg + PLDMLoss latent regularization** |
| Latent collapse prevention | None | **VCReg std_loss + cov_loss; temporal alignment; straightening** |
| TrainingState | loss_history, fisher_diagonal | **+ latent_history (rolling 32), pldm_pred_count, mppi_best_cost** |
| system_id | golias_v35 | **golias_v37** |

## Architecture (v37)

```
STAGE 1 — INGEST  (unchanged from v36)
  IngestRecord ← JSONL | WorldModel | CSV | text | live World.step()

STAGE 2 — TRAIN (θ* outer loop)
  golias_sequence(records, training_state)
    → IF1–IF7 (unchanged from v36)
    → OF1 latent_vector  ← v36: CLIP/MobileNet/DCT encoder
    → OF1 future_state   ← v37: PLDMNumpyPredictor(AdaLN-zero, depth=2)
    → OF1 action plan    ← v37: MPPILatentSolver(N=64, topK=16, H=4)
    → L_reg              ← v37: PLDMLoss(VCReg + temp_align + straighten)
    → L_Golias = λ1·L_coh + … + λ6·L_pred + λ7·L_reg
    → θ*_update (EWC + λ_adaptive, now includes λ7)

STAGE 3 — INFER  (unchanged from v36)
  InferenceSession → IF6 fused context → streaming terminal
```

## New Modules

### `core/pldm_predictor.py`
Closes **OF1_future_state_quality** gap.

```python
from core.pldm_predictor import get_pldm_predictor

pldm = get_pldm_predictor(input_dim=64, depth=2)
result = pldm.predict(latent_history=ts.latent_history, action=action_vec)
# result["predicted_obs"]  — next latent (AdaLN-zero Transformer)
# result["method"]         — "pldm_adaln_transformer"
```

Architecture: Conv1d Embedder + sinusoidal pos_embed + depth×AdaLN-zero blocks + LayerNorm + output_proj.
Directly mirrors stable-worldmodel PLDM (arxiv:2502.14819) in pure NumPy.

### `core/mppi_solver.py`
Closes **OF1_action_planning** gap.

```python
from core.mppi_solver import get_mppi_solver

mppi = get_mppi_solver(predictor=pldm, num_samples=64, n_steps=5, horizon=4)
result = mppi.solve(latent_history=ts.latent_history, goal_latent=goal_vec)
# result["actions"]       — (H, A) optimal action sequence
# result["best_cost"]     — L2 cost at best plan terminal state
```

Mirrors stable-worldmodel MPPISolver: TopK elite filter → softmax-weighted mean update.

### `core/loss_reg.py`
Closes **L_reg_latent_collapse** gap.

```python
from core.loss_reg import compute_latent_reg_loss

reg = compute_latent_reg_loss(ts.latent_history[-8:])
# reg["std_loss"]               — collapse prevention
# reg["cov_loss"]               — entanglement prevention
# reg["temp_align_loss"]        — temporal coherence
# reg["temporal_straight_loss"] — trajectory linearity (arxiv:2603.12231)
# reg["total"]                  — weighted composite for L_reg
```

## Architectural Sources (stable-worldmodel)

| Module | Source | Reference |
|---|---|---|
| AdaLN-zero ConditionalBlock | stable_worldmodel/wm/pldm/module.py | arxiv:2502.14819 |
| Autoregressive Predictor | stable_worldmodel/wm/pldm/pldm.py | |
| MPPI Solver | stable_worldmodel/solver/mppi.py | Model-predictive path integral |
| VCReg | stable_worldmodel/wm/loss.py | |
| SIGReg | stable_worldmodel/wm/loss.py | arxiv:2511.08544 |
| PLDMLoss | stable_worldmodel/wm/loss.py | arxiv:2502.14819 |
| TemporalStraighteningLoss | stable_worldmodel/wm/loss.py | arxiv:2603.12231 |

## Google Cloud / Hackathon Integration

This system targets **Track 2: Optimize (Existing Agents)** of the Google for Startups AI Agents Challenge.

- **Agent Platform feature critical**: Gemini ADK multi-step reasoning evaluation
- **Missing from Agent Platform**: Real-time latent trajectory visualization
- **API capability that would save 2+ hours**: Native PLDM rollout endpoint in Vertex AI

Submission deadline: June 5, 2026 @ 5:00 PM PT
