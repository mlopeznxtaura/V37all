"""core/__init__.py — Golias v37 core package.

v37 additions (stable-worldmodel architectural gains):
  - pldm_predictor:  PLDMNumpyPredictor — AdaLN-zero Transformer latent predictor
  - mppi_solver:     MPPILatentSolver   — MPPI model-predictive planning over latents
  - loss_reg:        PLDMLoss, VCReg, SIGReg, TemporalStraighteningLoss

v36 modules retained unchanged (no modifications):
  - worldmodel_live, m3_validator, of1_multimodal, golias_types, etc.
"""
from .golias_types import *
from .golias_engine import golias_forward, golias_sequence
from .ingest_bridge import auto_parse, compute_ingest_stats, stream_ingest
from .training_loop import run_training, TrainingConfig
from .inference_engine import InferenceSession, infer_once

# v36: existing capability modules
from .worldmodel_live import (
    WorldModelLiveSource,
    LiveEnvConfig,
    make_live_source,
    obs_to_geometry_tensor,
    RunningNormalizer,
)
from .m3_validator import (
    compute_m3_validator,
    m3_enhanced_if3,
    cosine_similarity,
    cosine_distance,
)
from .of1_multimodal import (
    OF1MultimodalPredictor,
    get_predictor,
    encode_image_obs,
    encode_vector_obs,
    latent_confidence,
)

# v37: new architectural modules from stable-worldmodel
from .pldm_predictor import (
    PLDMNumpyPredictor,
    get_pldm_predictor,
)
from .mppi_solver import (
    MPPILatentSolver,
    get_mppi_solver,
)
from .loss_reg import (
    VCReg,
    SIGReg,
    PLDMLoss,
    temporal_straightening_loss,
    compute_latent_reg_loss,
)
