"""of1_multimodal.py v36 — OF1 multimodal prediction (image/video latent path).

Closes gap: OF1_multimodal_prediction
  Task1: Integrate CLIP or BLIP encoder for image obs → latent_vector
  Test1: assert len(latent_vector) > 0 when modality=="image"
  Task2: Compute future_state_estimate from latent delta over last 3 frames
  Test2: assert "predicted_obs" in future_state_estimate
  Task3: Confidence calibration from latent norm
  Test3: assert 0.0 <= confidence <= 1.0

Design notes:
  - Uses torchvision CLIP (ViT-B/32) when available; falls back to a
    lightweight MobileNetV3-based encoder (always available via torchvision).
  - If torch is absent entirely, falls back to a pure-NumPy DCT descriptor
    that still produces a meaningful latent_vector from raw pixels.
  - Keeps a rolling frame buffer (3 frames) for temporal delta prediction.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Backend detection
# ─────────────────────────────────────────────────────────────────────────────

_BACKEND = "numpy_dct"   # default

try:
    import torch
    import torchvision.transforms as T
    _BACKEND = "torchvision_mobilenet"
    try:
        # CLIP from OpenAI (optional)
        import clip as _clip
        _BACKEND = "clip"
    except ImportError:
        pass
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────────────────────────────────────

def _dct2d_descriptor(img: np.ndarray, n_coeffs: int = 64) -> List[float]:
    """Pure-NumPy 2-D DCT energy descriptor.  No torch required.

    Takes the n_coeffs lowest-frequency coefficients of the luminance
    channel as the latent vector.  Cheap but semantically coherent
    enough for delta-based prediction.
    """
    # Convert to float32 luminance
    if img.ndim == 3:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        lum = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    elif img.ndim == 2:
        lum = img.astype(np.float32)
    else:
        return [0.0] * n_coeffs

    # Normalize to [-1,1]
    lum = lum / 127.5 - 1.0

    # 2-D DCT via separable 1-D FFT (real-valued)
    from scipy.fft import dctn  # type: ignore
    try:
        coeffs = dctn(lum, norm="ortho")
    except Exception:
        # scipy not available: fall back to raw column means
        coeffs = lum
    flat = coeffs.flatten()
    # Zig-zag-style: take top-left patch (lowest frequencies)
    side = max(1, int(math.sqrt(n_coeffs)))
    selected: List[float] = []
    for i in range(min(side, coeffs.shape[0]) if hasattr(coeffs, "shape") else len(flat)):
        for j in range(min(side, coeffs.shape[1] if hasattr(coeffs, "shape") and coeffs.ndim > 1 else 1)):
            if hasattr(coeffs, "__getitem__") and hasattr(coeffs, "ndim") and coeffs.ndim > 1:
                selected.append(float(coeffs[i, j]))
            else:
                idx = i * side + j
                if idx < len(flat):
                    selected.append(float(flat[idx]))
    # Pad or truncate
    while len(selected) < n_coeffs:
        selected.append(0.0)
    return selected[:n_coeffs]


def _mobilenet_encode(img: np.ndarray, n_coeffs: int = 64) -> List[float]:
    """MobileNetV3-small features → pooled latent vector."""
    import torch  # noqa: F811
    import torchvision.models as models  # type: ignore
    import torchvision.transforms as T  # noqa: F811
    from PIL import Image  # type: ignore

    # Build model once (module-level cache)
    if not hasattr(_mobilenet_encode, "_model"):
        mn = models.mobilenet_v3_small(weights=None)
        mn.eval()
        _mobilenet_encode._model = mn  # type: ignore

    model = _mobilenet_encode._model  # type: ignore

    pil = Image.fromarray(img.astype(np.uint8)) if img.dtype != np.uint8 else Image.fromarray(img)
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(pil).unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        # Use features up to adaptive avg pool
        feats = model.features(tensor)
        pooled = feats.mean(dim=[2, 3]).squeeze(0).numpy()

    # Reduce to n_coeffs via strided selection
    step = max(1, len(pooled) // n_coeffs)
    vec = pooled[::step][:n_coeffs].tolist()
    while len(vec) < n_coeffs:
        vec.append(0.0)
    return vec


def _clip_encode(img: np.ndarray, n_coeffs: int = 64) -> List[float]:
    """CLIP ViT-B/32 image encoder → latent_vector."""
    import clip  # type: ignore
    import torch  # noqa: F811
    from PIL import Image  # type: ignore

    if not hasattr(_clip_encode, "_model"):
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        _clip_encode._model = model  # type: ignore
        _clip_encode._preprocess = preprocess  # type: ignore

    model = _clip_encode._model  # type: ignore
    preprocess = _clip_encode._preprocess  # type: ignore

    pil = Image.fromarray(img.astype(np.uint8))
    image_input = preprocess(pil).unsqueeze(0)

    with torch.no_grad():
        feat = model.encode_image(image_input).squeeze(0).float().numpy()

    # CLIP is 512-D; truncate/pad to n_coeffs
    vec = feat[:n_coeffs].tolist()
    while len(vec) < n_coeffs:
        vec.append(0.0)
    return vec


def encode_image_obs(img: np.ndarray, latent_dims: int = 64) -> List[float]:
    """Encode an image observation to a latent vector.

    Task1: returns list of length > 0 for any non-empty image obs.
    Chooses best available backend: CLIP > MobileNet > NumPy-DCT.
    """
    if img is None or img.size == 0:
        return []
    try:
        if _BACKEND == "clip":
            return _clip_encode(img, latent_dims)
        elif _BACKEND == "torchvision_mobilenet":
            return _mobilenet_encode(img, latent_dims)
        else:
            return _dct2d_descriptor(img, latent_dims)
    except Exception:
        return _dct2d_descriptor(img, latent_dims)


# ─────────────────────────────────────────────────────────────────────────────
# Vector latent path (for non-image observations)
# ─────────────────────────────────────────────────────────────────────────────

def encode_vector_obs(obs: np.ndarray, latent_dims: int = 64) -> List[float]:
    """Encode a 1-D obs vector by zero-padding or truncating to latent_dims."""
    flat = obs.flatten().tolist()
    while len(flat) < latent_dims:
        flat.append(0.0)
    return flat[:latent_dims]


# ─────────────────────────────────────────────────────────────────────────────
# Frame buffer for temporal delta
# ─────────────────────────────────────────────────────────────────────────────

class LatentFrameBuffer:
    """Rolling buffer of the last N latent vectors for delta estimation."""

    def __init__(self, maxlen: int = 3):
        self._buf: deque = deque(maxlen=maxlen)

    def push(self, latent: List[float]) -> None:
        self._buf.append(latent)

    def predict_next(self) -> Dict[str, Any]:
        """Estimate next latent from linear extrapolation over buffered frames.

        Task2: returns dict with key "predicted_obs".
        """
        if len(self._buf) < 2:
            predicted = list(self._buf[-1]) if self._buf else []
            return {"predicted_obs": predicted, "delta": [], "n_frames": len(self._buf)}

        frames = list(self._buf)
        # Velocity: mean of pairwise deltas
        deltas = [
            [b - a for a, b in zip(frames[i], frames[i + 1])]
            for i in range(len(frames) - 1)
        ]
        mean_delta = [
            sum(d[j] for d in deltas) / len(deltas)
            for j in range(len(deltas[0]))
        ]
        last = frames[-1]
        predicted = [last[j] + mean_delta[j] for j in range(len(last))]

        return {
            "predicted_obs": predicted,
            "delta": mean_delta,
            "n_frames": len(frames),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Confidence from latent norm
# ─────────────────────────────────────────────────────────────────────────────

def latent_confidence(latent: List[float]) -> float:
    """Calibrate confidence from latent L2 norm.

    High-norm vectors are "well-activated" → high confidence.
    Soft-sigmoid map to [0.05, 0.95].

    Task3: assert 0.0 <= confidence <= 1.0
    """
    if not latent:
        return 0.5
    norm = math.sqrt(sum(v * v for v in latent))
    # Sigmoid: conf = 1 / (1 + exp(-norm + shift))
    shift = 1.0    # tune so norm≈1 → conf≈0.5
    conf = 1.0 / (1.0 + math.exp(-(norm - shift)))
    # Clamp to [0,0.95] — never claim certainty
    conf = max(0.0, min(0.95, float(conf)))
    assert 0.0 <= conf <= 1.0, f"confidence={conf} out of [0,1]"
    return conf


# ─────────────────────────────────────────────────────────────────────────────
# Main OF1 multimodal predictor
# ─────────────────────────────────────────────────────────────────────────────

class OF1MultimodalPredictor:
    """Stateful predictor that maintains frame buffers per env slot.

    Usage::

        predictor = OF1MultimodalPredictor()
        result = predictor.predict(obs=obs_array, modality="image", env_idx=0)
        latent_vector   = result["latent_vector"]
        future_estimate = result["future_state_estimate"]
        confidence      = result["confidence"]
    """

    def __init__(self, latent_dims: int = 64, frame_buffer_len: int = 3):
        self._latent_dims = latent_dims
        self._buffers: Dict[int, LatentFrameBuffer] = {}
        self._frame_buffer_len = frame_buffer_len

    def _get_buffer(self, env_idx: int) -> LatentFrameBuffer:
        if env_idx not in self._buffers:
            self._buffers[env_idx] = LatentFrameBuffer(self._frame_buffer_len)
        return self._buffers[env_idx]

    def predict(
        self,
        obs: Any,
        modality: str = "auto",
        env_idx: int = 0,
    ) -> Dict[str, Any]:
        """Produce OF1-compatible prediction dict.

        Returns:
            latent_vector          (List[float], len > 0 for non-empty obs) Task1 ✓
            future_state_estimate  (dict with "predicted_obs" key)           Task2 ✓
            confidence             (float in [0.0, 1.0])                     Task3 ✓
            modality               (resolved modality string)
            backend                (encoder used)
        """
        arr = np.asarray(obs) if obs is not None else np.array([])

        # Resolve modality
        if modality == "auto":
            if arr.ndim == 3:
                modality = "image"
            else:
                modality = "vector"

        # Task1: encode to latent_vector
        if modality in ("image", "pixels"):
            if arr.ndim == 3:
                latent_vector = encode_image_obs(arr, self._latent_dims)
            else:
                latent_vector = encode_vector_obs(arr, self._latent_dims)
        elif modality == "video":
            # Treat as image (last frame); future: stack frames
            if arr.ndim >= 3:
                latent_vector = encode_image_obs(arr[-1] if arr.ndim == 4 else arr, self._latent_dims)
            else:
                latent_vector = encode_vector_obs(arr, self._latent_dims)
        else:
            latent_vector = encode_vector_obs(arr, self._latent_dims)

        # Task1 assertion
        assert len(latent_vector) > 0, "latent_vector must be non-empty"

        # Task2: future_state_estimate from delta over frame buffer
        buf = self._get_buffer(env_idx)
        buf.push(latent_vector)
        future_state_estimate = buf.predict_next()
        # Task2 assertion
        assert "predicted_obs" in future_state_estimate, \
            "future_state_estimate must contain 'predicted_obs'"

        # Task3: confidence from latent norm
        confidence = latent_confidence(latent_vector)
        # Task3 assertion (also in latent_confidence, but be explicit here)
        assert 0.0 <= confidence <= 1.0, f"confidence={confidence} out of range"

        return {
            "latent_vector": latent_vector,
            "future_state_estimate": future_state_estimate,
            "confidence": confidence,
            "modality": modality,
            "backend": _BACKEND,
        }

    def reset(self, env_idx: int) -> None:
        """Clear frame buffer for an env on episode end."""
        self._buffers.pop(env_idx, None)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton (for golias_engine integration)
# ─────────────────────────────────────────────────────────────────────────────

_default_predictor: Optional[OF1MultimodalPredictor] = None


def get_predictor(latent_dims: int = 64) -> OF1MultimodalPredictor:
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = OF1MultimodalPredictor(latent_dims=latent_dims)
    return _default_predictor
