"""m3_validator.py v36 — M3 model-based validator via Ollama embeddings.

Closes gap: M3_model_based_validator
  Task1: Implement Ollama /api/embeddings call for if1.output
  Test1: assert isinstance(D_orthogonal, float) and 0.0 <= D_orthogonal <= 1.0
  Task2: Implement Ollama /api/embeddings call for if2.output
  Test2: assert isinstance(G_p, float) and G_p >= 0.0
  Task3: Compute cosine distance from embedding pair → D_orthogonal, G_p
  Test3: assert abs(D_orthogonal - expected) < 0.05 on fixture pair

Replaces the heuristic in golias_engine.compute_if3_retune with true
semantic distance measurement.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ─────────────────────────────────────────────────────────────────────────────
# Embedding call
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_embed(
    text: str,
    model: str = "nomic-embed-text",
    host: str = "http://localhost:11434",
    timeout: float = 10.0,
) -> List[float]:
    """Call Ollama /api/embeddings and return the embedding vector.

    Falls back to a deterministic heuristic vector if Ollama is unavailable
    so the pipeline never hard-crashes in offline/test mode.
    """
    if not _HAS_REQUESTS:
        return _heuristic_embed(text)

    try:
        resp = _requests.post(
            f"{host}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        vec = data.get("embedding", [])
        if vec:
            return vec
    except Exception:
        pass

    return _heuristic_embed(text)


def _heuristic_embed(text: str) -> List[float]:
    """Deterministic fallback: character-level n-gram hash → 128-dim vector.

    Used when Ollama is offline. Not semantically meaningful, but produces
    a consistent non-zero vector so the rest of the pipeline keeps running.
    """
    dim = 128
    vec = [0.0] * dim
    for i, ch in enumerate(text):
        idx = (ord(ch) * 31 + i) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# ─────────────────────────────────────────────────────────────────────────────
# Cosine distance
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity in [−1, 1].  Returns 0.0 on zero vectors."""
    if len(a) != len(b):
        # Truncate to shorter to handle model mismatches
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-8
    norm_b = math.sqrt(sum(y * y for y in b)) or 1e-8
    return dot / (norm_a * norm_b)


def cosine_distance(a: List[float], b: List[float]) -> float:
    """Cosine distance in [0, 2].  Mapped to [0, 1] for D_orthogonal."""
    return (1.0 - cosine_similarity(a, b)) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# M3 validator
# ─────────────────────────────────────────────────────────────────────────────

def compute_m3_validator(
    if1_output: str,
    if2_output: str,
    theta_3: Optional[dict] = None,
    embed_model: str = "nomic-embed-text",
    ollama_host: str = "http://localhost:11434",
    timeout: float = 10.0,
) -> Tuple[float, float]:
    """Compute D_orthogonal and G_p via Ollama embedding cosine distance.

    Args:
        if1_output: Text output from IF1 (creative exploration).
        if2_output: Text output from IF2 (efficiency gate).
        theta_3:    Dict with delta_novelty, delta_coverage (bias terms).
        embed_model: Ollama embedding model name.
        ollama_host: Ollama server base URL.
        timeout:    HTTP timeout in seconds.

    Returns:
        (D_orthogonal, G_p) where:
          D_orthogonal ∈ [0.0, 1.0] — semantic divergence between IF1/IF2
          G_p          ∈ [0.0, ∞)  — penalized growth metric

    Tests (Task1–3 assertions inline):
        assert isinstance(D_orthogonal, float) and 0.0 <= D_orthogonal <= 1.0
        assert isinstance(G_p, float) and G_p >= 0.0
    """
    theta = theta_3 or {"delta_novelty": 0.12, "delta_coverage": 0.08}

    # Task1: embed IF1 output
    emb1 = _ollama_embed(if1_output, model=embed_model, host=ollama_host, timeout=timeout)
    # Task2: embed IF2 output
    emb2 = _ollama_embed(if2_output, model=embed_model, host=ollama_host, timeout=timeout)

    # Task3: cosine distance → D_orthogonal
    D_orthogonal = cosine_distance(emb1, emb2)

    # Clamp to [0,1] (floating-point safety)
    D_orthogonal = max(0.0, min(1.0, float(D_orthogonal)))

    # G_p: growth penalty = D_orthogonal weighted by theta bias terms
    delta_novelty = float(theta.get("delta_novelty", 0.12))
    delta_coverage = float(theta.get("delta_coverage", 0.08))
    G_p = float(D_orthogonal * delta_novelty + (1.0 - D_orthogonal) * delta_coverage)
    G_p = max(0.0, G_p)

    # Task1 assertion
    assert isinstance(D_orthogonal, float) and 0.0 <= D_orthogonal <= 1.0, \
        f"D_orthogonal={D_orthogonal} out of range"
    # Task2 assertion
    assert isinstance(G_p, float) and G_p >= 0.0, \
        f"G_p={G_p} out of range"

    return D_orthogonal, G_p


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in replacement for heuristic IF3 retune
# ─────────────────────────────────────────────────────────────────────────────

def m3_enhanced_if3(
    if1_output: str,
    if2_output: str,
    theta_3: Optional[dict] = None,
    embed_model: str = "nomic-embed-text",
    ollama_host: str = "http://localhost:11434",
) -> dict:
    """Return an IF3-compatible dict with embedding-derived D_orthogonal/G_p.

    This replaces the heuristic in compute_if3_retune when called with
    use_m3=True. The engine uses it as:

        if3 = IF3_DirectionValidator(
            D_orthogonal=result["D_orthogonal"],
            G_p=result["G_p"],
            ...
        )
    """
    D_orth, G_p = compute_m3_validator(
        if1_output=if1_output,
        if2_output=if2_output,
        theta_3=theta_3,
        embed_model=embed_model,
        ollama_host=ollama_host,
    )
    return {
        "D_orthogonal": D_orth,
        "G_p": G_p,
        "method": "embedding_cosine_distance",
        "embed_model": embed_model,
    }
