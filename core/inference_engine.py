"""inference_engine.py v35 — Stage 3 adaptive inference.

Wraps Ollama local model calls with:
  - IF6 fused context injection into system prompt
  - Streaming output with token counting
  - Session history for multi-turn
  - Adaptive temperature from TrainingState.theta_1
  - Async streaming for NiceGUI terminal mirror
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Callable, Dict, List, Optional

from .golias_types import OllamaConfig, InferenceRequest, TrainingState


# ─────────────────────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────────────────────

class InferenceSession:
    """Multi-turn inference session backed by Ollama."""

    def __init__(
        self,
        ollama_config: Optional[OllamaConfig] = None,
        training_state: Optional[TrainingState] = None,
        session_id: str = "default",
    ):
        self.config = ollama_config or OllamaConfig()
        self.ts = training_state
        self.session_id = session_id
        self.history: List[Dict[str, str]] = []  # [{"role": ..., "content": ...}]
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_latency_ms = 0.0

    def _build_system_prompt(self, if6_context: Optional[str] = None) -> str:
        parts = [
            "You are Golias v35, an adaptive inference module.",
            "Respond concisely and accurately.",
        ]
        if if6_context:
            parts.append(f"World state context: {if6_context[:400]}")
        if self.ts:
            parts.append(
                f"Current θ* params: temperature={self.ts.theta_1.get('temperature', 1.0):.3f}, "
                f"step={self.ts.step}, perplexity={self.ts.perplexity:.2f}"
            )
        return " ".join(parts)

    def infer(
        self,
        prompt: str,
        if6_context: Optional[str] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        override_config: Optional[OllamaConfig] = None,
    ) -> InferenceRequest:
        """Run a single inference turn."""
        cfg = override_config or self.config

        # Adaptive temperature from training state
        if self.ts and not override_config:
            cfg = OllamaConfig(
                model=cfg.model,
                host=cfg.host,
                temperature=self.ts.theta_1.get("temperature", cfg.temperature),
                top_p=cfg.top_p,
                max_tokens=cfg.max_tokens,
                stream=cfg.stream,
                timeout_ms=cfg.timeout_ms,
            )

        system = self._build_system_prompt(if6_context)

        # Build messages payload (Ollama /api/chat)
        messages = [{"role": "system", "content": system}]
        messages.extend(self.history[-6:])  # last 3 turns
        messages.append({"role": "user", "content": prompt})

        req = InferenceRequest(
            prompt=prompt,
            system_prompt=system,
            ollama_config=cfg,
            if_context=if6_context,
            session_id=self.session_id,
        )

        t0 = time.time()
        try:
            payload = json.dumps({
                "model": cfg.model,
                "messages": messages,
                "stream": cfg.stream,
                "options": {
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                    "num_predict": cfg.max_tokens,
                },
            }).encode()

            r = urllib.request.Request(
                f"{cfg.host}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            collected = []
            tokens_out = 0

            with urllib.request.urlopen(r, timeout=cfg.timeout_ms / 1000.0) as resp:
                for raw in resp:
                    line = raw.decode().strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = obj.get("message", {}).get("content", "")
                    if chunk:
                        collected.append(chunk)
                        tokens_out += 1
                        if stream_callback:
                            stream_callback(chunk)
                    if obj.get("done"):
                        break

            req.response = "".join(collected)
            req.tokens_in = sum(len(m["content"].split()) for m in messages)
            req.tokens_out = tokens_out
            req.latency_ms = (time.time() - t0) * 1000.0
            req.done = True

        except Exception as e:
            req.response = f"[OLLAMA_ERROR] {str(e)[:120]}"
            req.latency_ms = (time.time() - t0) * 1000.0
            req.error = str(e)
            req.done = True
            if stream_callback:
                stream_callback(req.response)

        # Update session history
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": req.response})
        self.total_tokens_in += req.tokens_in
        self.total_tokens_out += req.tokens_out
        self.total_latency_ms += req.latency_ms

        return req

    def clear_history(self):
        self.history.clear()
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_latency_ms = 0.0

    def stats(self) -> dict:
        return {
            "session_id": self.session_id,
            "turns": len(self.history) // 2,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.total_latency_ms / max(len(self.history) // 2, 1),
            "model": self.config.model,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quick single-shot inference (no session state)
# ─────────────────────────────────────────────────────────────────────────────

def infer_once(
    prompt: str,
    config: Optional[OllamaConfig] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> InferenceRequest:
    sess = InferenceSession(ollama_config=config)
    return sess.infer(prompt, stream_callback=stream_callback)
