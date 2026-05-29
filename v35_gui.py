"""v35_gui.py — Golias v35 Three-Stage Adaptive NiceGUI Terminal Mirror

Stage 1: Adaptive Data Intake / Ingest Mirror
  - Parse JSONL / WorldModel / CSV / plain text → IngestRecord stream
  - Live terminal mirror of each record as it's parsed
  - Ingest stats panel (reward stats, tensor coverage, modality distribution)
  - Format auto-detection with manual override

Stage 2: Adaptive Training Terminal Mirror
  - θ* outer optimization loop (was agent gap in v34)
  - Flash mini-corpus cycling with configurable batch size
  - Per-step terminal echo: loss, temp, epsilon, perplexity
  - EWC λ display + adaptive lambda weight live update
  - Start / Stop / Resume controls

Stage 3: Adaptive Inference Terminal Mirror
  - Ollama local model chat with IF6 context injection
  - Streaming output terminal mirror
  - Model selector (auto-discovered from Ollama)
  - Session history + stats
  - Temperature adapts from TrainingState.theta_1

Run:
    pip install nicegui
    python v35_gui.py
    → http://localhost:8766
"""

import asyncio
import json
import sys
import time
import threading
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from nicegui import ui, app

from core.golias_types import (
    OllamaConfig, TrainingState, IngestRecord,
)
from core.ingest_bridge import auto_parse, compute_ingest_stats, stream_ingest
from core.golias_engine import golias_sequence, ollama_list_models_sync
from core.training_loop import run_training, TrainingConfig
from core.inference_engine import InferenceSession


# ─────────────────────────────────────────────────────────────────────────────
# Shared app state
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.ingest_records: list[IngestRecord] = []
        self.training_state: TrainingState = TrainingState()
        self.ollama_config: OllamaConfig = OllamaConfig()
        self.inference_session: Optional[InferenceSession] = None
        self._training_stop = threading.Event()
        self._training_thread: Optional[threading.Thread] = None
        self.available_models: list[str] = []

    def stop_training(self):
        self._training_stop.set()

    def is_training_running(self) -> bool:
        return self._training_thread is not None and self._training_thread.is_alive()

    def refresh_models(self):
        self.available_models = ollama_list_models_sync(self.ollama_config.host)


STATE = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# Shared CSS + helpers
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; }
  body { background: #020817; margin: 0; font-family: 'IBM Plex Sans', sans-serif; color: #e2e8f0; }
  .q-page { background: #020817 !important; }

  .stage-panel {
    background: #0a0f1e;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 0;
    height: calc(100vh - 130px);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .stage-header {
    padding: 10px 14px 8px;
    border-bottom: 1px solid #1e293b;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }
  .stage-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .15em;
    text-transform: uppercase;
  }
  .stage-controls {
    padding: 8px 12px;
    border-bottom: 1px solid #0f172a;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .terminal {
    flex: 1;
    background: #020817;
    border-radius: 0 0 10px 10px;
    padding: 10px 12px;
    overflow-y: auto;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    line-height: 1.6;
    color: #94a3b8;
  }
  .term-line { margin: 0; padding: 1px 0; }
  .term-ok { color: #4ade80; }
  .term-warn { color: #f59e0b; }
  .term-err { color: #f87171; }
  .term-info { color: #60a5fa; }
  .term-dim { color: #334155; }
  .term-purple { color: #a78bfa; }
  .term-teal { color: #2dd4bf; }
  .term-prompt { color: #475569; }
  .status-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 600;
  }
  .metric-row {
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    border-bottom: 1px solid #0f172a;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
  }
  .metric-key { color: #475569; }
  .metric-val { color: #e2e8f0; font-weight: 600; }
  .ctrl-row { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .nicegui-content { padding: 0 !important; }
  textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    background: #0f172a !important;
    color: #e2e8f0 !important;
  }
  .separator { height: 1px; background: #1e293b; margin: 4px 0; }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #0a0f1e; }
  ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }
</style>
"""

DEFAULT_JSONL = """\
{"frame_id":6,"language":"Static cover. Agents hold.","geometry":{"position":[71.45,14.10,0.0],"tensor":[71.45,14.10,0.0,89.50,10.45,1.0]},"substrate":{"hardware_state":240,"runtime_constraints":15,"memory_mb":4096,"exec_perms":255,"system_id":"golias_v35","deploy_boundary":"sandbox","scalar":171},"T_max":0.9,"tau_ms":150.0,"epsilon":0.65,"compute_cost":0.08,"latency_ms":38.0,"theta_3":{"delta_novelty":0.05,"delta_coverage":0.03},"next_frame_prediction_tokens":["IF1_EXP_X71.45_Y14.10","IF7_STABLE"],"latent_vector":[0.12,0.05,0.038],"confidence":0.91,"Y_target":"static coverage maintain","modality":"text"}
{"frame_id":11,"language":"Sudden acoustic report. React immediately.","geometry":{"position":[71.65,14.20,0.4],"tensor":[71.65,14.20,0.4,94.45,9.20,0.0]},"substrate":{"hardware_state":240,"runtime_constraints":15,"memory_mb":3800,"exec_perms":255,"system_id":"golias_v35","deploy_boundary":"sandbox","scalar":188},"T_max":1.4,"tau_ms":80.0,"epsilon":0.5,"compute_cost":0.22,"latency_ms":71.0,"theta_3":{"delta_novelty":0.22,"delta_coverage":0.14},"next_frame_prediction_tokens":["IF1_REACT_AUDIO_X72.05_Y14.40","IF3_DIVERSIFY","IF7_SPIKE"],"latent_vector":[0.68,0.22,0.071],"confidence":0.61,"Y_target":"react acoustic evasion","modality":"text"}
{"frame_id":22,"language":"Converge. Both agents close to objective.","geometry":{"position":[68.85,12.75,1.0],"tensor":[68.85,12.75,1.0,90.75,10.75,-1.0]},"substrate":{"hardware_state":240,"runtime_constraints":15,"memory_mb":4096,"exec_perms":255,"system_id":"golias_v35","deploy_boundary":"sandbox","scalar":163},"T_max":1.1,"tau_ms":200.0,"epsilon":0.6,"compute_cost":0.15,"latency_ms":52.0,"theta_3":{"delta_novelty":0.13,"delta_coverage":0.09},"next_frame_prediction_tokens":["IF1_CONVERGE_X69.85_Y13.10","IF6_FUSE_CONVERGENCE","IF7_ARBITRATE"],"latent_vector":[0.42,0.13,0.052],"confidence":0.77,"Y_target":"convergence objective reached","modality":"text"}
{"frame_id":96,"language":"Stable. Mission complete. All invariants hold.","geometry":{"position":[78.25,11.85,0.0],"tensor":[78.25,11.85,0.0,90.45,13.50,0.0]},"substrate":{"hardware_state":240,"runtime_constraints":15,"memory_mb":4096,"exec_perms":255,"system_id":"golias_v35","deploy_boundary":"sandbox","scalar":128},"T_max":0.7,"tau_ms":300.0,"epsilon":0.75,"compute_cost":0.04,"latency_ms":22.0,"theta_3":{"delta_novelty":0.02,"delta_coverage":0.01},"next_frame_prediction_tokens":["IF1_BASE_X78.25_Y11.85","IF7_INVARIANT_HOLD"],"latent_vector":[0.02,0.02,0.022],"confidence":0.98,"Y_target":"stable base state","modality":"text"}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _term_line(cls: str, text: str) -> str:
    escaped = str(text).replace("<", "&lt;").replace(">", "&gt;")
    return f'<p class="term-line {cls}">{escaped}</p>'


def _prompt(tag: str) -> str:
    return f'<span class="term-prompt">{tag}»</span> '


def _kv(key: str, val) -> str:
    if isinstance(val, float):
        val_s = f"{val:.5f}"
    else:
        val_s = str(val)
    return (
        f'<div class="metric-row">'
        f'<span class="metric-key">{key}</span>'
        f'<span class="metric-val">{val_s}</span>'
        f'</div>'
    )


def _badge(text: str, color: str = "#1e3a5f") -> str:
    return (
        f'<span class="status-pill" style="background:{color}">{text}</span>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/")
def index():
    ui.add_head_html(GLOBAL_CSS)

    # ── top bar ───────────────────────────────────────────────────────────────
    with ui.row().style(
        "width:100%;background:#0a0f1e;border-bottom:1px solid #1e293b;"
        "padding:10px 16px;align-items:center;gap:12px;height:52px"
    ):
        ui.html(
            '<span style="font-family:IBM Plex Mono,monospace;font-size:13px;'
            'font-weight:700;color:#e2e8f0;letter-spacing:.08em">GOLIAS V35</span>'
            '<span style="font-family:IBM Plex Mono,monospace;font-size:11px;'
            'color:#334155"> / ADAPTIVE PIPELINE</span>'
        )
        ui.space()
        ui.html(
            '<span style="font-family:IBM Plex Mono,monospace;font-size:10px;'
            'color:#334155">STAGE-1:INGEST · STAGE-2:TRAIN · STAGE-3:INFER</span>'
        )

    # ── shared JSONL input strip ───────────────────────────────────────────────
    with ui.row().style("width:100%;padding:8px 14px 0;gap:6px;align-items:flex-end"):
        shared_input = ui.textarea(
            label="SHARED CORPUS INPUT (JSONL / WorldModel / CSV / text)",
            value=DEFAULT_JSONL,
        ).style(
            "width:100%;font-family:IBM Plex Mono,monospace;font-size:11px"
        ).props("outlined dense dark rows=3")

        fmt_select = ui.select(
            options=["auto", "jsonl", "worldmodel", "csv", "text"],
            value="auto",
            label="fmt",
        ).props("outlined dense dark").style("width:110px")

    # ── three stages ──────────────────────────────────────────────────────────
    with ui.row().style("width:100%;padding:8px 14px 14px;gap:12px;flex:1"):

        # ══════════════════════════════════════════════════════════════════════
        # STAGE 1 — INGEST MIRROR
        # ══════════════════════════════════════════════════════════════════════
        with ui.column().classes("stage-panel").style("flex:1;min-width:0"):
            with ui.row().classes("stage-header"):
                ui.html(
                    '<span class="stage-title" style="color:#2dd4bf">■ STAGE 1</span>'
                    '<span style="color:#334155;font-family:IBM Plex Mono,monospace;'
                    'font-size:10px"> ADAPTIVE INGEST MIRROR</span>'
                )

            # Controls
            with ui.column().classes("stage-controls"):
                with ui.row().classes("ctrl-row"):
                    s1_run_btn = ui.button("INGEST", icon="input").props("color=teal-8 dense")
                    s1_clear_btn = ui.button("CLR", icon="clear").props("color=grey-9 dense flat")
                    s1_status = ui.html('<span class="status-pill" style="background:#1e293b;color:#64748b">idle</span>')

                # Stats mini-panel
                s1_stats = ui.html("")

            # Terminal
            s1_term = ui.html("").style(
                "flex:1;background:#020817;border-radius:0 0 10px 10px;"
                "padding:10px 12px;overflow-y:auto;"
                "font-family:'IBM Plex Mono',monospace;font-size:11px;line-height:1.6"
            )
            s1_term.classes("terminal")

            async def run_ingest():
                s1_run_btn.disable()
                s1_status.set_content(
                    '<span class="status-pill" style="background:#854d0e;color:#fef08a">⏳ parsing</span>'
                )
                s1_stats.set_content("")
                lines = []

                try:
                    text = shared_input.value.strip()
                    if not text:
                        raise ValueError("No input")

                    fmt = fmt_select.value

                    lines.append(_term_line("term-teal", f"[INGEST] format={fmt} bytes={len(text)}"))
                    s1_term.set_content("".join(lines))
                    await asyncio.sleep(0.01)

                    records = auto_parse(text, fmt)
                    STATE.ingest_records = records

                    lines.append(_term_line("term-dim", "─" * 50))

                    for i, rec in enumerate(records):
                        # Mirror each record
                        lines.append(_term_line(
                            "term-info",
                            f"[REC {i+1}/{len(records)}] id={rec.frame_id} "
                            f"lang={rec.language[:50]!r}"
                        ))
                        if rec.geometry_position and any(v != 0 for v in rec.geometry_position):
                            lines.append(_term_line(
                                "term-dim",
                                f"  geo=[{','.join(f'{v:.2f}' for v in rec.geometry_position[:3])}] "
                                f"tensor_len={len(rec.geometry_tensor)}"
                            ))
                        if rec.reward is not None:
                            col = "term-ok" if rec.reward >= 0 else "term-err"
                            lines.append(_term_line(col, f"  reward={rec.reward:.4f}"))
                        lines.append(_term_line(
                            "term-dim",
                            f"  T_max={rec.T_max} tau_ms={rec.tau_ms} mod={rec.modality}"
                        ))
                        s1_term.set_content("".join(lines))
                        await asyncio.sleep(0.02)

                    # Stats
                    stats = compute_ingest_stats(records)
                    stats_html = (
                        _kv("n_records", stats["n"]) +
                        _kv("n_with_reward", stats["n_with_reward"]) +
                        _kv("avg_reward", f"{stats['avg_reward']:.4f}" if stats.get("avg_reward") else "—") +
                        _kv("avg_tensor_len", f"{stats['avg_tensor_len']:.1f}") +
                        _kv("avg_T_max", f"{stats['avg_T_max']:.3f}") +
                        _kv("modalities", str(stats.get("modalities", {})))
                    )
                    s1_stats.set_content(stats_html)

                    lines.append(_term_line("term-dim", "─" * 50))
                    lines.append(_term_line("term-ok",
                        f"[INGEST DONE] {len(records)} records loaded → STATE.ingest_records"))
                    s1_term.set_content("".join(lines))
                    s1_status.set_content(
                        f'<span class="status-pill" style="background:#14532d;color:#4ade80">'
                        f'✓ {len(records)} records</span>'
                    )

                except Exception as e:
                    import traceback
                    lines.append(_term_line("term-err", f"[ERROR] {e}"))
                    lines.append(_term_line("term-err", traceback.format_exc()[:400]))
                    s1_term.set_content("".join(lines))
                    s1_status.set_content(
                        '<span class="status-pill" style="background:#7f1d1d;color:#f87171">✗ error</span>'
                    )
                finally:
                    s1_run_btn.enable()

            def clear_s1():
                s1_term.set_content("")
                s1_stats.set_content("")
                s1_status.set_content(
                    '<span class="status-pill" style="background:#1e293b;color:#64748b">idle</span>'
                )

            s1_run_btn.on_click(run_ingest)
            s1_clear_btn.on_click(clear_s1)


        # ══════════════════════════════════════════════════════════════════════
        # STAGE 2 — TRAINING TERMINAL MIRROR
        # ══════════════════════════════════════════════════════════════════════
        with ui.column().classes("stage-panel").style("flex:1;min-width:0"):
            with ui.row().classes("stage-header"):
                ui.html(
                    '<span class="stage-title" style="color:#a78bfa">■ STAGE 2</span>'
                    '<span style="color:#334155;font-family:IBM Plex Mono,monospace;'
                    'font-size:10px"> ADAPTIVE TRAINING MIRROR</span>'
                )

            with ui.column().classes("stage-controls"):
                with ui.row().classes("ctrl-row"):
                    s2_epochs = ui.number("epochs", value=2, min=1, max=20, step=1).props(
                        "outlined dense dark").style("width:80px")
                    s2_batch = ui.number("batch", value=4, min=1, max=32, step=1).props(
                        "outlined dense dark").style("width:70px")
                    s2_lr = ui.number("lr", value=0.05, min=0.001, max=1.0, step=0.005,
                                      format="%.3f").props("outlined dense dark").style("width:80px")
                    s2_ewc = ui.number("EWC λ", value=5000, min=0, max=50000, step=500).props(
                        "outlined dense dark").style("width:90px")

                with ui.row().classes("ctrl-row"):
                    s2_model = ui.input("model", value="llama3.2").props(
                        "outlined dense dark").style("width:130px")
                    s2_host = ui.input("host", value="http://localhost:11434").props(
                        "outlined dense dark").style("width:200px")
                    s2_run_btn = ui.button("TRAIN", icon="model_training").props("color=purple-8 dense")
                    s2_stop_btn = ui.button("STOP", icon="stop").props("color=red-9 dense flat")
                    s2_status = ui.html('<span class="status-pill" style="background:#1e293b;color:#64748b">idle</span>')

                s2_metrics = ui.html("")

            s2_term = ui.html("").classes("terminal")
            _s2_lines: list[str] = []

            def _s2_emit(cls: str, msg: str):
                _s2_lines.append(_term_line(cls, msg))
                s2_term.set_content("".join(_s2_lines[-300:]))

            def _on_step(info: dict):
                _s2_emit("term-purple",
                    f"[STEP] ep={info['epoch']} b={info['batch']}/{info['n_batches']} "
                    f"loss={info['avg_loss']:.5f} temp={info['theta_1_temp']:.4f} "
                    f"eps={info['theta_2_epsilon']:.4f}"
                )
                # Loss breakdown
                frames = info.get("frames", [])
                if frames:
                    fr = frames[-1]
                    _s2_emit("term-dim",
                        f"  coh={fr.get('loss_coherence',0):.4f} "
                        f"stab={fr.get('loss_stability',0):.4f} "
                        f"aln={fr.get('loss_alignment',0):.4f} "
                        f"ppl={info.get('perplexity',{}).get('text',0):.2f}"
                    )
                # Lambda weights
                lw = info.get("lambda_weights", {})
                if lw:
                    lw_str = " ".join(f"λ{i+1}={v:.2f}" for i, v in enumerate(lw.values()))
                    _s2_emit("term-dim", f"  [{lw_str}]")
                # Stream preview
                prev = info.get("stream_preview", "")
                if prev and not prev.startswith("[OLLAMA"):
                    _s2_emit("term-dim", f"  M1: {prev[:100]}")
                # Update metrics panel
                s2_metrics.set_content(
                    _kv("step", info["step"]) +
                    _kv("avg_loss", f"{info['avg_loss']:.5f}") +
                    _kv("temperature", f"{info['theta_1_temp']:.4f}") +
                    _kv("epsilon", f"{info['theta_2_epsilon']:.4f}") +
                    _kv("elapsed_ms", f"{info['elapsed_ms']:.0f}")
                )

            def _on_epoch(info: dict):
                _s2_emit("term-ok",
                    f"[EPOCH {info['epoch']} DONE] avg_loss={info['epoch_avg_loss']:.5f} "
                    f"ppl={info['perplexity']:.2f} frames={info['total_frames']} "
                    f"t={info['epoch_elapsed_s']:.2f}s"
                )

            def _train_thread(records, cfg):
                try:
                    final_ts = run_training(
                        records, cfg,
                        initial_state=STATE.training_state,
                        on_step=_on_step,
                        on_epoch=_on_epoch,
                        stop_flag=lambda: STATE._training_stop.is_set(),
                    )
                    STATE.training_state = final_ts
                except Exception as e:
                    _s2_emit("term-err", f"[TRAIN ERROR] {e}")
                finally:
                    s2_run_btn.enable()
                    s2_status.set_content(
                        f'<span class="status-pill" style="background:#14532d;color:#4ade80">'
                        f'✓ done step={STATE.training_state.step}</span>'
                    )

            async def run_train():
                if STATE.is_training_running():
                    return
                records = STATE.ingest_records
                if not records:
                    # Try parsing inline
                    try:
                        records = auto_parse(shared_input.value, fmt_select.value)
                        STATE.ingest_records = records
                    except Exception as e:
                        _s2_emit("term-err", f"[ERROR] Parse failed: {e}")
                        return

                STATE._training_stop.clear()
                oc = OllamaConfig(
                    model=s2_model.value,
                    host=s2_host.value,
                )
                STATE.ollama_config = oc
                cfg = TrainingConfig(
                    epochs=int(s2_epochs.value),
                    batch_size=int(s2_batch.value),
                    lr=float(s2_lr.value),
                    ewc_lambda=float(s2_ewc.value),
                    ollama_config=oc,
                )
                s2_run_btn.disable()
                s2_status.set_content(
                    '<span class="status-pill" style="background:#4a1d96;color:#c4b5fd">⏳ training</span>'
                )
                _s2_emit("term-purple",
                    f"[TRAIN START] epochs={cfg.epochs} batch={cfg.batch_size} "
                    f"lr={cfg.lr} EWC_λ={cfg.ewc_lambda} model={oc.model} n={len(records)}"
                )
                t = threading.Thread(target=_train_thread, args=(records, cfg), daemon=True)
                STATE._training_thread = t
                t.start()

            def stop_train():
                STATE.stop_training()
                _s2_emit("term-warn", "[TRAIN] stop requested")

            s2_run_btn.on_click(run_train)
            s2_stop_btn.on_click(stop_train)


        # ══════════════════════════════════════════════════════════════════════
        # STAGE 3 — INFERENCE TERMINAL MIRROR
        # ══════════════════════════════════════════════════════════════════════
        with ui.column().classes("stage-panel").style("flex:1;min-width:0"):
            with ui.row().classes("stage-header"):
                ui.html(
                    '<span class="stage-title" style="color:#f59e0b">■ STAGE 3</span>'
                    '<span style="color:#334155;font-family:IBM Plex Mono,monospace;'
                    'font-size:10px"> ADAPTIVE INFERENCE MIRROR</span>'
                )

            with ui.column().classes("stage-controls"):
                with ui.row().classes("ctrl-row"):
                    s3_model = ui.input("model", value="llama3.2").props(
                        "outlined dense dark").style("width:140px")
                    s3_host = ui.input("host", value="http://localhost:11434").props(
                        "outlined dense dark").style("width:200px")
                    s3_temp = ui.number("temp", value=1.0, min=0.0, max=2.0, step=0.05,
                                        format="%.2f").props("outlined dense dark").style("width:75px")
                    s3_refresh_btn = ui.button("", icon="refresh").props("color=grey-9 dense flat")

                with ui.row().classes("ctrl-row"):
                    s3_inject_ctx = ui.checkbox("inject IF6 ctx", value=True).style("font-size:11px")
                    s3_adapt_temp = ui.checkbox("adaptive temp (θ*)", value=True).style("font-size:11px")
                    s3_clear_btn = ui.button("CLR SESS", icon="clear").props("color=grey-9 dense flat")

                with ui.row().classes("ctrl-row"):
                    s3_prompt = ui.input(
                        placeholder="prompt → enter to run…",
                    ).props("outlined dense dark").style("flex:1")
                    s3_run_btn = ui.button("RUN", icon="send").props("color=amber-8 dense")

                s3_stats = ui.html("")

            s3_term = ui.html("").classes("terminal")
            _s3_lines: list[str] = []
            _s3_session: list[InferenceSession] = [None]  # mutable ref

            def _get_session() -> InferenceSession:
                oc = OllamaConfig(
                    model=s3_model.value,
                    host=s3_host.value,
                    temperature=s3_temp.value,
                    stream=True,
                )
                ts = STATE.training_state if s3_adapt_temp.value else None
                if _s3_session[0] is None:
                    _s3_session[0] = InferenceSession(oc, ts)
                else:
                    _s3_session[0].config = oc
                    _s3_session[0].ts = ts
                return _s3_session[0]

            def _s3_emit(cls: str, msg: str):
                _s3_lines.append(_term_line(cls, msg))
                s3_term.set_content("".join(_s3_lines[-400:]))

            def _refresh_models():
                STATE.ollama_config.host = s3_host.value
                STATE.refresh_models()
                models = STATE.available_models
                _s3_emit("term-teal",
                    f"[MODELS] {models if models else 'Ollama not reachable at ' + s3_host.value}"
                )

            async def run_inference():
                prompt = s3_prompt.value.strip()
                if not prompt:
                    return
                s3_run_btn.disable()
                s3_prompt.set_value("")

                sess = _get_session()

                # IF6 context injection
                if6_ctx = None
                if s3_inject_ctx.value and STATE.training_state.loss_history:
                    last_loss = STATE.training_state.loss_history[-1]
                    last_step = STATE.training_state.step
                    last_temp = STATE.training_state.theta_1.get("temperature", 1.0)
                    if6_ctx = (
                        f"θ*_step={last_step} loss={last_loss:.4f} "
                        f"temp={last_temp:.3f} ppl={STATE.training_state.perplexity:.2f}"
                    )

                _s3_emit("term-info", f"[USER] {prompt}")
                if if6_ctx:
                    _s3_emit("term-dim", f"  if6_ctx: {if6_ctx}")

                eff_temp = (
                    STATE.training_state.theta_1.get("temperature", s3_temp.value)
                    if s3_adapt_temp.value and STATE.training_state.step > 0
                    else s3_temp.value
                )
                _s3_emit("term-dim", f"  model={s3_model.value} temp={eff_temp:.3f}")

                # Stream output
                collected = []

                def _cb(chunk: str):
                    collected.append(chunk)

                _s3_emit("term-warn", "[GOLIAS] ")

                try:
                    loop = asyncio.get_event_loop()
                    req = await loop.run_in_executor(
                        None,
                        lambda: sess.infer(prompt, if6_ctx, _cb),
                    )
                    # Show response
                    resp = req.response
                    for para in resp.split("\n"):
                        if para.strip():
                            _s3_emit("term-warn" if not req.error else "term-err", f"  {para}")

                    _s3_emit("term-dim",
                        f"  [{req.tokens_out} tokens | {req.latency_ms:.0f}ms | "
                        f"{'✗ ' + req.error[:40] if req.error else '✓'}]"
                    )

                    # Stats
                    stats = sess.stats()
                    s3_stats.set_content(
                        _kv("turns", stats["turns"]) +
                        _kv("tok_out", stats["total_tokens_out"]) +
                        _kv("tok_in", stats["total_tokens_in"]) +
                        _kv("avg_lat_ms", f"{stats['avg_latency_ms']:.0f}")
                    )

                except Exception as e:
                    _s3_emit("term-err", f"[ERROR] {e}")
                finally:
                    s3_run_btn.enable()

            def clear_session():
                if _s3_session[0]:
                    _s3_session[0].clear_history()
                _s3_lines.clear()
                s3_term.set_content("")
                s3_stats.set_content("")
                _s3_emit("term-dim", "[SESSION CLEARED]")

            s3_run_btn.on_click(run_inference)
            s3_refresh_btn.on_click(_refresh_models)
            s3_clear_btn.on_click(clear_session)
            s3_prompt.on("keydown.enter", run_inference)


# ─────────────────────────────────────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────────────────────────────────────

@app.on_startup
async def startup():
    # Background model discovery
    STATE.refresh_models()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Golias V35 — Adaptive Pipeline",
        dark=True,
        port=8766,
        reload=False,
        favicon="⚡",
    )
