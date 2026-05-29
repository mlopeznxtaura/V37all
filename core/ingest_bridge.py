"""ingest_bridge.py v36 — Bridges stable-worldmodel Dataset/ReplayBuffer to Golias IF5 E_t.

v36 additions:
  - WorldModelLiveSource re-exported for convenience
  - parse_worldmodel_jsonl: tensor cap raised from 16 → 64, normalized

Maps:
  stable_worldmodel.data.dataset.Dataset  →  List[IngestRecord]
  JSONL (Golias IF schema)                →  List[IngestRecord]
  CSV / tabular                           →  List[IngestRecord]
  Raw text lines                          →  List[IngestRecord]

The ingest pipeline feeds Stage 1 of the GUI (adaptive data intake mirror).
"""

from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional

from .golias_types import IngestRecord


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_jsonl(text: str) -> List[IngestRecord]:
    """Parse Golias IF-schema JSONL."""
    records = []
    for i, line in enumerate(text.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            rec = IngestRecord.from_jsonl_dict(d)
            records.append(rec)
        except (json.JSONDecodeError, Exception) as e:
            raise ValueError(f"Line {i+1}: {e}")
    return records


def parse_worldmodel_jsonl(text: str) -> List[IngestRecord]:
    """Parse stable-worldmodel episode JSONL (one dict per step).

    Expected keys: observation, action, reward, done, [info]
    """
    records = []
    for i, line in enumerate(text.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            rec = IngestRecord.from_worldmodel_episode(d, frame_idx=i)
            records.append(rec)
        except Exception as e:
            raise ValueError(f"Line {i+1}: {e}")
    return records


def parse_csv(text: str) -> List[IngestRecord]:
    """Parse CSV with flexible column mapping to IngestRecord."""
    records = []
    reader = csv.DictReader(io.StringIO(text))
    for i, row in enumerate(reader):
        # Try to map common column names
        frame_id = row.get("frame_id", row.get("step", row.get("id", i)))
        lang = row.get("language", row.get("text", row.get("description", row.get("context", ""))))
        try:
            pos_x = float(row.get("x", row.get("pos_x", 0.0)))
            pos_y = float(row.get("y", row.get("pos_y", 0.0)))
            pos_z = float(row.get("z", row.get("pos_z", 0.0)))
        except (ValueError, TypeError):
            pos_x = pos_y = pos_z = 0.0

        try:
            reward = float(row.get("reward", row.get("rew", "")))
        except (ValueError, TypeError):
            reward = None

        tensor_str = row.get("tensor", row.get("e_t", ""))
        try:
            tensor = [float(v) for v in tensor_str.split(",") if v.strip()]
        except Exception:
            tensor = []

        rec = IngestRecord(
            frame_id=frame_id,
            language=str(lang),
            geometry_position=[pos_x, pos_y, pos_z],
            geometry_tensor=tensor,
            reward=reward,
            Y_target=row.get("target", row.get("goal", "")),
        )
        records.append(rec)
    return records


def parse_plain_text(text: str) -> List[IngestRecord]:
    """Parse plain text — one IngestRecord per non-empty line."""
    records = []
    for i, line in enumerate(text.strip().splitlines()):
        line = line.strip()
        if line:
            records.append(IngestRecord(
                frame_id=i,
                language=line,
            ))
    return records


def auto_parse(text: str, hint: str = "auto") -> List[IngestRecord]:
    """Auto-detect format and parse.

    hint: 'jsonl' | 'worldmodel' | 'csv' | 'text' | 'auto'
    """
    text = text.strip()
    if not text:
        return []

    if hint == "worldmodel":
        return parse_worldmodel_jsonl(text)
    if hint == "csv":
        return parse_csv(text)
    if hint == "text":
        return parse_plain_text(text)
    if hint == "jsonl":
        return parse_jsonl(text)

    # Auto-detect
    if text.startswith("{"):
        try:
            first = json.loads(text.splitlines()[0])
            # WorldModel style has observation/action/reward
            if any(k in first for k in ("observation", "action", "reward")):
                return parse_worldmodel_jsonl(text)
            # Golias IF style has language or frame_id
            return parse_jsonl(text)
        except Exception:
            pass

    if "," in text.splitlines()[0]:
        try:
            return parse_csv(text)
        except Exception:
            pass

    return parse_plain_text(text)


# ─────────────────────────────────────────────────────────────────────────────
# Streaming ingest — yields records one at a time for terminal mirror
# ─────────────────────────────────────────────────────────────────────────────

def stream_ingest(
    text: str,
    hint: str = "auto",
    on_record: Optional[Callable[[IngestRecord, int, int], None]] = None,
    delay_ms: float = 0.0,
) -> Generator[IngestRecord, None, None]:
    """Parse and stream records, calling on_record callback per record."""
    records = auto_parse(text, hint)
    total = len(records)
    for i, rec in enumerate(records):
        if on_record:
            on_record(rec, i, total)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        yield rec


# ─────────────────────────────────────────────────────────────────────────────
# Ingest stats
# ─────────────────────────────────────────────────────────────────────────────

def compute_ingest_stats(records: List[IngestRecord]) -> dict:
    """Compute summary statistics over a batch of IngestRecords."""
    if not records:
        return {"n": 0}

    n = len(records)
    rewards = [r.reward for r in records if r.reward is not None]
    tensor_lens = [len(r.geometry_tensor) for r in records]
    lang_lens = [len(r.language) for r in records]
    t_maxes = [r.T_max for r in records]
    modalities = {}
    for r in records:
        modalities[r.modality] = modalities.get(r.modality, 0) + 1

    return {
        "n": n,
        "n_with_reward": len(rewards),
        "avg_reward": sum(rewards) / max(len(rewards), 1),
        "min_reward": min(rewards) if rewards else None,
        "max_reward": max(rewards) if rewards else None,
        "avg_tensor_len": sum(tensor_lens) / n,
        "avg_lang_len": sum(lang_lens) / n,
        "avg_T_max": sum(t_maxes) / n,
        "modalities": modalities,
        "n_with_geometry": sum(1 for r in records if any(v != 0 for v in r.geometry_position)),
        "frame_ids": [str(r.frame_id) for r in records[:10]],
    }


# ─────────────────────────────────────────────────────────────────────────────
# v36: Live source convenience re-export
# ─────────────────────────────────────────────────────────────────────────────

try:
    from .worldmodel_live import (  # noqa: F401
        WorldModelLiveSource,
        LiveEnvConfig,
        make_live_source,
    )
    _LIVE_SOURCE_AVAILABLE = True
except ImportError:
    _LIVE_SOURCE_AVAILABLE = False
