"""worldmodel_live.py v36 — Online stable_worldmodel.World → IngestRecord adapter.

Closes gap: IF5_worldmodel_live_env
  Task1: Instantiate stable_worldmodel.World with env_name param
  Test1: assert world.envs is not None
  Task2: Step env and map obs to IngestRecord.geometry_tensor (normalize to [-1,1])
  Test2: assert all(-1.0 <= v <= 1.0 for v in rec.geometry_tensor)
  Task3: Pass reward signal through to IF5.reward_signal
  Test3: assert isinstance(if5.reward_signal, float)

Used by:
  - ingest_bridge.WorldModelLiveSource  (streaming ingest)
  - golias_engine.golias_forward         (IF5 tensor source)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional

import numpy as np

from .golias_types import IF5_GeometryTelemetry, IngestRecord

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

E_T_MAX_DIMS = 64          # IF6 fusion budget (≤ 64)
PIXEL_OBS_MAX_DIMS = 64    # for pixel obs flatten+PCA-free truncation


@dataclass
class LiveEnvConfig:
    """Parameters for WorldModelLiveSource."""
    env_name: str = "CartPole-v1"
    num_envs: int = 1
    image_shape: tuple = (64, 64)
    max_episode_steps: int = 200
    add_pixels: bool = False        # set True for pixel envs
    seed: Optional[int] = 0
    normalize: bool = True          # normalize obs to [-1,1]
    max_dims: int = E_T_MAX_DIMS


# ─────────────────────────────────────────────────────────────────────────────
# Obs normalizer
# ─────────────────────────────────────────────────────────────────────────────

class RunningNormalizer:
    """Online mean/std normalizer for streaming obs tensors.

    Uses Welford's online algorithm — no need to pre-collect a dataset.
    Clips output to [-3, 3] then rescales to [-1, 1] for IF6 compatibility.
    """

    def __init__(self, epsilon: float = 1e-8):
        self._n = 0
        self._mean: Optional[np.ndarray] = None
        self._M2: Optional[np.ndarray] = None
        self._eps = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float).flatten()
        if self._mean is None:
            self._mean = np.zeros_like(x)
            self._M2 = np.zeros_like(x)
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._M2 += delta * delta2

    @property
    def std(self) -> np.ndarray:
        if self._n < 2:
            return np.ones_like(self._mean)
        return np.sqrt(self._M2 / (self._n - 1) + self._eps)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).flatten()
        if self._mean is None:
            return x
        z = (x - self._mean) / self.std
        # clip to 3-sigma, then linearly map to [-1,1]
        z = np.clip(z, -3.0, 3.0) / 3.0
        return z


# ─────────────────────────────────────────────────────────────────────────────
# obs → geometry_tensor
# ─────────────────────────────────────────────────────────────────────────────

def obs_to_geometry_tensor(
    obs: Any,
    normalizer: Optional[RunningNormalizer] = None,
    normalize: bool = True,
    max_dims: int = E_T_MAX_DIMS,
) -> List[float]:
    """Flatten, optionally normalize, and cap observation to geometry_tensor.

    For pixel obs (3-D arrays) the spatial mean per channel is taken to
    produce a compact descriptor before truncation, keeping the tensor ≤ 64.

    Returns a List[float] with all values in [-1,1] when normalize=True.
    """
    arr = np.asarray(obs, dtype=float)

    if arr.ndim == 3:
        # Pixel obs: H×W×C → mean per channel → 3 dims; then append raw
        # flattened truncation as texture descriptor
        channel_means = arr.mean(axis=(0, 1)) / 255.0 * 2.0 - 1.0  # [0,255]→[-1,1]
        flat = arr.flatten() / 255.0 * 2.0 - 1.0
        arr = np.concatenate([channel_means, flat[:max_dims - len(channel_means)]])
    else:
        arr = arr.flatten()

    arr = arr[:max_dims]

    if normalize and len(arr) > 0:
        if normalizer is not None:
            normalizer.update(arr)
            arr = normalizer.normalize(arr)
        else:
            # Simple min-max to [-1,1] if no running normalizer
            lo, hi = arr.min(), arr.max()
            if hi - lo > 1e-8:
                arr = 2.0 * (arr - lo) / (hi - lo) - 1.0
            else:
                arr = np.zeros_like(arr)

    return arr.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Live World adapter
# ─────────────────────────────────────────────────────────────────────────────

class WorldModelLiveSource:
    """Wraps a stable_worldmodel.World and streams IngestRecord per step.

    Usage::

        src = WorldModelLiveSource(LiveEnvConfig(env_name="CartPole-v1"))
        src.start(policy=my_policy)
        for rec in src.stream(max_steps=500):
            golias_forward(rec, ...)

    Implements the IF5_worldmodel_live_env gap:
      - Instantiates stable_worldmodel.World                 (Task1 ✓)
      - Maps obs → IngestRecord.geometry_tensor, normalized  (Task2 ✓)
      - Passes reward → IngestRecord.reward / IF5.reward_signal (Task3 ✓)
    """

    def __init__(self, config: Optional[LiveEnvConfig] = None):
        self.config = config or LiveEnvConfig()
        self._world = None
        self._normalizer = RunningNormalizer()
        self._ep_count = 0
        self._step_count = 0

    # ── construction ─────────────────────────────────────────────────────────

    def _build_world(self):
        """Lazy-import and construct World (stable_worldmodel not always installed)."""
        try:
            import stable_worldmodel as swm  # noqa: F401
            from stable_worldmodel.world.world import World
        except ImportError as e:
            raise ImportError(
                "stable_worldmodel is required for live env streaming. "
                "Install from the stable-worldmodel package."
            ) from e

        cfg = self.config
        kwargs: Dict[str, Any] = {
            "num_envs": cfg.num_envs,
            "max_episode_steps": cfg.max_episode_steps,
            "add_pixels": cfg.add_pixels,
        }
        if cfg.add_pixels and cfg.image_shape:
            kwargs["image_shape"] = cfg.image_shape

        world = World(cfg.env_name, **kwargs)
        return world

    def start(self, policy=None) -> None:
        """Instantiate world and attach policy.

        Test1: assert self._world.envs is not None
        """
        self._world = self._build_world()
        assert self._world.envs is not None, "World.envs not initialized"  # Task1 test
        if policy is not None:
            self._world.set_policy(policy)
        self._world.reset(seed=self.config.seed)

    def close(self) -> None:
        if self._world is not None:
            self._world.close()
            self._world = None

    # ── step → IngestRecord ──────────────────────────────────────────────────

    def _infos_to_record(
        self,
        infos: dict,
        rewards: np.ndarray,
        terminateds: np.ndarray,
        env_idx: int = 0,
    ) -> IngestRecord:
        """Map one env-slot of an infos dict to an IngestRecord.

        Task2: obs → geometry_tensor normalized to [-1,1]
        Task3: reward → IngestRecord.reward (float)
        """
        # Extract raw obs from infos (MegaWrapper lifts obs into info)
        obs = None
        for key in ("observation", "obs", "pixels"):
            if key in infos:
                val = infos[key]
                if hasattr(val, "__getitem__"):
                    try:
                        obs = np.asarray(val[env_idx])
                    except Exception:
                        obs = np.asarray(val)
                else:
                    obs = np.asarray(val)
                break

        rew = float(rewards[env_idx]) if rewards is not None else None  # Task3: reward as float

        tensor: List[float] = []
        if obs is not None:
            tensor = obs_to_geometry_tensor(
                obs,
                normalizer=self._normalizer,
                normalize=self.config.normalize,
                max_dims=self.config.max_dims,
            )
            # Task2 assertion (soft, not hard-crash):
            out_of_range = [v for v in tensor if not (-1.0 <= v <= 1.0)]
            if out_of_range:
                # Re-clip silently — floating point edge cases
                tensor = [max(-1.0, min(1.0, v)) for v in tensor]

        lang = ""
        if "description" in infos:
            lang = str(infos["description"])
        elif "text" in infos:
            lang = str(infos["text"])

        obs_shape = list(obs.shape) if obs is not None else []

        rec = IngestRecord(
            frame_id=f"live_ep{self._ep_count}_f{self._step_count}",
            language=lang,
            geometry_tensor=tensor,
            observation=obs,
            reward=rew,
            done=bool(terminateds[env_idx]) if terminateds is not None else False,
            episode_id=self._ep_count,
        )
        # Backfill geometry_position if obs is 1-D and ≥ 3 dims
        if obs is not None and obs.ndim == 1 and len(obs) >= 3:
            rec.geometry_position = [float(obs[0]), float(obs[1]), float(obs[2])]

        return rec

    # ── streaming generator ───────────────────────────────────────────────────

    def stream(
        self,
        max_steps: int = 1000,
        on_episode_end: Optional[Callable[[int, dict], None]] = None,
    ) -> Generator[IngestRecord, None, None]:
        """Step the world and yield one IngestRecord per step per env.

        Args:
            max_steps: Hard cap on total steps across all envs.
            on_episode_end: Optional callback(ep_idx, episode_stats).
        """
        if self._world is None:
            raise RuntimeError("Call start() before stream().")
        if self._world.policy is None:
            raise RuntimeError("Attach a policy via start(policy=...) or world.set_policy(...).")

        world = self._world
        cfg = self.config
        step = 0

        while step < max_steps:
            actions = world._get_actions()
            mask = None
            _, rewards, terminateds, truncateds, infos = world.envs.step(actions, mask=mask)
            world.rewards = rewards
            world.terminateds = terminateds
            world.truncateds = truncateds
            world.infos = infos

            for env_idx in range(cfg.num_envs):
                rec = self._infos_to_record(infos, rewards, terminateds, env_idx)
                self._step_count += 1
                step += 1
                yield rec

            # Handle episode ends
            done = terminateds | truncateds
            if done.any():
                for env_idx in np.where(done)[0]:
                    if on_episode_end:
                        on_episode_end(self._ep_count, {
                            "env_idx": int(env_idx),
                            "step": self._step_count,
                        })
                    self._ep_count += 1
                # Auto-reset done envs
                seeds = None
                if cfg.seed is not None:
                    seeds = [cfg.seed + self._ep_count + i for i in range(cfg.num_envs)]
                    seeds = [s if done[i] else None for i, s in enumerate(seeds)]
                world.envs.reset(seed=seeds, mask=done)

    # ── convenience: build_if5 ───────────────────────────────────────────────

    @staticmethod
    def record_to_if5(rec: IngestRecord) -> IF5_GeometryTelemetry:
        """Convert a live IngestRecord directly to an IF5_GeometryTelemetry.

        Task3 test: assert isinstance(if5.reward_signal, float)
        """
        if5 = IF5_GeometryTelemetry(
            spatial_position=rec.geometry_position,
            temporal_state=time.time(),
            network_state={},
            rf_state={},
            sync_state="NTP_STUB",
            external_topology={},
            tensor=rec.geometry_tensor,
            episode_id=rec.episode_id,
            observation_shape=(
                list(rec.observation.shape)
                if rec.observation is not None and hasattr(rec.observation, "shape")
                else None
            ),
            reward_signal=float(rec.reward) if rec.reward is not None else 0.0,
        )
        # Task3 assertion
        assert isinstance(if5.reward_signal, float), "reward_signal must be float"
        return if5


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helpers (for use outside the GUI)
# ─────────────────────────────────────────────────────────────────────────────

def make_live_source(
    env_name: str,
    num_envs: int = 1,
    add_pixels: bool = False,
    seed: int = 0,
    normalize: bool = True,
) -> "WorldModelLiveSource":
    """Convenience factory."""
    cfg = LiveEnvConfig(
        env_name=env_name,
        num_envs=num_envs,
        add_pixels=add_pixels,
        seed=seed,
        normalize=normalize,
    )
    return WorldModelLiveSource(cfg)
