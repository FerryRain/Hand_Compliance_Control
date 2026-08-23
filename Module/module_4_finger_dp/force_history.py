"""Causal anti-aliased fingertip-force history for Finger DP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, sosfilt, sosfilt_zi

from Module.module_4_finger_dp.contracts import FORCE_HISTORY_STEPS, NUM_FINGERS


def _forces(value: ArrayLike, name: str) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != (NUM_FINGERS,) or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape ({NUM_FINGERS},)")
  if np.any(result < 0.0):
    raise ValueError(f"{name} must contain non-negative magnitudes")
  return np.array(result, dtype=np.float64, copy=True)


def _mask(value: ArrayLike, name: str) -> NDArray[np.bool_]:
  result = np.asarray(value, dtype=np.bool_)
  if result.shape != (NUM_FINGERS,):
    raise ValueError(f"{name} must have shape ({NUM_FINGERS},)")
  return np.array(result, dtype=np.bool_, copy=True)


@dataclass(frozen=True, slots=True)
class ForceHistoryConfig:
  raw_rate_hz: int = 500
  history_rate_hz: int = 100
  history_steps: int = FORCE_HISTORY_STEPS
  lowpass_cutoff_hz: float = 20.0
  butterworth_order: int = 4
  normalization_force_n: float = 2.0
  force_clip_n: float = 20.0

  def __post_init__(self) -> None:
    if self.raw_rate_hz <= 0 or self.history_rate_hz <= 0:
      raise ValueError("force sampling rates must be positive")
    if self.raw_rate_hz % self.history_rate_hz != 0:
      raise ValueError("raw_rate_hz must be an integer multiple of history_rate_hz")
    if self.history_steps < 2:
      raise ValueError("history_steps must be at least two")
    if self.butterworth_order < 2:
      raise ValueError("butterworth_order must be at least two")
    if not 0.0 < self.lowpass_cutoff_hz < 0.5 * self.history_rate_hz:
      raise ValueError("lowpass cutoff must be below the decimated Nyquist frequency")
    if self.normalization_force_n <= 0.0 or self.force_clip_n <= 0.0:
      raise ValueError("force normalization and clip values must be positive")

  @property
  def decimation(self) -> int:
    return self.raw_rate_hz // self.history_rate_hz


@dataclass(frozen=True, slots=True)
class ForceHistoryWindow:
  filtered_force_n: NDArray[np.float64]
  normalized_force: NDArray[np.float64]
  contact_history: NDArray[np.bool_]
  force_valid_history: NDArray[np.bool_]
  sample_dt_s: float

  def encoder_input(self) -> NDArray[np.float32]:
    return np.stack(
      (
        self.normalized_force,
        self.contact_history.astype(np.float64),
        self.force_valid_history.astype(np.float64),
      ),
      axis=-1,
    ).astype(np.float32)


class CausalForcePreprocessor:
  """Filter 500 Hz force causally and emit a 100 Hz, 200 ms history.

  The hard guard must consume the raw sample before this class.  Invalid force
  samples hold the filter input and are explicitly marked invalid instead of
  being aliased with a real zero-newton measurement.
  """

  def __init__(self, config: ForceHistoryConfig | None = None) -> None:
    self.config = config or ForceHistoryConfig()
    self._sos = butter(
      self.config.butterworth_order,
      self.config.lowpass_cutoff_hz,
      btype="lowpass",
      fs=self.config.raw_rate_hz,
      output="sos",
    )
    self.reset()

  def reset(self) -> None:
    self._zi: NDArray[np.float64] | None = None
    self._last_filter_input = np.zeros(NUM_FINGERS, dtype=np.float64)
    self._raw_count = 0
    self._history_count = 0
    shape = (NUM_FINGERS, self.config.history_steps)
    self._filtered_history = np.zeros(shape, dtype=np.float64)
    self._normalized_history = np.zeros(shape, dtype=np.float64)
    self._contact_history = np.zeros(shape, dtype=np.bool_)
    self._valid_history = np.zeros(shape, dtype=np.bool_)

  @property
  def ready(self) -> bool:
    return self._history_count >= self.config.history_steps

  @property
  def collected_history_steps(self) -> int:
    return min(self._history_count, self.config.history_steps)

  @property
  def latest_filtered_force_n(self) -> NDArray[np.float64]:
    result = np.array(self._filtered_history[:, -1], copy=True)
    result.setflags(write=False)
    return result

  def push(
    self,
    raw_normal_force_n: ArrayLike,
    contact_mask: ArrayLike,
    force_valid_mask: ArrayLike,
  ) -> bool:
    """Consume one raw tick; return true when a 100 Hz sample was emitted."""

    raw = _forces(raw_normal_force_n, "raw_normal_force_n")
    contact = _mask(contact_mask, "contact_mask")
    valid = _mask(force_valid_mask, "force_valid_mask")
    filter_input = np.where(valid, raw, self._last_filter_input)
    self._last_filter_input[:] = filter_input

    if self._zi is None:
      base = sosfilt_zi(self._sos)
      self._zi = base[:, :, None] * filter_input[None, None, :]
    filtered_batch, self._zi = sosfilt(
      self._sos,
      filter_input[None, :],
      axis=0,
      zi=self._zi,
    )
    filtered = np.maximum(filtered_batch[-1], 0.0)
    self._raw_count += 1
    if self._raw_count % self.config.decimation != 0:
      return False

    self._filtered_history[:, :-1] = self._filtered_history[:, 1:]
    self._normalized_history[:, :-1] = self._normalized_history[:, 1:]
    self._contact_history[:, :-1] = self._contact_history[:, 1:]
    self._valid_history[:, :-1] = self._valid_history[:, 1:]
    clipped = np.clip(filtered, 0.0, self.config.force_clip_n)
    normalized = np.arcsinh(clipped / self.config.normalization_force_n)
    self._filtered_history[:, -1] = filtered
    self._normalized_history[:, -1] = normalized
    self._contact_history[:, -1] = contact
    self._valid_history[:, -1] = valid
    self._history_count += 1
    return True

  def window(self) -> ForceHistoryWindow:
    if not self.ready:
      raise RuntimeError(
        f"force history is not ready: {self.collected_history_steps}/"
        f"{self.config.history_steps} samples"
      )
    arrays = (
      self._filtered_history.copy(),
      self._normalized_history.copy(),
      self._contact_history.copy(),
      self._valid_history.copy(),
    )
    for value in arrays:
      value.setflags(write=False)
    return ForceHistoryWindow(
      filtered_force_n=arrays[0],
      normalized_force=arrays[1],
      contact_history=arrays[2],
      force_valid_history=arrays[3],
      sample_dt_s=1.0 / self.config.history_rate_hz,
    )
