"""Long, finger-heterogeneous C-infinity height field for current E05.

The broad waves preserve a long traversable object, while the short oblique
wave, cross-wave and staggered local bumps make fingers separated by 40--50 mm
see different height changes.  That is intentional: one pad may be compressed
or lifted while its neighbours remain on a different part of the profile.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


X_HALF_M = 0.30
Y_HALF_M = 0.42
N_COL = 301
N_ROW = 421
BASE_DEPTH_M = 0.014

_LONG_WAVE_AMPLITUDE_M = 0.0035
_LONG_WAVE_LENGTH_M = 0.22
_DIAGONAL_WAVE_AMPLITUDE_M = 0.0018
_DIAGONAL_WAVE_LENGTH_M = 0.095
_DIAGONAL_DIRECTION = np.array([0.62, 0.78], dtype=np.float64)
_DIAGONAL_DIRECTION /= np.linalg.norm(_DIAGONAL_DIRECTION)
_CROSS_AMPLITUDE_M = 0.0014
_CROSS_X_LENGTH_M = 0.18
_CROSS_Y_LENGTH_M = 0.26
_FINGER_WAVE_AMPLITUDE_M = 0.0038
_FINGER_WAVE_LENGTH_M = 0.067
_FINGER_WAVE_DIRECTION = np.array([0.96, 0.28], dtype=np.float64)
_FINGER_WAVE_DIRECTION /= np.linalg.norm(_FINGER_WAVE_DIRECTION)
_FINGER_CROSS_AMPLITUDE_M = 0.0028
_FINGER_CROSS_X_LENGTH_M = 0.052
_FINGER_CROSS_Y_LENGTH_M = 0.118
_RIDGE_AMPLITUDE_M = 0.0038
_RIDGE_CENTER_M = 0.20
_RIDGE_WIDTH_M = 0.007
_RIDGE_DIRECTION = np.array([0.38, 0.925], dtype=np.float64)
_RIDGE_DIRECTION /= np.linalg.norm(_RIDGE_DIRECTION)

_GAUSSIANS = (
  # amplitude, x center, y center, x width, y width
  (0.0090, -0.10, -0.25, 0.055, 0.042),
  (-0.0100, 0.08, -0.08, 0.040, 0.030),
  (0.0065, -0.04, 0.10, 0.022, 0.016),
  (-0.0050, 0.13, 0.25, 0.028, 0.018),
  # Staggered finger-scale events.  Adjacent finger tracks encounter
  # different signs/times instead of rising and falling as one rigid row.
  (0.0085, -0.075, -0.290, 0.020, 0.018),
  (-0.0075, -0.028, -0.220, 0.018, 0.016),
  (0.0090, 0.018, -0.140, 0.019, 0.017),
  (-0.0090, 0.092, -0.060, 0.024, 0.019),
  (-0.0070, -0.075, -0.030, 0.020, 0.018),
  (0.0080, -0.028, 0.050, 0.019, 0.018),
  (-0.0080, 0.018, 0.130, 0.019, 0.017),
  (0.0090, 0.092, 0.205, 0.024, 0.020),
)


def _gaussian_terms(
  x_m: NDArray[np.float64],
  y_m: NDArray[np.float64],
  amplitude: float,
  x_center: float,
  y_center: float,
  x_width: float,
  y_width: float,
) -> tuple[
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
]:
  x_normalized = (x_m - x_center) / x_width
  y_normalized = (y_m - y_center) / y_width
  value = amplitude * np.exp(-(x_normalized**2 + y_normalized**2))
  dx = value * (-2.0 * x_normalized / x_width)
  dy = value * (-2.0 * y_normalized / y_width)
  dxx = value * ((4.0 * x_normalized**2 - 2.0) / x_width**2)
  dxy = value * (4.0 * x_normalized * y_normalized / (x_width * y_width))
  dyy = value * ((4.0 * y_normalized**2 - 2.0) / y_width**2)
  return value, dx, dy, dxx, dxy, dyy


def height_full_derivatives(
  x_m: ArrayLike,
  y_m: ArrayLike,
) -> tuple[
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
]:
  """Return height, dx, dy, dxx, dxy, and dyy."""

  x = np.asarray(x_m, dtype=np.float64)
  y = np.asarray(y_m, dtype=np.float64)
  height = np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=np.float64)
  dx = np.zeros_like(height)
  dy = np.zeros_like(height)
  dxx = np.zeros_like(height)
  dxy = np.zeros_like(height)
  dyy = np.zeros_like(height)

  long_wave_number = 2.0 * np.pi / _LONG_WAVE_LENGTH_M
  long_phase = long_wave_number * (y + 0.36)
  long_sine = np.sin(long_phase)
  height += _LONG_WAVE_AMPLITUDE_M * long_sine
  dy += _LONG_WAVE_AMPLITUDE_M * long_wave_number * np.cos(long_phase)
  dyy -= _LONG_WAVE_AMPLITUDE_M * long_wave_number**2 * long_sine

  diagonal_wave_number = 2.0 * np.pi / _DIAGONAL_WAVE_LENGTH_M
  diagonal_coordinate = (
    _DIAGONAL_DIRECTION[0] * x
    + _DIAGONAL_DIRECTION[1] * y
    + 0.27
  )
  diagonal_phase = diagonal_wave_number * diagonal_coordinate
  diagonal_sine = np.sin(diagonal_phase)
  diagonal_cosine = np.cos(diagonal_phase)
  diagonal_first = (
    _DIAGONAL_WAVE_AMPLITUDE_M
    * diagonal_wave_number
    * diagonal_cosine
  )
  diagonal_second = (
    -_DIAGONAL_WAVE_AMPLITUDE_M
    * diagonal_wave_number**2
    * diagonal_sine
  )
  height += _DIAGONAL_WAVE_AMPLITUDE_M * diagonal_sine
  dx += diagonal_first * _DIAGONAL_DIRECTION[0]
  dy += diagonal_first * _DIAGONAL_DIRECTION[1]
  dxx += diagonal_second * _DIAGONAL_DIRECTION[0] ** 2
  dxy += diagonal_second * np.prod(_DIAGONAL_DIRECTION)
  dyy += diagonal_second * _DIAGONAL_DIRECTION[1] ** 2

  cross_x_number = 2.0 * np.pi / _CROSS_X_LENGTH_M
  cross_y_number = 2.0 * np.pi / _CROSS_Y_LENGTH_M
  cross_x_sine = np.sin(cross_x_number * x)
  cross_x_cosine = np.cos(cross_x_number * x)
  cross_y_sine = np.sin(cross_y_number * y)
  cross_y_cosine = np.cos(cross_y_number * y)
  cross = _CROSS_AMPLITUDE_M * cross_x_sine * cross_y_cosine
  height += cross
  dx += _CROSS_AMPLITUDE_M * cross_x_number * cross_x_cosine * cross_y_cosine
  dy -= _CROSS_AMPLITUDE_M * cross_y_number * cross_x_sine * cross_y_sine
  dxx -= _CROSS_AMPLITUDE_M * cross_x_number**2 * cross_x_sine * cross_y_cosine
  dxy -= (
    _CROSS_AMPLITUDE_M
    * cross_x_number
    * cross_y_number
    * cross_x_cosine
    * cross_y_sine
  )
  dyy -= _CROSS_AMPLITUDE_M * cross_y_number**2 * cross_x_sine * cross_y_cosine

  finger_wave_number = 2.0 * np.pi / _FINGER_WAVE_LENGTH_M
  finger_coordinate = (
    _FINGER_WAVE_DIRECTION[0] * x
    + _FINGER_WAVE_DIRECTION[1] * y
    + 0.187
  )
  finger_phase = finger_wave_number * finger_coordinate
  finger_sine = np.sin(finger_phase)
  finger_cosine = np.cos(finger_phase)
  finger_first = _FINGER_WAVE_AMPLITUDE_M * finger_wave_number * finger_cosine
  finger_second = (
    -_FINGER_WAVE_AMPLITUDE_M * finger_wave_number**2 * finger_sine
  )
  height += _FINGER_WAVE_AMPLITUDE_M * finger_sine
  dx += finger_first * _FINGER_WAVE_DIRECTION[0]
  dy += finger_first * _FINGER_WAVE_DIRECTION[1]
  dxx += finger_second * _FINGER_WAVE_DIRECTION[0] ** 2
  dxy += finger_second * np.prod(_FINGER_WAVE_DIRECTION)
  dyy += finger_second * _FINGER_WAVE_DIRECTION[1] ** 2

  finger_x_number = 2.0 * np.pi / _FINGER_CROSS_X_LENGTH_M
  finger_y_number = 2.0 * np.pi / _FINGER_CROSS_Y_LENGTH_M
  finger_x_sine = np.sin(finger_x_number * (x + 0.011))
  finger_x_cosine = np.cos(finger_x_number * (x + 0.011))
  finger_y_sine = np.sin(finger_y_number * (y + 0.017))
  finger_y_cosine = np.cos(finger_y_number * (y + 0.017))
  finger_cross = (
    _FINGER_CROSS_AMPLITUDE_M * finger_x_sine * finger_y_cosine
  )
  height += finger_cross
  dx += (
    _FINGER_CROSS_AMPLITUDE_M
    * finger_x_number
    * finger_x_cosine
    * finger_y_cosine
  )
  dy -= (
    _FINGER_CROSS_AMPLITUDE_M
    * finger_y_number
    * finger_x_sine
    * finger_y_sine
  )
  dxx -= (
    _FINGER_CROSS_AMPLITUDE_M
    * finger_x_number**2
    * finger_x_sine
    * finger_y_cosine
  )
  dxy -= (
    _FINGER_CROSS_AMPLITUDE_M
    * finger_x_number
    * finger_y_number
    * finger_x_cosine
    * finger_y_sine
  )
  dyy -= (
    _FINGER_CROSS_AMPLITUDE_M
    * finger_y_number**2
    * finger_x_sine
    * finger_y_cosine
  )

  for gaussian in _GAUSSIANS:
    terms = _gaussian_terms(x, y, *gaussian)
    height += terms[0]
    dx += terms[1]
    dy += terms[2]
    dxx += terms[3]
    dxy += terms[4]
    dyy += terms[5]

  ridge_coordinate = (
    _RIDGE_DIRECTION[0] * x
    + _RIDGE_DIRECTION[1] * y
    - _RIDGE_CENTER_M
  )
  ridge_argument = ridge_coordinate / _RIDGE_WIDTH_M
  ridge_tanh = np.tanh(ridge_argument)
  ridge_sech_squared = 1.0 - ridge_tanh**2
  ridge_first = _RIDGE_AMPLITUDE_M / _RIDGE_WIDTH_M * ridge_sech_squared
  ridge_second = (
    -2.0
    * _RIDGE_AMPLITUDE_M
    / _RIDGE_WIDTH_M**2
    * ridge_sech_squared
    * ridge_tanh
  )
  height += _RIDGE_AMPLITUDE_M * ridge_tanh
  dx += ridge_first * _RIDGE_DIRECTION[0]
  dy += ridge_first * _RIDGE_DIRECTION[1]
  dxx += ridge_second * _RIDGE_DIRECTION[0] ** 2
  dxy += ridge_second * np.prod(_RIDGE_DIRECTION)
  dyy += ridge_second * _RIDGE_DIRECTION[1] ** 2
  return height, dx, dy, dxx, dxy, dyy


def height_derivatives(
  x_m: ArrayLike,
  y_m: ArrayLike,
) -> tuple[
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
]:
  """Compatibility view returning height, dx, dy, dxx, and dyy."""

  height, dx, dy, dxx, _, dyy = height_full_derivatives(x_m, y_m)
  return height, dx, dy, dxx, dyy


def maximum_principal_curvature(
  dx: ArrayLike,
  dy: ArrayLike,
  dxx: ArrayLike,
  dxy: ArrayLike,
  dyy: ArrayLike,
) -> NDArray[np.float64]:
  """Return max absolute principal curvature for a graph z=f(x,y)."""

  fx = np.asarray(dx, dtype=np.float64)
  fy = np.asarray(dy, dtype=np.float64)
  fxx = np.asarray(dxx, dtype=np.float64)
  fxy = np.asarray(dxy, dtype=np.float64)
  fyy = np.asarray(dyy, dtype=np.float64)
  first_e = 1.0 + fx**2
  first_f = fx * fy
  first_g = 1.0 + fy**2
  denominator = first_e * first_g - first_f**2
  normal_scale = np.sqrt(1.0 + fx**2 + fy**2)
  second_l = fxx / normal_scale
  second_m = fxy / normal_scale
  second_n = fyy / normal_scale
  mean = (
    first_e * second_n
    - 2.0 * first_f * second_m
    + first_g * second_l
  ) / (2.0 * denominator)
  gaussian = (second_l * second_n - second_m**2) / denominator
  discriminant = np.sqrt(np.maximum(mean**2 - gaussian, 0.0))
  first = mean + discriminant
  second = mean - discriminant
  return np.maximum(np.abs(first), np.abs(second))


def query_surface(
  x_m: float,
  y_m: float,
) -> tuple[float, NDArray[np.float64], float]:
  """Return height, outward normal, and maximum principal curvature."""

  height, dx, dy, dxx, dxy, dyy = height_full_derivatives(float(x_m), float(y_m))
  normal = np.array([-float(dx), -float(dy), 1.0], dtype=np.float64)
  normal /= np.linalg.norm(normal)
  curvature = maximum_principal_curvature(dx, dy, dxx, dxy, dyy)
  return float(height), normal, float(curvature)


def hfield_elevation() -> tuple[NDArray[np.float64], float, float]:
  """Return normalized row-major elevation plus physical min/span."""

  x = np.linspace(-X_HALF_M, X_HALF_M, N_COL)
  y = np.linspace(-Y_HALF_M, Y_HALF_M, N_ROW)
  grid_x, grid_y = np.meshgrid(x, y)
  height = height_full_derivatives(grid_x, grid_y)[0]
  minimum = float(np.min(height))
  span = float(np.max(height) - minimum)
  normalized = (height - minimum) / span
  # MuJoCo hfield row zero is +Y, while meshgrid above starts at -Y.
  return normalized[::-1].astype(np.float64), minimum, span


def profile_characteristics() -> dict[str, float]:
  """Dense deterministic two-dimensional characterization."""

  x = np.linspace(-X_HALF_M, X_HALF_M, 601)
  y = np.linspace(-Y_HALF_M, Y_HALF_M, 841)
  grid_x, grid_y = np.meshgrid(x, y)
  height, dx, dy, dxx, dxy, dyy = height_full_derivatives(grid_x, grid_y)
  slope = np.sqrt(dx**2 + dy**2)
  curvature = maximum_principal_curvature(dx, dy, dxx, dxy, dyy)
  maximum_curvature = float(np.max(curvature))
  return {
    "surface_width_m": 2.0 * X_HALF_M,
    "surface_length_m": 2.0 * Y_HALF_M,
    "height_range_m": float(np.max(height) - np.min(height)),
    "maximum_gradient_norm": float(np.max(slope)),
    "maximum_curvature_inv_m": maximum_curvature,
    "minimum_curvature_radius_m": float(1.0 / maximum_curvature),
  }
