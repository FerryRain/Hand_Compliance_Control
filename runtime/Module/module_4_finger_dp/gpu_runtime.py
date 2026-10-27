"""CUDA-only execution contract for Finger DP training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True, slots=True)
class CUDARuntimeInfo:
  requested_device: str
  resolved_device: str
  device_index: int
  device_name: str
  total_memory_bytes: int
  compute_capability: tuple[int, int]
  torch_version: str
  torch_cuda_version: str

  def to_dict(self) -> dict[str, object]:
    return asdict(self)


def require_cuda(device: str = "cuda:0") -> tuple[torch.device, CUDARuntimeInfo]:
  """Resolve a CUDA device and fail closed instead of falling back to CPU."""

  requested = torch.device(device)
  if requested.type != "cuda":
    raise RuntimeError(
      f"Finger DP requires CUDA; CPU device {device!r} is forbidden"
    )
  if not torch.cuda.is_available():
    raise RuntimeError(
      "Finger DP CUDA is unavailable. Training/inference was not started; "
      "CPU fallback is forbidden."
    )
  index = requested.index if requested.index is not None else torch.cuda.current_device()
  if index < 0 or index >= torch.cuda.device_count():
    raise RuntimeError(f"CUDA device index {index} is unavailable")
  resolved = torch.device(f"cuda:{index}")
  torch.cuda.set_device(resolved)
  properties = torch.cuda.get_device_properties(resolved)
  info = CUDARuntimeInfo(
    requested_device=device,
    resolved_device=str(resolved),
    device_index=index,
    device_name=properties.name,
    total_memory_bytes=int(properties.total_memory),
    compute_capability=(int(properties.major), int(properties.minor)),
    torch_version=torch.__version__,
    torch_cuda_version=str(torch.version.cuda),
  )
  return resolved, info


def synchronize_cuda(device: torch.device) -> None:
  if device.type != "cuda":
    raise RuntimeError("synchronize_cuda refuses a non-CUDA device")
  torch.cuda.synchronize(device)
