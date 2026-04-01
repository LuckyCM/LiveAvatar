import os
from typing import Literal, Optional

import torch


DeviceBackend = Literal["auto", "cuda", "npu", "cpu"]


def _has_torch_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return True


def resolve_device_backend(prefer: DeviceBackend = "auto") -> DeviceBackend:
    """Resolve runtime device backend.

    Priority:
    - If prefer != auto: return prefer (caller ensures availability)
    - Else: npu (if torch_npu + device exists) -> cuda -> cpu
    """
    if prefer != "auto":
        return prefer

    if _has_torch_npu():
        # torch_npu registers torch.npu and device type "npu"
        try:
            if hasattr(torch, "npu") and torch.npu.is_available():
                return "npu"
        except Exception:
            # torch_npu import succeeded but runtime not ready
            pass

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def device_count(backend: DeviceBackend) -> int:
    if backend == "npu":
        if not hasattr(torch, "npu"):
            return 0
        return int(torch.npu.device_count())
    if backend == "cuda":
        return int(torch.cuda.device_count())
    return 0


def set_device(local_rank: int, backend: DeviceBackend) -> None:
    if backend == "npu":
        # import torch_npu first to ensure torch.npu exists
        import torch_npu  # noqa: F401
        torch.npu.set_device(local_rank)
        return
    if backend == "cuda":
        torch.cuda.set_device(local_rank)
        return


def synchronize(backend: DeviceBackend) -> None:
    if backend == "npu" and hasattr(torch, "npu"):
        try:
            torch.npu.synchronize()
        except Exception:
            return
    if backend == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def default_dist_backend(backend: DeviceBackend) -> str:
    if backend == "npu":
        return "hccl"
    if backend == "cuda":
        return "nccl"
    return "gloo"


def parse_device_backend_arg(value: Optional[str]) -> DeviceBackend:
    if value is None:
        return "auto"
    v = value.strip().lower()
    if v in ("auto", "cuda", "npu", "cpu"):
        return v  # type: ignore[return-value]
    raise ValueError(f"Unsupported device_backend: {value}")


def env_device_backend(default: DeviceBackend = "auto") -> DeviceBackend:
    """Read device backend preference from env, used for compatibility.

    Env:
    - LIVEAVATAR_DEVICE_BACKEND: auto/cuda/npu/cpu
    """
    v = os.getenv("LIVEAVATAR_DEVICE_BACKEND")
    if not v:
        return default
    try:
        return parse_device_backend_arg(v)
    except Exception:
        return default

