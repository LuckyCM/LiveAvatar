import os
from contextlib import contextmanager
from typing import Literal, Optional

import torch


DeviceBackend = Literal["auto", "cuda", "npu", "cpu"]


_NPU_RUNTIME_COMPAT_PATCHED = False


def apply_npu_runtime_compat_patches() -> bool:
    """Apply best-effort runtime compatibility patches for Ascend NPU.

    背景：部分 PyTorch/Dynamo/Triton（torch.utils._triton）在初始化/探测 CUDA 能力时
    会调用 `torch.cuda.get_device_capability()` / `torch.cuda.get_device_properties()`。
    在 Ascend NPU 环境中，这些接口可能抛异常或返回 None，从而导致比较报错，甚至在
    torch.compile / Dynamo 期间触发崩溃。

    本函数集中做 NPU 场景下的兼容性 Monkey Patch（幂等）：
    - 强制 Triton 相关探测返回 False，避免误入 CUDA/TMA 分支
    - 兜底 CUDA capability / properties，避免 None/异常参与比较（默认回退为 (8, 0)）

    Returns:
        bool: True 表示已在当前进程中应用（或曾应用）；False 表示未应用（例如 torch.cuda 不存在）。
    """

    global _NPU_RUNTIME_COMPAT_PATCHED
    if _NPU_RUNTIME_COMPAT_PATCHED:
        return True

    # Ensure torch.npu exists when running on Ascend.
    if not hasattr(torch, "npu"):
        try:
            import torch_npu  # noqa: F401
        except Exception:
            pass

    # 1) Force Triton availability checks to False on NPU.
    try:
        import torch.utils._triton as _triton  # type: ignore

        if hasattr(_triton, "has_triton"):
            _triton.has_triton = lambda: False  # type: ignore[assignment]
        if hasattr(_triton, "has_triton_experimental_host_tma"):
            _triton.has_triton_experimental_host_tma = lambda: False  # type: ignore[assignment]
        if hasattr(_triton, "_device_supports_tma"):
            _triton._device_supports_tma = lambda *args, **kwargs: False  # type: ignore[assignment]
    except Exception:
        # Torch builds without _triton are OK.
        pass

    # 2) Fallback patch for torch.cuda.get_device_capability() returning None / raising.
    try:
        cuda = getattr(torch, "cuda", None)
        original_cap = getattr(cuda, "get_device_capability", None) if cuda is not None else None
        if callable(original_cap):
            def patched_cap(*args, **kwargs):
                try:
                    res = original_cap(*args, **kwargs)
                except Exception:
                    res = None
                # 在 NPU 场景下兜底为 (8, 0)：
                # - 保证是可比较的 tuple
                # - 同时对 Hopper/TMA(>=9.0) 等分支保持 False
                if not (isinstance(res, tuple) and len(res) >= 2 and res[0] is not None and res[1] is not None):
                    return (8, 0)
                try:
                    return (int(res[0]), int(res[1]))
                except Exception:
                    return (8, 0)

            cuda.get_device_capability = patched_cap  # type: ignore[assignment]
    except Exception:
        pass

    # 3) Fallback patch for get_device_properties().major/minor being None / raising.
    try:
        cuda = getattr(torch, "cuda", None)
        original_props = getattr(cuda, "get_device_properties", None) if cuda is not None else None
        if callable(original_props):
            from types import SimpleNamespace

            class _CudaPropsProxy:
                def __init__(self, p):
                    self._p = p

                @property
                def major(self):
                    v = getattr(self._p, "major", None)
                    return int(v) if v is not None else 8

                @property
                def minor(self):
                    v = getattr(self._p, "minor", None)
                    return int(v) if v is not None else 0

                def __getattr__(self, name):
                    return getattr(self._p, name)

            def patched_get_device_properties(*args, **kwargs):
                try:
                    p = original_props(*args, **kwargs)
                except Exception:
                    return SimpleNamespace(major=8, minor=0)

                if p is None:
                    return SimpleNamespace(major=8, minor=0)

                if getattr(p, "major", None) is None or getattr(p, "minor", None) is None:
                    return _CudaPropsProxy(p)
                # best-effort normalize
                try:
                    _ = int(getattr(p, "major"))
                    _ = int(getattr(p, "minor"))
                except Exception:
                    return _CudaPropsProxy(p)
                return p

            cuda.get_device_properties = patched_get_device_properties  # type: ignore[assignment]
    except Exception:
        pass

    _NPU_RUNTIME_COMPAT_PATCHED = True
    return True


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
        # NPU runtime compat patches must be applied before any torch.compile/Triton probing.
        apply_npu_runtime_compat_patches()
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


def empty_cache(backend: DeviceBackend) -> None:
    """Best-effort device cache cleanup (no-op if unsupported)."""
    if backend == "npu" and hasattr(torch, "npu"):
        try:
            torch.npu.empty_cache()
        except Exception:
            return
    if backend == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_backend_available(backend: DeviceBackend) -> bool:
    if backend == "cpu":
        return True
    if backend == "cuda":
        return torch.cuda.is_available()
    if backend == "npu":
        try:
            return hasattr(torch, "npu") and torch.npu.is_available()
        except Exception:
            return False
    return False


def preferred_infer_dtype(backend: DeviceBackend) -> torch.dtype:
    """Default inference dtype for the backend.

    Ascend 910B prefers bfloat16.
    """
    if backend == "npu":
        return torch.bfloat16
    # keep conservative default for cuda/cpu unless caller overrides
    return torch.float16 if backend == "cuda" else torch.float32


def autocast_device_type_from_tensors(*tensors: object, fallback: DeviceBackend = "auto") -> str:
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.device.type
    return resolve_device_backend(fallback)


@contextmanager
def autocast(
    device_type: str,
    dtype: torch.dtype = torch.bfloat16,
    enabled: bool = True,
):
    """Device-agnostic autocast context.

    Uses `torch.autocast` when available (PyTorch 2.6+), falls back to
    `torch.amp.autocast`.
    """
    try:
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled):
            yield
    except Exception:
        # Older or vendor builds may not expose torch.autocast
        with torch.amp.autocast(device_type, dtype=dtype, enabled=enabled):
            yield


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
