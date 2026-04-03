import torch

STREAMING_VAE = True

# torch.compile 会使用 Dynamo 缓存；这里保持一个较小的上限以避免长时间运行时缓存膨胀。
torch._dynamo.config.cache_size_limit = 128

# -----------------------------------------------------------------------------
# Ascend NPU + torch.compile compatibility patch
#
# 背景：在 Ascend NPU 环境开启 torch.compile 时，PyTorch Dynamo 内部会走到
# torch.utils._triton 的 CUDA/Triton 特性探测逻辑（如 TMA），但在 NPU 上
# torch.cuda.get_device_capability() 可能返回 None，或 get_device_properties().major
# 为空，从而触发 InternalTorchDynamoError / TypeError。
#
# 这里在调用 torch.compile 之前做 Monkey Patch：
# - 强制 Triton 检查返回 False，避免误入 CUDA/TMA 分支
# - 兜底 CUDA capability / properties，避免 None 参与比较
# -----------------------------------------------------------------------------

_NPU_TORCH_COMPILE_PATCHED = False


def patch_torch_compile_for_npu() -> bool:
    """Patch Dynamo/Triton CUDA feature checks for Ascend NPU.

    Returns:
        bool: True if patch is applied in current process; otherwise False.
    """

    global _NPU_TORCH_COMPILE_PATCHED
    if _NPU_TORCH_COMPILE_PATCHED:
        return True

    # Ensure `torch.npu` exists when running on Ascend.
    if not hasattr(torch, "npu"):
        try:
            import torch_npu  # noqa: F401
        except Exception:
            pass

    if not (hasattr(torch, "npu") and getattr(torch.npu, "is_available", lambda: False)()):
        return False

    # 1) Force Triton availability checks to False on NPU.
    try:
        import torch.utils._triton as _triton  # type: ignore

        if hasattr(_triton, "has_triton"):
            _triton.has_triton = lambda: False  # type: ignore[assignment]
        if hasattr(_triton, "has_triton_experimental_host_tma"):
            _triton.has_triton_experimental_host_tma = lambda: False  # type: ignore[assignment]
        # Some versions use internal helper; safest is to hard-disable.
        if hasattr(_triton, "_device_supports_tma"):
            _triton._device_supports_tma = lambda *args, **kwargs: False  # type: ignore[assignment]
    except Exception:
        # Torch builds without _triton are OK.
        pass

    # 2) Fallback patch for torch.cuda.get_device_capability() returning None / raising.
    try:
        original_cap = getattr(getattr(torch, "cuda", None), "get_device_capability", None)
        if callable(original_cap):
            def patched_cap(*args, **kwargs):
                try:
                    res = original_cap(*args, **kwargs)
                except Exception:
                    res = None
                # Return a tuple to avoid NoneType comparisons and keep CUDA-only paths disabled.
                return res if res is not None else (0, 0)

            torch.cuda.get_device_capability = patched_cap  # type: ignore[assignment]
    except Exception:
        pass

    # 3) Fallback patch for get_device_properties().major/minor being None / raising.
    try:
        original_props = getattr(getattr(torch, "cuda", None), "get_device_properties", None)
        if callable(original_props):
            from types import SimpleNamespace

            class _CudaPropsProxy:
                def __init__(self, p):
                    self._p = p

                @property
                def major(self):
                    v = getattr(self._p, "major", None)
                    return int(v) if v is not None else 0

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
                    # Provide a minimal shape used by Dynamo/Triton checks.
                    return SimpleNamespace(major=0, minor=0)

                if p is None:
                    return SimpleNamespace(major=0, minor=0)

                if getattr(p, "major", None) is None or getattr(p, "minor", None) is None:
                    return _CudaPropsProxy(p)
                return p

            torch.cuda.get_device_properties = patched_get_device_properties  # type: ignore[assignment]
    except Exception:
        pass

    _NPU_TORCH_COMPILE_PATCHED = True
    return True

NO_REFRESH_INFERENCE = False
