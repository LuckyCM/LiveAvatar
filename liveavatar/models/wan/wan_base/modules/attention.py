# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import torch
import math

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    import torch_npu
    from torch_npu import npu_fusion_attention
    FLASH_ATTN_2_NPU_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_NPU_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def _use_npu_fused_attention():
    return FLASH_ATTN_2_NPU_AVAILABLE and os.getenv(
        "LIVEAVATAR_DISABLE_NPU_FUSED_ATTN", "false"
    ).lower() not in ("1", "true", "yes", "on")


def _sdpa_attention(q, k, v, q_lens, k_lens, dropout_p, softmax_scale, causal, window_size, dtype):
    if window_size != (-1, -1):
        warnings.warn(
            'Windowed attention is disabled when using scaled_dot_product_attention fallback on NPU.'
        )
    attn_mask = None
    if q_lens is not None or k_lens is not None:
        if q_lens is None:
            q_lens = torch.full((q.size(0),), q.size(1), dtype=torch.int32, device=q.device)
        if k_lens is None:
            k_lens = torch.full((k.size(0),), k.size(1), dtype=torch.int32, device=k.device)
        attn_mask = make_attn_mask_from_lens(
            q_lens, k_lens, q.size(1), k.size(1), dtype=q.dtype, device=q.device
        ).bool()

    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )
    return out.transpose(1, 2).contiguous()


def make_attn_mask_from_lens(q_lens: torch.Tensor,
                             k_lens: torch.Tensor,
                             q_len: int,
                             k_len: int,
                             dtype: torch.dtype = torch.float32,
                             device: torch.device | None = None,
                             four_d: bool = True,
                             neginf_safe: bool = True) -> torch.Tensor:
    device = device or q_lens.device
    B = q_lens.size(0)
    q_range = torch.arange(q_len, device=device).unsqueeze(0).expand(B, -1)
    k_range = torch.arange(k_len, device=device).unsqueeze(0).expand(B, -1)
    q_valid = q_range < q_lens.unsqueeze(1)
    k_valid = k_range < k_lens.unsqueeze(1)
    mask = q_valid.unsqueeze(2) & k_valid.unsqueeze(1)
    if four_d:
        mask = mask.unsqueeze(1)
    return mask


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type in ['cuda', 'npu']
    assert q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), v.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_scale is not None:
        q = q * q_scale

    if q.device.type == 'npu' and _use_npu_fused_attention():
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if k_lens is None and q_lens is None:
            npu_attn_mask = None
        else:
            if q_lens is None:
                q_lens = torch.full((b, ), lq, dtype=torch.int32, device=q.device)
            if k_lens is None:
                k_lens = torch.full((b, ), lk, dtype=torch.int32, device=k.device)

            attn_mask = make_attn_mask_from_lens(
                q_lens, k_lens, lq, lk, dtype=q.dtype, device=q.device
            )
            npu_attn_mask = (~attn_mask.bool()).contiguous()

        q_bnsd = q.transpose(1, 2).contiguous()
        k_bnsd = k.transpose(1, 2).contiguous()
        v_bnsd = v.transpose(1, 2).contiguous()
        head_num = q_bnsd.size(1)

        npu_softmax_scale = (1.0 if softmax_scale is None else float(softmax_scale)) / math.sqrt(q.size(-1))

        x_bnsd = torch_npu.npu_fusion_attention(
            q_bnsd,
            k_bnsd,
            v_bnsd,
            head_num=head_num,
            input_layout="BNSD",
            pse=None,
            padding_mask=None,
            atten_mask=npu_attn_mask,
            scale=npu_softmax_scale,
            pre_tockens=2147483647,
            next_tockens=0 if causal else 2147483647,
            keep_prob=1.0 - dropout_p,
            inner_precise=0,
            sparse_mode=0,
        )[0]

        x = x_bnsd.transpose(1, 2).contiguous()
        return x.type(out_dtype)
    elif q.device.type == 'npu':
        return _sdpa_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            dtype=dtype,
        ).type(out_dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if q.device.type == 'npu':
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )

    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
