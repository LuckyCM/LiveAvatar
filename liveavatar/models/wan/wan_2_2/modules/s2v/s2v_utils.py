# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import numpy as np
import torch

def rope_precompute(x, grid_sizes, freqs, start=None):
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2

    # RoPE cache is purely real: [..., 2] = (cos, sin). Never use complex64.
    if type(freqs) is list:
        # [ (base_cos, base_sin), (trainable_cos, trainable_sin) ]
        (base_cos, base_sin) = freqs[0]
        (tf_cos, tf_sin) = freqs[1]
        tf_cos = tf_cos.to(dtype=torch.float32, device=x.device)
        tf_sin = tf_sin.to(dtype=torch.float32, device=x.device)
    else:
        (base_cos, base_sin) = freqs
        tf_cos, tf_sin = None, None

    base_cos = base_cos.to(dtype=torch.float32, device=x.device)
    base_sin = base_sin.to(dtype=torch.float32, device=x.device)

    split_sizes = [c - 2 * (c // 3), c // 3, c // 3]
    cos_split = base_cos.split(split_sizes, dim=1)
    sin_split = base_sin.split(split_sizes, dim=1)

    # Initialize cache to identity rotation (cos=1, sin=0) to avoid dirty data.
    output_float = torch.zeros((b, s, n, c, 2), dtype=torch.float32, device=x.device)
    output_float[..., 0] = 1.0

    seq_bucket = [0]
    if not type(grid_sizes) is list:
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not type(g) is list:
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]

            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1, seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(), (-t_f - f_o).item() + 1, seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1, seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1, seq_w).astype(int).tolist()

                    # 3. 完美的 Float32 切片 (NPU 毫无压力)
                    f0_cos = cos_split[0][f_sam]
                    f0_sin = sin_split[0][f_sam]
                    if f_o < 0:
                        f0_sin = -f0_sin  # 等同于复数的 conj()

                    f1_cos = cos_split[1][h_sam]
                    f1_sin = sin_split[1][h_sam]
                    
                    f2_cos = cos_split[2][w_sam]
                    f2_sin = sin_split[2][w_sam]

                    # 展开维度
                    f0_cos = f0_cos.view(seq_f, 1, 1, -1).expand(seq_f, seq_h, seq_w, -1)
                    f0_sin = f0_sin.view(seq_f, 1, 1, -1).expand(seq_f, seq_h, seq_w, -1)
                    
                    f1_cos = f1_cos.view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1)
                    f1_sin = f1_sin.view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1)
                    
                    f2_cos = f2_cos.view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1)
                    f2_sin = f2_sin.view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1)

                    # 分别合并 cos 和 sin
                    cat_cos = torch.cat([f0_cos, f1_cos, f2_cos], dim=-1).reshape(seq_len, 1, -1)
                    cat_sin = torch.cat([f0_sin, f1_sin, f2_sin], dim=-1).reshape(seq_len, 1, -1)
                    
                    # 组合成 [..., 2]
                    freqs_stacked = torch.stack([cat_cos, cat_sin], dim=-1)

                elif t_f < 0:
                    freqs_stacked = torch.stack([tf_cos, tf_sin], dim=-1).unsqueeze(1)
                
                # 4. 完美的 Float32 赋值 (绕过 NPU 无法 InplaceCopy 复数的 Bug)
                output_float[i, seq_bucket[-1]:seq_bucket[-1] + seq_len] = freqs_stacked
        seq_bucket.append(seq_bucket[-1] + seq_len)
        
    # 5. 直接返回纯实数 cache: [..., 2] = (cos, sin)
    return output_float


if __name__ == "__main__":
    # python path
    import sys
    sys.path.append('Causvid')
    x = torch.randn(1, 3375, 40,128).to('cuda')
    grid_sizes = [[torch.tensor([[0, 0, 0]]).to('cuda'), torch.tensor([[3, 45, 25]]).to('cuda'), torch.tensor([[3, 45, 25]]).to('cuda')]]
    # freqs = torch.randn(1, 1024, 16, 16).to('cuda')
    start = None
    d = 128
    from liveavatar.models.wan.wan_2_2.modules.model import rope_params
    cos0, sin0 = rope_params(16384, d - 4 * (d // 6))
    cos1, sin1 = rope_params(16384, 2 * (d // 6))
    cos2, sin2 = rope_params(16384, 2 * (d // 6))
    freqs = (torch.cat([cos0, cos1, cos2], dim=1).to('cuda'),
             torch.cat([sin0, sin1, sin2], dim=1).to('cuda'))
    torch.cuda.synchronize()
    import time;t_start = time.time()
    output = rope_precompute(x.to('cuda'), grid_sizes, freqs, start)
    torch.cuda.synchronize()
    print(f"rope precompute time: {time.time() - t_start}")
    x = torch.randn(1, 3375, 40,128).to('cuda')
    torch.cuda.synchronize()
    import time;t_start = time.time()
    output = rope_precompute(x.to('cuda'), grid_sizes, freqs, start)
    torch.cuda.synchronize()
    print(f"rope precompute time 2: {time.time() - t_start}")
    
    # print(output.shape)
    # from liveavatar.models.wan.wan_2_2.modules.s2v.s2v_utils_ori import rope_precompute as rope_precompute_ori
    # torch.cuda.synchronize()
    # import time;t_start = time.time()
    # output_ori = rope_precompute_ori(x.to('cuda'), grid_sizes, [freqs.to('cuda'), freqs.to('cuda'), freqs.to('cuda')], start)
    # print(f"rope precompute time ori: {time.time() - t_start}")

    # torch.cuda.synchronize()
    # import time;t_start = time.time()
    # output_ori = rope_precompute(x.to('cuda'), grid_sizes, [freqs.to('cuda'), freqs.to('cuda'), freqs.to('cuda')], start)
    # print(f"rope precompute time 2: {time.time() - t_start}")
