# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import numpy as np
import torch
from ....inference_utils import conditional_compile

# @conditional_compile
def rope_precompute(x, grid_sizes, freqs, start=None):
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2

    # 1. 第一时间将输入的复数频率拆解为纯实数 (Float32)
    if type(freqs) is list:
        trainable_freqs = freqs[1]
        freqs = freqs[0]
        tf_r = trainable_freqs.real.to(torch.float32)
        tf_i = trainable_freqs.imag.to(torch.float32)
    else:
        tf_r, tf_i = None, None

    f_r = freqs.real.to(torch.float32)
    f_i = freqs.imag.to(torch.float32)

    split_sizes = [c - 2 * (c // 3), c // 3, c // 3]
    f_r_split = f_r.split(split_sizes, dim=1)
    f_i_split = f_i.split(split_sizes, dim=1)

    # 2. 将输入张量转为纯 Float32 格式，尾部维度为 2 (实部, 虚部)
    output_float = x.detach().contiguous().to(torch.float32).reshape(b, s, n, -1, 2)

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
                    f0_r = f_r_split[0][f_sam]
                    f0_i = f_i_split[0][f_sam]
                    if f_o < 0:
                        f0_i = -f0_i  # 等同于复数的 conj()

                    f1_r = f_r_split[1][h_sam]
                    f1_i = f_i_split[1][h_sam]
                    
                    f2_r = f_r_split[2][w_sam]
                    f2_i = f_i_split[2][w_sam]

                    # 展开维度
                    f0_r = f0_r.view(seq_f, 1, 1, -1).expand(seq_f, seq_h, seq_w, -1)
                    f0_i = f0_i.view(seq_f, 1, 1, -1).expand(seq_f, seq_h, seq_w, -1)
                    
                    f1_r = f1_r.view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1)
                    f1_i = f1_i.view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1)
                    
                    f2_r = f2_r.view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1)
                    f2_i = f2_i.view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1)

                    # 分别合并实部和虚部
                    cat_r = torch.cat([f0_r, f1_r, f2_r], dim=-1).reshape(seq_len, 1, -1)
                    cat_i = torch.cat([f0_i, f1_i, f2_i], dim=-1).reshape(seq_len, 1, -1)
                    
                    # 组合成 [..., 2]
                    freqs_stacked = torch.stack([cat_r, cat_i], dim=-1)

                elif t_f < 0:
                    freqs_stacked = torch.stack([tf_r, tf_i], dim=-1).unsqueeze(1)
                
                # 4. 完美的 Float32 赋值 (绕过 NPU 无法 InplaceCopy 复数的 Bug)
                output_float[i, seq_bucket[-1]:seq_bucket[-1] + seq_len] = freqs_stacked
        seq_bucket.append(seq_bucket[-1] + seq_len)
        
    # 5. 最后一秒钟，转回復数格式返回
    return torch.view_as_complex(output_float)


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
    freqs = torch.cat([
            rope_params(16384, d - 4 * (d // 6)),
            rope_params(16384, 2 * (d // 6)),
            rope_params(16384, 2 * (d // 6))
        ],
                               dim=1).to('cuda')
    torch.cuda.synchronize()
    import time;t_start = time.time()
    output = rope_precompute(x.to('cuda'), grid_sizes, freqs.to('cuda'), start)
    torch.cuda.synchronize()
    print(f"rope precompute time: {time.time() - t_start}")
    x = torch.randn(1, 3375, 40,128).to('cuda')
    torch.cuda.synchronize()
    import time;t_start = time.time()
    output = rope_precompute(x.to('cuda'), grid_sizes, freqs.to('cuda'), start)
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
