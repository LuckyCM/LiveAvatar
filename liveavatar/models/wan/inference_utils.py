import torch

STREAMING_VAE = True

# torch.compile 会使用 Dynamo 缓存；这里保持一个较小的上限以避免长时间运行时缓存膨胀。
torch._dynamo.config.cache_size_limit = 128

NO_REFRESH_INFERENCE = False
