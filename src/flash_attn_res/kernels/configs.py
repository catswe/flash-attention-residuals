import triton

forward_attn_configs = [
    triton.Config({}, num_warps=1, num_stages=2),
    triton.Config({}, num_warps=1, num_stages=3),
    triton.Config({}, num_warps=1, num_stages=4),
    triton.Config({}, num_warps=2, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=3),
]

backward_attn_configs = [
    triton.Config({}, num_warps=1, num_stages=2),
    triton.Config({}, num_warps=2, num_stages=3),
    triton.Config({}, num_warps=2, num_stages=4),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=4),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=4),
]


def set_autotune_configs(kernel, configs):
    kernel.configs = list(configs)
    kernel.cache.clear()
