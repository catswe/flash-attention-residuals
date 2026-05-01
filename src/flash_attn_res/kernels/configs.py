import triton

forward_configs = [
    triton.Config({}, num_warps=1, num_stages=2),
    triton.Config({}, num_warps=1, num_stages=3),
    triton.Config({}, num_warps=1, num_stages=4),
    triton.Config({}, num_warps=2, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=3),
]

phase1_backward_configs = [
    triton.Config({}, num_warps=1, num_stages=2),
    triton.Config({}, num_warps=2, num_stages=3),
    triton.Config({}, num_warps=2, num_stages=4),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=4),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=4),
]


phase2_backward_configs = [
    triton.Config({"BLOCK_BT": 8}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_BT": 8}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_BT": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_BT": 16}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_BT": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_BT": 32}, num_warps=8, num_stages=1),
]


def set_autotune_configs(kernel, configs):
    kernel.configs = list(configs)
    kernel.cache.clear()
