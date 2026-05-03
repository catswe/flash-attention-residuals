## Flash Attention Residuals

> **2.2x faster training** vs. optimized torch.compile attention residuals implementation

*Benchmarked on A100 with activation checkpointing. L=32 | BLOCK_SIZE = 8 | T=16384, D=2048

Reference: https://arxiv.org/abs/2603.15031 (Kimi Team, MoonshotAI, 2026)

## Credits:
Thanks to Mohamed Osman (https://github.com/spaghettiSystems) and Cartesia (https://github.com/cartesia-ai) for advising on and supporting the development of this project.

## Install

```
pip install flash-attn-res
```

## Usage
This package contains Triton kernels, `triton_op` wrappers compatible with torch.compile, and an experimental high-performance Block AttenRes autograd implementation.

```python
from flash_attn_res.ops.phase_1 import phase_1_batched_attention_triton_op
from flash_attn_res.ops.phase_2 import phase_2_online_softmax_merge_triton_op

phase1_out, phase1_lse = phase_1_batched_attention_triton_op(
    values,
    pseudo_queries,
    eps,
)

merged = phase_2_online_softmax_merge_triton_op(
    curr_block,
    pseudo_query,
    phase1_out_i,
    phase1_lse_i,
    eps,
)
```

For peak performance, import `BlockAttentionResiduals` from `experimental` folder. For more detail on usage, see `src` and `benchmarks` folders.

<!-- TODO: -->
<!-- - Figure out first block phase 1 special case redundant computation output -->
<!-- - Determine redundant store -->
<!-- - Consider "phase_2_online_softmax_merge_intrablock_backward_kernel probably does not need atomic_add" -->
<!-- - Consider two-phase reduction -->

## Roadmap:
- Better autotuning set up
- Better benchmarks
- More robust autograd impl.
- Precision tuning
- Mixed FP16 and BF16 and store quantization scale
- Stochastic rounding
- CuTE, CUDA, and other DSLs implementation

## Development Notes:
- Normalizing in phase 1 keeps outputs bounded (convex combination of values) so bf16 error doesn't scale with softmax flatness. Phase 2 computes in fp32, and the reduction algebra matches split-KV Flash Attention.
- Certain dimensions, especially NUM_QUERIES_PER_BLOCK, are small so semi-elementwise (B, T) kernel with static_range is better than doing tl.dot
- Kernel is memory bound and doing semi-elementwise allows for kernel fusion
- NUM_SOURCE_BLOCKS and NUM_QUERIES_PER_BLOCK should be autotuning keys, unlike with torch.compile, which allows for faster kernels
- Small NUM_QUERIES_PER_BLOCK so eviction_policy should be "evict_last"
