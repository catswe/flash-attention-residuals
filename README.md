## Flash Attention Residuals

> **1.4x faster inference/training** vs. torch.compile impl. of the paper’s two-phase batched attention + online softmax

> **20% reduction in training memory** (without activation checkpointing)*

*Benchmarked on H100. Dependent on problem size and setup.

Reference: https://arxiv.org/abs/2603.15031 (Kimi Team, MoonshotAI, 2026)

## Credits:
Thanks to Mohamed Osman (https://github.com/spaghettiSystems) and Cartesia for advising on and supporting the development of this kernel.

## Install

```
pip install flash-attn-res
```

## Usage
This package contains Triton kernels, `triton_op` wrappers compatible with torch.compile, and an experimental high-performance Block AttenRes autograd implementation.
See `src` and `examples` folders.


<!-- TODO: -->
<!-- - Figure out first block phase 1 special case redundant computation output -->
<!-- - Determine redundant store -->
<!-- - Consider "phase_2_online_softmax_merge_intrablock_backward_kernel probably does not need atomic_add" -->
<!-- - Consider two-phase reduction -->

## Roadmap:
- CuTE, CUDA, and other DSLs implementation
- Tune precision
- Mixed FP16 and BF16 and store quantization scale
- Stochastic rounding
- Make into Python package

## Development Notes:
- Normalizing in phase 1 keeps outputs bounded (convex combination of values) so bf16 error doesn't scale with softmax flatness. Phase 2 computes in fp32, and the reduction algebra matches split-KV Flash Attention.
- Certain dimensions, especially NUM_QUERIES_PER_BLOCK, are small so semi-elementwise (B, T) kernel with static_range is better than doing tl.dot
- Kernel is memory bound and doing semi-elementwise allows for kernel fusion
- NUM_SOURCE_BLOCKS and NUM_QUERIES_PER_BLOCK should be autotuning keys, unlike with torch.compile, which allows for faster kernels
- Small NUM_QUERIES_PER_BLOCK so eviction_policy should be "evict_last"
