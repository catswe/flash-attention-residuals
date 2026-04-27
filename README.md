## Flash Attention Residuals
30% speedup over optimized torch.compile implementation of the paper’s two-phase batched inference with online softmax

## Credits:
Thanks to Mohamed Osman (https://github.com/spaghettiSystems) and Cartesia for advising on and supporting the development of this kernel.

<!-- TODO: -->
<!-- - Determine redundant store -->
<!-- - Consider "phase_2_online_softmax_merge_intrablock_backward_kernel probably does not need atomic_add" -->
<!-- - Consider two-phase reduction -->

## Roadmap:
- Proper backward eval
- Implement in CuTE and CUDA
- Tune precision
- Mixed FP16 and BF16 and store quantization scale
- Stochastic rounding
- Make into Python package

## Insights:
- Normalizing in phase 1 keeps outputs bounded (convex combination of values) so bf16 error doesn't scale with softmax flatness. Phase 2 computes in fp32, and the reduction algebra matches split-KV Flash Attention.
- Dimensions, especially NUM_QUERIES_PER_BLOCK, are small so semi-elementwise (B, T) kernel with static_range is better than doing tl.dot
- Kernel is memory bound and doing semi-elementwise allows for kernel fusion
- Small dimensions so online softmax is not necessary
- NUM_SOURCE_BLOCKS and NUM_QUERIES_PER_BLOCK should be autotuning keys, unlike with torch.compile, which allows for faster kernels
- Store branch when layer offset is 0 reduces wasted computation + memory
