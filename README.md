5x inference speedup over naive torch.compile implementation 

Credits:
Thank you Cartesia for providing support with developing this kernel

Roadmap:
- Backward implementation
- Implement in CuTE and CUDA

Key Insights:
- Normalizing in phase 1 keeps outputs bounded (convex combination of values) so bf16 error doesn't scale with softmax flatness. Phase 2 computes in fp32, and the reduction algebra matches split-KV Flash Attention.
- Dimensions, especially NUM_QUERIES_PER_BLOCK, are small so semi-elementwise (B, T) kernel with static_range is better than doing tl.dot
- Kernel is memory bound and doing semi-elementwise allows for kernel fusion
- Small dimensions so online softmax is not necessary
- NUM_SOURCE_BLOCKS and NUM_QUERIES_PER_BLOCK should be autotuning keys, unlike with torch.compile, which allows for faster kernels
- Store branch when layer offset is 0 reduces wasted computation + memory
