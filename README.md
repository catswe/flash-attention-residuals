5x inference speedup over naive torch.compile implementation 

Credits:
Thank you Cartesia for providing support with developing this kernel

Roadmap:
- Backward implementation
- Implement in CuTE and CUDA

Key Insights:
- Dimensions, especially NUM_QUERIES_PER_BLOCK, are small so semi-elementwise (B, T) kernel with static_range is better than doing tl.dot
- Kernel is memory bound and doing semi-elementwise allows for kernel fusion
- Small dimensions so online softmax is not necessary
- NUM_SOURCE_BLOCKS and NUM_QUERIES_PER_BLOCK should be autotuning keys, unlike with torch.compile, which allows for faster kernels
- Store branch when layer offset is 0 reduces wasted computation + memory
