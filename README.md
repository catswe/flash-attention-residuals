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
This package contains Triton kernels, `triton_op` wrappers compatible with torch.compile, and an experimental high-performance Block AttenRes autograd implementation. Example usage with checkpointing:
```python
from flash_attn_res.ops.phase_1 import phase_1_batched_attention_triton_op
from flash_attn_res.ops.phase_2 import phase_2_online_softmax_merge_triton_op

from torch.utils.checkpoint import checkpoint

def production_forward(
    inputs: torch.Tensor,
    pseudo_queries: torch.Tensor,
    layers,
    eps: float | None = None,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    if eps is None:
        eps = EPS

    blocks = [inputs]

    for block_start in range(0, len(layers), block_size):
        num_queries = min(block_size, len(layers) - block_start)
        block_queries = pseudo_queries[block_start : block_start + num_queries]

        def run_block(
            block_queries_arg,
            *prev_blocks,
            block_start=block_start,
            num_queries=num_queries,
        ):
            values = torch.stack(prev_blocks, dim=0)

            phase1_out, phase1_lse = phase_1_batched_attention_triton_op(
                values,
                block_queries_arg,
                eps,
            )

            # NOTE: unbind calls are highly important for performance
            block_queries_unbind = block_queries_arg.unbind(0)
            phase1_out_unbind = phase1_out.unbind(0)
            phase1_lse_unbind = phase1_lse.unbind(0)

            curr_block = None

            for i in range(num_queries):
                layer = layers[block_start + i]

                if i == 0:
                    curr_block = layer(phase1_out_unbind[i])
                else:
                    layer_input = phase_2_online_softmax_merge_triton_op(
                        curr_block,
                        block_queries_unbind[i],
                        phase1_out_unbind[i],
                        phase1_lse_unbind[i],
                        eps,
                    )

                    curr_block = curr_block + layer(layer_input)

            return curr_block

        curr_block = checkpoint(
            run_block,
            block_queries,
            *blocks,
            use_reentrant=False,
        )

        blocks.append(curr_block)

    final_out, _final_lse = phase_1_batched_attention_triton_op(
        torch.stack(blocks, dim=0),
        pseudo_queries[-1:],
        eps,
    )

    return final_out.squeeze(0).to(inputs.dtype)
```

For more detail on usage, see `src` folders. For peak performance (not recommended), check out `BlockAttentionResiduals` from `experimental` folder.

<!-- TODO: -->
<!-- - Figure out first block phase 1 special case redundant computation output -->
<!-- - Determine redundant store -->
<!-- - Consider "phase_2_online_softmax_merge_intrablock_backward_kernel probably does not need atomic_add" -->
<!-- - Consider two-phase reduction -->

## Roadmap:
- Pointer (indirection) table kernel impl. (potentially may not integrate cleanly with Torch)
- Better autotuning set up
- Better tests
- Better mixed precision setup
- Depth-wise RoPE
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

## Contributing:
- PRs are welcomed! It is highly recommended to file an issue before creating a pull request! 
