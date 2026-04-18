import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

DEVICE = "cuda"
DTYPE = torch.bfloat16
FORWARD_ONLY = True

BLOCK_SIZE = 8
NUM_LAYERS = 64

B, T, D = 32, 1024, 512


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


autotune_configs = [
    triton.Config({}, num_warps=num_warps, num_stages=num_stages)
    for num_warps in [1, 2, 4, 8, 16]
    for num_stages in [1, 2, 3, 4]
]


@triton.autotune(
    configs=autotune_configs,
    key=["NUM_SOURCE_BLOCKS", "HIDDEN_DIM", "NUM_QUERIES_PER_BLOCK", "PADDED_SRC"],
)
@triton.jit
def phase_1_batched_interblock_attention(
    block_representations_ptr,
    pseudo_queries_ptr,
    first_layer_normalized_output_ptr,
    interblock_normalized_output_ptr,
    interblock_lse_ptr,
    eps,
    NUM_SOURCE_BLOCKS: tl.constexpr,
    BT: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    NUM_QUERIES_PER_BLOCK: tl.constexpr,
    PADDED_SRC: tl.constexpr,
):
    batch_seq_idx = tl.program_id(0)

    source_block_range = tl.arange(0, PADDED_SRC)[:, None]
    hidden_dim_range = tl.arange(0, HIDDEN_DIM)[None, :]
    valid_block_mask_2d = source_block_range < NUM_SOURCE_BLOCKS

    valid_block_mask_1d = tl.arange(0, PADDED_SRC) < NUM_SOURCE_BLOCKS

    source_block_values = tl.load(
        block_representations_ptr
        + source_block_range * (BT * HIDDEN_DIM)
        + batch_seq_idx * HIDDEN_DIM
        + hidden_dim_range,
        mask=valid_block_mask_2d,
        other=0.0,
    ).to(tl.float32)

    squared_norm_sum = tl.sum(source_block_values * source_block_values, axis=1)
    inverse_rms_norm = tl.rsqrt(squared_norm_sum / float(HIDDEN_DIM) + eps)

    hidden_dim_range_1d = tl.arange(0, HIDDEN_DIM)

    for layer_offset in tl.static_range(NUM_QUERIES_PER_BLOCK):
        pseudo_query_vector = tl.load(
            pseudo_queries_ptr + layer_offset * HIDDEN_DIM + hidden_dim_range,
            eviction_policy="evict_last",
        ).to(tl.float32)

        attention_logits = (
            tl.sum(source_block_values * pseudo_query_vector, axis=1) * inverse_rms_norm
        )
        attention_logits = tl.where(
            valid_block_mask_1d, attention_logits, float("-inf")
        )

        max_attention_logit = tl.max(attention_logits)
        exp_attention_logits = tl.exp(attention_logits - max_attention_logit)
        exp_sum = tl.sum(exp_attention_logits)

        unnormalized_output = tl.sum(
            exp_attention_logits[:, None] * source_block_values, axis=0
        )
        normalized_output = (unnormalized_output / exp_sum).to(tl.bfloat16)

        if layer_offset == 0:
            tl.store(
                first_layer_normalized_output_ptr
                + batch_seq_idx * HIDDEN_DIM
                + hidden_dim_range_1d,
                normalized_output,
            )
        else:
            tl.store(
                interblock_normalized_output_ptr
                + (layer_offset - 1) * BT * HIDDEN_DIM
                + batch_seq_idx * HIDDEN_DIM
                + hidden_dim_range_1d,
                normalized_output,
            )
            tl.store(
                interblock_lse_ptr + (layer_offset - 1) * BT + batch_seq_idx,
                max_attention_logit + tl.log(exp_sum),
            )


@triton.autotune(
    configs=autotune_configs,
    key=["HIDDEN_DIM"],
)
@triton.jit
def phase_2_online_softmax_merge_intrablock(
    intrablock_partial_sum_ptr,
    pseudo_query_ptr,
    interblock_unnormalized_output_ptr,
    interblock_lse_ptr,
    merged_output_ptr,
    eps,
    HIDDEN_DIM: tl.constexpr,
):
    batch_seq_idx = tl.program_id(0)
    hidden_dim_range = tl.arange(0, HIDDEN_DIM)

    intrablock_partial_sum = tl.load(
        intrablock_partial_sum_ptr + batch_seq_idx * HIDDEN_DIM + hidden_dim_range
    ).to(tl.float32)
    pseudo_query_vector = tl.load(
        pseudo_query_ptr + hidden_dim_range, eviction_policy="evict_last"
    ).to(tl.float32)

    interblock_lse = tl.load(interblock_lse_ptr + batch_seq_idx)
    interblock_normalized_output = tl.load(
        interblock_unnormalized_output_ptr
        + batch_seq_idx * HIDDEN_DIM
        + hidden_dim_range
    ).to(tl.float32)

    squared_norm_sum = tl.sum(intrablock_partial_sum * intrablock_partial_sum)
    inverse_rms_norm = tl.rsqrt(squared_norm_sum / float(HIDDEN_DIM) + eps)

    intrablock_logit = (
        tl.sum(intrablock_partial_sum * pseudo_query_vector) * inverse_rms_norm
    )
    merged_max = tl.maximum(interblock_lse, intrablock_logit)
    interblock_weight = tl.exp(interblock_lse - merged_max)
    intrablock_weight = tl.exp(intrablock_logit - merged_max)
    merged_output = (
        interblock_weight * interblock_normalized_output
        + intrablock_weight * intrablock_partial_sum
    ) / (interblock_weight + intrablock_weight)

    tl.store(
        merged_output_ptr + batch_seq_idx * HIDDEN_DIM + hidden_dim_range,
        merged_output.to(tl.bfloat16),
    )


def production_forward(inputs, pseudo_queries, layers):
    block_representations = torch.zeros(
        math.ceil(len(layers) / BLOCK_SIZE) + 1,
        B,
        T,
        D,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    block_representations[0] = inputs
    curr_block_idx = 0

    residual_attention_output = torch.empty(
        (B, T, D), dtype=torch.bfloat16, device=DEVICE
    )
    interblock_normalized_outputs = torch.empty(
        (BLOCK_SIZE - 1, B, T, D),
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    interblock_lses = torch.empty(
        (BLOCK_SIZE - 1, B, T), dtype=torch.float32, device=DEVICE
    )

    for i in range(len(layers)):
        if i % BLOCK_SIZE == 0:
            curr_block_idx += 1
            num_queries = min(BLOCK_SIZE, len(layers) - i)

            phase_1_batched_interblock_attention[(B * T,)](
                block_representations,
                pseudo_queries[i : i + num_queries],
                residual_attention_output,
                interblock_normalized_outputs,
                interblock_lses,
                torch.finfo(torch.float32).eps,
                curr_block_idx,
                B * T,
                D,
                num_queries,
                triton.next_power_of_2(curr_block_idx),
            )
        else:
            offset = (i % BLOCK_SIZE) - 1

            phase_2_online_softmax_merge_intrablock[(B * T,)](
                block_representations[curr_block_idx],
                pseudo_queries[i],
                interblock_normalized_outputs[offset],
                interblock_lses[offset],
                residual_attention_output,
                torch.finfo(torch.float32).eps,
                D,
            )

        block_representations[curr_block_idx] += layers[i](residual_attention_output)

    phase_1_batched_interblock_attention[(B * T,)](
        block_representations,
        pseudo_queries[-1:],
        residual_attention_output,
        interblock_normalized_outputs,
        interblock_lses,
        torch.finfo(torch.float32).eps,
        curr_block_idx + 1,
        B * T,
        D,
        1,
        triton.next_power_of_2(curr_block_idx + 1),
    )
    return residual_attention_output


with torch.no_grad():
    inputs = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE)
    pseudo_queries = torch.randn(
        NUM_LAYERS + 1,
        D,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=not FORWARD_ONLY,
    )
    layers_identity = [Identity() for _ in range(NUM_LAYERS)]
    out = production_forward(inputs, pseudo_queries, layers_identity)
