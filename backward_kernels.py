from __future__ import annotations

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import os

from typing import Tuple
from torch.library import triton_op, wrap_triton


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

DEVICE = "cuda"
DTYPE = torch.bfloat16

L = 32
BLOCK_SIZE = 8
NUM_BLOCKS = math.ceil(L / BLOCK_SIZE) + 1

B, T, D = 64, 2048, 512
BT = B * T

EPS = torch.finfo(torch.float32).eps

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
def phase_1_batched_interblock_attention_kernel(
    block_representations_ptr,
    pseudo_queries_ptr,
    softmax_normalized_output_ptr,
    lse_ptr,
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

        tl.store(
            softmax_normalized_output_ptr
            + layer_offset * BT * HIDDEN_DIM
            + batch_seq_idx * HIDDEN_DIM
            + hidden_dim_range_1d,
            normalized_output,
        )
        tl.store(
            lse_ptr + layer_offset * BT + batch_seq_idx,
            max_attention_logit + tl.log(exp_sum),
        )


@triton_op("blockwise_attn::phase_1_batched_interblock_attention", mutates_args={})
def phase_1_batched_interblock_attention(
    block_representations: torch.Tensor,
    pseudo_queries: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_source_blocks = block_representations.shape[0]
    num_queries = pseudo_queries.shape[0]
    B_, T_, D_ = block_representations.shape[1:]
    BT_ = B_ * T_

    softmax_outputs = torch.empty(
        (num_queries, B_, T_, D_),
        device=block_representations.device,
        dtype=torch.bfloat16,
    )

    lses = torch.empty(
        (num_queries, B_, T_),
        device=block_representations.device,
        dtype=torch.float32,
    )

    wrap_triton(phase_1_batched_interblock_attention_kernel)[(BT_,)](
        block_representations,
        pseudo_queries,
        softmax_outputs,
        lses,
        eps,
        num_source_blocks,
        BT_,
        D_,
        num_queries,
        triton.next_power_of_2(num_source_blocks),
    )

    return softmax_outputs, lses


@triton.autotune(
    configs=autotune_configs,
    key=["HIDDEN_DIM"],
)
@triton.jit
def phase_2_online_softmax_merge_intrablock_out_kernel(
    intrablock_partial_sum_ptr,
    pseudo_query_ptr,
    interblock_normalized_output_ptr,
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
        interblock_normalized_output_ptr + batch_seq_idx * HIDDEN_DIM + hidden_dim_range
    ).to(tl.float32)

    squared_norm_sum = tl.sum(intrablock_partial_sum * intrablock_partial_sum)
    inverse_rms_norm = tl.rsqrt(squared_norm_sum / float(HIDDEN_DIM) + eps)

    intrablock_logit = (
        tl.sum(intrablock_partial_sum * pseudo_query_vector) * inverse_rms_norm
    )
    merged_max = tl.maximum(interblock_lse, intrablock_logit)
    interblock_weight = tl.exp(interblock_lse - merged_max)
    intrablock_weight = tl.exp(intrablock_logit - merged_max)
    exp_sum = interblock_weight + intrablock_weight
    merged_output = (
        interblock_weight * interblock_normalized_output
        + intrablock_weight * intrablock_partial_sum
    ) / exp_sum

    tl.store(
        merged_output_ptr + batch_seq_idx * HIDDEN_DIM + hidden_dim_range,
        merged_output.to(tl.bfloat16),
    )


@triton_op("blockwise_attn::phase_2_online_softmax_merge_intrablock", mutates_args={})
def phase_2_online_softmax_merge_intrablock(
    intrablock_partial_sum: torch.Tensor,
    pseudo_query: torch.Tensor,
    interblock_normalized_output: torch.Tensor,
    interblock_lse: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    B_, T_, D_ = intrablock_partial_sum.shape
    BT_ = B_ * T_

    merged_output = torch.empty(
        (B_, T_, D_),
        device=interblock_normalized_output.device,
        dtype=interblock_normalized_output.dtype,
    )

    wrap_triton(phase_2_online_softmax_merge_intrablock_out_kernel)[(BT_,)](
        intrablock_partial_sum,
        pseudo_query,
        interblock_normalized_output,
        interblock_lse,
        merged_output,
        eps,
        D_,
    )

    return merged_output


@triton.autotune(
    configs=autotune_configs,
    key=["NUM_SOURCE_BLOCKS", "HIDDEN_DIM", "NUM_QUERIES_PER_BLOCK", "PADDED_SRC"],
    restore_value=[
        "grad_block_representations_accumulator_ptr",
    ],
)
@triton.jit
def phase_1_batched_interblock_attention_backward_kernel(
    block_representations_ptr,
    pseudo_queries_ptr,
    lse_ptr,
    grad_softmax_normalized_output_ptr,
    grad_lse_ptr,
    grad_block_representations_accumulator_ptr,
    grad_pseudo_queries_partial_ptr,
    eps,
    NUM_SOURCE_BLOCKS: tl.constexpr,
    BT: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    NUM_QUERIES_PER_BLOCK: tl.constexpr,
    PADDED_SRC: tl.constexpr,
    HAS_GRAD_LSE: tl.constexpr,
):
    batch_seq_idx = tl.program_id(0)

    source_block_range = tl.arange(0, PADDED_SRC)[:, None]
    source_block_range_1d = tl.arange(0, PADDED_SRC)

    hidden_dim_range = tl.arange(0, HIDDEN_DIM)[None, :]
    hidden_dim_range_1d = tl.arange(0, HIDDEN_DIM)

    valid_block_mask_2d = source_block_range < NUM_SOURCE_BLOCKS
    valid_block_mask_1d = source_block_range_1d < NUM_SOURCE_BLOCKS

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
    inverse_rms_norm_cubed = inverse_rms_norm * inverse_rms_norm * inverse_rms_norm

    grad_source_accumulator = tl.zeros((PADDED_SRC, HIDDEN_DIM), tl.float32)

    for layer_offset in tl.static_range(NUM_QUERIES_PER_BLOCK):
        pseudo_query_vector = tl.load(
            pseudo_queries_ptr + layer_offset * HIDDEN_DIM + hidden_dim_range,
            eviction_policy="evict_last",
        ).to(tl.float32)

        grad_attention_output = tl.load(
            grad_softmax_normalized_output_ptr
            + layer_offset * BT * HIDDEN_DIM
            + batch_seq_idx * HIDDEN_DIM
            + hidden_dim_range_1d,
        ).to(tl.float32)

        if HAS_GRAD_LSE:
            grad_logsumexp = tl.load(
                grad_lse_ptr + layer_offset * BT + batch_seq_idx
            ).to(tl.float32)
        else:
            grad_logsumexp = 0.0

        forward_logsumexp = tl.load(lse_ptr + layer_offset * BT + batch_seq_idx).to(
            tl.float32
        )

        pseudo_query_source_dot = tl.sum(
            source_block_values * pseudo_query_vector,
            axis=1,
        )

        attention_logits = pseudo_query_source_dot * inverse_rms_norm
        attention_logits = tl.where(
            valid_block_mask_1d,
            attention_logits,
            float("-inf"),
        )

        softmax_probabilities = tl.exp(attention_logits - forward_logsumexp)

        grad_output_dot_source_values = tl.sum(
            source_block_values * grad_attention_output[None, :],
            axis=1,
        )

        grad_output_dot_expected_value = tl.sum(
            softmax_probabilities * grad_output_dot_source_values,
            axis=0,
        )

        grad_attention_logits = softmax_probabilities * (
            grad_logsumexp
            + grad_output_dot_source_values
            - grad_output_dot_expected_value
        )

        grad_source_from_value_path = (
            softmax_probabilities[:, None] * grad_attention_output[None, :]
        )

        grad_source_from_logit_path = grad_attention_logits[:, None] * (
            inverse_rms_norm[:, None] * pseudo_query_vector
            - pseudo_query_source_dot[:, None]
            * inverse_rms_norm_cubed[:, None]
            * source_block_values
            / float(HIDDEN_DIM)
        )

        grad_source_block_values = (
            grad_source_from_value_path + grad_source_from_logit_path
        )

        grad_source_accumulator += tl.where(
            valid_block_mask_2d,
            grad_source_block_values,
            0.0,
        )

        grad_pseudo_query = tl.sum(
            grad_attention_logits[:, None]
            * inverse_rms_norm[:, None]
            * source_block_values,
            axis=0,
        )

        tl.store(
            grad_pseudo_queries_partial_ptr
            + layer_offset * BT * HIDDEN_DIM
            + batch_seq_idx * HIDDEN_DIM
            + hidden_dim_range_1d,
            grad_pseudo_query,
        )

    # TODO: specialize for NUM_QUERIES_PER_BLOCK == 1.
    grad_block_ptr = (
        grad_block_representations_accumulator_ptr
        + source_block_range * (BT * HIDDEN_DIM)
        + batch_seq_idx * HIDDEN_DIM
        + hidden_dim_range
    )

    prev_grad_block = tl.load(
        grad_block_ptr,
        mask=valid_block_mask_2d,
        other=0.0,
    ).to(tl.float32)

    tl.store(
        grad_block_ptr,
        prev_grad_block + grad_source_accumulator,
        mask=valid_block_mask_2d,
    )


reduce_configs = [
    triton.Config(
        {
            "BLOCK_BATCH_SEQ": block_batch_seq,
            "BLOCK_HIDDEN": block_hidden,
        },
        num_warps=num_warps,
        num_stages=1,
    )
    for block_batch_seq in [128, 256]
    for block_hidden in [32, 64]
    for num_warps in [4, 8]
]


@triton.autotune(
    configs=reduce_configs,
    key=["NUM_BATCH_SEQ", "HIDDEN_DIM", "NUM_QUERIES_PER_BLOCK"],
    restore_value=[
        "grad_pseudo_queries_accumulator_ptr",
    ],
)
@triton.jit
def phase_1_reduce_grad_pseudo_queries_kernel(
    grad_pseudo_queries_partial_ptr,
    grad_pseudo_queries_accumulator_ptr,
    NUM_BATCH_SEQ: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    NUM_QUERIES_PER_BLOCK: tl.constexpr,
    BLOCK_BATCH_SEQ: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    batch_seq_block_idx = tl.program_id(0)
    query_idx = tl.program_id(1)
    hidden_block_idx = tl.program_id(2)

    batch_seq_offsets = batch_seq_block_idx * BLOCK_BATCH_SEQ + tl.arange(
        0, BLOCK_BATCH_SEQ
    )

    hidden_offsets = hidden_block_idx * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)

    grad_tile = tl.load(
        grad_pseudo_queries_partial_ptr
        + query_idx * NUM_BATCH_SEQ * HIDDEN_DIM
        + batch_seq_offsets[:, None] * HIDDEN_DIM
        + hidden_offsets[None, :],
        mask=(
            (batch_seq_offsets[:, None] < NUM_BATCH_SEQ)
            & (hidden_offsets[None, :] < HIDDEN_DIM)
            & (query_idx < NUM_QUERIES_PER_BLOCK)
        ),
        other=0.0,
    ).to(tl.float32)

    grad_reduced = tl.sum(grad_tile, axis=0)

    tl.atomic_add(
        grad_pseudo_queries_accumulator_ptr + query_idx * HIDDEN_DIM + hidden_offsets,
        grad_reduced,
        mask=((hidden_offsets < HIDDEN_DIM) & (query_idx < NUM_QUERIES_PER_BLOCK)),
        sem="relaxed",
    )


def phase_1_batched_interblock_attention_backward(
    block_representations,
    pseudo_queries,
    lses,
    grad_softmax_outputs,
    grad_lses,
    grad_block_representations,
    grad_pseudo_queries,
    grad_pseudo_queries_partial,
    eps=None,
):
    NUM_QUERIES = pseudo_queries.shape[0]
    NUM_SOURCE_BLOCKS = block_representations.shape[0]

    if eps is None:
        eps = EPS

    has_grad_lses = grad_lses is not None
    if grad_lses is None:
        grad_lses = lses

    phase_1_batched_interblock_attention_backward_kernel[(BT,)](
        block_representations,
        pseudo_queries,
        lses,
        grad_softmax_outputs,
        grad_lses,
        grad_block_representations,
        grad_pseudo_queries_partial,
        eps,
        NUM_SOURCE_BLOCKS,
        BT,
        D,
        NUM_QUERIES,
        triton.next_power_of_2(NUM_SOURCE_BLOCKS),
        has_grad_lses,
    )

    phase_1_reduce_grad_pseudo_queries_kernel[
        lambda META: (
            triton.cdiv(BT, META["BLOCK_BATCH_SEQ"]),
            NUM_QUERIES,
            triton.cdiv(D, META["BLOCK_HIDDEN"]),
        )
    ](
        grad_pseudo_queries_partial,
        grad_pseudo_queries,
        BT,
        D,
        NUM_QUERIES,
    )


@triton.autotune(
    configs=autotune_configs,
    key=["HIDDEN_DIM"],
    restore_value=[
        "grad_intrablock_partial_sum_accumulator_ptr",
    ],
)
@triton.jit
def phase_2_online_softmax_merge_intrablock_backward_kernel(
    intrablock_partial_sum_ptr,
    pseudo_query_ptr,
    phase1_interblock_normalized_output_ptr,
    phase1_interblock_logsumexp_ptr,
    grad_merged_attention_output_ptr,
    grad_intrablock_partial_sum_accumulator_ptr,
    grad_pseudo_query_partial_ptr,
    grad_phase1_interblock_normalized_output_ptr,
    grad_phase1_interblock_logsumexp_ptr,
    eps,
    HIDDEN_DIM: tl.constexpr,
):
    batch_seq_idx = tl.program_id(0)
    hidden_dim_range = tl.arange(0, HIDDEN_DIM)

    intrablock_partial_sum = tl.load(
        intrablock_partial_sum_ptr + batch_seq_idx * HIDDEN_DIM + hidden_dim_range
    ).to(tl.float32)

    pseudo_query = tl.load(
        pseudo_query_ptr + hidden_dim_range,
        eviction_policy="evict_last",
    ).to(tl.float32)

    phase1_interblock_normalized_output = tl.load(
        phase1_interblock_normalized_output_ptr
        + batch_seq_idx * HIDDEN_DIM
        + hidden_dim_range
    ).to(tl.float32)

    phase1_interblock_logsumexp = tl.load(
        phase1_interblock_logsumexp_ptr + batch_seq_idx
    ).to(tl.float32)

    grad_merged_attention_output = tl.load(
        grad_merged_attention_output_ptr + batch_seq_idx * HIDDEN_DIM + hidden_dim_range
    ).to(tl.float32)

    intrablock_partial_sum_squared_norm = tl.sum(
        intrablock_partial_sum * intrablock_partial_sum
    )
    intrablock_inverse_rms_norm = tl.rsqrt(
        intrablock_partial_sum_squared_norm / float(HIDDEN_DIM) + eps
    )

    pseudo_query_intrablock_dot = tl.sum(intrablock_partial_sum * pseudo_query)
    phase2_intrablock_logit = pseudo_query_intrablock_dot * intrablock_inverse_rms_norm

    online_softmax_shift = tl.maximum(
        phase1_interblock_logsumexp,
        phase2_intrablock_logit,
    )
    phase1_partition_weight = tl.exp(phase1_interblock_logsumexp - online_softmax_shift)
    phase2_partition_weight = tl.exp(phase2_intrablock_logit - online_softmax_shift)
    merged_partition_weight_sum = phase1_partition_weight + phase2_partition_weight

    phase1_merge_probability = phase1_partition_weight / merged_partition_weight_sum
    phase2_merge_probability = phase2_partition_weight / merged_partition_weight_sum

    grad_phase1_interblock_normalized_output = (
        phase1_merge_probability * grad_merged_attention_output
    )
    grad_intrablock_partial_sum_from_value_path = (
        phase2_merge_probability * grad_merged_attention_output
    )

    grad_output_dot_interblock_minus_intrablock = tl.sum(
        grad_merged_attention_output
        * (phase1_interblock_normalized_output - intrablock_partial_sum)
    )

    merge_probability_product = phase1_merge_probability * phase2_merge_probability

    grad_phase1_interblock_logsumexp = (
        merge_probability_product * grad_output_dot_interblock_minus_intrablock
    )

    grad_phase2_intrablock_logit = (
        -merge_probability_product * grad_output_dot_interblock_minus_intrablock
    )

    intrablock_inverse_rms_norm_cubed = (
        intrablock_inverse_rms_norm
        * intrablock_inverse_rms_norm
        * intrablock_inverse_rms_norm
    )

    grad_intrablock_partial_sum_from_logit_path = grad_phase2_intrablock_logit * (
        intrablock_inverse_rms_norm * pseudo_query
        - pseudo_query_intrablock_dot
        * intrablock_inverse_rms_norm_cubed
        * intrablock_partial_sum
        / float(HIDDEN_DIM)
    )

    grad_pseudo_query = (
        grad_phase2_intrablock_logit
        * intrablock_inverse_rms_norm
        * intrablock_partial_sum
    )
    grad_intrablock_partial_sum = (
        grad_intrablock_partial_sum_from_value_path
        + grad_intrablock_partial_sum_from_logit_path
    )

    grad_intrablock_ptr = (
        grad_intrablock_partial_sum_accumulator_ptr
        + batch_seq_idx * HIDDEN_DIM
        + hidden_dim_range
    )

    tl.store(
        grad_intrablock_ptr,
        tl.load(grad_intrablock_ptr).to(tl.float32) + grad_intrablock_partial_sum,
    )

    tl.store(
        grad_pseudo_query_partial_ptr + batch_seq_idx * HIDDEN_DIM + hidden_dim_range,
        grad_pseudo_query,
    )

    tl.store(
        grad_phase1_interblock_normalized_output_ptr
        + batch_seq_idx * HIDDEN_DIM
        + hidden_dim_range,
        grad_phase1_interblock_normalized_output,
    )

    tl.store(
        grad_phase1_interblock_logsumexp_ptr + batch_seq_idx,
        grad_phase1_interblock_logsumexp,
    )


@triton.autotune(
    configs=reduce_configs,
    key=["NUM_BATCH_SEQ", "HIDDEN_DIM"],
    restore_value=[
        "grad_pseudo_query_accumulator_ptr",
    ],
)
@triton.jit
def phase_2_reduce_grad_pseudo_query_kernel(
    grad_pseudo_query_partial_ptr,
    grad_pseudo_query_accumulator_ptr,
    NUM_BATCH_SEQ: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_BATCH_SEQ: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    batch_seq_block_idx = tl.program_id(0)
    hidden_block_idx = tl.program_id(1)

    batch_seq_offsets = batch_seq_block_idx * BLOCK_BATCH_SEQ + tl.arange(
        0, BLOCK_BATCH_SEQ
    )
    hidden_offsets = hidden_block_idx * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)

    grad_tile = tl.load(
        grad_pseudo_query_partial_ptr
        + batch_seq_offsets[:, None] * HIDDEN_DIM
        + hidden_offsets[None, :],
        mask=(
            (batch_seq_offsets[:, None] < NUM_BATCH_SEQ)
            & (hidden_offsets[None, :] < HIDDEN_DIM)
        ),
        other=0.0,
    ).to(tl.float32)

    grad_reduced = tl.sum(grad_tile, axis=0)

    tl.atomic_add(
        grad_pseudo_query_accumulator_ptr + hidden_offsets,
        grad_reduced,
        mask=hidden_offsets < HIDDEN_DIM,
        sem="relaxed",
    )


def phase_2_online_softmax_merge_intrablock_backward(
    intrablock_partial_sum,
    pseudo_query,
    phase1_interblock_normalized_output,
    phase1_interblock_logsumexp,
    grad_merged_attention_output,
    grad_intrablock_partial_sum,
    grad_pseudo_query,
    grad_phase1_interblock_normalized_output,
    grad_phase1_interblock_logsumexp,
    grad_pseudo_query_partial,
    eps=None,
):
    if eps is None:
        eps = EPS

    phase_2_online_softmax_merge_intrablock_backward_kernel[(BT,)](
        intrablock_partial_sum,
        pseudo_query,
        phase1_interblock_normalized_output,
        phase1_interblock_logsumexp,
        grad_merged_attention_output,
        grad_intrablock_partial_sum,
        grad_pseudo_query_partial,
        grad_phase1_interblock_normalized_output,
        grad_phase1_interblock_logsumexp,
        eps,
        D,
    )

    phase_2_reduce_grad_pseudo_query_kernel[
        lambda META: (
            triton.cdiv(BT, META["BLOCK_BATCH_SEQ"]),
            triton.cdiv(D, META["BLOCK_HIDDEN"]),
        )
    ](
        grad_pseudo_query_partial,
        grad_pseudo_query,
        BT,
        D,
    )


class BlockwiseAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, pseudo_queries, layers, eps, *flat_layer_params):
        block_representations = torch.empty(
            NUM_BLOCKS,
            B,
            T,
            D,
            device=DEVICE,
            dtype=inputs.dtype,
        )
        block_representations[0].copy_(inputs)

        for block_start in range(0, L, BLOCK_SIZE):
            curr_block_idx = block_start // BLOCK_SIZE + 1
            num_queries = min(BLOCK_SIZE, L - block_start)

            block_attn_out, block_lse = phase_1_batched_interblock_attention(
                block_representations[:curr_block_idx],
                pseudo_queries[block_start : block_start + num_queries],
                eps,
            )

            curr_block = block_representations[curr_block_idx]

            for query_offset in range(num_queries):
                i = block_start + query_offset

                if query_offset == 0:
                    layer_input = block_attn_out[query_offset]
                else:
                    layer_input = phase_2_online_softmax_merge_intrablock(
                        curr_block,
                        pseudo_queries[i],
                        block_attn_out[query_offset],
                        block_lse[query_offset],
                        eps,
                    )

                update = layers[i](layer_input)

                if query_offset == 0:
                    curr_block.copy_(update)
                else:
                    curr_block.add_(update)

        final_out, _final_lse = phase_1_batched_interblock_attention(
            block_representations,
            pseudo_queries[-1:],
            eps,
        )

        ctx.save_for_backward(
            block_representations,
            pseudo_queries,
        )
        ctx.layers = layers
        ctx.eps = eps
        ctx.num_layer_params = len(flat_layer_params)

        return final_out[0].to(inputs.dtype)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        if grad_output is None:
            return (None, None, None, None, *([None] * ctx.num_layer_params))

        block_representations, pseudo_queries = ctx.saved_tensors
        layers = ctx.layers
        eps = ctx.eps

        device = block_representations.device
        block_dtype = block_representations.dtype

        grad_output = grad_output.contiguous()

        layer_param_groups = [tuple(layer.parameters()) for layer in layers]
        flat_layer_params = [p for group in layer_param_groups for p in group]

        param_offsets = []
        offset = 0
        for group in layer_param_groups:
            param_offsets.append(offset)
            offset += len(group)

        grad_flat_layer_params = [
            torch.zeros_like(p, dtype=torch.float32) if p.requires_grad else None
            for p in flat_layer_params
        ]

        grad_block_representations = torch.zeros_like(
            block_representations,
            dtype=torch.float32,
        )
        grad_pseudo_queries = torch.zeros_like(
            pseudo_queries,
            dtype=torch.float32,
        )

        grad_pseudo_queries_partial = torch.empty(
            BLOCK_SIZE,
            B,
            T,
            D,
            device=device,
            dtype=torch.float32,
        )

        grad_phase2_pseudo_query_partial = torch.empty(
            B,
            T,
            D,
            device=device,
            dtype=torch.float32,
        )

        def run_saved_layer_backward(layer_idx, layer_input, update, grad_update_f32):
            params_i = layer_param_groups[layer_idx]
            active_param_indices = [
                j for j, p in enumerate(params_i) if p.requires_grad
            ]
            active_params = [params_i[j] for j in active_param_indices]

            grad_results = torch.autograd.grad(
                outputs=update,
                inputs=(layer_input, *active_params),
                grad_outputs=grad_update_f32.to(dtype=update.dtype),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )

            grad_layer_input = grad_results[0].to(torch.float32).contiguous()

            base = param_offsets[layer_idx]
            for local_idx, param_grad in zip(active_param_indices, grad_results[1:]):
                if param_grad is not None:
                    grad_flat_layer_params[base + local_idx].add_(
                        param_grad.to(torch.float32)
                    )

            return grad_layer_input

        with torch.no_grad():
            _final_out_recomputed, final_lse = phase_1_batched_interblock_attention(
                block_representations,
                pseudo_queries[-1:],
                eps,
            )

        phase_1_batched_interblock_attention_backward(
            block_representations,
            pseudo_queries[-1:],
            final_lse,
            grad_output.unsqueeze(0),
            None,
            grad_block_representations,
            grad_pseudo_queries[-1:],
            grad_pseudo_queries_partial[:1],
            eps=eps,
        )

        grad_block_phase1_out_scratch = torch.empty(
            BLOCK_SIZE,
            B,
            T,
            D,
            device=device,
            dtype=torch.float32,
        )
        grad_block_lse_scratch = torch.empty(
            BLOCK_SIZE,
            B,
            T,
            device=device,
            dtype=torch.float32,
        )

        intrablock_partial_before_scratch = torch.empty(
            max(BLOCK_SIZE - 1, 1),
            B,
            T,
            D,
            device=device,
            dtype=block_dtype,
        )

        partial_recompute = torch.empty(
            B,
            T,
            D,
            device=device,
            dtype=block_dtype,
        )

        last_block_start = ((L - 1) // BLOCK_SIZE) * BLOCK_SIZE

        for block_start in range(last_block_start, -1, -BLOCK_SIZE):
            curr_block_idx = block_start // BLOCK_SIZE + 1
            num_queries = min(BLOCK_SIZE, L - block_start)

            grad_phase1_out = grad_block_phase1_out_scratch[:num_queries]
            grad_phase1_lse = grad_block_lse_scratch[:num_queries]

            grad_phase1_lse[0].zero_()

            layer_input_recomputed_list = []
            layer_update_list = []

            with torch.no_grad():
                phase1_out, phase1_lse = phase_1_batched_interblock_attention(
                    block_representations[:curr_block_idx],
                    pseudo_queries[block_start : block_start + num_queries],
                    eps,
                )

            for query_offset in range(num_queries):
                layer_idx = block_start + query_offset

                if query_offset == 0:
                    layer_input_recomputed = phase1_out[query_offset]
                else:
                    partial_before = intrablock_partial_before_scratch[query_offset - 1]
                    partial_before.copy_(partial_recompute)

                    with torch.no_grad():
                        layer_input_recomputed = (
                            phase_2_online_softmax_merge_intrablock(
                                partial_before,
                                pseudo_queries[layer_idx],
                                phase1_out[query_offset],
                                phase1_lse[query_offset],
                                eps,
                            )
                        )

                with torch.enable_grad():
                    layer_input_for_grad = (
                        layer_input_recomputed.detach().requires_grad_(True)
                    )
                    update = layers[layer_idx](layer_input_for_grad)

                layer_input_recomputed_list.append(layer_input_for_grad)
                layer_update_list.append(update)

                with torch.no_grad():
                    if query_offset == 0:
                        partial_recompute.copy_(update.detach())
                    else:
                        partial_recompute.add_(update.detach())

            grad_curr_partial = grad_block_representations[curr_block_idx]

            for query_offset in range(num_queries - 1, -1, -1):
                layer_idx = block_start + query_offset

                grad_layer_input = run_saved_layer_backward(
                    layer_idx,
                    layer_input_recomputed_list[query_offset],
                    layer_update_list[query_offset],
                    grad_curr_partial,
                )

                if query_offset == 0:
                    grad_phase1_out[query_offset].copy_(grad_layer_input)
                else:
                    phase_2_online_softmax_merge_intrablock_backward(
                        intrablock_partial_before_scratch[query_offset - 1],
                        pseudo_queries[layer_idx],
                        phase1_out[query_offset],
                        phase1_lse[query_offset],
                        grad_layer_input,
                        grad_curr_partial,
                        grad_pseudo_queries[layer_idx],
                        grad_phase1_out[query_offset],
                        grad_phase1_lse[query_offset],
                        grad_phase2_pseudo_query_partial,
                        eps=eps,
                    )

            phase_1_batched_interblock_attention_backward(
                block_representations[:curr_block_idx],
                pseudo_queries[block_start : block_start + num_queries],
                phase1_lse,
                grad_phase1_out,
                grad_phase1_lse,
                grad_block_representations[:curr_block_idx],
                grad_pseudo_queries[block_start : block_start + num_queries],
                grad_pseudo_queries_partial[:num_queries],
                eps=eps,
            )

        grad_inputs = (
            grad_block_representations[0].to(block_dtype)
            if ctx.needs_input_grad[0]
            else None
        )

        grad_pseudo_queries_out = (
            grad_pseudo_queries.to(pseudo_queries.dtype)
            if ctx.needs_input_grad[1]
            else None
        )

        grad_flat_layer_params_out = []
        for j, (param, grad_param) in enumerate(
            zip(flat_layer_params, grad_flat_layer_params)
        ):
            needs_grad = ctx.needs_input_grad[4 + j]
            if not needs_grad or grad_param is None:
                grad_flat_layer_params_out.append(None)
            else:
                grad_flat_layer_params_out.append(grad_param.to(param.dtype))

        return (
            grad_inputs,
            grad_pseudo_queries_out,
            None,
            None,
            *grad_flat_layer_params_out,
        )


def production_forward(inputs, pseudo_queries, layers, eps=None):
    if eps is None:
        eps = EPS

    flat_layer_params = tuple(p for layer in layers for p in layer.parameters())

    return BlockwiseAttentionFunction.apply(
        inputs,
        pseudo_queries,
        layers,
        eps,
        *flat_layer_params,
    )


# TODO: do max-autotune
@torch.compile(mode="max-autotune-no-cudagraphs")
def naive_attention_residual(pseudo_query, values):
    keys = F.rms_norm(values, (values.shape[-1],), eps=EPS)

    logits = torch.einsum("d, n b t d -> n b t", pseudo_query, keys)
    logits = logits - logits.max(dim=0, keepdim=True).values

    return torch.einsum(
        "n b t, n b t d -> b t d",
        logits.softmax(0),
        values,
    ).to(DTYPE)


def paper_forward(inputs, pseudo_queries, layers):
    inputs = inputs.to(torch.float32)
    pseudo_queries = pseudo_queries.to(torch.float32)

    blocks = [inputs]

    for i in range(len(layers)):
        outputs = naive_attention_residual(
            pseudo_queries[i],
            torch.stack(blocks, dim=0),
        )

        update = layers[i](outputs)

        if i % BLOCK_SIZE == 0:
            blocks.append(update)
        else:
            blocks[-1] = blocks[-1] + update

    return naive_attention_residual(
        pseudo_queries[-1],
        torch.stack(blocks, dim=0),
    )


@torch.compile(mode="max-autotune-no-cudagraphs")
def phase_1_fn(query, value):
    query = query.to(torch.float32)
    value = value.to(torch.float32)

    D_ = value.shape[-1]

    squared_norm_sum = (value * value).sum(dim=-1)
    inverse_rms_norm = torch.rsqrt(squared_norm_sum / float(D_) + EPS)
    raw_dot = torch.einsum("nbtd,sd->nbts", value, query)
    logits = raw_dot * inverse_rms_norm.unsqueeze(-1)

    max_logits = logits.amax(dim=0)
    exp_weights = torch.exp(logits - max_logits.unsqueeze(0))
    exp_sum = exp_weights.sum(dim=0)

    weighted_sum = (exp_weights.unsqueeze(-1) * value.unsqueeze(3)).sum(dim=0)
    normalized = (weighted_sum / exp_sum[..., None]).permute(2, 0, 1, 3).contiguous()

    lse = (max_logits + torch.log(exp_sum)).permute(2, 0, 1).contiguous()

    h = normalized[0]
    return lse, normalized.to(torch.bfloat16), h


@torch.compile(mode="max-autotune-no-cudagraphs")
def phase_2_fn(current_block_values, query_vector, prev_lse, prev_normalized):
    query_vector_f32 = query_vector.to(torch.float32)
    prev_normalized_f32 = prev_normalized.to(torch.float32)

    current_block_values_f32 = current_block_values.to(torch.float32)

    squared_norm_sum = (current_block_values_f32 * current_block_values_f32).sum(dim=-1)

    inverse_rms_norm = torch.rsqrt(
        squared_norm_sum / current_block_values_f32.shape[-1] + EPS
    )

    current_logit = (current_block_values_f32 @ query_vector_f32) * inverse_rms_norm

    merged_max = torch.maximum(prev_lse, current_logit)
    interblock_weight = torch.exp(prev_lse - merged_max)
    intrablock_weight = torch.exp(current_logit - merged_max)

    out = (
        interblock_weight[..., None] * prev_normalized_f32
        + intrablock_weight[..., None] * current_block_values_f32
    ) / (interblock_weight + intrablock_weight)[..., None]

    return out.to(torch.bfloat16)


def torch_compile_phases_forward(inputs, query_w, layers):
    blocks = [inputs]

    for i in range(len(layers)):
        offset = i % BLOCK_SIZE

        if offset == 0:
            values = torch.stack(blocks, dim=0)

            lse, normalized, h = phase_1_fn(query_w[i : i + BLOCK_SIZE], values)
            blocks.append(layers[i](h.to(inputs.dtype)))
        else:
            h = phase_2_fn(
                blocks[-1],
                query_w[i],
                lse[offset],
                normalized[offset],
            )

            blocks[-1] = blocks[-1] + layers[i](h.to(inputs.dtype))

    _, _, h = phase_1_fn(query_w[-1:], torch.stack(blocks, dim=0))
    return h.to(inputs.dtype)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(D, device=DEVICE, dtype=DTYPE, eps=EPS)
        self.linear1 = nn.Linear(D, D * 2, bias=False, device=DEVICE, dtype=DTYPE)
        self.linear2 = nn.Linear(D, D, bias=False, device=DEVICE, dtype=DTYPE)

    def forward(self, x):
        h1, gate = self.linear1(self.norm(x)).chunk(2, dim=-1)
        return self.linear2(F.silu(gate) * h1)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def bench_memory(fn, inputs, pseudo_queries, layers, grad_out):
    targets = grad_targets(inputs, pseudo_queries, layers)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    out = fn(inputs, pseudo_queries, layers)

    fwd_alloc = torch.cuda.max_memory_allocated()
    fwd_reserved = torch.cuda.max_memory_reserved()

    torch.autograd.grad(
        outputs=out,
        inputs=targets,
        grad_outputs=grad_out,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    torch.cuda.synchronize()

    total_alloc = torch.cuda.max_memory_allocated()
    total_reserved = torch.cuda.max_memory_reserved()

    return {
        "fwd_alloc_gib": fwd_alloc / 1024**3,
        "fwd_reserved_gib": fwd_reserved / 1024**3,
        "fwd_bwd_alloc_gib": total_alloc / 1024**3,
        "fwd_bwd_reserved_gib": total_reserved / 1024**3,
    }


def grad_targets(inputs, pseudo_queries, layers):
    params = tuple(p for layer in layers for p in layer.parameters() if p.requires_grad)
    return (inputs, pseudo_queries, *params)


def bench_fwd_bwd(fn, inputs, pseudo_queries, layers, grad_out, warmup=3, runs=10):
    targets = grad_targets(inputs, pseudo_queries, layers)

    for _ in range(warmup):
        out = fn(inputs, pseudo_queries, layers)
        torch.autograd.grad(
            outputs=out,
            inputs=targets,
            grad_outputs=grad_out,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(runs):
        out = fn(inputs, pseudo_queries, layers)
        torch.autograd.grad(
            outputs=out,
            inputs=targets,
            grad_outputs=grad_out,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

    torch.cuda.synchronize()

    return (time.perf_counter() - t0) / runs * 1000


def collect_grads(fn, inputs, pseudo_queries, layers, grad_out):
    targets = grad_targets(inputs, pseudo_queries, layers)

    out = fn(inputs, pseudo_queries, layers)

    grads = torch.autograd.grad(
        outputs=out,
        inputs=targets,
        grad_outputs=grad_out,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )

    grads = [grad.detach().to(torch.float32) for grad in grads]
    return out.detach(), grads


def compare_grads(
    ref_name, ref_fn, test_name, test_fn, inputs, pseudo_queries, layers, grad_out
):
    ref_out, ref_grads = collect_grads(ref_fn, inputs, pseudo_queries, layers, grad_out)
    test_out, test_grads = collect_grads(
        test_fn, inputs, pseudo_queries, layers, grad_out
    )

    out_abs = (ref_out.to(torch.float32) - test_out.to(torch.float32)).abs()
    print(
        f"{test_name} vs {ref_name} output: "
        f"mean_abs={out_abs.mean()}, max_abs={out_abs.max()}"
    )

    for idx, (rg, tg) in enumerate(zip(ref_grads, test_grads)):
        if rg is None or tg is None:
            print(
                f"{test_name} grad[{idx}] vs {ref_name}: "
                f"None mismatch: ref_is_none={rg is None}, test_is_none={tg is None}"
            )
            continue

        diff = (rg - tg).abs()
        rel = diff / (rg.abs() + 1e-3)

        norm_rel = (rg - tg).norm() / (rg.norm() + 1e-12)

        rg_abs_avg = rg.abs().mean()
        tg_abs_avg = tg.abs().mean()

        print(
            f"{test_name} grad[{idx}] vs {ref_name}: "
            f"mean_abs={diff.mean()}, max_abs={diff.max()}, "
            f"mean_rel={rel.mean()}, max_rel={rel.max()}, "
            f"norm_rel={norm_rel}, "
            f"ref_abs_avg={rg_abs_avg}, test_abs_avg={tg_abs_avg}"
        )


def bench_backward_only(
    fn, inputs, pseudo_queries, layers, grad_out, warmup=3, runs=10
):
    targets = grad_targets(inputs, pseudo_queries, layers)

    for _ in range(warmup):
        out = fn(inputs, pseudo_queries, layers)
        torch.cuda.synchronize()

        torch.autograd.grad(
            outputs=out,
            inputs=targets,
            grad_outputs=grad_out,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        torch.cuda.synchronize()

    total = 0.0

    for _ in range(runs):
        out = fn(inputs, pseudo_queries, layers)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        torch.autograd.grad(
            outputs=out,
            inputs=targets,
            grad_outputs=grad_out,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        torch.cuda.synchronize()

        total += time.perf_counter() - t0

    return total / runs * 1000


def print_bench_group(args):
    for name, func in funcs_to_bench:
        fwd_bwd = bench_fwd_bwd(func, *args, grad_out)
        bwd = bench_backward_only(func, *args, grad_out)
        mem = bench_memory(func, *args, grad_out)

        print(f"{name} fwd+bwd:  {fwd_bwd:.3f} ms")
        print(f"{name} bwd-only: {bwd:.3f} ms")
        print(
            f"{name} peak allocated: "
            f"fwd={mem['fwd_alloc_gib']:.3f} GiB, "
            f"fwd+bwd={mem['fwd_bwd_alloc_gib']:.3f} GiB"
        )
        print(
            f"{name} peak reserved:  "
            f"fwd={mem['fwd_reserved_gib']:.3f} GiB, "
            f"fwd+bwd={mem['fwd_bwd_reserved_gib']:.3f} GiB"
        )
    print()


for i in range(5):
    inputs = torch.randn(
        B,
        T,
        D,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )

    layers_swiglu = [SwiGLU() for _ in range(L)]
    layers_identity = [Identity() for _ in range(L)]

    pseudo_queries_randn = torch.randn(
        L + 1,
        D,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    ) / math.sqrt(D)

    grad_out = torch.randn(
        B,
        T,
        D,
        device=DEVICE,
        dtype=DTYPE,
    )

    args_swiglu_randn = (inputs, pseudo_queries_randn, layers_swiglu)
    args_identity_randn = (inputs, pseudo_queries_randn, layers_identity)

    funcs_to_bench = [
        ("torch_compile_phases_forward", torch_compile_phases_forward),
        ("production_forward", production_forward),
        ("production_forward2", production_forward2),
        ("paper_forward", paper_forward),
    ]

    random.shuffle(funcs_to_bench)

    print("identity layers + randn queries")
    print_bench_group(args_identity_randn)

    print("grads check for swiglu layers + randn queries")
    compare_grads(
        "paper_forward",
        paper_forward,
        "production_forward",
        production_forward,
        *args_swiglu_randn,
        grad_out,
    )
    compare_grads(
        "paper_forward",
        paper_forward,
        "torch_compile_phases_forward",
        torch_compile_phases_forward,
        *args_swiglu_randn,
        grad_out,
    )
    compare_grads(
        "paper_forward",
        paper_forward,
        "production_forward2",
        production_forward2,
        *args_swiglu_randn,
        grad_out,
    )
