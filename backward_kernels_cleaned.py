import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

DEVICE = "cuda"
DTYPE = torch.bfloat16

L = 64
BLOCK_SIZE = 8
NUM_BLOCKS = math.ceil(L / BLOCK_SIZE) + 1

B, T, D = 32, 1024, 512
BT = BT


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


def phase_1_batched_interblock_attention(
    block_representations,
    pseudo_queries,
    softmax_outputs,
    lses,
    eps=None,
):
    NUM_QUERIES = pseudo_queries.shape[0]
    NUM_SOURCE_BLOCKS = block_representations.shape[0]

    if eps is None:
        eps = torch.finfo(torch.float32).eps

    phase_1_batched_interblock_attention_kernel[(BT,)](
        block_representations,
        pseudo_queries,
        softmax_outputs,
        lses,
        eps,
        NUM_SOURCE_BLOCKS,
        BT,
        D,
        NUM_QUERIES,
        triton.next_power_of_2(NUM_SOURCE_BLOCKS),
    )


@triton.autotune(
    configs=autotune_configs,
    key=["HIDDEN_DIM"],
    restore_value=[
        "interblock_normalized_output_ptr",
        "interblock_lse_ptr",
    ],
)
@triton.jit
def phase_2_online_softmax_merge_intrablock_kernel(
    intrablock_partial_sum_ptr,
    pseudo_query_ptr,
    interblock_normalized_output_ptr,
    interblock_lse_ptr,
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
        interblock_normalized_output_ptr
        + batch_seq_idx * HIDDEN_DIM
        + hidden_dim_range,
        merged_output.to(tl.bfloat16),
    )
    tl.store(
        interblock_lse_ptr + batch_seq_idx,
        merged_max + tl.log(exp_sum),
    )


def phase_2_online_softmax_merge_intrablock(
    intrablock_partial_sum,
    pseudo_query,
    interblock_normalized_output,
    interblock_lse,
    eps=None,
):
    if eps is None:
        eps = torch.finfo(torch.float32).eps

    phase_2_online_softmax_merge_intrablock_kernel[(BT,)](
        intrablock_partial_sum,
        pseudo_query,
        interblock_normalized_output,
        interblock_lse,
        eps,
        D,
    )


def production_forward(inputs, pseudo_queries, layers):
    block_representations = torch.zeros(
        NUM_BLOCKS,
        B,
        T,
        D,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    block_representations[0] = inputs

    attn_out = torch.empty(
        (L + 1, B, T, D),
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    attn_lse = torch.empty(
        (L + 1, B, T),
        dtype=torch.float32,
        device=DEVICE,
    )

    for i in range(L):
        curr_block_idx = i // BLOCK_SIZE + 1

        if i % BLOCK_SIZE == 0:
            num_queries = min(BLOCK_SIZE, L - i)

            phase_1_batched_interblock_attention(
                block_representations[:curr_block_idx],
                pseudo_queries[i : i + num_queries],
                attn_out[i : i + num_queries],
                attn_lse[i : i + num_queries],
            )
        else:
            phase_2_online_softmax_merge_intrablock(
                block_representations[curr_block_idx],
                pseudo_queries[i],
                attn_out[i],
                attn_lse[i],
            )

        block_representations[curr_block_idx] += layers[i](attn_out[i])

    phase_1_batched_interblock_attention(
        block_representations,
        pseudo_queries[-1:],
        attn_out[-1:],
        attn_lse[-1:],
    )
    return attn_out[-1]


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(D, device=DEVICE, dtype=DTYPE)
        self.linear1 = nn.Linear(D, D * 2, bias=False, device=DEVICE, dtype=DTYPE)
        self.linear2 = nn.Linear(D, D, bias=False, device=DEVICE, dtype=DTYPE)

    def forward(self, x):
        h1, gate = self.linear1(self.norm(x)).chunk(2, dim=-1)
        return self.linear2(F.silu(gate) * h1)


inputs = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE)
pseudo_queries = torch.randn(
    L + 1,
    D,
    device=DEVICE,
    dtype=DTYPE,
    requires_grad=True,
)
layers_swiglu = [SwiGLU() for _ in range(L)]
out = production_forward(inputs, pseudo_queries, layers_swiglu)


@triton.autotune(
    configs=autotune_configs,
    key=["NUM_SOURCE_BLOCKS", "HIDDEN_DIM", "NUM_QUERIES_PER_BLOCK", "PADDED_SRC"],
    restore_value=[
        "grad_block_representations_accumulator_ptr",
        "grad_pseudo_queries_accumulator_ptr",
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
    grad_pseudo_queries_accumulator_ptr,
    eps,
    NUM_SOURCE_BLOCKS: tl.constexpr,
    BT: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    NUM_QUERIES_PER_BLOCK: tl.constexpr,
    PADDED_SRC: tl.constexpr,
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

        grad_logsumexp = tl.load(grad_lse_ptr + layer_offset * BT + batch_seq_idx).to(
            tl.float32
        )

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

        grad_source_block_values = tl.where(
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

        tl.atomic_add(
            grad_block_representations_accumulator_ptr
            + source_block_range * (BT * HIDDEN_DIM)
            + batch_seq_idx * HIDDEN_DIM
            + hidden_dim_range,
            grad_source_block_values,
            mask=valid_block_mask_2d,
            sem="relaxed",
        )

        tl.atomic_add(
            grad_pseudo_queries_accumulator_ptr
            + layer_offset * HIDDEN_DIM
            + hidden_dim_range_1d,
            grad_pseudo_query,
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
    eps=None,
):
    NUM_QUERIES = pseudo_queries.shape[0]
    NUM_SOURCE_BLOCKS = block_representations.shape[0]

    if eps is None:
        eps = torch.finfo(torch.float32).eps

    if grad_lses is None:
        grad_lses = torch.zeros_like(lses)

    phase_1_batched_interblock_attention_backward_kernel[(BT,)](
        block_representations,
        pseudo_queries,
        lses,
        grad_softmax_outputs,
        grad_lses,
        grad_block_representations,
        grad_pseudo_queries,
        eps,
        NUM_SOURCE_BLOCKS,
        BT,
        D,
        NUM_QUERIES,
        triton.next_power_of_2(NUM_SOURCE_BLOCKS),
    )


@triton.autotune(
    configs=autotune_configs,
    key=["HIDDEN_DIM"],
    restore_value=[
        "grad_intrablock_partial_sum_accumulator_ptr",
        "grad_phase1_interblock_normalized_output_ptr",
        "grad_phase1_interblock_logsumexp_ptr",
    ],
)
@triton.jit
def phase_2_online_softmax_merge_intrablock_backward_kernel(
    intrablock_partial_sum_ptr,
    pseudo_query_ptr,
    phase1_interblock_normalized_output_ptr,
    phase1_interblock_logsumexp_ptr,
    grad_merged_attention_output_ptr,
    grad_merged_logsumexp_ptr,
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

    grad_merged_logsumexp = tl.load(grad_merged_logsumexp_ptr + batch_seq_idx).to(
        tl.float32
    )

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

    grad_phase1_interblock_logsumexp = (
        phase1_merge_probability * grad_merged_logsumexp
        + phase1_merge_probability
        * phase2_merge_probability
        * grad_output_dot_interblock_minus_intrablock
    )
    grad_phase2_intrablock_logit = (
        phase2_merge_probability * grad_merged_logsumexp
        - phase1_merge_probability
        * phase2_merge_probability
        * grad_output_dot_interblock_minus_intrablock
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

    tl.atomic_add(
        grad_intrablock_partial_sum_accumulator_ptr
        + batch_seq_idx * HIDDEN_DIM
        + hidden_dim_range,
        grad_intrablock_partial_sum,
        sem="relaxed",
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


phase_2_reduce_configs = [
    triton.Config(
        {
            "BLOCK_BATCH_SEQ": block_batch_seq,
            "BLOCK_HIDDEN": block_hidden,
        },
        num_warps=num_warps,
        num_stages=1,
    )
    for block_batch_seq in [64, 128, 256]
    for block_hidden in [16, 32]
    for num_warps in [4, 8]
]


@triton.autotune(
    configs=phase_2_reduce_configs,
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
    grad_merged_logsumexp,
    grad_intrablock_partial_sum,
    grad_pseudo_query,
    grad_phase1_interblock_normalized_output,
    grad_phase1_interblock_logsumexp,
    eps=None,
):
    if eps is None:
        eps = torch.finfo(torch.float32).eps

    if grad_merged_logsumexp is None:
        grad_merged_logsumexp = torch.zeros_like(phase1_interblock_logsumexp)

    grad_pseudo_query_partial = torch.empty(
        (BT, D),
        device=intrablock_partial_sum.device,
        dtype=torch.float32,
    )

    phase_2_online_softmax_merge_intrablock_backward_kernel[(BT,)](
        intrablock_partial_sum,
        pseudo_query,
        phase1_interblock_normalized_output,
        phase1_interblock_logsumexp,
        grad_merged_attention_output,
        grad_merged_logsumexp,
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
        block_representations = torch.zeros(
            NUM_BLOCKS,
            B,
            T,
            D,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        block_representations[0].copy_(inputs)

        attn_lse = torch.empty(
            L + 1,
            B,
            T,
            device=inputs.device,
            dtype=torch.float32,
        )

        for block_start in range(0, L, BLOCK_SIZE):
            curr_block_idx = block_start // BLOCK_SIZE + 1
            num_queries = min(BLOCK_SIZE, L - block_start)

            block_attn_out = torch.empty(
                num_queries,
                B,
                T,
                D,
                device=inputs.device,
                dtype=torch.bfloat16,
            )

            phase_1_batched_interblock_attention(
                block_representations[:curr_block_idx],
                pseudo_queries[block_start : block_start + num_queries],
                block_attn_out,
                attn_lse[block_start : block_start + num_queries],
                eps=eps,
            )

            for query_offset in range(num_queries):
                i = block_start + query_offset

                if query_offset != 0:
                    phase_2_online_softmax_merge_intrablock(
                        block_representations[curr_block_idx],
                        pseudo_queries[i],
                        block_attn_out[query_offset],
                        attn_lse[i],
                        eps=eps,
                    )

                block_representations[curr_block_idx].add_(
                    layers[i](block_attn_out[query_offset])
                )

        final_out = torch.empty(
            B,
            T,
            D,
            device=inputs.device,
            dtype=inputs.dtype,
        )

        phase_1_batched_interblock_attention(
            block_representations,
            pseudo_queries[-1:],
            final_out.unsqueeze(0),
            attn_lse[-1:],
            eps=eps,
        )

        ctx.save_for_backward(
            block_representations,
            attn_lse,
            pseudo_queries,
            *flat_layer_params,
        )
        ctx.layers = layers
        ctx.eps = eps
        ctx.num_layer_params = len(flat_layer_params)

        return final_out

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]


def production_forward_custom_autograd(inputs, pseudo_queries, layers, eps=None):
    if eps is None:
        eps = torch.finfo(torch.float32).eps

    flat_layer_params = tuple(p for layer in layers for p in layer.parameters())

    return BlockwiseAttentionFunction.apply(
        inputs,
        pseudo_queries,
        layers,
        eps,
        *flat_layer_params,
    )
