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

    phase_1_batched_interblock_attention_kernel[(B * T,)](
        block_representations,
        pseudo_queries,
        softmax_outputs,
        lses,
        eps,
        NUM_SOURCE_BLOCKS,
        B * T,
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

    phase_2_online_softmax_merge_intrablock_kernel[(B * T,)](
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
    key=["HIDDEN_DIM"],
)
@triton.jit
def phase_2_online_softmax_merge_intrablock_backward_kernel(
    intrablock_partial_sum_ptr,
    pseudo_query_ptr,
    prev_interblock_normalized_output_ptr,
    prev_interblock_lse_ptr,
    grad_merged_output_ptr,
    grad_merged_lse_ptr,
    grad_intrablock_partial_sum_ptr,
    grad_pseudo_query_ptr,
    grad_prev_interblock_normalized_output_ptr,
    grad_prev_interblock_lse_ptr,
    eps,
    HIDDEN_DIM: tl.constexpr,
):
    batch_seq_idx = tl.program_id(0)
    hidden_dim_range = tl.arange(0, HIDDEN_DIM)

    x = tl.load(
        intrablock_partial_sum_ptr + batch_seq_idx * HIDDEN_DIM + hidden_dim_range
    ).to(tl.float32)

    q = tl.load(
        pseudo_query_ptr + hidden_dim_range,
        eviction_policy="evict_last",
    ).to(tl.float32)

    y0 = tl.load(
        prev_interblock_normalized_output_ptr
        + batch_seq_idx * HIDDEN_DIM
        + hidden_dim_range
    ).to(tl.float32)

    l0 = tl.load(prev_interblock_lse_ptr + batch_seq_idx).to(tl.float32)

    grad_y = tl.load(
        grad_merged_output_ptr + batch_seq_idx * HIDDEN_DIM + hidden_dim_range
    ).to(tl.float32)

    grad_l = tl.load(grad_merged_lse_ptr + batch_seq_idx).to(tl.float32)

    squared_norm_sum = tl.sum(x * x)
    inverse_rms_norm = tl.rsqrt(squared_norm_sum / float(HIDDEN_DIM) + eps)

    dot_xq = tl.sum(x * q)
    l1 = dot_xq * inverse_rms_norm

    merged_max = tl.maximum(l0, l1)
    w0 = tl.exp(l0 - merged_max)
    w1 = tl.exp(l1 - merged_max)
    exp_sum = w0 + w1

    alpha = w0 / exp_sum
    beta = w1 / exp_sum

    grad_y0 = alpha * grad_y
    grad_x_from_value = beta * grad_y

    dot_grad_y_y0_minus_x = tl.sum(grad_y * (y0 - x))

    grad_l0 = alpha * grad_l + alpha * beta * dot_grad_y_y0_minus_x
    grad_l1 = beta * grad_l - alpha * beta * dot_grad_y_y0_minus_x

    inv_rms_cubed = inverse_rms_norm * inverse_rms_norm * inverse_rms_norm

    grad_x_from_logit = grad_l1 * (
        inverse_rms_norm * q - dot_xq * inv_rms_cubed * x / float(HIDDEN_DIM)
    )

    grad_q = grad_l1 * inverse_rms_norm * x
    grad_x = grad_x_from_value + grad_x_from_logit

    tl.atomic_add(
        grad_intrablock_partial_sum_ptr + batch_seq_idx * HIDDEN_DIM + hidden_dim_range,
        grad_x,
        sem="relaxed",
    )

    tl.atomic_add(
        grad_pseudo_query_ptr + hidden_dim_range,
        grad_q,
        sem="relaxed",
    )

    tl.store(
        grad_prev_interblock_normalized_output_ptr
        + batch_seq_idx * HIDDEN_DIM
        + hidden_dim_range,
        grad_y0,
    )

    tl.store(
        grad_prev_interblock_lse_ptr + batch_seq_idx,
        grad_l0,
    )


def phase_2_online_softmax_merge_intrablock_backward(
    intrablock_partial_sum,
    pseudo_query,
    prev_interblock_normalized_output,
    prev_interblock_lse,
    grad_merged_output,
    grad_merged_lse,
    grad_intrablock_partial_sum,
    grad_pseudo_query,
    grad_prev_interblock_normalized_output,
    grad_prev_interblock_lse,
    eps=None,
):
    if eps is None:
        eps = torch.finfo(torch.float32).eps

    if grad_merged_lse is None:
        grad_merged_lse = torch.zeros_like(prev_interblock_lse)

    phase_2_online_softmax_merge_intrablock_backward_kernel[(B * T,)](
        intrablock_partial_sum,
        pseudo_query,
        prev_interblock_normalized_output,
        prev_interblock_lse,
        grad_merged_output,
        grad_merged_lse,
        grad_intrablock_partial_sum,
        grad_pseudo_query,
        grad_prev_interblock_normalized_output,
        grad_prev_interblock_lse,
        eps,
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
