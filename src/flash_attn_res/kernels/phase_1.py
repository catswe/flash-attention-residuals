import triton
import triton.language as tl
from .configs import forward_attn_configs, backward_attn_configs


@triton.autotune(
    configs=forward_attn_configs,
    key=["NUM_SOURCE_BLOCKS", "HIDDEN_DIM", "NUM_QUERIES_PER_BLOCK", "PADDED_SRC"],
)
@triton.jit
def phase_1_batched_attention_forward_kernel(
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


@triton.autotune(
    configs=backward_attn_configs,
    key=["NUM_SOURCE_BLOCKS", "HIDDEN_DIM", "NUM_QUERIES_PER_BLOCK", "PADDED_SRC"],
    restore_value=[
        "grad_block_representations_accumulator_ptr",
        "grad_pseudo_queries_partial_ptr",
    ],
)
@triton.jit
def phase_1_batched_attention_backward_kernel(
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
