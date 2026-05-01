import triton
import triton.language as tl
from .configs import (
    forward_configs,
    phase2_backward_configs,
)


@triton.autotune(
    configs=forward_configs,
    key=["NUM_SOURCE_BLOCKS", "HIDDEN_DIM", "NUM_QUERIES_PER_BLOCK", "PADDED_SRC"],
)
@triton.jit
def phase_2_online_softmax_merge_forward_kernel(
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
    intrablock_prob = tl.sigmoid(intrablock_logit - interblock_lse)

    merged_output = interblock_normalized_output + intrablock_prob * (
        intrablock_partial_sum - interblock_normalized_output
    )

    tl.store(
        merged_output_ptr + batch_seq_idx * HIDDEN_DIM + hidden_dim_range,
        merged_output.to(tl.bfloat16),
    )


@triton.autotune(
    configs=phase2_backward_configs,
    key=["HIDDEN_DIM"],
    restore_value=[
        "grad_intrablock_partial_sum_accumulator_ptr",
        "grad_pseudo_query_accumulator_ptr",
    ],
)
@triton.jit
def phase_2_online_softmax_merge_backward_kernel(
    intrablock_partial_sum_ptr,
    pseudo_query_ptr,
    phase1_interblock_normalized_output_ptr,
    phase1_interblock_logsumexp_ptr,
    grad_merged_attention_output_ptr,
    grad_intrablock_partial_sum_accumulator_ptr,
    grad_pseudo_query_accumulator_ptr,
    grad_phase1_interblock_normalized_output_ptr,
    grad_phase1_interblock_logsumexp_ptr,
    eps,
    BT: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_BT: tl.constexpr,
):
    bt_block_idx = tl.program_id(0)

    bt_offsets = bt_block_idx * BLOCK_BT + tl.arange(0, BLOCK_BT)
    hidden_offsets = tl.arange(0, HIDDEN_DIM)

    valid_bt = bt_offsets < BT
    mask_2d = valid_bt[:, None]

    offsets_2d = bt_offsets[:, None] * HIDDEN_DIM + hidden_offsets[None, :]

    intrablock_partial_sum = tl.load(
        intrablock_partial_sum_ptr + offsets_2d,
        mask=mask_2d,
        other=0.0,
    ).to(tl.float32)

    pseudo_query = tl.load(
        pseudo_query_ptr + hidden_offsets,
        eviction_policy="evict_last",
    ).to(tl.float32)

    phase1_interblock_normalized_output = tl.load(
        phase1_interblock_normalized_output_ptr + offsets_2d,
        mask=mask_2d,
        other=0.0,
    ).to(tl.float32)

    phase1_interblock_logsumexp = tl.load(
        phase1_interblock_logsumexp_ptr + bt_offsets,
        mask=valid_bt,
        other=float("-inf"),
    ).to(tl.float32)

    grad_merged_attention_output = tl.load(
        grad_merged_attention_output_ptr + offsets_2d,
        mask=mask_2d,
        other=0.0,
    ).to(tl.float32)

    intrablock_partial_sum_squared_norm = tl.sum(
        intrablock_partial_sum * intrablock_partial_sum,
        axis=1,
    )

    intrablock_inverse_rms_norm = tl.rsqrt(
        intrablock_partial_sum_squared_norm / float(HIDDEN_DIM) + eps
    )

    pseudo_query_intrablock_dot = tl.sum(
        intrablock_partial_sum * pseudo_query[None, :],
        axis=1,
    )

    phase2_intrablock_logit = pseudo_query_intrablock_dot * intrablock_inverse_rms_norm

    phase2_merge_probability = tl.sigmoid(
        phase2_intrablock_logit - phase1_interblock_logsumexp
    )
    phase1_merge_probability = 1.0 - phase2_merge_probability

    grad_phase1_interblock_normalized_output = (
        phase1_merge_probability[:, None] * grad_merged_attention_output
    )

    grad_intrablock_partial_sum_from_value_path = (
        phase2_merge_probability[:, None] * grad_merged_attention_output
    )

    grad_output_dot_interblock_minus_intrablock = tl.sum(
        grad_merged_attention_output
        * (phase1_interblock_normalized_output - intrablock_partial_sum),
        axis=1,
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

    grad_intrablock_partial_sum_from_logit_path = grad_phase2_intrablock_logit[
        :, None
    ] * (
        intrablock_inverse_rms_norm[:, None] * pseudo_query[None, :]
        - pseudo_query_intrablock_dot[:, None]
        * intrablock_inverse_rms_norm_cubed[:, None]
        * intrablock_partial_sum
        / float(HIDDEN_DIM)
    )

    grad_pseudo_query_per_row = (
        grad_phase2_intrablock_logit[:, None]
        * intrablock_inverse_rms_norm[:, None]
        * intrablock_partial_sum
    )

    grad_pseudo_query_tile = tl.sum(
        tl.where(mask_2d, grad_pseudo_query_per_row, 0.0),
        axis=0,
    )

    grad_intrablock_partial_sum = (
        grad_intrablock_partial_sum_from_value_path
        + grad_intrablock_partial_sum_from_logit_path
    )

    grad_intrablock_ptr = grad_intrablock_partial_sum_accumulator_ptr + offsets_2d

    prev_grad_intrablock = tl.load(
        grad_intrablock_ptr,
        mask=mask_2d,
        other=0.0,
    ).to(tl.float32)

    tl.store(
        grad_intrablock_ptr,
        prev_grad_intrablock + grad_intrablock_partial_sum,
        mask=mask_2d,
    )

    tl.store(
        grad_phase1_interblock_normalized_output_ptr + offsets_2d,
        grad_phase1_interblock_normalized_output,
        mask=mask_2d,
    )

    tl.store(
        grad_phase1_interblock_logsumexp_ptr + bt_offsets,
        grad_phase1_interblock_logsumexp,
        mask=valid_bt,
    )

    tl.atomic_add(
        grad_pseudo_query_accumulator_ptr + hidden_offsets,
        grad_pseudo_query_tile,
        sem="relaxed",
    )
