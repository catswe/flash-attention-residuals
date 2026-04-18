import os
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

DEVICE = "cuda"
DTYPE = torch.bfloat16
FORWARD_ONLY = True

BLOCK_SIZE = 8
NUM_LAYERS = 64

B, T, D = 32, 1024, 512


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.RMSNorm(dim, device=DEVICE, dtype=DTYPE)
        self.linear1 = nn.Linear(dim, dim * 2, bias=False, device=DEVICE, dtype=DTYPE)
        self.linear2 = nn.Linear(dim, dim, bias=False, device=DEVICE, dtype=DTYPE)

    def forward(self, x):
        h1, gate = self.linear1(self.norm(x)).chunk(2, dim=-1)
        return self.linear2(F.silu(gate) * h1)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


@torch.compile(mode="max-autotune-no-cudagraphs")
def naive_attention_residual(pseudo_queries, values):
    keys = F.rms_norm(values, (values.shape[-1],))

    logits = torch.einsum("d, n b t d -> n b t", pseudo_queries, keys)
    logits = logits - logits.max(dim=0, keepdim=True).values

    return torch.einsum("n b t, n b t d -> b t d", logits.softmax(0), values).to(DTYPE)


def paper_forward(inputs, pseudo_queries, layers):
    block_representations = torch.zeros(
        math.ceil(len(layers) / BLOCK_SIZE) + 1,
        *inputs.shape,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    curr_block_idx = 0

    block_representations[curr_block_idx] = inputs
    for i in range(len(layers)):
        outputs = naive_attention_residual(
            pseudo_queries[i].to(torch.float32),
            block_representations[: curr_block_idx + 1].to(torch.float32),
        )

        if i % BLOCK_SIZE == 0:
            curr_block_idx += 1

        block_representations[curr_block_idx] += layers[i](outputs)

    return naive_attention_residual(pseudo_queries[-1], block_representations)


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
    interblock_exp_sum_ptr,
    interblock_max_logit_ptr,
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
                interblock_exp_sum_ptr + (layer_offset - 1) * BT + batch_seq_idx,
                exp_sum,
            )
            tl.store(
                interblock_max_logit_ptr + (layer_offset - 1) * BT + batch_seq_idx,
                max_attention_logit,
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
    interblock_exp_sum_ptr,
    interblock_max_logit_ptr,
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

    interblock_max_logit = tl.load(interblock_max_logit_ptr + batch_seq_idx)
    interblock_exp_sum = tl.load(interblock_exp_sum_ptr + batch_seq_idx)
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

    merged_max_logit = tl.maximum(interblock_max_logit, intrablock_logit)
    interblock_rescale_factor = tl.exp(interblock_max_logit - merged_max_logit)
    intrablock_rescale_factor = tl.exp(intrablock_logit - merged_max_logit)

    interblock_weight = interblock_rescale_factor * interblock_exp_sum
    intrablock_weight = intrablock_rescale_factor

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
    interblock_exp_sums = torch.empty(
        (BLOCK_SIZE - 1, B, T), dtype=torch.float32, device=DEVICE
    )
    interblock_max_logits = torch.empty(
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
                interblock_exp_sums,
                interblock_max_logits,
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
                interblock_exp_sums[offset],
                interblock_max_logits[offset],
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
        interblock_exp_sums,
        interblock_max_logits,
        torch.finfo(torch.float32).eps,
        curr_block_idx + 1,
        B * T,
        D,
        1,
        triton.next_power_of_2(curr_block_idx + 1),
    )
    return residual_attention_output


def bench(fn, *args, warmup=5, runs=20):
    for _ in range(warmup):
        fn(*args)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn(*args)
    torch.cuda.synchronize()

    return (time.perf_counter() - t0) / runs * 1000


torch._dynamo.reset()
torch.compiler.reset()
torch._inductor.codecache.FxGraphCache.clear()
torch._inductor.config.force_disable_caches = True

for i in range(10):
    with torch.no_grad():
        inputs = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE)
        layers_swiglu = [SwiGLU(D) for _ in range(NUM_LAYERS)]
        layers_identity = [Identity() for _ in range(NUM_LAYERS)]

        pseudo_queries_zeros = torch.zeros(
            NUM_LAYERS + 1,
            D,
            device=DEVICE,
            dtype=DTYPE,
            requires_grad=not FORWARD_ONLY,
        )
        pseudo_queries_randn = torch.randn(
            NUM_LAYERS + 1,
            D,
            device=DEVICE,
            dtype=DTYPE,
            requires_grad=not FORWARD_ONLY,
        )

        args_identity = (inputs, pseudo_queries_randn, layers_identity)

        args_swiglu_zeros = (inputs, pseudo_queries_zeros, layers_swiglu)
        args_swiglu_randn = (inputs, pseudo_queries_randn, layers_swiglu)

        out_paper_zeros = paper_forward(*args_swiglu_zeros)
        out_paper_randn = paper_forward(*args_swiglu_randn)

        print(f"mean abs randn paper: {out_paper_zeros.abs().mean()}")
        print(f"mean abs zeros paper: {out_paper_randn.abs().mean()}")

        funcs_to_bench = [
            ("production_forward", production_forward),
            ("paper_forward", paper_forward),
        ]
        random.shuffle(funcs_to_bench)

        for name, func in funcs_to_bench:
            print(f"{name}: {bench(func, *args_identity)} ms")

            abs_difference_zeros = (out_paper_zeros - func(*args_swiglu_zeros)).abs()
            abs_difference_randn = (out_paper_randn - func(*args_swiglu_randn)).abs()

            print(f"mean abs difference zeros: {abs_difference_zeros.mean()}")
            print(f"mean abs difference randn: {abs_difference_randn.mean()}")
            print(
                f"mean relative difference: {(abs_difference_zeros / (out_paper_zeros.abs() + 1e-3)).mean()}"
            )
            print(
                f"mean relative difference: {(abs_difference_randn / (out_paper_randn.abs() + 1e-3)).mean()}"
            )
        print()
