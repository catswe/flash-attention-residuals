import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc

from flash_attn_res.experimental.autograd import BlockAttentionResiduals
from flash_attn_res.ops.phase_1 import phase_1_batched_attention_triton_op
from flash_attn_res.ops.phase_2 import phase_2_online_softmax_merge_triton_op
from liger_kernel.transformers.functional import liger_attn_res
from torch.utils.checkpoint import checkpoint

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

DEVICE = "cuda"
DTYPE = torch.bfloat16

L = 32
BLOCK_SIZE = 8
NUM_BLOCKS = math.ceil(L / BLOCK_SIZE) + 1

B, T, D = 4, 2048 * 8, 2048
BT = B * T

EPS = torch.finfo(torch.float32).eps

LIGER_W_NORM = torch.ones(D, device=DEVICE, dtype=DTYPE)


def pytorch_attn_res(V, w_query, w_norm, eps=1e-6):
    """
    Reference PyTorch implementation.
    V: [N, B, T, D], w_query: [D], w_norm: [D]
    """
    V_f32 = V.float()
    rms = torch.sqrt(V_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    K = (V_f32 / rms).to(V.dtype) * w_norm

    scores = torch.einsum("d, n b t d -> n b t", w_query.float(), K.float())
    alpha = scores.softmax(dim=0)

    h = torch.einsum("n b t, n b t d -> b t d", alpha, V.float()).to(V.dtype)
    return h


def maybe_ckpt(fn, *args):
    if torch.is_grad_enabled():
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def liger_torch_forward(inputs, pseudo_queries, layers):
    blocks = [inputs]
    w_norm = LIGER_W_NORM.to(device=pseudo_queries.device, dtype=pseudo_queries.dtype)

    for i in range(len(layers)):
        layer = layers[i]

        def step(blocks_t, q, layer=layer):
            outputs = pytorch_attn_res(blocks_t, q, w_norm, eps=EPS)
            return layer(outputs.to(inputs.dtype))

        update = maybe_ckpt(step, torch.stack(blocks, dim=0), pseudo_queries[i])

        if i % BLOCK_SIZE == 0:
            blocks.append(update)
        else:
            blocks[-1] = blocks[-1] + update

    def final_step(blocks_t, q):
        return pytorch_attn_res(blocks_t, q, w_norm, eps=EPS)

    return maybe_ckpt(
        final_step,
        torch.stack(blocks, dim=0),
        pseudo_queries[-1],
    ).to(inputs.dtype)


def liger_kernel_forward(inputs, pseudo_queries, layers):
    blocks = [inputs]
    w_norm = LIGER_W_NORM.to(device=pseudo_queries.device, dtype=pseudo_queries.dtype)

    for i in range(len(layers)):
        layer = layers[i]

        def step(blocks_t, q, layer=layer):
            outputs = liger_attn_res(blocks_t, q, w_norm, eps=EPS)
            return layer(outputs.to(inputs.dtype))

        update = maybe_ckpt(step, torch.stack(blocks, dim=0), pseudo_queries[i])

        if i % BLOCK_SIZE == 0:
            blocks.append(update)
        else:
            blocks[-1] = blocks[-1] + update

    def final_step(blocks_t, q):
        return liger_attn_res(blocks_t, q, w_norm, eps=EPS)

    return maybe_ckpt(
        final_step,
        torch.stack(blocks, dim=0),
        pseudo_queries[-1],
    ).to(inputs.dtype)


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


def torch_compile_phases_forward(
    inputs, query_w, layers, checkpoint_blocks: bool = True
):
    blocks = [inputs]
    input_dtype = inputs.dtype

    for block_start in range(0, len(layers), BLOCK_SIZE):
        num_queries = min(BLOCK_SIZE, len(layers) - block_start)
        query_block = query_w[block_start : block_start + num_queries]

        def run_block(query_block_arg, *prev_blocks, block_start=block_start):
            values = torch.stack(prev_blocks, dim=0)

            lse, normalized, h = phase_1_fn(query_block_arg, values)
            curr_block = layers[block_start](h.to(input_dtype))

            for offset in range(1, num_queries):
                layer_idx = block_start + offset

                h = phase_2_fn(
                    curr_block,
                    query_block_arg[offset],
                    lse[offset],
                    normalized[offset],
                )

                curr_block = curr_block + layers[layer_idx](h.to(input_dtype))

            return curr_block

        if checkpoint_blocks and torch.is_grad_enabled():
            curr_block = checkpoint(
                run_block,
                query_block,
                *blocks,
                use_reentrant=False,
            )
        else:
            curr_block = run_block(query_block, *blocks)

        blocks.append(curr_block)

    def run_final(final_query, *block_args):
        _, _, h = phase_1_fn(final_query, torch.stack(block_args, dim=0))
        return h.to(input_dtype)

    if checkpoint_blocks and torch.is_grad_enabled():
        return checkpoint(
            run_final,
            query_w[-1:],
            *blocks,
            use_reentrant=False,
        )

    return run_final(query_w[-1:], *blocks)


def production_forward(inputs, pseudo_queries, layers, eps=None):
    if eps is None:
        eps = EPS

    flat_layer_params = tuple(p for layer in layers for p in layer.parameters())

    return BlockAttentionResiduals.apply(
        inputs,
        pseudo_queries,
        layers,
        BLOCK_SIZE,
        eps,
        *flat_layer_params,
    )


def _blockwise_attention_block_forward(
    block_start: int,
    block_size: int,
    layers,
    pseudo_queries: torch.Tensor,
    eps: float,
    *prev_blocks: torch.Tensor,
) -> torch.Tensor:
    num_queries = min(block_size, len(layers) - block_start)

    values = torch.stack(prev_blocks, dim=0)

    phase1_out, phase1_lse = phase_1_batched_attention_triton_op(
        values,
        pseudo_queries[block_start : block_start + num_queries],
        eps,
    )

    curr_block = None

    for query_offset in range(num_queries):
        layer_idx = block_start + query_offset

        if query_offset == 0:
            layer_input = phase1_out[query_offset]
            curr_block = layers[layer_idx](layer_input)
        else:
            layer_input = phase_2_online_softmax_merge_triton_op(
                curr_block,
                pseudo_queries[layer_idx],
                phase1_out[query_offset],
                phase1_lse[query_offset],
                eps,
            )
            curr_block = curr_block + layers[layer_idx](layer_input)

    return curr_block


def production_forward2(
    inputs: torch.Tensor,
    pseudo_queries: torch.Tensor,
    layers,
    eps: float | None = None,
    block_size: int = BLOCK_SIZE,
    checkpoint_blocks: bool = True,
) -> torch.Tensor:
    if eps is None:
        eps = EPS

    blocks = [inputs]

    for block_start in range(0, len(layers), block_size):

        if checkpoint_blocks:

            def run_block(pseudo_queries_arg, *prev_blocks, block_start=block_start):
                return _blockwise_attention_block_forward(
                    block_start,
                    block_size,
                    layers,
                    pseudo_queries_arg,
                    eps,
                    *prev_blocks,
                )

            curr_block = checkpoint(
                run_block,
                pseudo_queries,
                *blocks,
                use_reentrant=False,
            )
        else:
            curr_block = _blockwise_attention_block_forward(
                block_start,
                block_size,
                layers,
                pseudo_queries,
                eps,
                *blocks,
            )

        blocks.append(curr_block)

    final_out, _final_lse = phase_1_batched_attention_triton_op(
        torch.stack(blocks, dim=0),
        pseudo_queries[-1:],
        eps,
    )

    return final_out[0].to(inputs.dtype)


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

        rg_f = rg.detach().float()
        tg_f = tg.detach().float()

        delta = rg_f - tg_f
        diff = delta.abs()
        rel = diff / (rg_f.abs() + 1e-3)

        rg_norm = rg_f.norm()
        tg_norm = tg_f.norm()

        norm_rel = delta.norm() / (rg_norm + 1e-12)
        norm_ratio = tg_norm / (rg_norm + 1e-12)

        if rg_norm > 0 and tg_norm > 0:
            cos = torch.nn.functional.cosine_similarity(
                rg_f.flatten(),
                tg_f.flatten(),
                dim=0,
            )
        else:
            cos = torch.tensor(float("nan"), device=rg_f.device)

        rg_abs_avg = rg_f.abs().mean()
        tg_abs_avg = tg_f.abs().mean()

        print(
            f"{test_name} grad[{idx}] vs {ref_name}: "
            f"mean_abs={diff.mean()}, max_abs={diff.max()}, "
            f"mean_rel={rel.mean()}, max_rel={rel.max()}, "
            f"norm_rel={norm_rel}, norm_ratio={norm_ratio}, cos={cos}, "
            f"ref_abs_avg={rg_abs_avg}, test_abs_avg={tg_abs_avg}"
        )


def bench_forward_inference(fn, inputs, pseudo_queries, layers, warmup=10, runs=50):
    with torch.inference_mode():
        for _ in range(warmup):
            fn(inputs, pseudo_queries, layers)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(runs):
            fn(inputs, pseudo_queries, layers)
        end.record()

        torch.cuda.synchronize()

    return start.elapsed_time(end) / runs


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


def print_bench(
    funcs_to_bench, args_identity_randn, args_swiglu_randn, out_paper_randn
):
    for name, func in funcs_to_bench:
        fwd_bwd = bench_fwd_bwd(func, *args_identity_randn, grad_out)
        fwd = bench_forward_inference(func, *args_identity_randn)
        bwd = bench_backward_only(func, *args_identity_randn, grad_out)
        mem = bench_memory(func, *args_identity_randn, grad_out)

        print(f"{name} fwd+bwd:  {fwd_bwd:.3f} ms")
        print(f"{name} fwd-only: {fwd:.3f} ms")
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

        with torch.inference_mode():
            test_out = func(*args_swiglu_randn)
            abs_difference_randn = (out_paper_randn - test_out).abs()

            mean_abs = abs_difference_randn.mean().item()
            mean_rel = (
                (abs_difference_randn / (out_paper_randn.abs() + 1e-3)).mean().item()
            )

        print(f"mean abs difference randn: {mean_abs}")
        print(f"mean relative difference randn: {mean_rel}")

        del test_out, abs_difference_randn
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    print()


for i in range(2):
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
        ("liger_torch_forward", liger_torch_forward),
        ("torch_compile_phases_forward", torch_compile_phases_forward),
        ("liger_kernel_forward", liger_kernel_forward),
        ("production_forward", production_forward),
    ]
    random.shuffle(funcs_to_bench)

    print("identity layers + randn queries")
    with torch.no_grad():
        out_paper_randn = liger_torch_forward(*args_swiglu_randn).detach()
    print(f"mean abs randn paper: {out_paper_randn.abs().mean()}")
    print_bench(funcs_to_bench, args_identity_randn, args_swiglu_randn, out_paper_randn)

    print("grads check for swiglu layers + randn queries")
    compare_grads(
        "liger_torch_forward",
        liger_torch_forward,
        "production_forward",
        production_forward,
        *args_swiglu_randn,
        grad_out,
    )
    compare_grads(
        "liger_torch_forward",
        liger_torch_forward,
        "liger_kernel_forward",
        liger_kernel_forward,
        *args_swiglu_randn,
        grad_out,
    )
    compare_grads(
        "liger_torch_forward",
        liger_torch_forward,
        "torch_compile_phases_forward",
        torch_compile_phases_forward,
        *args_swiglu_randn,
        grad_out,
    )


SUBLAYER_BLOCK_SIZE = 2 * BLOCK_SIZE


class AttnSublayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(D, device=DEVICE, dtype=DTYPE, eps=EPS)
        self.linear = nn.Linear(D, D, bias=False, device=DEVICE, dtype=DTYPE)

    def forward(self, x):
        return self.linear(self.norm(x))


def liger_torch_forward_sublayer(inputs, pseudo_queries, sublayers):
    blocks = [inputs]
    w_norm = LIGER_W_NORM.to(device=pseudo_queries.device, dtype=pseudo_queries.dtype)

    for i in range(len(sublayers)):
        layer = sublayers[i]

        def step(blocks_t, q, layer=layer):
            outputs = pytorch_attn_res(blocks_t, q, w_norm, eps=EPS)
            return layer(outputs.to(inputs.dtype))

        update = maybe_ckpt(step, torch.stack(blocks, dim=0), pseudo_queries[i])

        if i % SUBLAYER_BLOCK_SIZE == 0:
            blocks.append(update)
        else:
            blocks[-1] = blocks[-1] + update

    def final_step(blocks_t, q):
        return pytorch_attn_res(blocks_t, q, w_norm, eps=EPS)

    return maybe_ckpt(
        final_step,
        torch.stack(blocks, dim=0),
        pseudo_queries[-1],
    ).to(inputs.dtype)


def production_forward_sublayer(inputs, pseudo_queries, sublayers, eps=None):
    if eps is None:
        eps = EPS

    flat_layer_params = tuple(p for layer in sublayers for p in layer.parameters())

    return BlockAttentionResiduals.apply(
        inputs,
        pseudo_queries,
        sublayers,
        SUBLAYER_BLOCK_SIZE,
        eps,
        *flat_layer_params,
    )


L_SUBLAYER = 2 * L

print("\n" + "=" * 60)
print(
    f"PER-SUBLAYER ROUTING (Paper Figure 2): "
    f"{L} transformer layers, {L_SUBLAYER} sublayers, "
    f"block_size={SUBLAYER_BLOCK_SIZE}"
)
print("=" * 60)

for i in range(2):
    inputs_sl = torch.randn(
        B, T, D, device=DEVICE, dtype=DTYPE, requires_grad=True,
    )

    attn_layers = [AttnSublayer() for _ in range(L)]
    mlp_layers = [SwiGLU() for _ in range(L)]

    sublayers_interleaved = []
    for a, m in zip(attn_layers, mlp_layers):
        sublayers_interleaved.append(a)
        sublayers_interleaved.append(m)

    pseudo_queries_sl = torch.randn(
        L_SUBLAYER + 1, D, device=DEVICE, dtype=DTYPE, requires_grad=True,
    ) / math.sqrt(D)

    grad_out_sl = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE)

    args_sl = (inputs_sl, pseudo_queries_sl, sublayers_interleaved)

    print(f"\nIteration {i}")

    compare_grads(
        "liger_torch_sublayer",
        liger_torch_forward_sublayer,
        "production_sublayer",
        production_forward_sublayer,
        *args_sl,
        grad_out_sl,
    )

    fwd_bwd = bench_fwd_bwd(
        production_forward_sublayer, *args_sl, grad_out_sl,
    )
    fwd = bench_forward_inference(
        production_forward_sublayer, *args_sl,
    )
    bwd = bench_backward_only(
        production_forward_sublayer, *args_sl, grad_out_sl,
    )
    mem = bench_memory(
        production_forward_sublayer, *args_sl, grad_out_sl,
    )

    print(f"production_sublayer fwd+bwd:  {fwd_bwd:.3f} ms")
    print(f"production_sublayer fwd-only: {fwd:.3f} ms")
    print(f"production_sublayer bwd-only: {bwd:.3f} ms")
    print(
        f"production_sublayer peak allocated: "
        f"fwd={mem['fwd_alloc_gib']:.3f} GiB, "
        f"fwd+bwd={mem['fwd_bwd_alloc_gib']:.3f} GiB"
    )

    del inputs_sl, pseudo_queries_sl, grad_out_sl
    del attn_layers, mlp_layers, sublayers_interleaved
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
