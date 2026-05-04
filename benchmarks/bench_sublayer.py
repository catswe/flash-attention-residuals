import math
import time
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flash_attn_res.experimental.autograd import BlockAttentionResiduals
from flash_attn_res.ops.phase_1 import phase_1_batched_attention_triton_op
from flash_attn_res.ops.phase_2 import phase_2_online_softmax_merge_triton_op

DEVICE = "cuda"
DTYPE = torch.bfloat16

L = 32
BLOCK_SIZE = 8
SUBLAYER_BLOCK_SIZE = 2 * BLOCK_SIZE

B, T, D = 4, 2048 * 8, 2048

EPS = torch.finfo(torch.float32).eps


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(D, device=DEVICE, dtype=DTYPE, eps=EPS)
        self.linear1 = nn.Linear(D, D * 2, bias=False, device=DEVICE, dtype=DTYPE)
        self.linear2 = nn.Linear(D, D, bias=False, device=DEVICE, dtype=DTYPE)

    def forward(self, x):
        h1, gate = self.linear1(self.norm(x)).chunk(2, dim=-1)
        return self.linear2(F.silu(gate) * h1)


class AttnSublayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(D, device=DEVICE, dtype=DTYPE, eps=EPS)
        self.linear = nn.Linear(D, D, bias=False, device=DEVICE, dtype=DTYPE)

    def forward(self, x):
        return self.linear(self.norm(x))


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def pytorch_attn_res(V, w_query, eps=EPS):
    V_f32 = V.float()
    rms = torch.sqrt(V_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    K = (V_f32 / rms).to(V.dtype)
    scores = torch.einsum("d, n b t d -> n b t", w_query.float(), K.float())
    alpha = scores.softmax(dim=0)
    return torch.einsum("n b t, n b t d -> b t d", alpha, V.float()).to(V.dtype)


def maybe_ckpt(fn, *args):
    if torch.is_grad_enabled():
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def reference_forward(inputs, pseudo_queries, layers, block_size):
    blocks = [inputs]
    for i in range(len(layers)):
        layer = layers[i]

        def step(blocks_t, q, layer=layer):
            outputs = pytorch_attn_res(blocks_t, q)
            return layer(outputs.to(inputs.dtype))

        update = maybe_ckpt(step, torch.stack(blocks, dim=0), pseudo_queries[i])
        if i % block_size == 0:
            blocks.append(update)
        else:
            blocks[-1] = blocks[-1] + update

    def final_step(blocks_t, q):
        return pytorch_attn_res(blocks_t, q)

    return maybe_ckpt(
        final_step, torch.stack(blocks, dim=0), pseudo_queries[-1],
    ).to(inputs.dtype)


def production_1x(inputs, pseudo_queries, layers):
    flat = tuple(p for layer in layers for p in layer.parameters())
    return BlockAttentionResiduals.apply(
        inputs, pseudo_queries, layers, BLOCK_SIZE, EPS, *flat,
    )


def production_2x(inputs, pseudo_queries, sublayers):
    flat = tuple(p for layer in sublayers for p in layer.parameters())
    return BlockAttentionResiduals.apply(
        inputs, pseudo_queries, sublayers, SUBLAYER_BLOCK_SIZE, EPS, *flat,
    )


def grad_targets(inputs, pseudo_queries, layers):
    params = tuple(p for layer in layers for p in layer.parameters() if p.requires_grad)
    return (inputs, pseudo_queries, *params)


def bench_fwd_bwd(fn, inputs, pseudo_queries, layers, grad_out, warmup=3, runs=10):
    targets = grad_targets(inputs, pseudo_queries, layers)
    for _ in range(warmup):
        out = fn(inputs, pseudo_queries, layers)
        torch.autograd.grad(out, targets, grad_out, retain_graph=False, allow_unused=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        out = fn(inputs, pseudo_queries, layers)
        torch.autograd.grad(out, targets, grad_out, retain_graph=False, allow_unused=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


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


def bench_memory(fn, inputs, pseudo_queries, layers, grad_out):
    targets = grad_targets(inputs, pseudo_queries, layers)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    out = fn(inputs, pseudo_queries, layers)
    fwd_alloc = torch.cuda.max_memory_allocated()
    torch.autograd.grad(out, targets, grad_out, retain_graph=False, allow_unused=True)
    torch.cuda.synchronize()
    total_alloc = torch.cuda.max_memory_allocated()
    return fwd_alloc / 1024**3, total_alloc / 1024**3


def collect_grads(fn, inputs, pseudo_queries, layers, grad_out):
    targets = grad_targets(inputs, pseudo_queries, layers)
    out = fn(inputs, pseudo_queries, layers)
    grads = torch.autograd.grad(out, targets, grad_out, allow_unused=False)
    return out.detach(), [g.detach().float() for g in grads]


def check_correctness(ref_name, ref_fn, test_name, test_fn, inputs, pq, layers, grad_out):
    ref_out, ref_grads = collect_grads(ref_fn, inputs, pq, layers, grad_out)
    test_out, test_grads = collect_grads(test_fn, inputs, pq, layers, grad_out)

    out_abs = (ref_out.float() - test_out.float()).abs()
    print(f"  output: mean_abs={out_abs.mean():.2e}, max_abs={out_abs.max():.2e}")

    max_cos = 1.0
    min_cos = 1.0
    for i, (rg, tg) in enumerate(zip(ref_grads, test_grads)):
        cos = F.cosine_similarity(rg.flatten(), tg.flatten(), dim=0).item()
        min_cos = min(min_cos, cos)
        max_cos = max(max_cos, cos)
    print(f"  grads: min_cos={min_cos:.6f}, n_grads={len(ref_grads)}")
    return min_cos > 0.999


def reset_cuda():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def run_perf(name, fn, inputs, pq, layers, grad_out):
    fwd_bwd = bench_fwd_bwd(fn, inputs, pq, layers, grad_out)
    fwd = bench_forward_inference(fn, inputs, pq, layers)
    fwd_mem, fwd_bwd_mem = bench_memory(fn, inputs, pq, layers, grad_out)
    reset_cuda()
    return {"name": name, "fwd_bwd": fwd_bwd, "fwd": fwd, "fwd_mem": fwd_mem, "fwd_bwd_mem": fwd_bwd_mem}


def print_table(results):
    print(f"  {'':30s} {'fwd+bwd':>10s} {'fwd':>10s} {'mem fwd':>10s} {'mem f+b':>10s}")
    for r in results:
        print(
            f"  {r['name']:30s} "
            f"{r['fwd_bwd']:>8.1f}ms "
            f"{r['fwd']:>8.1f}ms "
            f"{r['fwd_mem']:>8.2f}GiB "
            f"{r['fwd_bwd_mem']:>8.2f}GiB"
        )


gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name}")
print(f"Config: L={L}, BLOCK_SIZE={BLOCK_SIZE}, B={B}, T={T}, D={D}")
print(f"Sublayer: L_sublayer={2*L}, SUBLAYER_BLOCK_SIZE={SUBLAYER_BLOCK_SIZE}")

for iteration in range(2):
    print(f"\n{'='*60}")
    print(f"Iteration {iteration}")
    print(f"{'='*60}")

    torch.manual_seed(iteration)

    # --- 1x setup ---
    layers_1x_swiglu = [SwiGLU() for _ in range(L)]
    layers_1x_identity = [Identity() for _ in range(L)]
    pq_1x = torch.randn(L + 1, D, device=DEVICE, dtype=DTYPE, requires_grad=True) / math.sqrt(D)

    # --- 2x setup ---
    attn_layers = [AttnSublayer() for _ in range(L)]
    mlp_layers = [SwiGLU() for _ in range(L)]
    sublayers_2x = []
    for a, m in zip(attn_layers, mlp_layers):
        sublayers_2x.append(a)
        sublayers_2x.append(m)
    sublayers_2x_identity = []
    for _ in range(L):
        sublayers_2x_identity.append(Identity())
        sublayers_2x_identity.append(Identity())
    pq_2x = torch.randn(2 * L + 1, D, device=DEVICE, dtype=DTYPE, requires_grad=True) / math.sqrt(D)

    inputs = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    grad_out = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE)

    # --- Correctness ---
    print("\nCorrectness (production vs reference):")

    ref_1x = lambda i, p, l: reference_forward(i, p, l, BLOCK_SIZE)
    print(f"  1x routing:")
    ok_1x = check_correctness("ref", ref_1x, "prod", production_1x, inputs, pq_1x, layers_1x_swiglu, grad_out)
    reset_cuda()

    ref_2x = lambda i, p, l: reference_forward(i, p, l, SUBLAYER_BLOCK_SIZE)
    print(f"  2x sublayer:")
    ok_2x = check_correctness("ref", ref_2x, "prod", production_2x, inputs, pq_2x, sublayers_2x, grad_out)
    reset_cuda()

    print(f"  {'PASS' if ok_1x and ok_2x else 'FAIL'}")

    # --- Performance: identity layers (isolates routing overhead) ---
    print("\nPerformance (identity layers — routing overhead only):")
    results = []
    reset_cuda()
    results.append(run_perf("1x production", production_1x, inputs, pq_1x, layers_1x_identity, grad_out))
    results.append(run_perf("2x production_sublayer", production_2x, inputs, pq_2x, sublayers_2x_identity, grad_out))
    print_table(results)

    # --- Performance: real layers ---
    print("\nPerformance (SwiGLU / AttnSublayer+SwiGLU layers):")
    results = []
    reset_cuda()
    results.append(run_perf("1x production (SwiGLU)", production_1x, inputs, pq_1x, layers_1x_swiglu, grad_out))
    results.append(run_perf("2x production (Attn+SwiGLU)", production_2x, inputs, pq_2x, sublayers_2x, grad_out))
    print_table(results)

    del inputs, grad_out, layers_1x_swiglu, layers_1x_identity
    del attn_layers, mlp_layers, sublayers_2x, sublayers_2x_identity
    reset_cuda()

print(f"\n{'='*60}")
print("Done")
