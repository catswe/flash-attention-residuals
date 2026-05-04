"""Smoke test for per-sublayer routing (paper Figure 2).

Validates that BlockAttentionResiduals with interleaved sublayers produces
correct outputs and gradients vs a naive PyTorch reference.

Usage: python tests/test_sublayer_routing.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn_res.experimental.autograd import BlockAttentionResiduals

DEVICE = "cuda"
DTYPE = torch.bfloat16
EPS = torch.finfo(torch.float32).eps

B, T, D = 2, 256, 256
L = 4
BLOCK_SIZE = 2
SUBLAYER_BLOCK_SIZE = 2 * BLOCK_SIZE


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


def naive_attention_residual(pseudo_query, values):
    keys = F.rms_norm(values.float(), (values.shape[-1],), eps=EPS)
    logits = torch.einsum("d, n b t d -> n b t", pseudo_query.float(), keys)
    logits = logits - logits.max(dim=0, keepdim=True).values
    return torch.einsum(
        "n b t, n b t d -> b t d", logits.softmax(0), values.float(),
    ).to(DTYPE)


def naive_forward(inputs, pseudo_queries, sublayers, block_size):
    blocks = [inputs]
    for i in range(len(sublayers)):
        outputs = naive_attention_residual(
            pseudo_queries[i], torch.stack(blocks, dim=0),
        )
        update = sublayers[i](outputs)
        if i % block_size == 0:
            blocks.append(update)
        else:
            blocks[-1] = blocks[-1] + update
    return naive_attention_residual(
        pseudo_queries[-1], torch.stack(blocks, dim=0),
    )


def production_forward(inputs, pseudo_queries, sublayers, block_size):
    flat_params = tuple(p for layer in sublayers for p in layer.parameters())
    return BlockAttentionResiduals.apply(
        inputs, pseudo_queries, sublayers, block_size, EPS, *flat_params,
    )


def check(name, ref_out, test_out, ref_grads, test_grads):
    out_err = (ref_out.float() - test_out.float()).abs()
    print(f"  {name} output: mean_abs={out_err.mean():.2e}, max_abs={out_err.max():.2e}")

    ok = True
    for i, (rg, tg) in enumerate(zip(ref_grads, test_grads)):
        diff = (rg.float() - tg.float()).abs()
        cos = F.cosine_similarity(rg.flatten().float(), tg.flatten().float(), dim=0)
        print(f"  {name} grad[{i}]: mean_abs={diff.mean():.2e}, max_abs={diff.max():.2e}, cos={cos:.6f}")
        if cos < 0.99:
            print(f"  WARN: low cosine similarity on grad[{i}]")
            ok = False
    return ok


def run_test(name, sublayers, pseudo_queries, block_size):
    print(f"\n{name}: {len(sublayers)} sublayers, block_size={block_size}")

    inputs = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    grad_out = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE)
    targets = (inputs, pseudo_queries, *[p for s in sublayers for p in s.parameters()])

    ref_out = naive_forward(inputs, pseudo_queries, sublayers, block_size)
    ref_grads = torch.autograd.grad(ref_out, targets, grad_out, allow_unused=True)

    test_out = production_forward(inputs, pseudo_queries, sublayers, block_size)
    test_grads = torch.autograd.grad(test_out, targets, grad_out, allow_unused=True)

    ref_grads = [g for g in ref_grads if g is not None]
    test_grads = [g for g in test_grads if g is not None]

    return check(name, ref_out, test_out, ref_grads, test_grads)


torch.manual_seed(42)

# 1x routing (existing behavior)
layers_1x = [SwiGLU() for _ in range(L)]
queries_1x = torch.randn(
    L + 1, D, device=DEVICE, dtype=DTYPE, requires_grad=True,
) / math.sqrt(D)
ok_1x = run_test("1x routing", layers_1x, queries_1x, BLOCK_SIZE)

# 2x sublayer routing (paper Figure 2)
attn_layers = [AttnSublayer() for _ in range(L)]
mlp_layers = [SwiGLU() for _ in range(L)]
sublayers = []
for a, m in zip(attn_layers, mlp_layers):
    sublayers.append(a)
    sublayers.append(m)
queries_2x = torch.randn(
    2 * L + 1, D, device=DEVICE, dtype=DTYPE, requires_grad=True,
) / math.sqrt(D)
ok_2x = run_test("2x sublayer", sublayers, queries_2x, SUBLAYER_BLOCK_SIZE)

print("\n" + "=" * 40)
if ok_1x and ok_2x:
    print("ALL PASSED")
else:
    print("FAILED")
    exit(1)
