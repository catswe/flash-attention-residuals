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
) -> torch.Tensor:
    if eps is None:
        eps = EPS

    blocks = [inputs]

    for block_start in range(0, len(layers), block_size):
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


def paper_forward(inputs, pseudo_queries, layers, block_size=BLOCK_SIZE):
    inputs = inputs.to(torch.float32)
    pseudo_queries = pseudo_queries.to(torch.float32)

    blocks = [inputs]

    for i in range(len(layers)):
        outputs = naive_attention_residual(
            pseudo_queries[i],
            torch.stack(blocks, dim=0),
        )
        update = layers[i](outputs)

        if i % block_size == 0:
            blocks.append(update)
        else:
            blocks[-1] = blocks[-1] + update

    return naive_attention_residual(
        pseudo_queries[-1],
        torch.stack(blocks, dim=0),
    )


def paper_forward_sublayer(
    inputs, pseudo_queries, attn_sublayers, mlp_sublayers, block_size=BLOCK_SIZE,
):
    sublayers = []
    for attn_fn, mlp_fn in zip(attn_sublayers, mlp_sublayers):
        sublayers.append(attn_fn)
        sublayers.append(mlp_fn)

    sublayer_block_size = 2 * block_size
    return paper_forward(inputs, pseudo_queries, sublayers, sublayer_block_size)


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


def torch_compile_phases_forward(inputs, query_w, layers, block_size=BLOCK_SIZE):
    blocks = [inputs]
    input_dtype = inputs.dtype

    for block_start in range(0, len(layers), block_size):
        num_queries = min(block_size, len(layers) - block_start)
        query_block = query_w[block_start : block_start + num_queries]
        values = torch.stack(blocks, dim=0)

        lse, normalized, h = phase_1_fn(query_block, values)
        curr_block = layers[block_start](h.to(input_dtype))

        for offset in range(1, num_queries):
            layer_idx = block_start + offset

            h = phase_2_fn(
                curr_block,
                query_block[offset],
                lse[offset],
                normalized[offset],
            )

            curr_block = curr_block + layers[layer_idx](h.to(input_dtype))

        blocks.append(curr_block)

    _, _, h = phase_1_fn(query_w[-1:], torch.stack(blocks, dim=0))
    return h.to(input_dtype)


def interleave_sublayers(attn_sublayers, mlp_sublayers):
    sublayers = []
    for attn_fn, mlp_fn in zip(attn_sublayers, mlp_sublayers):
        sublayers.append(attn_fn)
        sublayers.append(mlp_fn)
    return sublayers


def production_forward_sublayer(
    inputs, pseudo_queries, attn_sublayers, mlp_sublayers,
    eps=None, block_size=BLOCK_SIZE, **kwargs,
):
    sublayers = interleave_sublayers(attn_sublayers, mlp_sublayers)
    sublayer_block_size = 2 * block_size
    return production_forward2(
        inputs, pseudo_queries, sublayers,
        eps=eps, block_size=sublayer_block_size, **kwargs,
    )
