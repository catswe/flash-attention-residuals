import math

import torch
from ..ops import phase_1
from ..ops import phase_2


class BlockwiseAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, inputs, pseudo_queries, layers, BLOCK_SIZE, eps, *flat_layer_params
    ):
        B, T, D = inputs.shape
        L = len(layers)
        NUM_BLOCKS = math.ceil(L / BLOCK_SIZE) + 1
        DEVICE = inputs.device

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

            block_attn_out, block_lse = phase_1.phase_1_batched_attention_triton_op(
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
                    layer_input = phase_2.phase_2_online_softmax_merge_triton_op(
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

                del layer_input, update

            del (block_attn_out,)

        final_out, _final_lse = phase_1.phase_1_batched_attention_triton_op(
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
        ctx.BLOCK_SIZE = BLOCK_SIZE

        return final_out[0].to(inputs.dtype)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        if grad_output is None:
            return (None, None, None, None, None, *([None] * ctx.num_layer_params))

        block_representations, pseudo_queries = ctx.saved_tensors
        layers = ctx.layers
        eps = ctx.eps

        NUM_BLOCKS, B, T, D = block_representations.shape
        L = len(layers)
        BLOCK_SIZE = ctx.BLOCK_SIZE

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
            dtype=torch.bfloat16,
        )
        grad_pseudo_queries = torch.zeros_like(
            pseudo_queries,
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
            _final_out_recomputed, final_lse = (
                phase_1.phase_1_batched_attention_triton_op(
                    block_representations,
                    pseudo_queries[-1:],
                    eps,
                )
            )

            del _final_out_recomputed

        grad_pseudo_queries_partial_final = torch.empty(
            1,
            B,
            T,
            D,
            device=device,
            dtype=torch.float32,
        )

        phase_1._batched_attention_backward_accumulate(
            block_representations,
            pseudo_queries[-1:],
            final_lse,
            grad_output.unsqueeze(0),
            None,
            grad_block_representations,
            grad_pseudo_queries[-1:],
            grad_pseudo_queries_partial_final,
            eps=eps,
        )

        del grad_pseudo_queries_partial_final

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
                phase1_out, phase1_lse = (
                    phase_1.phase_1_batched_attention_triton_op(
                        block_representations[:curr_block_idx],
                        pseudo_queries[block_start : block_start + num_queries],
                        eps,
                    )
                )

            partial_recompute = torch.empty(
                B,
                T,
                D,
                device=device,
                dtype=block_dtype,
            )

            for query_offset in range(num_queries):
                layer_idx = block_start + query_offset

                if query_offset == 0:
                    layer_input_recomputed = phase1_out[query_offset]
                else:
                    with torch.no_grad():
                        layer_input_recomputed = (
                            phase_2.phase_2_online_softmax_merge_triton_op(
                                partial_recompute,
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

                layer_input = layer_input_recomputed_list[query_offset]
                layer_update = layer_update_list[query_offset]

                grad_layer_input = run_saved_layer_backward(
                    layer_idx,
                    layer_input,
                    layer_update,
                    grad_curr_partial,
                )

                layer_input_recomputed_list[query_offset] = None
                layer_update_list[query_offset] = None

                if query_offset == 0:
                    grad_phase1_out[query_offset].copy_(grad_layer_input)
                else:
                    with torch.no_grad():
                        partial_recompute.sub_(layer_update.detach())

                    phase_2._online_softmax_merge_backward_accumulate(
                        partial_recompute,
                        pseudo_queries[layer_idx],
                        phase1_out[query_offset],
                        phase1_lse[query_offset],
                        grad_layer_input,
                        grad_curr_partial,
                        grad_pseudo_queries[layer_idx],
                        grad_phase1_out[query_offset],
                        grad_phase1_lse[query_offset],
                        grad_layer_input,
                        eps=eps,
                    )

                del grad_layer_input
                del layer_input
                del layer_update

            del layer_input_recomputed_list
            del layer_update_list
            del phase1_out

            phase_1._batched_attention_backward_accumulate(
                block_representations[:curr_block_idx],
                pseudo_queries[block_start : block_start + num_queries],
                phase1_lse,
                grad_phase1_out,
                grad_phase1_lse,
                grad_block_representations[:curr_block_idx],
                grad_pseudo_queries[block_start : block_start + num_queries],
                grad_phase1_out,
                eps=eps,
            )

            del phase1_lse

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
            needs_grad = ctx.needs_input_grad[5 + j]
            if not needs_grad or grad_param is None:
                grad_flat_layer_params_out.append(None)
            else:
                grad_flat_layer_params_out.append(grad_param.to(param.dtype))

        return (
            grad_inputs,
            grad_pseudo_queries_out,
            None,
            None,
            None,
            *grad_flat_layer_params_out,
        )
