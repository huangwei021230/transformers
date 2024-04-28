import json
import numpy as np
import torch

# TOOD: The code is reused from the clip benchmark, we need to refactor the code to make it as a dependened package
def sparsify_matrix_for_FC_layer(hidden_states, sparsity_percentage, dim_of_neurals, enable_random_sparsity_selection=False):
    topk = int(sparsity_percentage * dim_of_neurals)
    # assert len(hidden_states.size()) == 3
    _logit = hidden_states.reshape(-1, dim_of_neurals).float()
    if enable_random_sparsity_selection:
        _top_indices = torch.randint(0, dim_of_neurals, (_logit.size()[0], topk), device='cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        _, _top_indices = _logit.topk(topk, dim=1)
    _top_k_indices = _top_indices[:, :topk]
    mask = torch.zeros_like(_logit).scatter(1, _top_k_indices, 1).bool().half()
    hidden_states = hidden_states * mask.reshape(hidden_states.size())
    return hidden_states

def mask_attention_result(hidden_states: torch.Tensor, sparsity_percentage, num_attention_heads):
    bsz, num_head, token_length, head_size = hidden_states.size()

    with torch.no_grad():
        topk = int(sparsity_percentage * num_attention_heads)
        # Calculate the Matrix L2-norm (Frobenius norm) of the attention heads (based on the last two dimentions)
        _logit = torch.linalg.norm(hidden_states, dim=(-1, -2), ord = 'fro')

        # for each element in a batch, select the K most activated attention heads
        _, _top_k_indices = _logit.topk(int(topk), dim=1)
        _head_mask = torch.zeros(
            bsz,
            num_head,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ).scatter_(1, _top_k_indices, 1)
        # Mask unactivated attention heads
        _head_mask = _head_mask.unsqueeze(-1).unsqueeze(-1).expand_as(hidden_states)
    
    return hidden_states * _head_mask
