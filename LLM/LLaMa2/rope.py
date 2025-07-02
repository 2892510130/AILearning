import torch
from typing import Tuple

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    num_head of Q maybe larger then kv_head
    """
    # input shape：(batch_size, seq_len, kv_head, head_dim)
    bs, slen, n_kv_heads, head_dim = x.shape
    
    if n_rep == 1:
        return x
    
    # expand and reshape
    return (
        x[:, :, :, None, :]  # add a new dim before head_dim
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)    # expand this new dim to n_rep, repeat accomplished
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # reshape back to num of head
    )

# Note：dim here is actually dim//n_head，becayse we do RoPE to every head
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float(), why [: (dim // 2)] is needed?
    # Because if dim is odd, there will be problem, torch.arange(0, dim, 2) will return even data, which is not correct.
    # divide dim，take 1 / theta，get freq
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float() # outer_ij is t_i * freqs_j
    freqs_cos = torch.cos(freqs) # get real part
    freqs_sin = torch.sin(freqs) # get imaginary part
    return freqs_cos, freqs_sin

# Note: reshape freqs_cis to align with x
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    
    # for the broadcast, other dim with shape 1
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # seperate odd and even position, reshape from (*, d) to (*, -1, 2), here -1 is d//2, 
    # then unbind last dim, return two tensor with shape (*, d//2)
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # stack last dimension, then return to origin shape
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)