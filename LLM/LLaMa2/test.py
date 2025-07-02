import torch

from attention import *
from rope import *
from norm import *

def test_rope():
    batch_size, seq_len, n_head, head_dim = 1, 10, 8, 16
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len)
    print("freqs_cos shape: ", freqs_cos.shape)
    print("freqs_cos:\n", freqs_cos)
    xq = torch.zeros((batch_size, seq_len, n_head, head_dim))
    xk = torch.zeros((batch_size, seq_len, n_head, head_dim))
    xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
    print("xq shape: ", xq.shape) # same as origin x dim

def test_norm():
    batch_size, seq_len, n_head, head_dim = 1, 10, 8, 16
    xq = torch.arange(batch_size * seq_len * n_head * head_dim).reshape((batch_size, seq_len, n_head, head_dim)).to(torch.float32)
    norm = RMSNorm(head_dim, 1e-6)
    x = norm.forward(xq)
    print(x.shape)
    print(xq[0, 0])
    print(x[0, 0])

def random_test():
    xq = torch.arange(4*4).reshape((4, 4))
    print(xq.transpose(0, 1))

if __name__ == "__main__":
    # random_test()
    # test_rope()
    test_norm()