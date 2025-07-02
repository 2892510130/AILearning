import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        # weight is learned param initialize as 1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # rsqrt is 1 / sqrt
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # -> float type -> origin type
        output = self._norm(x.float()).type_as(x)
        return output * self.weight