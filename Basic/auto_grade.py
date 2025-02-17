import torch
import numpy as np

def test_grade():
    x = torch.tensor([1, 2, 3], requires_grad=True)

    y = x.dot(x)

    z = x @ x

    y.backward()
