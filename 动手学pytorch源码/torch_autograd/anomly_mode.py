import torch
from torch import autograd

class MyFunc(autograd.Function):

    @staticmethod
    def forward(ctx, inp):
        return inp.clone()

    @staticmethod
    def backward(ctx, gO):
        raise RuntimeError("Some error in backward")
        return gO.clone()


def run_fn(a):
    out = MyFunc.apply(a)
    return out.sum()

with autograd.detect_anomaly():
    inp = torch.rand(10, 10, requires_grad=True)
    out = run_fn(inp)
    out.backward()