import torch
from torch.autograd.function import Function


class Sigmoid(Function):

    @staticmethod
    def forward(ctx, x):
        output = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_x = output * (1 - output) * grad_output
        return grad_x


test_input = torch.randn(4, dtype=torch.float32, requires_grad=True)  # tensor([-0.4646, -0.4403,  1.2525, -0.5953], requires_grad=True)
print(test_input.dtype)

a = torch.autograd.gradcheck(Sigmoid.apply, (test_input,), eps=1e-3)  # pass
b = torch.autograd.gradcheck(torch.sigmoid, (test_input,), eps=1e-3)  # pass
c = torch.autograd.gradcheck(Sigmoid.apply, (test_input,), eps=1e-4)  # fail
d = torch.autograd.gradcheck(torch.sigmoid, (test_input,), eps=1e-4)  # fail
print(a,b,c,d)