import torch
import torch.autograd


# 两种查询中间grad
# 1-- torch.autograd.grad
A = torch.tensor(2., requires_grad=True)
B = torch.tensor(.5, requires_grad=True)
C = A * B
D = C.exp()
torch.autograd.grad(D, (C, A))  # (tensor(2.7183), tensor(1.3591)), 返回的梯度为tuple类型, grad接口支持对多个变量计算梯度


# 2. 使用hook机制
def variable_hook(grad):                        # hook注册在Tensor上，输入为反传至这一tensor的梯度

    grad = grad * 2

    print('the gradient of A is：', grad)
    return grad

A = torch.tensor(2., requires_grad=True)
B = torch.tensor(.5, requires_grad=True)
C = A * B
# hook_handle = A.register_hook(lambda grad: grad*2)  # 两种写法
hook_handle = A.register_hook(variable_hook)    # 在中间变量C上注册hook
D = C.exp()
D.backward()                                    # 反传时打印：the gradient of C is： tensor(2.7183)
hook_handle.remove()                            # 如不再需要，可remove掉这一hook
print(A.grad)
print(B.grad)
A = A + A.grad
B = B + B.grad
print(A)
print(B)

import torchvision.datasets.FashionMNIST

