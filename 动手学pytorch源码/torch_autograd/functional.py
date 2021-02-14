import torch.autograd as at
import torch
import torch.autograd.function


A = torch.tensor(2., requires_grad=True)
B = torch.tensor(.5, requires_grad=True)
E = torch.tensor(1., requires_grad=True)
C = A * B
D = C.exp()
F = D + E
print(F)        # tensor(3.7183, grad_fn=<AddBackward0>) 打印计算结果，可以看到F的grad_fn指向AddBackward，即产生F的运算
print([x.is_leaf for x in [A, B, C, D, E, F]])  # [True, True, False, False, True, False] 打印是否为叶节点，由用户创建，且requires_grad设为True的节点为叶节点
print([x.grad_fn for x in [F, D, C, A]])    # [<AddBackward0 object at 0x7f972de8c7b8>, <ExpBackward object at 0x7f972de8c278>, <MulBackward0 object at 0x7f972de8c2b0>, None]  每个变量的grad_fn指向产生其算子的backward function，叶节点的grad_fn为空
print(F.grad_fn.next_functions) # ((<ExpBackward object at 0x7f972de8c390>, 0), (<AccumulateGrad object at 0x7f972de8c5f8>, 0)) 由于F = D + E， 因此F.grad_fn.next_functions也存在两项，分别对应于D, E两个变量，每个元组中的第一项对应于相应变量的grad_fn，第二项指示相应变量是产生其op的第几个输出。E作为叶节点，其上没有grad_fn，但有梯度累积函数，即AccumulateGrad（由于反传时多出可能产生梯度，需要进行累加）
F.backward()   # 进行梯度反传
print(A.grad, B.grad, E.grad)   # tensor(1.3591) tensor(5.4366) tensor(1.) 算得每个变量梯度，与求导得到的相符
print(C.grad, D.grad)   # None None 为节约空间，梯度反传完成后，中间节点的梯度并不会保留


#######################################
######  torch.autograd.functional.jacobian() / hessian()
###  可惜了, 1.3.1没有这个的实现

'''
这两个函数的输入为运算函数（接受输入 tensor，返回输出 tensor）和输入 tensor，返回 jacobian 和 hessian 矩阵。
对于jacobian接口，输入输出均可以为 n 维张量，
对于hessian接口，输出必需为一标量。
jacobian返回的张量 shape 为output_dim x input_dim（若函数输出为标量，则 output_dim 可省略），
hessian返回的张量为input_dim x input_dim。除此之外，这两个自动微分接口同时支持运算函数接收和输出多个 tensor。


'''

# from torch.autograd.functional import jacobian, hessian
# from torch.nn import Linear, AvgPool2d
#
# fc = Linear(4, 2)
# pool = AvgPool2d(kernel_size=2)
#
# def scalar_func(x):
#     y = x ** 2
#     z = torch.sum(y)
#     return z
#
# def vector_func(x):
#     y = fc(x)
#     return y
#
# def mat_func(x):
#     x = x.reshape((1, 1,) + x.shape)
#     x = pool(x)
#     x = x.reshape(x.shape[2:])
#     return x ** 2
#
# vector_input = torch.randn(4, requires_grad=True)
# mat_input = torch.randn((4, 4), requires_grad=True)
#
# j = jacobian(scalar_func, vector_input)
# assert j.shape == (4, )
# assert torch.all(jacobian(scalar_func, vector_input) == 2 * vector_input)
# h = hessian(scalar_func, vector_input)
# assert h.shape == (4, 4)
# assert torch.all(hessian(scalar_func, vector_input) == 2 * torch.eye(4))
# j = jacobian(vector_func, vector_input)
# assert j.shape == (2, 4)
# assert torch.all(j == fc.weight)
# j = jacobian(mat_func, mat_input)
# assert j.shape == (2, 2, 4, 4)