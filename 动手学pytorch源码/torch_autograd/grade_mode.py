# 四种取消梯度跟踪（动态计算图构建grad_fn那些信息的构建跟踪）的方法，都一样的

from torch.autograd import grad_mode
import torch


x = torch.tensor([1.], requires_grad=True)
with grad_mode.no_grad():
    y = x*2

print(y.requires_grad)  # False


with torch.no_grad() as f:  #f --> None
    z = x*2

print(z.requires_grad)  # False

@torch.no_grad()  # __call__(self, func) --> 定义了修饰器
def double(x):
    return x*2

c = double(x)
print(c.requires_grad)  # False

@grad_mode.no_grad()
def double2(x):
    return x*2

d = double2(x)
print(d.requires_grad)  # False


'''
在1.7 版本（反正比1.3.1高的某个版本）改变了实现方式：

参考官方源码
https://pytorch.org/docs/stable/_modules/torch/autograd/grad_mode.html#no_grad
'''