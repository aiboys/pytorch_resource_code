import torch
import numpy
# 参考 https://zhuanlan.zhihu.com/p/64551412

a = torch.from_numpy(numpy.array([[3,4,5],[1,2,3]]))

# torch.reshape() == torch.contigous().view()

# reshape 会创建一个新的tensor
# view只是给了一个view而已，不会创建新的，共享内存
# 上述的两种性质和numpy里是一致的