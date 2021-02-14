import torch
from torch import nn
import torch.nn.functional as F

torch.random.seed()

# 首先我们定义一个模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4, 4)
        self.layer1 = nn.Sequential(
            nn.Linear(4,10),
            nn.Linear(10,20),
            nn.Linear(20, 4)
        )

        self.fc2 = nn.Linear(4, 1)


    def forward(self, x):
        x = self.fc1(x)
        o = F.relu(x)
        o = self.layer1(o)
        o = F.relu(o)
        o = self.fc2(o)
        return o

def init_weight(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(1.)
        m.bias.data.fill_(0.)

model = Model()

'''
1. named_children , children

'''

for name, children in model.named_children():
    print(name, children)

#%% output
# fc1 Linear(in_features=4, out_features=4, bias=True)
# layer1 Sequential(
#   (0): Linear(in_features=4, out_features=10, bias=True)
#   (1): Linear(in_features=10, out_features=20, bias=True)
#   (2): Linear(in_features=20, out_features=4, bias=True)
# )
# fc2 Linear(in_features=4, out_features=1, bias=True)


for children in model.children():
    print(children)

#%% md
# Linear(in_features=4, out_features=4, bias=True)
# Sequential(
#   (0): Linear(in_features=4, out_features=10, bias=True)
#   (1): Linear(in_features=10, out_features=20, bias=True)
#   (2): Linear(in_features=20, out_features=4, bias=True)
# )
# Linear(in_features=4, out_features=1, bias=True)

'''
2. named_modules, modules

和children的区别参考：https://blog.csdn.net/dss_dssssd/article/details/83958518

'''

for name, module in model.named_modules():
    print(name, module)

#%% md
#  Model(
#   (fc1): Linear(in_features=4, out_features=4, bias=True)
#   (layer1): Sequential(
#     (0): Linear(in_features=4, out_features=10, bias=True)
#     (1): Linear(in_features=10, out_features=20, bias=True)
#     (2): Linear(in_features=20, out_features=4, bias=True)
#   )
#   (fc2): Linear(in_features=4, out_features=1, bias=True)
# )
# fc1 Linear(in_features=4, out_features=4, bias=True)
# layer1 Sequential(
#   (0): Linear(in_features=4, out_features=10, bias=True)
#   (1): Linear(in_features=10, out_features=20, bias=True)
#   (2): Linear(in_features=20, out_features=4, bias=True)
# )
# layer1.0 Linear(in_features=4, out_features=10, bias=True)
# layer1.1 Linear(in_features=10, out_features=20, bias=True)
# layer1.2 Linear(in_features=20, out_features=4, bias=True)
# fc2 Linear(in_features=4, out_features=1, bias=True)

for module in model.modules():
    print(module)

'''
3. named_buffers, buffers

'''