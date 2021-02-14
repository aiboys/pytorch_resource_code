import torch
from torch import nn


torch.random.seed()

'''
torch.save
'''

#


# 首先我们定义一个模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4, 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)


    def forward(self, x):
        # x = self.fc1(x)  # 被删除
        o = self.relu1(x)
        o = self.fc2(o)
        return o

def init_weight(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(1.)
        m.bias.data.fill_(0.)

model = Model()


'''
state_dict
'''
print(model.state_dict().keys())



'''

torch.save

'''
