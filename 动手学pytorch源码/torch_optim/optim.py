import  torch
import torch.optim as optim
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CivilNet(nn.Module):
    def __init__(self):
        super(CivilNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 22 * 22, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Sequential(nn.Linear(84, 10),
                                nn.ReLU(),
                                nn.Linear(10,2))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 22 * 22)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CivilNet().to(device)

# opt = optim.SGD(list(model.parameters()), lr=0.1)
# opt = optim.SGD(model.parameters(), lr=0.1) # 传入的param=迭代器或者list都可以




#########################
# 分模块赋学习率进行优化更新

# lr = 1e-3
# conv1_params = list(map(id, model.conv1.parameters()))
# conv2_params = list(map(id, model.conv2.parameters()))
# base_params = filter(lambda p: id(p) not in conv1_params + conv2_params, model.parameters())
# base_pa = list(base_params)
# print(len(base_pa))
# params = [{'params': base_pa},
#           {'params': list(model.conv1.parameters()), 'lr': lr * 100},
#           {'params': list(model.conv2.parameters()), 'lr': lr * 100}]
# opt = torch.optim.SGD(params, lr=lr, momentum=0.9)

#############################
### 分为weight和bias

lr = 0.001
params = []
fc_params = dict(model.fc.named_parameters())
for key, value in fc_params.items():
        if 'bias' not in key:
            params += [{'params': [value], 'lr': lr}]
        else:
            params += [{'params': [value], 'lr': lr * 0.1}]
conv4_param = dict(model.conv1.named_parameters())
for key, value in conv4_param.items():
        if 'bias' not in key:
            params += [{'params': [value], 'lr': lr}]
        else:
            params += [{'params': [value], 'lr': lr * 0.1}]

params += [{'params': model.conv2.parameters(), 'lr': lr * 0.1}]
conv1_id = list(map(id, model.conv1.parameters()))
conv2_id = list(map(id, model.conv2.parameters()))
fc_id = list(map(id,model.fc.parameters()))
base_param = list(filter(lambda p: id(p) not in conv1_id+ conv2_id+ fc_id, model.parameters()))

params += [{'params': base_param, 'lr': lr*0.2}]

opt= optim.Adam(params)




input = torch.randn((1,3,100,100))
input = input.to(device)

loss_fn = nn.MSELoss()

for i in range(3):
    opt.zero_grad()
    output = model(input)
    label = torch.empty(1, 2, dtype=torch.float).random_(5).to(device)
    loss = loss_fn(output, label)
    loss.backward()
    opt.step()
    print(i)




