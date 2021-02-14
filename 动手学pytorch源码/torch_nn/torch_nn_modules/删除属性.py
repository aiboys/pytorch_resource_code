'''

pytorch的属性删除都是针对底层的三类数据进行删除的: self._parameters, self._modules, self._buggers （对于
不属于该三类的，当做一般属性进行删除）
在删除的时候，pytorch通过self.__delattr__()会挨个检查变量是属于哪种类型, 然后直接从字典里删除

删除模块的方法:
(1)知道了某一个模块的名字,直接手动删除,比如： del model.fc1
(2)知道了某一个模块的名字，手动底层对三种数据类型进行删除: del model._modules['fc1']
(3)删除一类模块, 利用named_modules进行查询模块名称,然后用delattr(model, module_name)进行删除
（4）在预训练的时候,比如只需要加载前面部分的网络结构, 那么可以
这个时候其实要把握两点：1-- 对网络的state_dict进行删除， del state_dict[key]; 2-- 把握nn.Sequtial(*list([]))
比如： model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
或者 model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
'''


import torch
from torch import nn


torch.random.seed()

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

# for k, v in model._modules.items():
#     print(k, v)

# del model.fc1     # 利用__delatttr__进行删除： 可以删除三种类型参数：_modules, _parameters, _buffers
#
# del model._modules['fc1']  # 直接对module进行删除

model.apply(init_weight)
print(model)


a = torch.tensor([1.,2.,3.,4.],requires_grad=False)

# print(model(a))
new = []


# 3. 通过named_modules找到指定的特殊模块进行删除
for name, module in model.named_modules():
    # print(name, module)
    if name !='':
        if name.find('fc2') != -1:
            print("ok")
            a = str(name)
            delattr(model, name)  # 至于为啥不能用 del model.name 见 https://blog.csdn.net/windscloud/article/details/79732014

            # del model._modules[name]
        new.append(module)

print(model)
print(new)
new_model = nn.Sequential(*new)

for name, module in new_model.named_modules():
    print(name, module)

new_model.apply(init_weight)

print(new_model(a))





