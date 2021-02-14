# 参考 https://www.bilibili.com/video/BV1MV411t7td?from=search&seid=4733570378288382581

'''

forward

'''


'''
在整个动态图构建的时候,网络的节点分为两种: leaf和中间节点。其中leaf节点是网络的更新值,会要求在后向传播后保存grad,
而中间节点就不会保存梯度信息--为了节省内存（但如何进行保存以及查看参考hook机制）。

步骤:(主要的重点步骤)
(1) 节点加载 -- 比如 x, y 
(2) 检查节点是否有hook-- x._backward_hooks --> dict; (参考hook机制主题--注意retain_grad的位置影响) 
(3) 根据 计算操作mul-- x*y 生成中间节点z = x*y, 并且此时会保存fn,便于后续反传
(4) 检查是否中间节点有hook(注意中间节点的hook会和grad_fn有关联)
(5) 搭建完毕 
'''

'''

backward

'''

'''

backward 图是根据前向传播的各个节点的信息以及fn来构建的,是forward图的反向，只是说要结合hook机制来修正下；
一般来说hook是为空的, 因此就比较简单。

(1) 假如说最终的节点为w, 得到w的初始梯度为1, 根据hook更新grad
(2) 计算当前节点对之前节点的梯度
(3) 将梯度传递到下一个fn: next_functions--> tuple
(4) 根据链式法则计算梯度,并把每一个节点的梯度保存在grad里, 直到初始节点
(5) 中间节点不会保存grad

'''