'''

def scale_lr(optimizer, scale):
    """Scale the learning rate of the optimizer"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= scale


optimizer: defaults {dict},
            param_groups: {list}: 保存每一层的各种信息，当然包括学习率等. 每一层都是一个dict [params, lr,weight_decay]
'''