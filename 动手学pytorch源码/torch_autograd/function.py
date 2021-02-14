# 在torch.autograd.function的包中最重要的就是Function类，该类作为autograd的扩展，可以实现自定义nn.Module的功能
# 参考 https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function

# 该类的要求是：
'''
Recall that Function s are what autograd uses to compute the results and gradients,
and encode the operation history. Every new function requires you to implement 2 methods:

forward() - the code that performs the operation. It can take as many arguments as you want,
            with some of them being optional, if you specify the default values. All kinds
            of Python objects are accepted here. Tensor arguments that track history
            (i.e., with requires_grad=True) will be converted to ones that don’t track history
            before the call, and their use will be registered in the graph. Note that this logic
            won’t traverse lists/dicts/any other data structures and will only consider Tensor s
            that are direct arguments to the call. You can return either a single Tensor output,
            or a tuple of Tensor s if there are multiple outputs. Also, please refer to the
            docs of Function to find descriptions of useful methods that can be called only from forward().

backward() - gradient formula. It will be given as many Tensor arguments as there were outputs,
            with each of them representing gradient w.r.t. that output. It should return as many
            Tensor s as there were inputs, with each of them containing the gradient w.r.t. its
            corresponding input. If your inputs didn’t require gradient (needs_input_grad is
            a tuple of booleans indicating whether each input needs gradient computation),
            or were non-Tensor objects, you can return None. Also, if you have optional arguments
             to forward() you can return more gradients than there were inputs, as long as they’re
             all None
'''

# example1

import torch.nn as nn
from torch.autograd.function import Function
import torch.tensor

class Exp(Function):                    # 此层计算e^x

    @staticmethod
    def forward(ctx, input_tensor):

        print(ctx) # <torch.autograd.function.ExpBackward object at 0x0000025FB5221A48>
        '''
         
        :param ctx: 参考 https://stackoverflow.com/questions/49516188/difference-between-ctx-and-self-in-python?newreg=02a3c546d6ed4a5e9c98def3b84081ab
                    这里的ctx不是什么self，它是在计算图构建的时候的每个节点（中间节点）的那些Backward模块对象，关于这个模块详见b站之前看的那个autograd机制的视频
                    额外的： ctx还可以通过对保存的数据进行指派要不要计算梯度，详细的见文档
        
        :param i:
        :return:
        '''
        result = input_tensor.exp()

        ctx.save_for_backward(result)   # 保存所需内容，以备backward时使用，所需的结果会被保存在saved_tensors元组中；此处仅能保存tensor类型变量，若其余类型变量（Int等），可直接赋予ctx作为成员变量，也可以达到保存效果
        return result

    @staticmethod
    def backward(ctx, grad_output):     # 模型梯度反传
        '''

        :param ctx: 第一个参数必须是ctx(context)(当然参数名随便都可以)
        :param grad_output: 相对于该节点的梯度
        :return:
        '''
        result, = ctx.saved_tensors     # 取出forward中保存的result
        return grad_output * result     # 计算梯度并返回

# 尝试使用
x = torch.tensor([1.], requires_grad=True)  # 需要设置tensor的requires_grad属性为True，才会进行梯度反传
ret = Exp.apply(x)                          # 使用apply方法调用自定义autograd function
print(ret)                                  # tensor([2.7183], grad_fn=<ExpBackward>)
print(x.grad)
ret.backward()                              # 反传梯度
print(x.grad)                               # tensor([2.7183])
print(x.is_leaf) # True

# 关于 save_for_backward()的解释：
'''
    def save_for_backward(self, *tensors):
        r"""Saves given tensors for a future call to :func:`~Function.backward`.

        **This should be called at most once, and only from inside the**
        :func:`forward` **method.**

        Later, saved tensors can be accessed through the :attr:`saved_tensors`
        attribute. Before returning them to the user, a check is made to ensure
        they weren't used in any in-place operation that modified their content.

        Arguments can also be ``None``.
        """
        self.to_save = tensors

'''

# example2

'''
不仅仅可以forward传入需要处理的tensor，还可以传入一些额外的数据
'''

class GradCoeff(Function):

    @staticmethod
    def forward(ctx, x, coeff):  # 模型前向,
        ctx.coeff = coeff  # 将coeff存为ctx的成员变量

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # 模型梯度反传
        return ctx.coeff * grad_output, None  # backward的输出个数，应与forward的输入个数相同，此处coeff不需要梯度，因此返回None


# 尝试使用
x = torch.tensor([2.], requires_grad=True)
ret = GradCoeff.apply(x, -0.1)  # 前向需要同时提供x及coeff，设置coeff为-0.1
ret = ret ** 2
print(ret)  # tensor([4.], grad_fn=<PowBackward0>)
ret.backward()
print(x.grad)  # tensor([-0.4000])，梯度已乘以相应系数


#########################################################################
###         对Funtion模块的使用: Function.apply()

# Inherit from Function
class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )