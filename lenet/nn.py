import torch
from torch import Tensor

from functional import (
    # typing
    FAN_IN, FAN_OUT, KERNEL_SIZE, STRIDES,
    # nn layers
    linear_forward, linear_backward,
    reshape_forward, reshape_backward,
    conv2d_forward, conv2d_backward,
    maxpool2d_forward, maxpool2d_backward,
    # nn activations
    relu_forward, relu_backward,
    softmax_forward, softmax_backward,
    # nn losses
    cross_entropy_forward, cross_entropy_backward,
)

# Siiilllly
_sqrt = lambda x: x**0.5
_prod = lambda x: x[0]*x[1]


# Base Module
class Module:
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    def backward(self, *args, **kwargs):
        raise NotImplementedError
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def to(self, device):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Tensor):
                setattr(self, attr_name, attr.to(device))
            elif isinstance(attr, Module):
                attr.to(device)
        return self
    

# Layers
class Linear(Module):
    def __init__(self, fan_in:int, fan_out:int):
        super().__init__()
        bound = _sqrt(2/fan_in)
        self.wie = torch.randn(fan_out, fan_in).uniform_(-bound, bound)
        self.bias = torch.randn(fan_out)

    def forward(self, x:Tensor):
        return linear_forward(x, self.wie, self.bias)
    def backward(self, x:Tensor, dL_dO:Tensor):
        return linear_backward(x, self.wie, self.bias, dL_dO)
    

class Reshape(Module):
    def __init__(self, shape:tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x:Tensor):
        self.x_shape = x.shape
        reshaped, _ = reshape_forward(x, self.shape)
        return reshaped
    def backward(self, dL_dO:Tensor):
        return reshape_backward(dL_dO, self.x_shape)
    

class Conv2d(Module):
    def __init__(
        self,
        in_channels:FAN_IN,
        out_channels:FAN_OUT,
        kernel_size:KERNEL_SIZE,
        strides:STRIDES,
        bias:bool=True
    ):
        bound = _sqrt(2/_prod(kernel_size))
        self.wei = torch.empty(size=(out_channels, in_channels, *kernel_size)).uniform_(-bound, bound)
        self.bias = torch.zeros(size=(out_channels,)) if bias else None
        self.strides = strides

    def forward(self, x:Tensor):
        self.x = x; return conv2d_forward(x, self.wei, self.bias, self.strides)
    def backward(self, dL_dO:Tensor):
        return conv2d_backward(self.x, self.wei, self.bias, dL_dO, self.strides)


class Maxpool2d(Module):
    def __init__(self, kernel_size:KERNEL_SIZE, strides:STRIDES):
        super().__init__()
        self.kernel_size = kernel_size
        self.strides = strides

    def forward(self, x:Tensor):
        out, self.idx, self.x_shape = maxpool2d_forward(x, self.kernel_size, self.strides)
        return out
    def backward(self, dL_dO:Tensor):
        return maxpool2d_backward(dL_dO, self.x_shape, self.idx)
    

# Activations
class ReLU(Module):
    def forward(self, x:Tensor):
        self.relu = relu_forward(x)
        return self.relu
    def backward(self, dL_dO:Tensor):
        return relu_backward(self.relu, dL_dO)


class Softmax(Module):
    """    Hardcoded Across the last dimension -1     """
    def forward(self, x:Tensor, axis:int=-1):
        assert axis == -1, "Softmax only supports the last axis... for now :/"
        self.probs = softmax_forward(x)
        return self.probs
    def backward(self, dL_dO:Tensor):
        return softmax_backward(self.probs, dL_dO)

# Losses
class CrossEntropy:
    def forward(self, y_true:Tensor, y_proba:Tensor):
        self.y_true = y_true
        self.y_proba = y_proba
        return cross_entropy_forward(y_true, y_proba)
    def backward(self):
        return cross_entropy_backward(self.y_true, self.y_proba)
    

if __name__ == "__main__":
    ...