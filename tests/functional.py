import torch
from torch.nn import functional as F

from aurus_py.functional import (
    linear_forward, linear_backward,
    reshape_forward, reshape_backward,
    conv2d_forward, conv2d_backward,
    maxpool2d_forward, maxpool2d_backward,

    relu_forward, relu_backward,
    softmax_forward, softmax_backward,
    
    cross_entropy_forward, cross_entropy_backward,
)

def test_conv2d():
    x = torch.randn(size=(2, 1, 28, 28), requires_grad=True)
    w = torch.randn(size=(6, 1, 5, 5), requires_grad=True)
    b = torch.randn(size=(6,), requires_grad=True)

    my_conv2d = conv2d_forward(x, w, b)
    my_conv2d.retain_grad()

    torch_conv2d = F.conv2d(x, w, b, stride=1)
    torch.testing.assert_close(my_conv2d.detach(), torch_conv2d)

    loss = my_conv2d.mean()
    loss.backward()

    torch_dL_dx, torch_dL_dw, torch_dL_db = x.grad, w.grad, b.grad
    dL_dO = my_conv2d.grad.clone()

    my_dL_dx, my_dL_dw, my_dL_db = conv2d_backward(x, w, b, dL_dO)
    torch.testing.assert_close(my_dL_dx, torch_dL_dx)
    torch.testing.assert_close(my_dL_dw, torch_dL_dw)
    torch.testing.assert_close(my_dL_db, torch_dL_db)


def test_maxpool2d():
    x = torch.randn(size=(2, 1, 28, 28), requires_grad=True)
    x.requires_grad = True

    torch_maxpool, torch_idx = F.max_pool2d_with_indices(x, kernel_size=(2, 2), stride=(2, 2))
    torch_maxpool.retain_grad()

    torch_loss = torch_maxpool.mean()
    torch_loss.backward()

    torch_dL_dX, torch_dL_dO = x.grad, torch_maxpool.grad.clone()
    assert not any([torch_dL_dX is None, torch_dL_dO is None])

    my_maxpool, indices, x_shape = maxpool2d_forward(x.clone(), kernel_size=(2, 2), strides=(2, 2))
    torch.testing.assert_close(my_maxpool, torch_maxpool)
    my_dL_dx = maxpool2d_backward(torch_dL_dO, x_shape, indices)

    torch.testing.assert_close(my_dL_dx, torch_dL_dX)
