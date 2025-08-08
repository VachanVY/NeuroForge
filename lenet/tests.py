import torch
import torch.nn.functional as F
from functional import (
    linear_forward, linear_backward,
    reshape_forward, reshape_backward,
    conv2d_forward, conv2d_backward,
    maxpool2d_forward, maxpool2d_backward,

    relu_forward, relu_backward,
    softmax_forward, softmax_backward,
    
    cross_entropy_forward, cross_entropy_backward,
)

def test_conv2d():
    torch.manual_seed(0)
    x = torch.randn(size=(2, 1, 28, 28), requires_grad=True)
    w = torch.randn(size=(6, 1, 5, 5), requires_grad=True)
    b = torch.randn(size=(6,), requires_grad=True)

    my_conv2d = conv2d_forward(x, w, b)

    torch_conv2d = F.conv2d(x, w, b, stride=1)
    torch.testing.assert_close(my_conv2d.detach(), torch_conv2d)

    loss = my_conv2d.mean()
    loss.backward()

    torch_dL_dx, torch_dL_dw, torch_dL_db = x.grad, w.grad, b.grad

    # For mean loss, dL/dO is constant 1/numel
    dL_dO = torch.ones_like(my_conv2d) / my_conv2d.numel()

    my_dL_dx, my_dL_dw, my_dL_db = conv2d_backward(x, w, b, dL_dO)
    torch.testing.assert_close(my_dL_dx, torch_dL_dx)
    torch.testing.assert_close(my_dL_dw, torch_dL_dw)
    torch.testing.assert_close(my_dL_db, torch_dL_db)


def test_maxpool2d():
    torch.manual_seed(0)
    x = torch.randn(size=(2, 1, 28, 28), requires_grad=True)

    torch_maxpool, torch_idx = F.max_pool2d_with_indices(x, kernel_size=(2, 2), stride=(2, 2))

    torch_loss = torch_maxpool.mean()
    torch_loss.backward()

    torch_dL_dX = x.grad

    my_maxpool, indices, x_shape = maxpool2d_forward(x.clone().detach().requires_grad_(True), kernel_size=(2, 2), strides=(2, 2))
    torch.testing.assert_close(my_maxpool, torch_maxpool)

    # For mean loss, dL/dO is constant 1/numel
    dL_dO = torch.ones_like(my_maxpool) / my_maxpool.numel()
    my_dL_dx = maxpool2d_backward(dL_dO, x_shape, indices)

    torch.testing.assert_close(my_dL_dx, torch_dL_dX)
    

def test_linear_autograd():
    torch.manual_seed(0)
    B, fi, fo = 4, 8, 5
    x = torch.randn(B, fi, requires_grad=True)
    w = torch.randn(fo, fi, requires_grad=True)
    b = torch.randn(fo, requires_grad=True)

    y = linear_forward(x, w, b)
    torch.testing.assert_close(y.detach(), F.linear(x, w, b))

    (y.sum()).backward()

    dL_dO = torch.ones_like(y)
    my_dx, my_dw, my_db = linear_backward(x, w, b, dL_dO)

    torch.testing.assert_close(my_dx, x.grad)
    torch.testing.assert_close(my_dw, w.grad)
    torch.testing.assert_close(my_db, b.grad)


def test_reshape_autograd():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 5, requires_grad=True)

    y, x_shape = reshape_forward(x, (2, 3 * 4 * 5))
    torch.testing.assert_close(y.detach(), x.reshape(2, 3 * 4 * 5))

    (y.sum()).backward()

    my_dx = reshape_backward(torch.ones_like(y), x_shape)
    torch.testing.assert_close(my_dx, x.grad)


def test_relu_autograd():
    torch.manual_seed(0)
    x = torch.randn(7, 11, requires_grad=True)

    y = relu_forward(x)
    torch.testing.assert_close(y.detach(), F.relu(x))

    (y.sum()).backward()

    dL_dO = torch.ones_like(y)
    my_dx = relu_backward(y.detach(), dL_dO)

    torch.testing.assert_close(my_dx, x.grad)


def test_softmax_cross_entropy_autograd():
    torch.manual_seed(0)
    B, C = 8, 6
    logits = torch.randn(B, C, requires_grad=True)
    y = torch.randint(low=0, high=C, size=(B,))

    probs = softmax_forward(logits.detach().clone())
    loss = cross_entropy_forward(y, probs)

    logits_ref = logits.detach().clone().requires_grad_(True)
    loss_ref = F.cross_entropy(logits_ref, y)
    grad_autograd = torch.autograd.grad(loss_ref, logits_ref)[0]

    dL_dprobs = cross_entropy_backward(y, probs.detach())
    dL_dlogits = softmax_backward(probs.detach(), dL_dprobs)

    torch.testing.assert_close(dL_dlogits, grad_autograd)


def run_all():
    test_conv2d()
    test_maxpool2d()
    test_linear_autograd()
    test_reshape_autograd()
    test_relu_autograd()
    test_softmax_cross_entropy_autograd()
    print("All tests passed.")


if __name__ == "__main__":
    run_all()
