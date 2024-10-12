import typing as tp

import torch
from torch import Tensor, _assert
from torch.nn import functional as F

FAN_IN:tp.TypeAlias = int
FAN_OUT:tp.TypeAlias = int
KERNEL_SIZE:tp.TypeAlias = tuple[int, int]
STRIDES:tp.TypeAlias = tuple[int, int]


def linear_forward(
    x:Tensor,      # (B, fi)
    wie:Tensor,    # (fo, fi)
    bias:Tensor,   # (fo,)
):
    return x @ wie.T + bias.unsqueeze(0) 

def linear_backward(
    x:Tensor,       # (B, fi)
    wie:Tensor,     # (fo, fi)
    bias:Tensor,    # (fo,)
    dL_dO:Tensor,   # (B, fo)
):
    dL_dx = dL_dO @ wie      # (B, fi) <= (B, fo) @ (fo, fi)
    dL_dwie = dL_dO.T @ x    # (fo, fi) <= (B, fo).T @ (B, fi)
    dL_db = dL_dO.sum(dim=0) # (fo,) <= (B, fo)
    return dL_dx, dL_dwie, dL_db


def reshape_forward(x:Tensor, shape:tuple): # (B, C, H, W)
    return x.reshape(shape), x.shape        # (B, C*H*W), (B, C, H, W)


def reshape_backward(dL_dO:Tensor, x_shape:torch.Size): # (B, C*H*W)
    return dL_dO.reshape(x_shape)                       # (B, C, H, W)


def _conv2d(
    x:Tensor, # (H, W)
    w:Tensor, # (h, w)
    full:bool=False,
    convolve:bool=False
):
    x = x[None, None, ...] # (1, 1, H, W)
    w = w[None, None, ...] # (1, 1, h, w)
    if full:
        pad_h = w.size(-2) - 1
        pad_w = w.size(-1) - 1
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='constant')
    if convolve:
        w = w.flip((2, 3))
    return F.conv2d(
        input=x, # (1, 1, H, W)
        weight=w, # (1, 1, h, w)
        stride=1
    )[0, 0, :, :]


def _dilate_matrix(x:Tensor, dilation:tuple[int, int]):
    """`x: shape(B, C, H, W)`\n `dilation:tuple`"""
    (B, C, H, W), (Hd, Wd)  = x.shape, dilation
    dilated = torch.zeros((B, C, Hd*(H-1)+1, Wd*(W-1)+1 ))
    dilated[:, :, ::Hd, ::Wd] = x
    return dilated


def conv2d_forward(
    x:Tensor,               # (B, fi, H, W) 
    wie:Tensor,               # (fo, fi, h, w)
    bias:tp.Optional[Tensor],  # (fo,)
    stride:STRIDES = (1, 1)
):
    B, C, H, W = x.shape
    fo, fi, h, w = wie.shape
    sh, sw = stride
    assert C == fi, f"Expected {C} == {fi}"
    assert H >= h, f"Expected {H} >= {h}"
    assert W >= w, f"Expected {W} >= {w}"
    assert bias.shape[0] == fo, f"Expected {bias.shape[0]} == {fo}"

    output_shape = (B, fo, int((H-h)//sh + 1), int((W-w)//sw + 1))
    O = torch.zeros(output_shape) # (B, C_out, H1, W1)
    for fan_out in range(fo):
        for fan_in in range(fi):
            for bdim in range(B):
                O[bdim, fan_out] += _conv2d(x[bdim, fan_in], wie[fan_out, fan_in])[::sh, ::sw]

    if bias is not None:
        O += bias.view(1, -1, 1, 1)
    return O


def conv2d_backward(
    x:Tensor,                   # (B, fi, H, W)
    wei:Tensor,                 # (fo, fi, h, w)
    bias:tp.Optional[Tensor],   # (fo,)
    dL_dO:Tensor,               # (B, fo, H1, W1)
    stride:STRIDES = (1, 1)
):  
    fo, fi, h, w = wei.shape
    B, C, H, W = x.shape

    dL_dx, dL_dwei = torch.zeros_like(x), torch.zeros_like(wei)
    dL_dO = _dilate_matrix(dL_dO, dilation=stride)
    for fan_out in range(fo): # C_out
        for bdim in range(B): # B
            for fan_in in range(fi): # C_in
                dL_dwei[fan_out, fan_in] += _conv2d(x[bdim, fan_in], dL_dO[bdim, fan_out]) # (H, W)*(H1, W1) => (Hk, Wk)
                dL_dx[bdim, fan_in] += _conv2d(dL_dO[bdim, fan_out], wei[fan_out, fan_in], full=True, convolve=True) # (H1, W1)*(Hk, Wk) => (H, W)
    
    dL_db = dL_dO.sum(dim=(0, 2, 3)) if bias is not None else None
    return dL_dx, dL_dwei, dL_db


def _maxpool(matrix:Tensor, kernel_size:KERNEL_SIZE, strides:STRIDES): # (H, W)
        (H, W), (Hk, Wk), (Hs, Ws) = matrix.shape, kernel_size, strides
        output_shape = ((H-Hk+1)//Hs + 1, (W-Wk+1)//Ws + 1)
        indices, maxpooled = [], []
        for i in range(0, H - Hk + 1, Hs):
            for j in range(0, W - Wk + 1, Ws):
                window = matrix[i:i+Hk, j:j+Wk]
                max_index = torch.unravel_index(torch.argmax(window), window.shape)
                max_index_global = (max_index[0] + i, max_index[1] + j)
                indices.append(max_index_global)
                maxpooled.append(window[max_index])
        maxpooled = torch.tensor(maxpooled).reshape(output_shape)
        indices = torch.tensor(indices) # (H1*W1, 2)
        # (H1, W1), ((H1, W1), (H1, W1))
        return maxpooled, (indices[:, 0].reshape(output_shape), indices[:, 1].reshape(output_shape))

# channeled_maxpool = torch.vmap(_maxpool, in_dims=(0, None, None), out_dims=0) # (C, H, W)
# vmaxpool = torch.vmap(channeled_maxpool, in_dims=(0, None, None), out_dims=0) # (B, C, H, W)

def channeled_maxpool(matrix:Tensor, kernel_size:KERNEL_SIZE, strides:STRIDES): # (C, H, W)
    (C, H, W) = matrix.shape
    maxpooled, Rindices, Cindices = [], [], []
    for c in range(C):
        maxpooled_, (indices_r, indices_c) = _maxpool(matrix[c], kernel_size, strides)
        maxpooled.append(maxpooled_)
        Rindices.append(indices_r)
        Cindices.append(indices_c)
    return torch.stack(maxpooled), (torch.stack(Rindices), torch.stack(Cindices))


def vmaxpool(matrix:Tensor, kernel_size:KERNEL_SIZE, strides:STRIDES): # (B, C, H, W)
    (B, C, H, W) = matrix.shape
    maxpooled, Rindices, Cindices = [], [], []
    for b in range(B):
        maxpooled_b, (indices_r, indices_c) = channeled_maxpool(matrix[b], kernel_size, strides)
        maxpooled.append(maxpooled_b)
        Rindices.append(indices_r)
        Cindices.append(indices_c)
    return torch.stack(maxpooled), (torch.stack(Rindices), torch.stack(Cindices))


def maxpool2d_forward(
    x:Tensor,
    kernel_size:KERNEL_SIZE,
    strides:STRIDES,
) -> tuple[Tensor, tuple[Tensor, Tensor], torch.Size]:
    O, (ridx, cidx) = vmaxpool(x, kernel_size, strides)
    return O, (ridx, cidx), x.shape


def maxpool2d_backward(
    dL_dO:Tensor,
    x_shape:torch.Size,
    indices:tuple[Tensor, Tensor],
):
    """SOMETHING IS WRONG"""
    (ridx, cidx) = indices

    dL_dY = torch.zeros(x_shape) # (B, C, H, W)
    dL_dY[:, :, ridx, cidx] += dL_dO

    dY_dX = torch.zeros(x_shape) # (B, C, H, W)
    dY_dX[:, :, ridx, cidx] += 1.0

    dL_dX = dL_dY*dY_dX # (B, C, H, W)
    return dL_dX # (B, C, H, W)


def relu_forward(x:Tensor):
    return torch.maximum(x, torch.tensor(0))


def relu_backward(relu:Tensor, dL_dO:Tensor):
    dO_dx = relu * dL_dO
    dL_dx = dL_dO * dO_dx
    return dL_dx


def softmax_forward(logits:Tensor):
    max_val, idx = logits.max(-1, keepdim=True) 
    logits -= max_val
    exp = torch.exp(logits)
    proba = exp / exp.sum(-1, keepdim=True)
    return proba

def softmax_backward(probs:Tensor, dL_dprobs:Tensor):
    nc = probs.shape[-1]
    t1 = torch.einsum("ij,ik->ijk", probs, probs) # (B, nc, nc)
    t2 = torch.einsum("ij,jk->ijk", probs, torch.eye(nc, nc)) # (B, nc, nc)
    dprobs_dlogits = t2 - t1 # (B, nc, nc)

    dL_dlogits = (dL_dprobs[:, None, :] @ dprobs_dlogits)[:, 0, :] # ((B, 1, nc) @ (B, nc, nc))[:, 0, :]
    return dL_dlogits # (B, nc)


def cross_entropy_forward(y_true:Tensor, y_proba:Tensor):
    log_probs = torch.log(y_proba) # (B, nc)
    loss = -log_probs[torch.arange(len(y_true)), y_true].mean()
    return loss

def cross_entropy_backward(y_true:Tensor, y_proba:Tensor):
    B = len(y_true)
    dL_dlogprobas = torch.zeros_like(y_proba) # (B, nc)
    dL_dlogprobas[torch.arange(B), y_true] = -1/B

    dlogprobas_dprobas = 1/y_proba # (B, nc)

    dL_dprobas = dL_dlogprobas * dlogprobas_dprobas # (B, nc)
    return dL_dprobas # (B, nc)

