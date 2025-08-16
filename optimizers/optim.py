import torch
from torch import Tensor, nn
del torch.optim # ;)


class Optimizer:
    def __init__(self, *args, **kwargs): ...
    def step(self): raise NotImplementedError("Optimizer step not implemented")
    def zero_grad(self):
        for param in self.parameters:
            param.grad.zero_()


class SGD(Optimizer):
    def __init__(self, parameters:list[Tensor], learning_rate:float):
        super().__init__()
        self.parameters = parameters
        self.lr = learning_rate
    
    @torch.no_grad()
    def step(self):
        for param in self.parameters:
            param -= self.lr * param.grad


class SGDMomentum(Optimizer):
    """beta: Usually set to 0.9"""
    def __init__(self, parameters:list[Tensor], learning_rate:float, beta:float):
        super().__init__()
        self.parameters = parameters
        self.lr = learning_rate
        self.beta = beta
        self.vdw = [torch.zeros_like(param) for param in parameters]

    @torch.no_grad()
    def step(self):
        for param, velocity in zip(self.parameters, self.vdw):
            velocity.mul_(self.beta).add_(param.grad, alpha=1 - self.beta)
            param -= self.lr * velocity


class RMSProp(Optimizer):
    """beta2: Usually set to 0.99"""
    def __init__(self, parameters:list[Tensor], learning_rate:float, beta2:float):
        super().__init__()
        self.parameters = parameters
        self.lr = learning_rate
        self.VSdw = [torch.zeros_like(param) for param in parameters]
        self.beta2 = beta2

    @torch.no_grad()
    def step(self):
        for param, vgrad_sq in zip(self.parameters, self.VSdw):
            vgrad_sq.mul_(self.beta2).add_(param.grad.square(), alpha=1 - self.beta2)
            param -= self.lr * param.grad / (torch.sqrt(vgrad_sq) + 1e-8)


class Adam(Optimizer):
    """| beta1: Usually set to 0.9 | beta2: Usually set to 0.999 |"""
    def __init__(self, parameters:list[Tensor], learning_rate:float, beta1:float, beta2:float, eps:float=1e-8):
        self.parameters = parameters
        self.lr = learning_rate
        self.betas = (beta1, beta2)
        self.eps = eps
        self.vdw = [torch.zeros_like(param) for param in parameters]
        self.VSdw = [torch.zeros_like(param) for param in parameters]
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        # velocity ema; squared velocity ema
        for param, vel_ema, sqvel_ema in zip(self.parameters, self.vdw, self.VSdw):
            vel_ema.mul_(self.betas[0]).add_(param.grad, alpha=1 - self.betas[0])
            sqvel_ema.mul_(self.betas[1]).addcmul_(param.grad, param.grad, value=1 - self.betas[1])
            # bias correction: affects only during first few steps
            m_hat = vel_ema / (1 - self.betas[0] ** self.t)
            v_hat = sqvel_ema / (1 - self.betas[1] ** self.t)
            param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)