import torch
from torch import Tensor, nn
del torch.optim # ;)


class Optimizer:
    def step(self): 
        raise NotImplementedError("Optimizer step not implemented")
    def zero_grad(self):
        for param in self.parameters: # type: ignore # vs code was giving annoying swiggly lines
            param.grad = None


class SGD(Optimizer):
    def __init__(self, parameters:list[Tensor], learning_rate:float):
        super().__init__()
        self.parameters = parameters
        self.lr = learning_rate
    
    @torch.no_grad()
    def step(self):
        for param in self.parameters:
            if param.grad is None: continue
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
            if param.grad is None: continue
            velocity.mul_(self.beta).add_(param.grad, alpha=self.lr)
            param -= velocity
        
        # NOTE(VachanVY): Compared this and above, the above converges faster and torch too follows the above method!
        # for param, velocity in zip(self.parameters, self.vdw):
        #     velocity.mul_(self.beta).add_(param.grad, alpha=1 - self.beta)
        #     param -= self.lr * velocity


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
            if param.grad is None: continue
            vgrad_sq.mul_(self.beta2).add_(param.grad.square(), alpha=1 - self.beta2)
            param -= self.lr * param.grad / (torch.sqrt(vgrad_sq) + 1e-8)


class Adam(Optimizer):
    """| beta1: Usually set to 0.9 | beta2: Usually set to 0.999 |"""
    def __init__(self, parameters:list[Tensor], learning_rate:float=1e-3, beta1:float=0.9, beta2:float=0.999, eps:float=1e-8):
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
            if param.grad is None: continue
            # vel_ema = vel_ema * beta1 + (1 - beta1) * grad
            vel_ema.mul_(self.betas[0]).add_(param.grad, alpha=1 - self.betas[0])
            # sqvel_ema = sqvel_ema * beta2 + (1 - beta2) * grad^2
            sqvel_ema.mul_(self.betas[1]).addcmul_(param.grad, param.grad, value=1 - self.betas[1])
            # bias correction: affects only during first few steps
            vel_ema_hat = vel_ema / (1 - self.betas[0] ** self.t)
            sqvel_ema_hat = sqvel_ema / (1 - self.betas[1] ** self.t)
            param -= self.lr * vel_ema_hat / (torch.sqrt(sqvel_ema_hat) + self.eps)


class Muon(Optimizer):
    """MomentUm Orthogonalized Newton-Shultz"""
    def __init__(self, parameters:list[Tensor], learning_rate:float, beta1:float)