import torch
from torch import Tensor, nn
del torch.optim # ;)


class Optimizer:
    def step(self): 
        raise NotImplementedError("Optimizer step not implemented")
    def zero_grad(self):
        for param in self.parameters: # type: ignore # vs code giving annoying swiggly lines
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
    """
    MomentUm Orthogonalized Newton-Schultz\n
    Reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
    """
    def __init__(self, parameters:list[Tensor], learning_rate:tuple[float, float], beta:float=0.95, adam_betas:tuple[float, float]=(0.9, 0.999)):
        self.muon_params = [param for param in parameters if param.ndim > 1]
        self.lr = learning_rate[0]
        self.beta1 = beta
        self.vdw = [torch.zeros_like(param) for param in self.muon_params]

        self.other_params = [param for param in parameters if param.ndim <= 1]
        self.adam = Adam(
            parameters=self.other_params,
            learning_rate=learning_rate[1],
            beta1=adam_betas[0],
            beta2=adam_betas[1]
        )
    
    @staticmethod
    def zeropower_via_newtonschulz5(G:Tensor, steps:int):
        """
        Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. 
        This iteration therefore does not produce UV^T but rather something like US'V^T
        where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
        performance at all relative to UV^T, where USV^T = G is the SVD.
        """
        assert G.ndim >= 2, f"Input tensor must be at least 2D, but got {G.ndim}D"  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
        a, b, c = (3.4445, -4.7750,  2.0315)
        X = G.bfloat16()
        if G.size(-2) > G.size(-1): # If the input matrix is "tall" (more rows than columns), transpose it to make it "wide"
            X = X.mT # mT transposes the last 2 dimensions

        # Ensure spectral norm is at most 1
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        # Perform the NS iterations
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
            X = a * X + B @ X
        
        if G.size(-2) > G.size(-1):
            X = X.mT # convert back to original shape
        return X
    
    @staticmethod
    def muon_update(grad:Tensor, velocity:Tensor, beta:float=0.95, ns_steps:int=5, nesterov:bool=True):
        shape = grad.shape
        # momentum = beta * momentum + (1 - beta) * grad
        velocity.lerp_(grad, 1 - beta)
        # if nesterov: update = beta * momentum + (1 - beta) * grad
        # else:        update = momentum
        update = grad.lerp_(velocity, beta) if nesterov else velocity
        if update.ndim == 4: # for the case of conv filters
            update = update.view(len(update), -1)
        update = Muon.zeropower_via_newtonschulz5(update, steps=ns_steps)
        update.mul_(max(1, grad.size(-2) / grad.size(-1))**0.5) # scale by max(1, sqrt(rows/cols)) # IDK WHY?
        return update.reshape(shape)

    @torch.no_grad()
    def step(self):
        for param, velocity in zip(self.muon_params, self.vdw):
            if param.grad is None: continue
            # compute the velocity
            # Newton-Schultz: Orthogonalize the the velocity
            orthog_vel = Muon.muon_update(param.grad, velocity, beta=self.beta1, ns_steps=5, nesterov=True)
            # update the parameter using orthogonalized velocity
            param -= self.lr * orthog_vel.reshape(param.shape)

        self.adam.step()

    def zero_grad(self):
        for param in self.muon_params:
            param.grad = None
        self.adam.zero_grad()