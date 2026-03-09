import torch
import math
from typing import Optional
from collections.abc import Callable, Iterable

class SGD(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr=1e-3
            ):
        if lr < 0:
            raise ValueError(
                f"Invalid learning rate: {lr}")
        defaults = {"lr":lr}
        super().__init__(
                params, 
                defaults)

    def step(
            self,
            closure:Optional[Callable]=None
            ):
        loss = None if closure is None \
                else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue 
                state = self.state[p] 
                t = state.get("t", 0)
                grad = p.grad.data 
                p.data -= lr/math.sqrt(t+ 1)*grad
                state["t"] = t + 1
        
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params:Optional[\
                list[torch.nn.Parameter]]=None,
            lr:float=1e-3,
            betas:tuple[float,float]=(0.9, 0.999),
            eps:float=1e-8,
            weight_decay:float=0.01
            ):
        if lr < 0:
            raise ValueError(
                f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(
                f"Invalid learning rate: {eps}")
        if weight_decay < 0:
            raise ValueError(
            f"Invalid learning rate: {weight_decay}")
        if not 0.0 <= betas[0] < 1.0 or \
                not 0.0 <= betas[1] < 1.0:
            raise ValueError(
            f"Invalid beta: {betas[0]}, {betas[1]}")
        defaults = {
                "lr":lr,
                "betas":betas,
                "eps":eps,
                "weight_decay":weight_decay
                }
        if params is None:
            params = []
        super().__init__(
                params,
                defaults
                )
    def step(
            self,
            closure:Optional[Callable]=None
            ):
        loss = None if closure is None \
                else closure()
        for group in self.param_groups:
            lr = group['lr'] 
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = \
                    group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get(
                        "m", 
                        torch.zeros(p.data.shape))
                v = state.get(
                        "v", 
                        torch.zeros(p.data.shape))
                grad = p.grad.data
                #update shift avg grad and avg grad^2
                m = beta1*m+(1-beta1)*grad
                v = beta2*v+(1-beta2)*grad**2
                #bias correction
                _m = m/(1-beta1**t)
                _v = v/(1-beta2**t)
                #update parameters
                p.data -= lr*_m/(torch.sqrt(_v)+eps)
                #decoupled weight decay
                p.data -= lr*weight_decay*p.data
                #update state
                state['t'] = t + 1
                state['m'] = m
                state['v'] = v

        return loss







