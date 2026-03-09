import math
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int

def softmax(
        x:torch.Tensor,
        dim:int
        )->torch.Tensor:
    """
    Args:
        x:[...,dim] tensor
        dim: applied in softmax
     Returns:
        output:[...,dim] tensor
    """
    max_v = torch.amax(
            x, dim=dim, keepdim=True)
    shift_x = x - max_v
    exp_x = torch.exp(shift_x)
    exp_sum = torch.sum(
            exp_x, dim=dim, keepdim=True)
    return exp_x/exp_sum

def log_softmax(
        x:Tensor,
        dim:int
        )->Tensor:
    """
    Args:
        x:[...,dim] tensor
        dim: applied in log_softmax
     Returns:
        output:[...,dim] tensor
    """
    max_v = torch.amax(
            x, dim=dim, keepdim=True)
    shift_x = x - max_v
    exp_x = torch.exp(shift_x)
    exp_sum = torch.sum(
            exp_x, dim=dim, keepdim=True)
    log_sum = torch.log(exp_sum)
    return shift_x - log_sum

def scaled_dot_product_attention(
        q:torch.Tensor,
        k:torch.Tensor,
        v:torch.Tensor,
        mask:torch.Tensor
        )->torch.Tensor:
    """
    Args:
        q:[..., n, d_k] query
        k:[..., m, d_k] key
        v:[..., m, d_v] value
        mask:[..., n, m] bool mask
     Returns:
        output:[..., n, d_v] attention
    """
    d_k = q.shape[-1]
    scores = torch.einsum(
        '...nk,...mk->...nm', q, k)\
        / math.sqrt(d_k)
    mask_value = torch.where(
            mask, 0, -float('inf'))
    scores = scores + mask_value
    attn_weights = softmax(
                    x=scores, 
                    dim=-1) 
    result = torch.einsum(
        '...nm,...mv->...nv',
        attn_weights, v)
    return result

def cross_entropy(
        x:Tensor,
        targets:Tensor
        )->Tensor:
    """
    Args:
        x:[batch_size,vocab_size]
        target:[batch_size]
    Returns:
        output:float
    """
    log_probs = log_softmax(x=x, dim=-1)
    selected_log_probs = log_probs.gather(
            dim = 1,
            index=targets.unsqueeze(1))
    loss = -selected_log_probs.mean()
    return loss

def get_lr_cosine_schedule(
        it:int,
        min_lr:float,
        max_lr:float,
        warmup_iters:int,
        cosine_cycle_iters:int
        )->float:
    """
    Args:
        it: current step
        min_lr: minimum learning rate
        max_lr: maximum learning rate
        warmup_iters: steps for linear warmup
        cosine_cycle_iters: steps for cosine decay
    Returns:
        lr: learning rate for current step
    """
    # linear warmup
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    # cosine decay
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) \
            / (cosine_cycle_iters - warmup_iters)
        cosine_factor = (1 + \
                math.cos(progress * math.pi))/2
        return min_lr + cosine_factor * \
                (max_lr - min_lr)
    # after cosine cycle, keep min_lr
    return min_lr

def gradient_clipping(
        params:list[torch.nn.Parameter],
        max_l2_norm:float,
        eps:float=10e-6
        )->None:
    """
    Args:
        params: list of model parameters
        max_l2_norm: maximum allowed L2 norm of gradients
        eps: small value to avoid division by zero
    """
    total_norm = 0
    # compute total L2 norm of gradients
    grads = [p.grad.data for p in params\
                if p.grad is not None]
    for grad in grads:
        total_norm += grad.norm()**2
    total_norm = math.sqrt(total_norm)
    
    if total_norm < max_l2_norm:
        return
    
    # compute clipping factor and scale gradients
    clip_factor = \
        max_l2_norm / (total_norm + eps)
    for grad in grads:
        grad *= clip_factor
    
