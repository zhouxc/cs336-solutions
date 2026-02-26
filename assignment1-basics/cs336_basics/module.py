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


class Linear(nn.Module):
    def __init__(
            self, 
            din:int, 
            dout:int, 
            device:torch.device|None=None, 
            dtype:torch.dtype|None=None):
        super().__init__() 
        self.w = nn.Parameter(
                torch.empty(
                    dout, 
                    din,
                    device=device,
                    dtype=dtype
                    )
                )
        Linear._initialize_weight(
                self.w,
                dout=dout,
                din=din
            )
    
    @staticmethod
    def _initialize_weight(
            weight:nn.Parameter,
            dout:int,
            din:int):
        variance = 2.0/(din+dout)
        std = math.sqrt(variance)
        torch.nn.init.trunc_normal_(
                weight,
                mean=0.0,
                std=std,
                a=-3*std,
                b=3*std
        )

    def forward(
            self,
            x:Tensor
            )->Tensor:
        """
        Args:
            x:[...,d_in] tensor
        Returns:
            output:[...,d_out] tensor
        """
        y = torch.matmul(x, self.w.T)
        return y

class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings:int,
            embedding_dim:int,
            device:torch.device|None=None,
            dtype:torch.dtype|None=None):
        super().__init__()
        self.w = nn.Parameter(
                    torch.empty(
                        num_embeddings,
                        embedding_dim,
                        device=device,
                        dtype=dtype
                        )
                    )
        Embedding._initialize_weight(self.w)
    
    @staticmethod
    def _initialize_weight(
            weight:nn.Parameter):
        torch.nn.init.trunc_normal_(
                weight,
                mean=0.0,
                std=1,
                a=-3,
                b=3
            )

    def forward(
            self,
            token_ids:Tensor
            )->Tensor:
        """
        Args:
            token_ids:[...] tensor
        Returns:
            output:[...,d_model] tensor
        """
        return self.w[token_ids]

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model:int,
            eps:float=1e-5,
            device:torch.device|None=None,
            dtype:torch.dtype|None=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.w = nn.Parameter(
                    torch.empty(
                        d_model,
                        device=device,
                        dtype=dtype)
                )
        RMSNorm._initialize_weight(self.w)
    
    @staticmethod
    def _initialize_weight(
            weight:nn.Parameter):
        nn.init.ones_(weight)
    
    def _norm(
            eps:float,
            x:Tensor
            )->Tensor:
        mean_square = torch.mean(
                    x**2, 
                    dim=-1, 
                    keepdim=True)
        rms = torch.sqrt(
                mean_square + eps
            )
        return rms

    def forward(
            self,
            x:Tensor
            )->Tensor:
        """
        Args:
            x:[...,d_model] tensor
        Returns:
            output:[...,d_model] tensor
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = RMSNorm._norm(self.eps, x)
        x_norm = x/rms
        result = x_norm * self.w
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model:int,
            d_ff:int
            ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = \
                int(round(8/3*d_model))
        self.W1 = Linear(d_model, d_ff)
        self.W2 = Linear(d_ff, d_model)
        self.W3 = Linear(d_model, d_ff)
    
    @staticmethod
    def _silu(
            x:Tensor
            )->Tensor:
        sigmoid_x = torch.sigmoid(x)
        return x * sigmoid_x
    
    def _glu(
            x:Tensor,
            W1:Linear,
            W2:Linear
            )->Tensor:
        w1_x = W1.forward(x)
        w2_x = W2.forward(x)
        return torch.sigmoid(x_w1)*w2_x
    
    def _swiglu(
            x:Tensor,
            W1:Linear,
            W2:Linear,
            W3:Linear
            )->Tensor:
        w1_x = W1.forward(x)
        silu_w1x = SwiGLU._silu(w1_x)
        w3_x = W3.forward(x)
        glu_x = silu_w1x*w3_x
        return W2.forward(glu_x)
    
    def load_state_dict(
            self,
            w1:Float[Tensor,"d_ff d_model"],
            w2:Float[Tensor,"d_model d_ff"],
            w3:Float[Tensor,"d_ff d_model"],
            ):
        self.W1.load_state_dict({'w': w1})
        self.W2.load_state_dict({'w': w2})
        self.W3.load_state_dict({'w': w3})

    def forward(
            self,
            x:Tensor
            )->Tensor:
        """
        Args:
            x:[...,d_model] tensor
        Returns:
            output:[...,d_model] tensor
        """
        result = SwiGLU._swiglu(
                    x,
                    self.W1,
                    self.W2,
                    self.W3)
        return result

class RoPE(nn.Module):
    def __init__(
            self,
            theta:float,
            d_k:int,
            max_seq_len:int,
            device:torch.device|None=None
            ):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        angles = RoPE._compute_rope_angles(
                theta, d_k, max_seq_len, 
                device=device)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        self.register_buffer(
            'cos', cos, persistent=False)
        self.register_buffer(
            'sin', sin, persistent=False)
    
    def _compute_rope_angles(
            theta:float,
            d_k:int,
            max_seq_len:int,
            dtype:torch.dtype=torch.float32,
            device:torch.device|None=None
            )->Tensor:
        i = torch.arange(
                    max_seq_len, 
                    dtype=dtype, 
                    device=device)
        k = torch.arange(
                    0, d_k, 2, 
                    dtype=dtype,
                    device=device
                    )/d_k
        theta_k = 1.0/(theta**k)
        angles = torch.einsum(
            'i,j->ij',i,theta_k)
        return angles
        
    def forward(
            self,
            x:Tensor,
            token_positions:Tensor
            )->Tensor:
        """
        Args:
            x:[...,seq_len,d_k] tensor
            token_positions:[...,seq_len] tensor
        Returns:
            output:[...,seq_len,d_k] tensor
        """
        seq_len = x.shape[-2]
        d_k = x.shape[-1]
        assert d_k == self.d_k, \
            f"RoPE:dim {d_k}!={self.d_k}"
        assert d_k % 2 == 0, \
            f"RoPE:d_k:{dk} must even"
        x_reshape = x.view(
            *x.shape[:-1], d_k//2, 2)
        x0 = x_reshape[...,0]
        x1 = x_reshape[...,1]
        max_pos = token_positions.max().item()
        assert max_pos < self.max_seq_len,\
            f"RoPE:token position {max_pos}"\
            f">= max_seq_len {self.max_seq_len}"
        sin_pos = self.sin[token_positions]
        cos_pos = self.cos[token_positions]
        x0_rot = \
            torch.einsum(
            '...sk,...sk->...sk',x0,cos_pos) - \
            torch.einsum(
            '...sk,...sk->...sk',x1,sin_pos)
        x1_rot = \
            torch.einsum(
            '...sk,...sk->...sk',x0,sin_pos) + \
            torch.einsum(
            '...sk,...sk->...sk',x1,cos_pos)
        x_rot = torch.stack(
                [x0_rot, x1_rot], 
                dim=-1
                ).view(x.shape)
        
        return x_rot

class MHSAttention(nn.Module):
    def __init__(
            self,
            d_model:int,
            num_heads:int,
            theta:float|None=None,
            max_seq_len:int|None=None,
            device:torch.device|None=None,
            dtype:torch.dtype|None=None):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = Linear(
                d_model,
                3*self.head_dim*self.num_heads,
                device=device, 
                dtype=dtype)
        self.o = Linear(
                self.head_dim*self.num_heads,
                d_model,
                device=device, 
                dtype=dtype)
        self.rope = None
        if theta is not None and \
                max_seq_len is not None:
            self.rope = RoPE(
                    theta=theta, 
                    d_k=self.head_dim,
                    max_seq_len=max_seq_len,
                    device=device)
    
    def load_state_dict(
            self,
            q_w:Tensor,
            k_w:Tensor,
            v_w:Tensor,
            o_w:Tensor
            ):
        qkv_w = torch.concat([q_w,k_w,v_w], dim=0)
        self.qkv.load_state_dict({'w':qkv_w})
        self.o.load_state_dict({'w': o_w})
    
    def calc_qkv(
            self,
            x:Tensor
            )->tuple[Tensor, Tensor, Tensor]:
        #x [...,seq_len,d_model]
        #qkv.w [3*num_heads*head_dim,d_model]
        qkv = self.qkv.forward(x)
        #qkv [...,seq_len,3*num_heads*head_dim]
        qkv = qkv.view(
                *qkv.shape[:-2], 
                qkv.shape[-2],
                3,
                self.num_heads,
                self.head_dim
                )
        #qkv [...,seq_len,3,num_heads,head_dim]
        q = qkv[...,:,0,:,:].transpose(-3,-2) 
        k = qkv[...,:,1,:,:].transpose(-3,-2)
        v = qkv[...,:,2,:,:].transpose(-3,-2)
        #q,k,v [...,num_heads,seq_len,head_dim]
        return q, k, v

    def calc_mhsa(
            self,
            q:Tensor,
            k:Tensor,
            v:Tensor,
            causal_mask:Tensor
            )->Tensor:
        #q, k, v [...,num_heads,seq_len,head_dim]
        mhsa = scaled_dot_product_attention(
                    q=q,
                    k=k,
                    v=v,
                    mask=causal_mask
                )
        #mhsa.shape [...,num_heads,seq_len,head_dim]
        mhsa = mhsa.transpose(-3,-2).contiguous()
        #mhsa.shape [...,seq_len,num_heads,head_dim]
        mhsa = mhsa.view(
                    *mhsa.shape[:-3],
                    mhsa.shape[-3],
                    self.head_dim*self.num_heads
                    )
        #mhsa.shape [...,seq_len,d_model]
        return mhsa

    def forward(
            self,
            x:Tensor,
            token_positions:Tensor|None=None
            )->Tensor:
        """
        Args:
            x:[...,seq_len,d_in] tensor
        Returns:
            output:[...,seq_len,d_out] tensor
        """
        q, k, v = self.calc_qkv(x)
        if self.rope is not None and \
                token_positions is not None:
            q = self.rope.forward(
                    q, token_positions)
            k = self.rope.forward(
                    k, token_positions)
        causal_mask = torch.tril(
            torch.ones(
                x.shape[-2], 
                x.shape[-2],
                device=self.device)).bool()
        mhsa = self.calc_mhsa(
                q, k, v, causal_mask)
        output = self.o.forward(mhsa)
        return output



