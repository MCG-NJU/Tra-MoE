import math
import os
from collections import defaultdict
from functools import partial
from random import random
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
#from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import rearrange, reduce, repeat
#from .mar.diffloss import DiffLoss, SimpleMLPAdaLN
from omegaconf import DictConfig
from timm.models.vision_transformer import PatchEmbed, LayerScale, Mlp, DropPath, Attention, Block

# from timm.models.vision_transformer import Attention as _Attention
from torch import nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

#from .diffusion.mask_generator import LowdimMaskGenerator
#from .diffusion.conditional_unet1d import ConditionalUnet1D
from atm.policy.vilt_modules.language_modules import *
from atm.utils.flow_utils import ImageUnNormalize, tracks_to_video
from atm.utils.pos_embed_utils import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed
from einops.layers.torch import Rearrange


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


import math
from inspect import isfunction

MIN_EXPERT_CAPACITY = 4


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.gamma = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        
        self.register_buffer("beta", torch.zeros(dim, dtype=torch.float32))

    def forward(self, x):
        
        if x.dtype != torch.float32:
            x = x.to(torch.float32)  

        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)




def default2(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

# tensor related helper functions

def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error
def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)


# expert class

class Experts(nn.Module):
    def __init__(self,
        dim,
        num_experts = 6,
        hidden_dim = None,
        activation = nn.GELU):
        super().__init__()

        hidden_dim = default2(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim, dtype=torch.float32)
        w2 = torch.zeros(*num_experts, hidden_dim, dim, dtype=torch.float32)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

        self.dropout = nn.Dropout(0.2)

        self.norm = LayerNorm(dim)


    def forward(self, x):

        x = x.to(torch.float32)
        x = self.norm(x)
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        hidden = self.dropout(self.act(hidden))

        out = self.dropout(torch.einsum('...nh,...hd->...nd', hidden, self.w2))

        return out


class Top2Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        outer_expert_dims = tuple(),
        second_policy_train = 'none',
        second_policy_eval = 'none',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train= 6.0,
        capacity_factor_eval= 6.0):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates, dtype=torch.float32))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    def log(self, t, eps = 1e-20):
        return torch.log(t.clamp(min = eps))

    
    def gumbel_noise(self, t):
        noise = torch.zeros_like(t).uniform_(0, 1)
        return -self.log(-self.log(noise))
    



    def forward(self, x, importance = None):
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval
            capacity_factor = self.capacity_factor_eval

        #print(444)
        #print(x.dtype)

        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)


        router_z_loss = torch.logsumexp(raw_gates, dim = -1)
        router_z_loss = torch.square(router_z_loss)
        router_z_loss = router_z_loss.mean()        

        raw_gates = raw_gates.softmax(dim=-1)


        gate_1, index_1 = top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()
        density_1_proxy = raw_gates

        if importance is not None:
            equals_one_mask = (importance == 1.).float()
            mask_1 *= equals_one_mask[..., None]
            gate_1 *= equals_one_mask
            density_1_proxy = density_1_proxy * equals_one_mask[..., None]
            del equals_one_mask

        gates_without_top_1 = raw_gates * (1. - mask_1)

        gate_2, index_2 = top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        if importance is not None:
            greater_zero_mask = (importance > 0.).float()
            mask_2 *= greater_zero_mask[..., None]
            del greater_zero_mask

        # normalize top2 gate scores
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        # BALANCING LOSSES
        # shape = [batch, experts]
        # We want to equalize the fraction of the batch assigned to each expert
        density_1 = mask_1.mean(dim=-2)
        # Something continuous that is correlated with what we want to equalize.
        density_1_proxy = density_1_proxy.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

        # Depending on the policy in the hparams, we may drop out some of the
        # second-place experts.
        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # COMPUTE ASSIGNMENT TO EXPERTS
        # [batch, group, experts]
        # This is the position within the expert's mini-batch for this sequence
        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
        # Remove the elements that don't fit. [batch, group, experts]
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        # [batch, experts]
        # How many examples in this sequence go to this expert
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        # [batch, group] - mostly ones, but zeros where something didn't fit
        mask_1_flat = mask_1.sum(dim=-1)
        # [batch, group]
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        # Weight assigned to first expert.  [batch, group]
        gate_1 *= mask_1_flat

        position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)

        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 *= mask_2_flat
        
        # [batch, group, experts, expert_capacity]
        combine_tensor = (
            gate_1[..., None, None]
            * mask_1_flat[..., None, None]
            * F.one_hot(index_1, num_gates)[..., None]
            * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :] +
            gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * F.one_hot(index_2, num_gates)[..., None]
            * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        )

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        #return dispatch_tensor, combine_tensor, loss
        return dispatch_tensor, combine_tensor, router_z_loss









class MoE2(nn.Module):
    def __init__(self,
        dim,
        num_experts = 6,
        hidden_dim = None,
        activation = nn.GELU,
        second_policy_train = 'none',
        second_policy_eval = 'none',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train= 6.0,
        capacity_factor_eval= 6.0,
        loss_coef = 1e-4,
        experts = None):
        super().__init__()

        self.num_experts = num_experts
        #self.norm = LayerNorm(dim)

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        self.gate = Top2Gating(dim, num_gates = num_experts, **gating_kwargs)
        self.experts = default2(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef
        

    def forward(self, inputs, **kwargs):
        b, n, d, e = *inputs.shape, self.num_experts

        dispatch_tensor, combine_tensor, loss = self.gate(inputs)

        inputs = inputs.to(torch.float32)
        dispatch_tensor = dispatch_tensor.to(torch.float32)

    
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        expert_inputs = expert_inputs.to(torch.float32)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)


        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)


        expert_outputs = expert_outputs.to(torch.float32)
        combine_tensor = combine_tensor.to(torch.float32)



        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)



        output = output.to(torch.float32)
        loss = loss.to(torch.float32)

        #return output, 0.0

        return output



class MoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = 6,
        hidden_dim = None,
        activation = nn.GELU,
        second_policy_train = 'none',
        second_policy_eval = 'none',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train= 6.0,
        capacity_factor_eval= 6.0,
        loss_coef = 0.0,
        experts = None):
        super().__init__()

        self.num_experts = num_experts
        #self.norm = LayerNorm(dim)

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        self.gate = Top2Gating(dim, num_gates = num_experts, **gating_kwargs)
        self.experts = default2(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef
        

    def forward(self, inputs, **kwargs):

        bb, nn, tt, dd = inputs.shape
        inputs = inputs.reshape(bb, nn * tt, dd)  

        b, n, d, e = *inputs.shape, self.num_experts

        dispatch_tensor, combine_tensor, loss = self.gate(inputs)

        inputs = inputs.to(torch.float32)
        dispatch_tensor = dispatch_tensor.to(torch.float32)

    
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        expert_inputs = expert_inputs.to(torch.float32)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)


        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        expert_outputs = expert_outputs.to(torch.float32)
        combine_tensor = combine_tensor.to(torch.float32)

        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)

        #output = output.reshape(bb, nn, tt, dd).permute(0, 1, 3, 2)  # 恢复为 bntd
        output = output.reshape(bb, nn, tt, dd)  # 恢复为 bntd

        output = output.to(torch.float32)
        loss = loss.to(torch.float32)


        return output


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, cond_fn=None):
        x = self.norm(x)

        if exists(cond_fn):
            x = cond_fn(x)
        #breakpoint()

        return self.net(x)


























































































class CrossAttention(Attention):
    def __init__(self, dim: int, qkv_bias: bool = False, **kwargs):
        super().__init__(dim=dim, qkv_bias=qkv_bias, **kwargs)
        del self.qkv
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, q_input: torch.Tensor, kv_input: torch.Tensor) -> torch.Tensor:
        """
        `q_input`: query input tensor of shape `(B, N, T, C)`
        `kv_input`: key-value input tensor of shape `(B, L, C)`
        """
        B, N, T, C = q_input.shape
        _, L, _ = kv_input.shape

        q = (
            self.q_proj(q_input)
            .reshape(B, N, T, self.num_heads, self.head_dim)
            .permute(0, 3, 1, 2, 4)
        )
        kv = (
            self.kv_proj(kv_input)
            .reshape(B, L, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        k, v = k.unsqueeze(2), v.unsqueeze(2)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * (self.head_dim**-0.5)
            attn = q.unsqueeze(2) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = rearrange(x, "b h n t c -> b n t (h c)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class EncoderBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.mlp = MoE2(dim = 384)  # 使用 MoE

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        # self.mlp = mlp_layer(
        #     in_features=dim,
        #     hidden_features=int(dim * mlp_ratio),
        #     act_layer=act_layer,
        #     drop=proj_drop,
        # )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
        




class DecoderBlock(Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio=4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        moe: bool = False,  # 新增参数，控制是否使用 MoE
        ff_dropout: float = 0.0,  # FeedForward 的 dropout 参数
        **kwargs,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            norm_layer=norm_layer,
            drop_path=drop_path,
            act_layer=act_layer,
            mlp_layer=mlp_layer,
            **kwargs,
        )
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm3 = norm_layer(dim)
        self.ls3 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 动态初始化 self.mlp
        if moe:
            self.mlp = MoE(dim=dim)  # 使用 MoE
        else:
            self.mlp = FeedForward(dim=dim, dropout=0.2)  # 使用 FeedForward

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        b, n, _, c = tgt.shape
        tgt = rearrange(tgt, "b n t c -> (b n) t c")
        # Self-attention
        tgt = tgt + self.drop_path1(self.ls1(self.attn(self.norm1(tgt))))
        tgt = rearrange(tgt, "(b n) t c -> b n t c", n=n)
        # Cross-attention
        tgt = tgt + self.drop_path2(self.ls2(self.cross_attn(self.norm2(tgt), memory)))
        # Feed-forward
        tgt = tgt + self.drop_path3(self.ls3(self.mlp(self.norm3(tgt))))
        return tgt



# class DecoderBlock(Block):
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         mlp_ratio=4.0,
#         qkv_bias: bool = False,
#         qk_norm: bool = False,
#         proj_drop: float = 0.0,
#         attn_drop: float = 0.0,
#         init_values: Optional[float] = None,
#         drop_path: float = 0.0,
#         act_layer: nn.Module = nn.GELU,
#         norm_layer: nn.Module = nn.LayerNorm,
#         mlp_layer: nn.Module = Mlp,
#         **kwargs,
#     ) -> None:
#         super().__init__(
#             dim=dim,
#             num_heads=num_heads,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             qk_norm=qk_norm,
#             proj_drop=proj_drop,
#             attn_drop=attn_drop,
#             init_values=init_values,
#             norm_layer=norm_layer,
#             drop_path=drop_path,
#             act_layer=act_layer,
#             mlp_layer=mlp_layer,
#             **kwargs,
#         )
#         self.cross_attn = CrossAttention(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_norm=qk_norm,
#             attn_drop=attn_drop,
#             proj_drop=proj_drop,
#             norm_layer=norm_layer,
#         )
#         self.norm3 = norm_layer(dim)
#         self.ls3 = (
#             LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         )
#         self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

#     def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
#         b, n, _, c = tgt.shape
#         tgt = rearrange(tgt, "b n t c -> (b n) t c")
#         # Self-attention
#         tgt = tgt + self.drop_path1(self.ls1(self.attn(self.norm1(tgt))))
#         tgt = rearrange(tgt, "(b n) t c -> b n t c", n=n)
#         # Cross-attention
#         tgt = tgt + self.drop_path2(self.ls2(self.cross_attn(self.norm2(tgt), memory)))
#         # Feed-forward
#         tgt = tgt + self.drop_path3(self.ls3(self.mlp(self.norm3(tgt))))
#         return tgt


class TrackACT(nn.Module):
    img_patch_size = 16

    def __init__(
        self,
        num_track_ts,
        num_track_ids,
        track_dim,
        frame_stack,
        cond_dim,
        include_intrinsics,
        # mar
        inference_config,
        encoder_embed_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=16,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        mask_ratio_min=0.7,
        label_drop_prob=0.1,
        attn_dropout=0.1,
        proj_dropout=0.1,
        grad_checkpointing=False,
        # others
        data_mean=None,
        data_std=None,
        pe_L=12,
        load_path=None,
        use_ar_loss=True,
        ar_loss_weight=1.0,
        pure_3d=False,
        **kwargs,
    ):
        super().__init__()
        track_patch_size = 1

        #breakpoint()

        self.frame_stack = frame_stack
        self.num_track_ts = num_track_ts
        self.num_track_ids = num_track_ids
        self.track_dim = track_dim
        self.cond_dim = cond_dim
        self.include_intrinsics = include_intrinsics and self.track_dim > 2

        self.token_embed_dim = track_dim
        self.inference_config = inference_config
        self.label_drop_prob = label_drop_prob
        self.track_patch_size = track_patch_size
        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.img_size = img_size = 128
        self.seq_len = self.num_track_ts // track_patch_size
        self.grad_checkpointing = grad_checkpointing
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        # self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = (
            (img_size // self.img_patch_size) ** 2 + 1 + int(self.include_intrinsics)
        )  # + self.num_track_ids
        self.encoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.buffer_size, encoder_embed_dim)
        )

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    encoder_embed_dim,
                    encoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_dropout,
                    attn_drop=attn_dropout,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(
            torch.zeros(1, self.seq_len - 1, decoder_embed_dim)
        )
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim)
        )

        # self.decoder_blocks = nn.ModuleList(
        #     [
        #         DecoderBlock(
        #             decoder_embed_dim,
        #             decoder_num_heads,
        #             mlp_ratio,
        #             qkv_bias=True,
        #             norm_layer=norm_layer,
        #             proj_drop=proj_dropout,
        #             attn_drop=attn_dropout,
        #             moe = False,
        #         )
        #         for _ in range(decoder_depth)
        #     ]
        # )



        self.decoder_blocks = nn.ModuleList(
        [
        DecoderBlock(
            decoder_embed_dim,
            decoder_num_heads,
            mlp_ratio,
            qkv_bias=True,
            norm_layer=norm_layer,
            proj_drop=proj_dropout,
            attn_drop=attn_dropout,
            moe=(i == 0 or i == decoder_depth // 2 or i == decoder_depth - 1)  # begin middle last
        )
        for i in range(decoder_depth)
        ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)


        self.use_ar_loss = True
        #self.use_ar_loss = use_ar_loss
        self.ar_loss_weight = ar_loss_weight
        self.pure_3d = pure_3d

        if self.use_ar_loss:
            self.ar_head = nn.Sequential(
                nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(
                    decoder_embed_dim,
                    self.token_embed_dim * self.track_patch_size,
                    bias=True,
                ),
                Rearrange(
                    "b ... l (p n) -> b ... (l p) n",
                    n=self.token_embed_dim,
                    p=self.track_patch_size,
                ),
            )

        self.pe_L = pe_L
        self.register_data_mean_std(data_mean, data_std)

        self._build_model()

        self.initialize_weights()

        if load_path is not None:
            self.load(load_path)
            print(f"loaded model from {load_path}")

        self.kwargs = kwargs

    def _build_model(self):
        self.img_size = (128, 128)

        self.img_normalizer = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.img_unnormalizer = ImageUnNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.img_patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.img_patch_size,
            in_chans=3 * self.frame_stack,
            embed_dim=self.cond_dim,
        )
        self.num_img_patches = self.img_patch_embed.num_patches

        self.img_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_img_patches, self.cond_dim), requires_grad=False
        )  # fixed sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.img_pos_embed.shape[-1],
            (int(self.num_img_patches**0.5), int(self.num_img_patches**0.5)),
        )
        self.img_pos_embed.data.copy_(torch.from_numpy(pos_embed).unsqueeze(0))

        self.query_track_embed = MLPEncoder(
            input_size=2 + 2 * self.pe_L if not self.pure_3d else 3 + 3 * self.pe_L,
            hidden_size=128,
            num_layers=1,
            output_size=self.cond_dim,
        )

        self.language_encoder = MLPEncoder(
            input_size=768,
            hidden_size=128,
            num_layers=1,
            output_size=self.cond_dim,
        )

        if self.include_intrinsics:
            self.intrinsics_encoder = MLPEncoder(
                input_size=4,
                hidden_size=128,
                num_layers=1,
                output_size=self.cond_dim,
            )

    def initialize_weights(self):
        # parameters
        # torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def track_embedding(self, coords):
        L = self.pe_L
        b, t, n, c = coords.shape
        if not self.pure_3d:
            assert c == 2, c
            x = coords[:, :, :, 0]
            y = coords[:, :, :, 1]
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-1)
            x = x / (2 ** (torch.arange(L, device=x.device, dtype=x.dtype) / L))
            y = y / (2 ** (torch.arange(L, device=y.device, dtype=y.dtype) / L))
            # x = x.squeeze(-2)
            # y = y.squeeze(-2)
            return torch.cat([x, y], dim=-1)
        else:
            assert c == 3, c
            x = coords[:, :, :, 0]
            y = coords[:, :, :, 1]
            z = coords[:, :, :, 2]
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-1)
            z = z.unsqueeze(-1)
            x = x / (2 ** (torch.arange(L, device=x.device, dtype=x.dtype) / L))
            y = y / (2 ** (torch.arange(L, device=y.device, dtype=y.dtype) / L))
            z = z / (2 ** (torch.arange(L, device=z.device, dtype=z.dtype) / L))
            return torch.cat([x, y, z], dim=-1)

    def register_data_mean_std(
        self,
        mean: Union[str, float, Sequence],
        std: Union[str, float, Sequence],
        namespace: str = "data",
    ):
        """
        Register mean and std of data as tensor buffer.

        Args:
            mean: the mean of data.
            std: the std of data.
            namespace: the namespace of the registered buffer.
        """
        for k, v in [("mean", mean), ("std", std)]:
            if isinstance(v, str):
                if v.endswith(".npy"):
                    v = torch.from_numpy(np.load(v))
                elif v.endswith(".pt"):
                    v = torch.load(v)
                else:
                    raise ValueError(f"Unsupported file type {v.split('.')[-1]}.")
            else:
                v = torch.tensor(v)
            self.register_buffer(f"{namespace}_{k}", v.float())

    def forward(self, mode, batch: Any):
        if mode == "loss":  # training
            return self.forward_train(batch)
        elif mode == "pred":  # validation
            return self.forward_pred(batch)
        elif mode == "inference":  # inference
            return self.forward_inference(batch)
        elif mode == "vis":
            return self.forward_vis(batch)

    def forward_encoder(self, gt, mask, condition_tokens):
        dtype = condition_tokens.dtype
        x = condition_tokens

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        return x

    def forward_decoder(self, x, mask, query_track_tokens):
        dtype = x.dtype

        x = self.decoder_embed(x)
        kv = x + self.decoder_pos_embed_learned[:, : self.buffer_size]

        mask_tokens = repeat(
            self.mask_token, "1 tl c -> b n tl c", b=x.shape[0], n=self.num_track_ids
        )
        mask_tokens = torch.cat([query_track_tokens, mask_tokens], dim=2)
        q = mask_tokens + self.decoder_pos_embed_learned[:, self.buffer_size :]

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, q, kv)
        else:
            for block in self.decoder_blocks:
                x = block(q, kv)
        x = self.decoder_norm(x)

        return x

    def forward_train(self, batch):
        condition_tokens, x, loss_weights, query_track_tokens = self._preprocess_batch(
            batch
        )
        gt_latents = x.clone().detach()
        mask = self.masking(x)

        # mae encoder
        x = self.forward_encoder(x, mask, condition_tokens)

        # mae decoder
        z = self.forward_decoder(x, mask, query_track_tokens)

        return_dict = dict()
        loss = 0
        if self.use_ar_loss:
            # ar loss
            ar_pred = self.ar_head(z)
            ar_loss = F.mse_loss(gt_latents, ar_pred, reduction="none") * loss_weights
            ar_loss = ar_loss.mean()
            return_dict["ar_loss"] = ar_loss.item()
            loss += self.ar_loss_weight * ar_loss

        return_dict["loss"] = loss

        #breakpoint()
        return loss, return_dict

    def forward_pred(self, batch):
        vids = batch["vids"].clone()
        # place them side by side
        combined_image = torch.cat(
            [vids[:1, -1], vids[:1, -1]], dim=-1
        )  # only visualize the current frame
        combined_image = torch.clamp(combined_image, 0, 255)

        condition_tokens, x, loss_weights, query_track_tokens = self._preprocess_batch(
            batch
        )
        gt_latents = x.clone().detach()
        mask = self.masking(x)

        # mae encoder
        x = self.forward_encoder(x, mask, condition_tokens)
        # mae decoder
        z = self.forward_decoder(x, mask, query_track_tokens)
        if self.use_ar_loss:
            ar_tokens = self.ar_head(z)

        torch.cuda.empty_cache()

        action = self._unstack_and_unnormalize(gt_latents)
        if self.use_ar_loss:
            ar_action_pred = self._unstack_and_unnormalize(ar_tokens)

        track = action[:1].clone()
        ar_rec_track_vid = None
        rec_track = ar_action_pred[:1].clone()
        rec_track_vid = tracks_to_video(rec_track, img_size=128)
        track_vid = tracks_to_video(track, img_size=128)
        combined_track_vid = torch.cat([track_vid, rec_track_vid], dim=-1)
        if ar_rec_track_vid is not None:
            combined_track_vid = torch.cat(
                [combined_track_vid, ar_rec_track_vid], dim=-1
            )
        combined_track_vid = combined_image * 0.25 + combined_track_vid * 0.75

        loss_weights = rearrange(
            loss_weights, "b n tl c -> b tl n c", n=self.num_track_ids
        )

        return_dict = dict(
            combined_track_vid=combined_track_vid.cpu().numpy().astype(np.uint8)
        )
        loss = 0
        if self.use_ar_loss:
            ar_loss = (
                F.mse_loss(action, ar_action_pred, reduction="none") * loss_weights
            )
            ar_loss = ar_loss.mean()
            return_dict["ar_loss"] = ar_loss.item()
            loss += self.ar_loss_weight * ar_loss

        return_dict["track_loss"] = return_dict["ar_loss"]

        return_dict["loss"] = loss
        return return_dict

    @torch.inference_mode()
    def forward_inference(self, batch, mini_batch_size=10240):
        batch["is_inference"] = True
        condition_tokens_, x_, _, query_track_tokens_ = self._preprocess_batch(batch)
        all_tokens = []
        for i in range(0, x_.shape[0], mini_batch_size):
            condition_tokens = condition_tokens_[i : i + mini_batch_size]
            query_track_tokens = query_track_tokens_[i : i + mini_batch_size]
            x = x_[i : i + mini_batch_size]

            gt_latents = x.clone().detach()
            mask = self.masking(x)

            # mae encoder
            x = self.forward_encoder(x, mask, condition_tokens)
            # mae decoder
            z = self.forward_decoder(x, mask, query_track_tokens)
            tokens = self.ar_head(z)

            torch.cuda.empty_cache()
            all_tokens.append(tokens)
        all_tokens = torch.cat(all_tokens, dim=0)
        action_pred = self._unstack_and_unnormalize(all_tokens)
        torch.cuda.empty_cache()
        return action_pred

    def forward_vis(self, batch):
        vids = batch["vid"].clone()
        # place them side by side
        combined_image = vids[:1, -1].clone()
        combined_image = torch.clamp(combined_image, 0, 255)

        action_pred = self.forward_inference(batch)
        torch.cuda.empty_cache()

        rec_track = action_pred[:1].clone()
        rec_track_vid = tracks_to_video(rec_track, img_size=128)

        combined_track_vid = combined_image * 0.25 + rec_track_vid * 0.75

        ret_dict = {
            #"combined_image": combined_image.cpu().numpy().astype(np.uint8),
            "combined_track_vid": combined_track_vid.cpu().numpy().astype(np.uint8),
        }


        return combined_track_vid.cpu().numpy().astype(np.uint8), ret_dict

    def masking(self, x):
        return torch.ones(x.shape[0:2], device=x.device, dtype=x.dtype)

    def _patchify_video(self, imgs):
        # imgs: b, fs, c, h, w
        imgs = rearrange(imgs, "b fs c h w -> (b fs) c h w")
        assert torch.max(imgs) >= 2
        imgs = self.img_normalizer(imgs / 255.0)
        patches = self.img_patch_embed(imgs) + self.img_pos_embed
        return patches

    def _patchify_query_tracks(self, tracks):
        # tracks: b, t=1, n, c=2
        b, t, n, c = tracks.shape
        assert t == 1, t

        track_embed = torch.cat(
            [self.track_embedding(tracks).squeeze(1), tracks.squeeze(1)], dim=-1
        )  # b, n, 2L+2
        track_embed = self.query_track_embed(track_embed)  # b, n, c

        return track_embed

    def _get_intrinsic_tokens(self, intrinsics):
        intrinsics = (
            torch.cat(
                [
                    intrinsics[:, 0, 0].unsqueeze(1),
                    intrinsics[:, 1, 1].unsqueeze(1),
                    intrinsics[:, 0, 2].unsqueeze(1),
                    intrinsics[:, 1, 2].unsqueeze(1),
                ],
                dim=-1,
            )
            / 128
        )
        intrinsics = self.intrinsics_encoder(intrinsics).unsqueeze(1)  # b, 1, c
        return intrinsics

    def _preprocess_batch(self, batch):
        batch_size, n_frames = batch["vid"].shape[:2]
        #n_points = batch["tracks_2d"].shape[2]
        assert n_frames == self.frame_stack  # currently only support frame_stack == 1

        img_tokens = self._patchify_video(batch["vid"])
        # img_tokens = repeat(img_tokens, "b l c -> (b n) l c", n=n_points)

        if not self.pure_3d:
            query_track_tokens = self._patchify_query_tracks(
                batch["track"][:, :1]
            )  # b n 2
        else:
            query_track_tokens = self._patchify_query_tracks(
                batch["tracks_3d"][:, :1]
            )   # b n 3
        # query_track_tokens = rearrange(query_track_tokens, "b n c -> (b n) 1 c")
        query_track_tokens = rearrange(query_track_tokens, "b n c -> b n 1 c")
        language_tokens = self.language_encoder(batch["task_emb"])
        # language_tokens = repeat(language_tokens, "b c -> (b n) 1 c", n=n_points)
        language_tokens = repeat(language_tokens, "b c -> b 1 c")
        condition_tokens = torch.cat(
            [img_tokens, language_tokens], dim=1
        )  # b, l+n+1, c
        if self.include_intrinsics:
            intrinsic_tokens = self._get_intrinsic_tokens(batch["intrinsics"])
            # intrinsic_tokens = repeat(intrinsic_tokens, "b 1 c -> (b n) 1 c", n=n_points)
            condition_tokens = torch.cat([condition_tokens, intrinsic_tokens], dim=1)
        # condition_tokens = torch.cat([condition_tokens, query_track_tokens], dim=1)  # b, l+n+2, c

        if not batch.get("is_inference", False):
            if self.track_dim == 2:
                gt_tokens = batch["track"]
            elif self.track_dim == 3:
                gt_tokens = self._normalize_x(batch["tracks_3d"])
            elif self.track_dim == 5:
                gt_tokens = torch.cat(
                    [batch["tracks_2d"], self._normalize_x(batch["tracks_3d"])], dim=-1
                )
            # gt_tokens = rearrange(gt_tokens, "b tl n c -> (b n) tl c")
            gt_tokens = rearrange(gt_tokens, "b tl n c -> b n tl c")

            #masks = batch["vis"]  # b, tl, n
            #masks[masks == 0] = 0.1

            b, tl, n, _ = batch["track"].shape
            masks = torch.ones((b, tl, n)).to(batch["track"].device)


            masks = repeat(masks, "b tl n -> b tl n c", c=self.track_dim)
            if self.track_dim > 2:
                # remove outliers
                masks[batch["tracks_3d"][..., -3] > 10][..., -3:] = 0.0
                masks[batch["tracks_3d"][..., -2] > 10][..., -3:] = 0.0
                masks[batch["tracks_3d"][..., -1] > 10][..., -3:] = 0.0

                # remove outside of image
                masks[batch["tracks_2d"][..., 0] < 0][..., -3:] = 0.0
                masks[batch["tracks_2d"][..., 1] < 0][..., -3:] = 0.0
                masks[batch["tracks_2d"][..., 0] > 1][..., -3:] = 0.0
                masks[batch["tracks_2d"][..., 1] > 1][..., -3:] = 0.0

            # masks = rearrange(masks, "b tl n c -> (b n) tl c")
            masks = rearrange(masks, "b tl n c -> b n tl c")
        else:
            gt_tokens = torch.zeros(
                (*batch["track"].shape[:3], self.track_dim),
                device=condition_tokens.device,
                dtype=condition_tokens.dtype,
            )
            # gt_tokens = rearrange(gt_tokens, "b tl n c -> (b n) tl c")
            gt_tokens = rearrange(gt_tokens, "b tl n c -> b n tl c")
            masks = torch.ones_like(gt_tokens)

        return condition_tokens, gt_tokens, masks, query_track_tokens

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return xs * std + mean

    def _unstack_and_unnormalize(self, xs):
        # xs = rearrange(
        #     xs, "(b n) tl c -> b tl n c", n=self.num_track_ids, c=self.track_dim, tl=self.num_track_ts
        # )
        xs = rearrange(
            xs,
            "b n tl c -> b tl n c",
            n=self.num_track_ids,
            c=self.track_dim,
            tl=self.num_track_ts,
        )
        if self.track_dim > 2:
            xs[..., -3:] = self._unnormalize_x(xs[..., -3:])
        return xs

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))
