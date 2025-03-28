import torch
import torch.nn.functional as F
from torch import nn, einsum

import torch.nn as nn

import os

from einops import rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

import math
from inspect import isfunction

MIN_EXPERT_CAPACITY = 4 #default

# float32 for downstream (policy), bfloat16 for upstream (track)

def get_dtype():
    use_bfloat16 = os.getenv("USE_BFLOAT16", "false").lower() == "true"
    return torch.bfloat16 if use_bfloat16 else torch.float32



class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        data_dtype = get_dtype()
        
        self.gamma = nn.Parameter(torch.ones(dim, dtype=data_dtype))
        
        self.register_buffer("beta", torch.zeros(dim, dtype=data_dtype))

    def forward(self, x):
        data_dtype = get_dtype()
        x = x.to(data_dtype)  

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
# default 4 experts
class Experts(nn.Module):
    def __init__(self,
        dim,
        num_experts = 4,
        hidden_dim = None,
        activation = nn.GELU):
        super().__init__()

        hidden_dim = default2(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)
        data_dtype = get_dtype()

        w1 = torch.zeros(*num_experts, dim, hidden_dim, dtype=data_dtype)
        w2 = torch.zeros(*num_experts, hidden_dim, dim, dtype=data_dtype)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

        self.dropout = nn.Dropout(0.2)

        self.norm = LayerNorm(dim)


    def forward(self, x):
        data_dtype = get_dtype()

        x = x.to(data_dtype)
        x = self.norm(x)
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        hidden = self.dropout(self.act(hidden))

        out = self.dropout(torch.einsum('...nh,...hd->...nd', hidden, self.w2))

        return out

# Top2Gating class
# default top-1
# default w/o balancing_loss, w/o noise, with router_z_loss
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
        capacity_factor_train = 4.0,
        capacity_factor_eval = 4.0):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        data_dtype = get_dtype()
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates, dtype=data_dtype))

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

# moe class
# default num_experts = 4
# balancing_loss = 0, w/o noise
class MoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = 4,
        hidden_dim = None,
        activation = nn.GELU,
        second_policy_train = 'none',
        second_policy_eval = 'none',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 4.0,
        capacity_factor_eval = 4.0,
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
        data_dtype = get_dtype()

        inputs = inputs.to(data_dtype)
        dispatch_tensor = dispatch_tensor.to(data_dtype)

        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        expert_inputs = expert_inputs.to(data_dtype)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)

        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        expert_outputs = expert_outputs.to(data_dtype)
        combine_tensor = combine_tensor.to(data_dtype)

        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)

        output = output.to(data_dtype)
        loss = loss.to(data_dtype)

        return output, loss * self.loss_coef



# baseline FFN block
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

        return self.net(x)


class TransformerAttention(nn.Module):
    def __init__(self, dim, causal=False, dim_head=None, dim_context=None, heads=8, norm_context=False, dropout=0.1):
        super().__init__()
        self.heads = heads
        if dim_head is None:
            dim_head = dim // heads
            inner_dim = dim_head * heads
        else:
            inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.causal = causal

        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, attn_bias=None, attn_mask=None, cond_fn=None):
        b = x.shape[0]
        if exists(context):
            context = self.context_norm(context)
        kv_input = default(context, x)

        x = self.norm(x)

        if exists(cond_fn):
            x = cond_fn(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        q = q * self.scale

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        if exists(attn_bias):
            sim = sim + attn_bias

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



# sparse moe
class Transformer(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, depth=6, attn_dropout=0., ff_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        middle_layer = depth // 2

        for i in range(depth):
            # default 3 moe layer
            if i in {0, middle_layer, depth - 1}:
            #if i in {0, 1, 2, 5, 6, 7}:
                ff_layer = MoE(dim=dim)
            else:
                ff_layer = FeedForward(dim=dim, dropout=ff_dropout)

            self.layers.append(
                nn.ModuleList(
                    [
                        TransformerAttention(dim=dim, heads=heads, dropout=attn_dropout, dim_head=dim_head),
                        ff_layer
                    ]))

    def forward(self, x, cond_fns=None, attn_mask=None):

        if not exists(cond_fns):
            cond_fns = (None,) * len(self.layers) * 2

        cond_fns = iter(cond_fns)

        total_loss = 0

        for attn, ff in self.layers:

            x = attn(x, attn_mask=attn_mask, cond_fn=next(cond_fns)) + x
            if isinstance(ff, MoE):

                x0, moe_loss = ff(x, cond_fn=next(cond_fns))
                x = x + x0

                total_loss += moe_loss
            else:
                x = ff(x, cond_fn=next(cond_fns)) + x

        return x, total_loss
        


# dense baseline
class Transformer_baselibe(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, depth=6, attn_dropout=0., ff_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        TransformerAttention(dim=dim, heads=heads, dropout=attn_dropout, dim_head=dim_head),
                        FeedForward(dim=dim, dropout=ff_dropout)
                    ]))

    def forward(self, x, cond_fns=None, attn_mask=None):
        if not exists(cond_fns):
            cond_fns = (None,) * len(self.layers) * 2

        cond_fns = iter(cond_fns)

        for attn, ff in self.layers:

            x = attn(x, attn_mask=attn_mask, cond_fn=next(cond_fns)) + x

            x = ff(x, cond_fn=next(cond_fns)) + x

        return x, 0.0



















































