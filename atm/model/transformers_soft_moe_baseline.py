import torch
from torch.nn import Module
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn, einsum, Tensor

from einops import rearrange, pack, unpack
from torch.autograd import Function

import math
from inspect import isfunction

import torch.nn as nn




def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def chunk_num(num, chunks):
    num_per_chunk, remainder = divmod(num, chunks)

    out = []
    for i in range(chunks):
        n = num_per_chunk
        out.append(n + int(i < remainder))

    return out


def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

def all_gather_same_dim(t):
    world_size = dist.get_world_size()
    t = t.contiguous()
    gathered_tensors = [torch.empty_like(t, device = t.device, dtype = t.dtype) for i in range(world_size)]
    dist.all_gather(gathered_tensors, t)
    return gathered_tensors

def gather_sizes(t, *, dim):
    size = torch.tensor(t.shape[dim], device = t.device, dtype = torch.long)
    sizes = all_gather_same_dim(size)
    return torch.stack(sizes)

def has_only_one_value(t):
    return (t == t[0]).all()

def all_gather_variable_dim(t, dim = 0, sizes = None):
    device, rank, world_size = t.device, dist.get_rank(), dist.get_world_size()

    if not exists(sizes):
        sizes = gather_sizes(t, dim = dim)

    if has_only_one_value(sizes):
        gathered_tensors = all_gather_same_dim(t)
        gathered_tensors = torch.cat(gathered_tensors, dim = dim)
        return gathered_tensors, sizes

    max_size = sizes.amax().item()

    padded_t = pad_dim_to(t, max_size, dim = dim)
    gathered_tensors = all_gather_same_dim(padded_t)

    gathered_tensors = torch.cat(gathered_tensors, dim = dim)
    seq = torch.arange(max_size, device = device)

    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    gathered_tensors = gathered_tensors.index_select(dim, indices)

    return gathered_tensors, sizes

class AllGatherFunction(Function):
    @staticmethod
    def forward(ctx, x, dim, sizes):
        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        ctx.batch_sizes = batch_sizes.tolist()
        ctx.dim = dim
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        return grads_by_rank[rank], None, None

class AllGather(nn.Module):
    def __init__(self, *, dim = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x, sizes = None):
        return AllGatherFunction.apply(x, self.dim, sizes)

def split_by_rank(x):
    rank = dist.get_rank()
    out = x[rank]
    return out


def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return F.normalize(t, dim = - 1)

def cumsum_exclusive(t, dim = -3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim = dim)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.gamma = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))
        
        self.register_buffer("beta", torch.zeros(dim, dtype=torch.bfloat16))

    def forward(self, x):
        if x.dtype != torch.bfloat16:
           
            x = x.to(torch.bfloat16)  

        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)



class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return l2norm(x) * self.scale * self.gamma

# expert
def FeedForwardxxx(
    dim,
    mult = 4,
    dropout = 0.2
):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim),
        nn.Dropout(dropout)
    )

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def GLUFeedForward(
    dim,
    mult = 4,
    dropout = 0.2
):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.Linear(dim, dim_hidden * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim),
        nn.Dropout(dropout)
    )

# experts
class Experts(nn.Module):
    def __init__(
        self,
        experts,
        is_distributed = False,
        offload_unused_experts_to_cpu = True
    ):
        super().__init__()
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)

        self.is_distributed = is_distributed
        if not exists(self.is_distributed):
            self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        # whether to offload unused experts to cpu, will require optimizer handles conversion of gradients to right device when accumulating
        self.offload_unused_experts_to_cpu = offload_unused_experts_to_cpu

        self.all_gather = AllGather()
        self.register_buffer('dummy', torch.ones(1), persistent = False)


    @property
    def device(self):
        return self.dummy.device

    def all_experts_to_cpu_besides(self, selection):
        if not self.offload_unused_experts_to_cpu:
            return

        if isinstance(selection, int):
            experts = [self.experts[selection]]
        if isinstance(selection, slice):
            experts = self.experts[selection]
        else:
            experts = selection

        experts_set = set(experts)

        for expert in self.experts:
            device = self.device if expert in experts_set else 'cpu'
            expert.to(device)

    def forward(
        self,
        x,
        is_distributed = False
    ):
        """
        einops notation:
        b - batch
        r - rank (device / machines)
        e - experts
        n - sequence (number of tokens per expert)
        d - feature dimension
        """

        is_distributed = default(is_distributed, self.is_distributed)
        shape, num_experts = x.shape, self.num_experts

        # for now naively all gather across batch dimension if distributed, optimize later

        if is_distributed:
            seq_sizes = gather_sizes(x, dim = -2)
            assert has_only_one_value(seq_sizes), 'number of tokens per expert must be the same'

            x, batch_sizes = self.all_gather(x)
            total_batch_size = x.shape[0]

            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        # the experts in use on the rank

        if is_distributed:
            if world_size <= num_experts:
                num_experts_across_ranks = chunk_num(num_experts, world_size)
                start_indices = cumsum_exclusive(torch.tensor(num_experts_across_ranks), dim = -1)

                num_experts_per_rank = num_experts_across_ranks[rank]
                num_experts_batches_across_ranks = tuple(i * total_batch_size for i in num_experts_across_ranks)

                expert_start_index = start_indices[rank].item()
            else:
                num_batch_chunks = world_size // num_experts
                total_ranks_in_use = num_batch_chunks * num_experts

                expert_start_index = rank // num_batch_chunks

                batch_splits = chunk_num(total_batch_size, num_batch_chunks)
                num_experts_batches_across_ranks = batch_splits * num_experts

                # for now, remaining machines just process nothing

                remain_ranks = world_size % num_experts
                num_experts_batches_across_ranks += (0,) * remain_ranks

                num_experts_per_rank = int(rank < total_ranks_in_use)

            assert len(num_experts_batches_across_ranks) == world_size

            expert_slice = slice(expert_start_index, expert_start_index + num_experts_per_rank)
        else:
            num_experts_per_rank = num_experts
            expert_slice = slice(0, num_experts)

        # if distributed, each machine only handles subset of experts and batch

        x = rearrange(x, 'b e n d -> e b n d')

        if is_distributed:
            x, expert_batch_packed_shape = pack_one(x, '* n d')
            x = x.split(num_experts_batches_across_ranks, dim = 0)
            x = split_by_rank(x)

            if num_experts_per_rank > 0:
                x = rearrange(x, '(e b) n d -> e b n d', e = num_experts_per_rank)
            else:
                x = x.reshape(num_experts, *x.shape)

        # get the experts in use

        self.all_experts_to_cpu_besides(expert_slice)

        experts = self.experts[expert_slice]

        # route tokens to appropriate experts

        outs = []
        for expert, expert_input in zip(experts, x):
            out = expert(expert_input)
            outs.append(out)

        if len(outs) > 0:
            outs = torch.stack(outs)
        else:
            outs = torch.empty_like(x).requires_grad_()

        # all gather across merged expert batches dimensions
        # then split the batch dimension back

        if is_distributed:
            outs = rearrange(outs, 'e b n d -> (e b) n d')
            outs, _ = self.all_gather(outs)
            outs = unpack_one(outs, expert_batch_packed_shape, '* n d')

        outs = rearrange(outs, 'e b n d -> b e n d')

        if is_distributed:
            outs = outs.split(batch_sizes.tolist())
            outs = split_by_rank(outs)

        assert outs.shape == shape
        return outs

# main class
class SoftMoE(Module):
    def __init__(
        self,
        dim,
        *,
        seq_len = None,
        num_experts = 4,
        num_slots = None,
        expert_mult = 4,
        dropout = 0.2,
        geglu = False,
        is_distributed = False,
        offload_unused_experts_to_cpu = True,
        use_layernorm = True
    ):
        super().__init__()
        assert exists(seq_len) ^ exists(num_slots), 'either seq_len, or num_slots must be passed into SoftMoE'

        if exists(seq_len):
            num_slots = default(num_slots, seq_len // num_experts)
        elif exists(num_slots):
            seq_len = num_slots * num_experts

        norm_klass = LayerNorm if use_layernorm else RMSNorm
        self.norm = norm_klass(dim)

        self.slot_norm = norm_klass(dim)
        self.slot_embeds = nn.Parameter(torch.randn(num_experts, num_slots, dim))

        expert_klass = GLUFeedForward if geglu else FeedForwardxxx

        self.experts = Experts(
            experts = [expert_klass(dim = dim, mult = expert_mult, dropout = dropout) for _ in range(num_experts)],
            is_distributed = is_distributed,
            offload_unused_experts_to_cpu = offload_unused_experts_to_cpu
        )

    def forward(self, x, mask = None, add_noise = False, noise_mult = 1.):
        """
        einstein notation
        b - batch
        n - sequence length
        e - number of experts
        s - number of slots per expert
        d - feature dimension
        """

        is_single_token = x.ndim == 2
        is_image = x.ndim == 4

        if is_image:
            x = rearrange(x, 'b d h w -> b h w d')
            x, ps = pack([x], 'b * d')
        elif is_single_token:
            x = rearrange(x, 'b d -> b 1 d')

        # following Algorithm 1, with the normalization they proposed, but with scaling of both (the now popular rmsnorm + gamma)

        x = self.norm(x)
        slot_embeds = self.slot_norm(self.slot_embeds)


        logits = einsum('b n d, e s d -> b n e s', x, slot_embeds)


        # noised dispatch and combine gate logits, with annealing if needed

        if add_noise:
            noise = gumbel_noise(logits) * noise_mult
            logits = logits + noise

        # account for key padding mask

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

        # get dispatch and combine weights (softmax across right dimensions)

        dispatch_weights = logits.softmax(dim = 1)

        combine_weights = rearrange(logits, 'b n e s -> b n (e s)')
        combine_weights = combine_weights.softmax(dim = -1)

        # derive slots by weighted average of input tokens using the dispatch weights from above

        slots = einsum('b n d, b n e s -> b e s d', x, dispatch_weights)

        # route the slots per expert to each expert

        out = self.experts(slots)

        # combine back out

        out = rearrange(out, ' b e s d -> b (e s) d')
        out = einsum('b s d, b n s -> b n d', out, combine_weights)

        if is_image:
            out, = unpack(out, ps, 'b * d')
            out = rearrange(out, 'b h w d -> b d h w')
        elif is_single_token:
            out = rearrange(out, 'b 1 d -> b d')

        return out, 0.0





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




class Transformer(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, depth=6, attn_dropout=0., ff_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        middle_layer = depth // 2

        for i in range(depth):
            if i in {0, middle_layer, depth - 1}:
                #ff_layer = FeedForward3(dim=dim, dropout=ff_dropout)
                ff_layer = SoftMoE(dim = 384, seq_len = 400, num_experts = 6 )
                
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

        total_load_balancing_loss = 0

        for attn, ff in self.layers:

            
            x = attn(x, attn_mask=attn_mask, cond_fn=next(cond_fns)) + x
            if isinstance(ff, SoftMoE):
                #x, load_balancing_loss = ff(x) 
                
                x0, load_balancing_loss = ff(x)
                x = x + x0

                total_load_balancing_loss += load_balancing_loss
            else:
                x = ff(x, cond_fn=next(cond_fns)) + x


        return x, 0.0
        #return x




