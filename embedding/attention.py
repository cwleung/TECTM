import math

import torch
from torch import nn, einsum
import torch.nn.functional as F
from collections import namedtuple
from functools import wraps

from einops import rearrange

from embedding.sliding_chunks import sliding_chunks_matmul_pv, sliding_chunks_matmul_qk


# helper
def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


def scaled_dot_product(q, k, v, mask=None, w=16, val_mul="normal"):
    d_k = q.size()[-1]
    attn_logits = sliding_chunks_matmul_qk(q=q, k=k, w=w, padding_value=0)
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    # TODO add dropout
    if val_mul == "longformer":
        values = sliding_chunks_matmul_pv(prob=attention, v=v, w=w)
    elif val_mul == "global_attention":
        pass
    elif val_mul == "flash_attention":
        pass
    elif val_mul == "normal":
        values = torch.matmul(attention, v)
    else:
        raise NotImplementedError
    return values, attention


def flash_attn(q, k, v, mask=None, training=False, causal=False, dropout=0.0):
    _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

    # Recommended for multi-query single-key-value attention by Tri Dao
    # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

    k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)
    v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

    # Check if mask exists and expand to compatible shape
    # The mask is B L, so it would have to be expanded to B H N L

    if mask is not None:
        mask = rearrange(mask, 'b j -> b 1 1 j')
        mask = mask.expand(-1, heads, q_len, -1)

    # Check if there is a compatible device for flash attention

    # config = self.cuda_config if is_cuda else self.cpu_config

    # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask,
        dropout_p=dropout if training else 0.,
        is_causal=causal
    )

    return out


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        # project to [batch,seqlen->3*emb_dim(n_heads*head_dim)]
        qkv = self.qkv_proj(x)
        # project to [bz, seqlen n_head, head_dim]
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        # [Batch, Head, SeqLen, Dims]
        # qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        # choose attention
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        # values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        # reshape to (bsz, seqlen, emb_dim(n_head*head_dim))
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)
        if return_attention:
            return o, attention
        else:
            return o

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)


class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()

        # select self attention
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps


if __name__ == '__main__':
    print('test attention')

    print('test transformer encoder')

    print('test multihead attention')
