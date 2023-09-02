from embedding.sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv
import torch
import math
import torch.nn.functional as F
from torch import nn


def scaled_dot_product(q, k, v, mask=None, w=16):
    d_k = q.size()[-1]
    attn_logits = sliding_chunks_matmul_qk(q=q, k=k, w=w, padding_value=0)
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


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
