import torch
import torch.nn as nn
import math
import einops
from fancy_einsum import einsum
from config import Config
import transformer_lens.utils as utils
from transformer_lens.hook_points import (HookPoint,)
from transformer_lens import HookedTransformer,FactoredMatrix
class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        scale = (x.pow(2).mean(dim=-1, keepdim=True) + self.cfg.layer_norm_eps).sqrt()
        return x / scale * self.w + self.b


class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_E = nn.Parameter(torch.empty(cfg.d_vocab, cfg.d_model))
        nn.init.normal_(self.W_E, std=cfg.init_range)

    def forward(self, tokens):
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_pos = nn.Parameter(torch.empty(cfg.n_ctx, cfg.d_model))
        nn.init.normal_(self.W_pos, std=cfg.init_range)

    def forward(self, tokens):
        pos = self.W_pos[:tokens.size(1)]
        return pos.unsqueeze(0).repeat(tokens.size(0), 1, 1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        self.W_K = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        self.W_V = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        self.W_O = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        self.b_Q = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))
        self.b_K = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))
        self.b_V = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))
        self.b_O = nn.Parameter(torch.zeros(cfg.d_model))
        nn.init.normal_(self.W_Q, std=cfg.init_range)
        nn.init.normal_(self.W_K, std=cfg.init_range)
        nn.init.normal_(self.W_V, std=cfg.init_range)
        nn.init.normal_(self.W_O, std=cfg.init_range)

    def forward(self, x):
        q = einsum("b p d, h d dh -> b p h dh", x, self.W_Q) + self.b_Q
        k = einsum("b p d, h d dh -> b p h dh", x, self.W_K) + self.b_K
        v = einsum("b p d, h d dh -> b p h dh", x, self.W_V) + self.b_V
        attn_scores = einsum("b q h d, b k h d -> b h q k", q, k)
        attn_scores /= math.sqrt(self.cfg.d_head)
        # causal mask
        mask = torch.triu(torch.ones_like(attn_scores), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask.bool(), float("-inf"))
        pattern = attn_scores.softmax(dim=-1)
        z = einsum("b h q k, b k h d -> b q h d", pattern, v)
        out = einsum("b p h d, h d m -> b p m", z, self.W_O) + self.b_O
        return out


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_in = nn.Parameter(torch.empty(cfg.d_model, cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty(cfg.d_mlp, cfg.d_model))
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))
        nn.init.normal_(self.W_in, std=cfg.init_range)
        nn.init.normal_(self.W_out, std=cfg.init_range)

    def forward(self, x):
        x = einsum("b p d, d m -> b p m", x, self.W_in) + self.b_in
        x = utils.gelu_new(x)
        return einsum("b p m, m d -> b p d", x, self.W_out) + self.b_out


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # 残差
        x = x + self.mlp(self.ln2(x))
        return x


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_U = nn.Parameter(torch.empty(cfg.d_model, cfg.d_vocab))
        nn.init.normal_(self.W_U, std=cfg.init_range)

    def forward(self, x):
        return einsum("b p d, d v -> b p v", x, self.W_U)


class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])  # 12层
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens):
        x = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.unembed(x)
