import torch
import torch.nn as nn
import torch.optim as optim
import math
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True):
    # Adapted from MAE paper: https://github.com/facebookresearch/mae
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='ij')  # (H, W)
    grid = torch.stack(grid, dim=0)  # (2, H, W)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 4 == 0
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])  # x
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])  # y
    return torch.cat([emb_h, emb_w], dim=1)

def get_1d_sincos_pos_embed(embed_dim, pos):
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)  # (dim/2,)

    pos = pos.reshape(-1)  # flatten
    out = torch.einsum('p,d->pd', pos, omega)  # (pos, dim/2)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)  # (pos, dim)


###### FIX FOR ERROR IN FINDING SECOND DERIVATIVES

class MHSAPlain(nn.Module):
    """Multi-head self-attention implemented with plain matmuls & softmax.
       This uses only autograd-friendly ops and supports double-backward.
    """
    def __init__(self, dim, num_heads, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x, mask=None):
        # x: (B, N, dim)
        B, N, D = x.shape
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, heads, N, head_dim)

        # scaled dot product attention (plain ops)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn_scores, dim=-1)  # (B, heads, N, N)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)  # (B, N, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class EncoderLayerPlain(nn.Module):
    def __init__(self, dim, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MHSAPlain(dim, nhead, attn_dropout=dropout, proj_dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout),
        )

    def forward(self, src, src_mask=None):
        # src: (B, N, dim)
        src2 = self.self_attn(src, mask=src_mask)
        src = src + src2
        src = self.norm1(src)
        src2 = self.ff(src)
        src = src + src2
        src = self.norm2(src)
        return src



class SimpleViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=10, dim=256, depth=4, heads=4, mlp_dim=512):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by patch size.'

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim), requires_grad=True)
        pos_embed = get_2d_sincos_pos_embed(dim, int(math.sqrt(self.num_patches)))
        self.register_buffer('pos_embedding', pos_embed.unsqueeze(0))  # [1, num_patches+1, dim]

        # encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.layers = nn.ModuleList([
            EncoderLayerPlain(dim, nhead=heads, dim_feedforward=mlp_dim, dropout=0.0)
            for _ in range(depth)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p).contiguous()
        x = x.view(B, C, -1, p, p).permute(0, 2, 1, 3, 4)
        x = x.reshape(B, self.num_patches, -1)
        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding

        # x = self.transformer(x)
        for layer in self.layers:
            x = layer(x)
        return self.mlp_head(x[:, 0])
