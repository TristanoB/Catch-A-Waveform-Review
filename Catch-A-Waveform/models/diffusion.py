import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float() * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels)
        )

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        time_out = self.time_mlp(t_emb).unsqueeze(-1)
        h = h + time_out
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class DiffusionUNet1D(nn.Module):
    def __init__(self, params):
        super().__init__()
        base_channels = params.hidden_channels
        time_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_proj = nn.Conv1d(2, base_channels, kernel_size=1)
        self.blocks = nn.ModuleList([ResBlock(base_channels, time_dim) for _ in range(params.num_layers)])
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, 1, kernel_size=1)
        )

    def forward(self, x, t, cond=None):
        if cond is None:
            cond = torch.zeros_like(x)
        if cond.shape[2] != x.shape[2]:
            # pad / crop cond to match
            if cond.shape[2] > x.shape[2]:
                cond = cond[:, :, :x.shape[2]]
            else:
                cond = torch.cat((cond, cond.new_zeros(cond.shape[0], cond.shape[1], x.shape[2] - cond.shape[2])), dim=2)
        h = torch.cat([x, cond], dim=1)
        h = self.input_proj(h)
        t = t.view(-1)
        t_emb = self.time_mlp(t)
        if t_emb.dim() == 1:
            t_emb = t_emb.unsqueeze(0)
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out(h)
