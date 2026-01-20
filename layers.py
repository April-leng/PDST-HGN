# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionTCN(nn.Module):
    def __init__(self, d_model, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model // 2, 3, padding=2 * dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(d_model, d_model // 2, 6, padding=5 * dilation, dilation=dilation)
        self.gate = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x):
        h = torch.cat([self.conv1(x)[..., :x.shape[-1]], self.conv2(x)[..., :x.shape[-1]]], dim=1)
        return torch.tanh(h) * torch.sigmoid(self.gate(x)) + x

class HSDHGNNLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, x_n, H, A_int):
        deg_e = torch.sum(H, dim=1, keepdim=True) + 1e-6
        x_e = torch.bmm(H.transpose(1, 2), x_n) / deg_e.transpose(1, 2)
        agg = torch.bmm(H, torch.bmm(A_int, x_e))
        return self.mlp(torch.cat([x_n, agg], dim=-1))

class DynamicRegionGrowing(nn.Module):
    def __init__(self, adj_mx, num_seeds=30):
        super().__init__()
        self.k = num_seeds
        self.register_buffer('adj', torch.FloatTensor(adj_mx))

    def forward(self, phases):
        B, N, _ = phases.shape
        _, seeds_idx = torch.topk(phases.transpose(1, 2), self.k, dim=-1)
        sim = torch.bmm(phases, phases.transpose(1, 2))
        A = self.adj.unsqueeze(0) * (sim > 0.5).float()
        M = A
        for _ in range(2): M = torch.bmm(M, A).clamp(0, 1)
        M = M + torch.eye(N, device=phases.device).unsqueeze(0)
        H_loc = M.gather(2, seeds_idx.view(B, -1).unsqueeze(1).expand(-1, N, -1))
        A_loc = (torch.bmm(H_loc.transpose(1, 2), H_loc) > 0).float()
        return H_loc, A_loc