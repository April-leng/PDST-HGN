# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import InceptionTCN, HSDHGNNLayer, DynamicRegionGrowing

class HSDHN_Hybrid_SOTA(nn.Module):
    def __init__(self, num_nodes, adj_mx, mi_mx, seq_len, pred_len, periodicity_features,
                 d_model=128, num_hyperedges=64, num_seeds=30, alpha=0.5):
        super().__init__()
        self.num_nodes, self.d_model, self.alpha = num_nodes, d_model, alpha
        dev = periodicity_features.device
        self.node_emb = nn.Parameter(torch.randn(num_nodes, d_model) * 0.1)
        self.register_buffer('p_feat', periodicity_features)
        self.register_buffer('mi_mx', mi_mx)
        self.time_of_day_emb = nn.Embedding(288, 16)
        self.day_of_week_emb = nn.Embedding(7, 16)
        self.p_emb = nn.Linear(2, 16)
        self.start_proj = nn.Linear(1 + 32 + 16, d_model)
        self.tcn = nn.ModuleList([InceptionTCN(d_model, d) for d in [1, 2, 4]])
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(d_model + 1, 64), nn.ReLU(), nn.Linear(64, 3), nn.Softmax(dim=-1))

        self.grower = DynamicRegionGrowing(adj_mx, num_seeds=num_seeds)
        self.hgnn_local = HSDHGNNLayer(d_model)
        self.hgnn_remote = HSDHGNNLayer(d_model)

        self.prototypes = nn.Parameter(torch.randn(d_model, num_hyperedges))
        self.A_remote = nn.Parameter(torch.eye(num_hyperedges, device=dev))

        self.temporal_mlp = nn.Sequential(nn.Linear(seq_len * d_model, d_model * 2), nn.ReLU(),
                                          nn.Linear(d_model * 2, d_model))
        self.fusion_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, pred_len))

    def forward(self, x_seq, time_features):
        B, T, N, _ = x_seq.shape
        tod = self.time_of_day_emb(time_features[..., 1]).unsqueeze(2).expand(-1, -1, N, -1)
        dow = self.day_of_week_emb(time_features[..., 0]).unsqueeze(2).expand(-1, -1, N, -1)
        pe = self.p_emb(self.p_feat).view(1, 1, N, -1).expand(B, T, -1, -1)
        h = self.start_proj(torch.cat([x_seq, tod, dow, pe], dim=-1)) + self.node_emb.view(1, 1, N, -1)

        h = h.permute(0, 2, 3, 1).reshape(B * N, self.d_model, T)
        for layer in self.tcn: h = layer(h)
        h_seq = h.reshape(B, N, self.d_model, T)
        h_t, h_tm1 = h_seq[..., -1], h_seq[..., -2]
        Hlt, Alt = self.grower(self.classifier(torch.cat([h_t, x_seq[:, -1, :, :]], dim=-1)))
        Hltm1, _ = self.grower(self.classifier(torch.cat([h_tm1, x_seq[:, -2, :, :]], dim=-1)))

        aff_t = h_t @ self.prototypes
        Hrt = F.softmax(aff_t + self.alpha * torch.matmul(self.mi_mx, aff_t), dim=-1)
        Hr_tm1 = F.softmax((h_tm1 @ self.prototypes) + self.alpha * torch.matmul(self.mi_mx, h_tm1 @ self.prototypes), dim=-1)
        h_l = self.hgnn_local(h_t, Hlt, Alt)
        h_r = self.hgnn_remote(h_t, Hrt, self.A_remote.unsqueeze(0).expand(B, -1, -1))

        h_for_gru = (h_l + h_r).unsqueeze(1).expand(-1, T, -1, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1)
        _, gru_sum = self.gru(h_for_gru)
        gru_sum = gru_sum.squeeze(0).reshape(B, N, -1)
        mlp_sum = self.temporal_mlp(h_seq.permute(0, 1, 3, 2).reshape(B, N, -1))
        final_h = self.fusion_norm(gru_sum + mlp_sum)
        return self.output_proj(final_h).permute(0, 2, 1).unsqueeze(-1), (Hlt, Hltm1), (Hrt, Hr_tm1)