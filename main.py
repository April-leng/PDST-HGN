# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import logging
import argparse
import os
import warnings

# 导入自定义模块
from utils import (load_data, normalize_data, generate_time_features, create_sequences,
                   precompute_mi_matrix, calculate_periodicity_features,
                   academic_mape, aggregate_mape, masked_mae_custom)
from model import HSDHN_Hybrid_SOTA

warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_path', type=str, default='/data/PEMS04/PEMS04.npz')
    parser.add_argument('--dist_path', type=str, default='/data/PEMS04/PEMS04.csv')
    parser.add_argument('--save_dir', type=str, default='/quanzhong')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=110)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_hyperedges', type=int, default=256)
    parser.add_argument('--num_seeds', type=int, default=30)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--lambda_consist', type=float, default=0.01)
    parser.add_argument('--mape_thresh', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    raw, adj = load_data(args.data_path, args.dist_path)
    s_idx = int(raw.shape[0] * 0.6)
    ds_name = os.path.basename(args.data_path).split('.')[0]
    best_path = os.path.join(args.save_dir, f'best_model_{ds_name}.pth')
    mi_cache_path = os.path.join(args.save_dir, f'mi_matrix_{ds_name}_trainlen{s_idx}.pt')

    if os.path.exists(mi_cache_path):
        logging.info(f"Loading cached MI matrix...")
        mi_mx = torch.load(mi_cache_path, map_location=device, weights_only=True)
    else:
        mi_mx = precompute_mi_matrix(raw[:s_idx]).to(device)
        torch.save(mi_mx, mi_cache_path)

    p_feat = calculate_periodicity_features(raw).to(device)
    norm, mean, std = normalize_data(raw, train_len=s_idx)
    X, Y, XT = create_sequences(norm, generate_time_features(raw.shape[0]), 12, 12)
    s, sv = int(len(X) * 0.6), int(len(X) * 0.8)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X[:s]), torch.FloatTensor(Y[:s]), torch.LongTensor(XT[:s])),
        batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X[s:sv]), torch.FloatTensor(Y[s:sv]), torch.LongTensor(XT[s:sv])),
        batch_size=args.batch_size)
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X[sv:]), torch.FloatTensor(Y[sv:]), torch.LongTensor(XT[sv:])),
        batch_size=args.batch_size)

    model = HSDHN_Hybrid_SOTA(adj.shape[0], adj, mi_mx, 12, 12, p_feat, d_model=args.d_model,
                              num_hyperedges=args.num_hyperedges, num_seeds=args.num_seeds, alpha=args.alpha).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_l = []
        for x, y, xt in train_loader:
            optimizer.zero_grad()
            out, (Hlt, Hltm1), (Hrt, Hrtm1) = model(x.to(device).unsqueeze(-1), xt.to(device))
            loss = masked_mae_custom(out, y.to(device).unsqueeze(-1)) + args.lambda_consist * (
                        F.mse_loss(Hlt, Hltm1) + F.mse_loss(Hrt, Hrtm1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            train_l.append(loss.item())

        model.eval()
        v_mae_list = []
        with torch.no_grad():
            for x, y, xt in val_loader:
                out, _, _ = model(x.to(device).unsqueeze(-1), xt.to(device))
                v_mae_list.append(masked_mae_custom(out.squeeze(-1) * std + mean, y.to(device) * std + mean).item())
        curr = np.mean(v_mae_list)
        print(f"Epoch {epoch + 1:03d} | Train: {np.mean(train_l):.4f} | Val MAE: {curr:.4f}")
        scheduler.step(curr)
        if curr < best_val:
            best_val = curr
            torch.save(model.state_dict(), best_path)

    # 测试阶段
    model.load_state_dict(torch.load(best_path))
    model.eval()
    t_mae, t_rmse, t_y, t_pred = [], [], [], []
    with torch.no_grad():
        for x, y, xt in test_loader:
            out, _, _ = model(x.to(device).unsqueeze(-1), xt.to(device))
            real_o, real_y = out.squeeze(-1) * std + mean, y.to(device) * std + mean
            t_mae.append(masked_mae_custom(real_o, real_y).item())
            t_rmse.append(torch.sqrt(torch.mean((real_o - real_y) ** 2)).item())
            t_y.append(real_y)
            t_pred.append(real_o)

    print(
        f"\n[{ds_name}] FINAL TEST | MAE: {np.mean(t_mae):.4f} | RMSE: {np.mean(t_rmse):.4f} | WMAPE: {aggregate_mape(torch.cat(t_pred), torch.cat(t_y)):.2f}% | Academic MAPE: {academic_mape(torch.cat(t_pred), torch.cat(t_y), null_val=args.mape_thresh):.2f}%")