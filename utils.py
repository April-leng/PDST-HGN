# -*- coding: utf-8 -*-
import torch
import numpy as np
import logging
from sklearn.feature_selection import mutual_info_regression

def load_data(npz_file, csv_file):
    data = np.load(npz_file)['data']
    traffic_data = data[:, :, 0]
    num_nodes = traffic_data.shape[1]
    dist_df = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    dist_mx = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(dist_mx, 0)
    for row in dist_df:
        u, v, dist = int(row[0]), int(row[1]), row[2]
        if u < num_nodes and v < num_nodes:
            dist_mx[u, v], dist_mx[v, u] = dist, dist
    return traffic_data, (dist_mx < np.inf).astype(float)

def normalize_data(data, train_len):
    train_data = data[:train_len]
    mean, std = np.mean(train_data), np.std(train_data)
    return (data - mean) / std, mean, std

def generate_time_features(num_samples):
    time_ind = np.arange(num_samples)
    return np.stack(((time_ind // 288) % 7, time_ind % 288), axis=-1)

def create_sequences(data, time_features, seq_len, pred_len):
    xs, ys, xts = [], [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        xs.append(data[i:i + seq_len])
        ys.append(data[i + seq_len:i + seq_len + pred_len])
        xts.append(time_features[i:i + seq_len])
    return np.array(xs), np.array(ys), np.array(xts)

def precompute_mi_matrix(data):
    logging.info(f"Calculating Mutual Information Matrix (Samples: {data.shape[0]})...")
    sample = data
    num_nodes = data.shape[1]
    mi_mx = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        target_y = sample[:, i]
        mi_row = mutual_info_regression(sample, target_y, discrete_features=False, random_state=42, n_neighbors=3)
        mi_mx[i, :] = mi_row
        if (i + 1) % 50 == 0:
            logging.info(f"Computed MI for node {i + 1}/{num_nodes}")
    mi_mx = (mi_mx + mi_mx.T) / 2
    return torch.FloatTensor((mi_mx - mi_mx.min()) / (mi_mx.max() - mi_mx.min() + 1e-6))

def calculate_periodicity_features(data):
    num_nodes, samples = data.shape[1], data.shape[0]
    fft_coeffs = np.fft.rfft(data, axis=0)
    daily = np.abs(fft_coeffs[samples // 288, :])
    weekly = np.abs(fft_coeffs[samples // 2016, :]) if samples >= 2016 else daily
    feat = np.stack([daily, weekly], axis=-1)
    return torch.FloatTensor((feat - feat.min()) / (feat.max() - feat.min() + 1e-6))

def academic_mape(preds, labels, null_val):
    mask = (labels > null_val).float()
    mask /= (torch.mean(mask) + 1e-8)
    loss = torch.abs(preds - labels) / (labels + 1e-5)
    return torch.mean(loss * mask) * 100

def aggregate_mape(preds, labels):
    return (torch.sum(torch.abs(preds - labels)) / (torch.sum(torch.abs(labels)) + 1e-5)) * 100

def masked_mae_custom(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= (torch.mean(mask) + 1e-6)
    return torch.mean(torch.abs(preds - labels) * torch.where(torch.isnan(mask), torch.zeros_like(mask), mask))