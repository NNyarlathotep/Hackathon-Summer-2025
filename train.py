
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced supervised training script for predicting double-perturbation expression from single-perturbation profiles.

Improvements added:
1) Richer input features (toggle via --features):
    - base:        [xA, xB]
    - +baseline:   [xA, xB, baseline]
    - +interact:   add |xA-xB|, xA*xB, (xA+xB)/2
    => e.g. --features base+baseline+interact

2) Hybrid loss with correlation regularization (toggle via --corr_w):
    - loss = MSE + corr_w * (1 - PearsonCorr(y_hat, y))
    - Corr is computed per sample across genes, then averaged across batch.

3) Thresholding for downstream binarization (optional; not used in RMSD metric):
    - quantile:  tau = Quantile(|y-baseline|) on train (default; --quantile)
    - absolute:  tau = --abs_tau (e.g., 0.5)
    - youden:    heuristic grid over percentiles

4) Optional per-gene weights for MSE (toggle via --gene_weight_csv):
    - CSV with one column 'weight' (length = #genes in row order) for weighted MSE.

Outputs:
    - model.pt        : best checkpoint (by lowest val RMSD)
    - history.csv     : epoch metrics
    - rmsd_curve.png  : RMSD curves (train/val)
    - loss_curve.png  : Loss curves (train/val)
"""

# --- OpenMP runtime guard (Windows/libiomp5md.dll duplication) ---
# Must run BEFORE importing numpy/torch to avoid OMP Error #15.
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
#################################################################

import argparse
import os
import re
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_base_condition(col: str) -> str:
    return re.sub(r"\.\d+$", "", col)


def split_pair(cond: str) -> Tuple[str, str]:
    parts = cond.split('+')
    if len(parts) != 2:
        raise ValueError(f"Invalid condition name (expected 'a+b'): {cond}")
    return parts[0], parts[1]


def collapse_replicates(df: pd.DataFrame) -> pd.DataFrame:
    by_base: Dict[str, List[str]] = {}
    for col in df.columns:
        base = parse_base_condition(col)
        by_base.setdefault(base, []).append(col)
    out_cols = {}
    for base, cols in by_base.items():
        if len(cols) == 1:
            out_cols[base] = df[cols[0]].astype(np.float32)
        else:
            out_cols[base] = df[cols].astype(np.float32).mean(axis=1)
    collapsed = pd.DataFrame(out_cols, index=df.index)
    collapsed.index.name = df.index.name
    return collapsed


def find_baseline_ctrl(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    if 'ctrl+ctrl' in df.columns:
        return df['ctrl+ctrl'].values.astype(np.float32), 'ctrl+ctrl'
    single_ctrl_cols = [c for c in df.columns if c.endswith('+ctrl') or c.startswith('ctrl+')]
    if len(single_ctrl_cols) > 0:
        baseline = df[single_ctrl_cols].median(axis=1).values.astype(np.float32)
        return baseline, 'median_single_ctrl'
    baseline = df.median(axis=1).values.astype(np.float32)
    return baseline, 'global_median'


def build_samples(df: pd.DataFrame) -> Tuple[List[Tuple[str, str, str]], Dict[str, np.ndarray]]:
    singles = set([c for c in df.columns if '+ctrl' in c or 'ctrl+' in c])
    doubles = set()
    for c in df.columns:
        if '+ctrl' in c or 'ctrl+' in c:
            continue
        if '+' in c:
            lhs, rhs = split_pair(c)
            if lhs != 'ctrl' and rhs != 'ctrl':
                doubles.add(c)

    gene_to_single = {}
    for c in singles:
        lhs, rhs = split_pair(c)
        if lhs != 'ctrl' and rhs == 'ctrl':
            gene_to_single[lhs] = c
        elif lhs == 'ctrl' and rhs != 'ctrl':
            gene_to_single[rhs] = c

    samples: List[Tuple[str, str, str]] = []
    for d in sorted(doubles):
        a, b = split_pair(d)
        a_col = gene_to_single.get(a, None)
        b_col = gene_to_single.get(b, None)
        if a_col is None or b_col is None:
            continue
        samples.append((a_col, b_col, d))

    cache: Dict[str, np.ndarray] = {}
    for c in (list(singles) + list(doubles)):
        cache[c] = df[c].values.astype(np.float32)

    if len(samples) == 0:
        raise RuntimeError("No usable samples found. Ensure single-ctrl columns exist for doubles.")
    return samples, cache


class DoublePerturbDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str, str]], cache: Dict[str, np.ndarray], baseline: Optional[np.ndarray], feature_flags: Dict[str, bool]):
        self.samples = samples
        self.cache = cache
        self.genes = next(iter(cache.values())).shape[0]
        self.baseline = baseline
        self.flags = feature_flags

    def __len__(self):
        return len(self.samples)

    def _make_features(self, xA, xB):
        feats = [xA, xB]
        if self.flags.get('baseline', False) and self.baseline is not None:
            feats.append(self.baseline)
        if self.flags.get('interact', False):
            feats.append(np.abs(xA - xB))
            feats.append(xA * xB)
            feats.append(0.5 * (xA + xB))
        X = np.stack(feats, axis=-1)  # (G, F)
        return X

    def __getitem__(self, idx):
        a_col, b_col, d_col = self.samples[idx]
        xA = self.cache[a_col]
        xB = self.cache[b_col]
        y  = self.cache[d_col]
        X = self._make_features(xA, xB)
        return X, y


class PerGeneMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(max(depth - 1, 1)):
            layers += [nn.Linear(dim, hidden), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, G, F)
        B, G, F = x.shape
        x = x.reshape(B * G, F)
        out = self.net(x).reshape(B, G)
        return out


def pearson_corr_loss(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Compute 1 - Pearson correlation across genes for each sample, then mean over batch.
    y_hat = y_hat - y_hat.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    num = (y_hat * y).sum(dim=1)
    den = torch.sqrt((y_hat.square().sum(dim=1) + eps) * (y.square().sum(dim=1) + eps))
    corr = num / den
    return (1.0 - corr).mean()


def compute_threshold(train_loader: DataLoader, baseline_vec: np.ndarray, mode: str, quantile: float, abs_tau: float, device: torch.device) -> float:
    if mode == 'absolute':
        return abs_tau

    # gather deltas |y - baseline| from train
    deltas = []
    base = torch.from_numpy(baseline_vec).to(device)
    with torch.no_grad():
        for _, y in train_loader:
            y = y.to(device)
            b = base.unsqueeze(0).expand_as(y)
            deltas.append(torch.abs(y - b).flatten().cpu().numpy())
    d = np.concatenate(deltas, axis=0)

    if mode == 'quantile':
        return float(np.quantile(d, quantile))

    if mode == 'youden':
        # grid over percentiles 50->99 step 1, select a balanced threshold
        taus = np.percentile(d, np.arange(50, 100, 1))
        best_tau, best_score = taus[0], -1.0
        # Choose tau maximizing positive/negative separability using label balance heuristic: maximize 2p(1-p)
        for tau in taus:
            p = (d >= tau).mean()
            score = 2 * p * (1 - p)  # balancedness proxy
            if score > best_score:
                best_score, best_tau = score, tau
        return float(best_tau)

    raise ValueError(f"Unknown threshold_mode: {mode}")


def rmse_metric(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    # Root Mean Squared Deviation (RMSD) between predictions and targets
    return float(torch.sqrt(torch.mean((y_hat - y) ** 2)).item())


def plot_curves(history: pd.DataFrame, rmsd_path: str, loss_path: str):
    plt.figure(figsize=(7,5))
    plt.plot(history['epoch'], history['train_rmse'], label='train RMSD')
    plt.plot(history['epoch'], history['val_rmse'], label='val RMSD')
    plt.xlabel('Epoch'); plt.ylabel('RMSD'); plt.title('RMSD (train vs val)')
    plt.legend(); plt.tight_layout(); plt.savefig(rmsd_path, dpi=150); plt.close()

    plt.figure(figsize=(7,5))
    plt.plot(history['epoch'], history['train_loss'], label='train loss')
    plt.plot(history['epoch'], history['val_loss'], label='val loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss (train vs val)')
    plt.legend(); plt.tight_layout(); plt.savefig(loss_path, dpi=150); plt.close()


def make_feature_flags(features: str) -> Dict[str, bool]:
    flags = {'baseline': False, 'interact': False}
    toks = [t.strip().lower() for t in features.split('+') if t.strip()]
    for t in toks:
        if t == 'base':
            pass
        elif t == 'baseline':
            flags['baseline'] = True
        elif t == 'interact':
            flags['interact'] = True
        else:
            raise ValueError(f"Unknown feature token '{t}'. Use combination of: base, baseline, interact.")
    return flags


class WeightedMSE(nn.Module):
    def __init__(self, weight_vec: Optional[torch.Tensor] = None):
        super().__init__()
        self.weight = weight_vec  # shape (G,) on device
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_hat, y):
        loss = self.mse(y_hat, y)  # (B,G)
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)
        return loss.mean()


def train_epoch(model, loader, optimizer, device, base_loss, corr_w, baseline_vec_t, tau):
    model.train()
    tot_loss = 0.0; tot_rmse = 0.0; n = 0
    for X, y in loader:
        X = X.to(device); y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(X)
        loss = base_loss(y_hat, y)
        if corr_w > 0:
            loss = loss + corr_w * pearson_corr_loss(y_hat, y)
        loss.backward(); optimizer.step()
        tot_loss += float(loss.item())

        tot_rmse += rmse_metric(y_hat, y)
        n += 1
    return tot_loss / max(n,1), tot_rmse / max(n,1)


@torch.no_grad()
def eval_epoch(model, loader, device, base_loss, corr_w, baseline_vec_t, tau):
    model.eval()
    tot_loss = 0.0; tot_rmse = 0.0; n = 0
    for X, y in loader:
        X = X.to(device); y = y.to(device)
        y_hat = model(X)
        loss = base_loss(y_hat, y)
        if corr_w > 0:
            loss = loss + corr_w * pearson_corr_loss(y_hat, y)
        tot_loss += float(loss.item())

        tot_rmse += rmse_metric(y_hat, y)
        n += 1
    return tot_loss / max(n,1), tot_rmse / max(n,1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default='data/train_set.csv')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=6)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_dir', type=str, default=None)

    # Features & losses
    ap.add_argument('--features', type=str, default='base+baseline+interact', help='Combine: base, baseline, interact')
    ap.add_argument('--corr_w', type=float, default=0.1, help='Weight for (1 - corr) in loss')
    ap.add_argument('--gene_weight_csv', type=str, default=None, help='CSV with a single column "weight" of length #genes')

    # Thresholding
    ap.add_argument('--threshold_mode', type=str, default='quantile', choices=['quantile','absolute','youden'])
    ap.add_argument('--quantile', type=float, default=0.85)
    ap.add_argument('--abs_tau', type=float, default=0.5)

    args = ap.parse_args()
    set_seed(args.seed)

    out_dir = args.out_dir or os.path.join('runs', time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading: {args.csv}")
    # 将第一列（基因 ID，如 g0001）作为行索引，避免被当作数值列参与计算
    raw_df = pd.read_csv(args.csv, index_col=0, low_memory=False)
    print(f"Raw shape: {raw_df.shape}")
    df = collapse_replicates(raw_df)
    print(f"Collapsed shape: {df.shape}")
    samples, cache = build_samples(df)
    print(f"Usable samples: {len(samples)}")

    # Baseline
    baseline_vec_np, baseline_src = find_baseline_ctrl(df)
    print(f"Baseline source: {baseline_src}")

    # Shuffle & split
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(samples)); rng.shuffle(idx)
    split = int(len(samples) * (1 - args.val_ratio))
    train_samples = [samples[i] for i in idx[:split]]
    val_samples   = [samples[i] for i in idx[split:]]

    # Feature flags
    flags = make_feature_flags(args.features)

    # Datasets
    baseline_input = baseline_vec_np if flags.get('baseline', False) else None
    train_ds = DoublePerturbDataset(train_samples, cache, baseline=baseline_input, feature_flags=flags)
    val_ds   = DoublePerturbDataset(val_samples,   cache, baseline=baseline_input, feature_flags=flags)

    genes = train_ds.genes
    feat_dim = 2 + (1 if flags.get('baseline', False) and baseline_input is not None else 0) + (3 if flags.get('interact', False) else 0)
    approx_mb = (genes * (feat_dim + 1) * 4) / (1024*1024)
    print(f"Genes={genes}, feat_dim={feat_dim}, approx memory/sample ≈ {approx_mb:.2f} MB.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Threshold
    device = torch.device(args.device)
    tau = compute_threshold(train_loader, baseline_vec_np, args.threshold_mode, args.quantile, args.abs_tau, device)
    print(f"Threshold mode={args.threshold_mode}, tau={tau:.6f}")

    # Gene weights
    weight_vec_t = None
    if args.gene_weight_csv is not None:
        gw = pd.read_csv(args.gene_weight_csv)
        if 'weight' not in gw.columns:
            raise ValueError('gene_weight_csv must have a column named "weight"')
        if len(gw) != genes:
            raise ValueError(f'gene_weight_csv length {len(gw)} != #genes {genes}')
        weight_vec_t = torch.tensor(gw['weight'].values.astype(np.float32), device=device)

    # Model & optim
    model = PerGeneMLP(in_dim=feat_dim, hidden=args.hidden, depth=args.depth, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    base_loss = WeightedMSE(weight_vec_t)

    baseline_vec_t = torch.from_numpy(baseline_vec_np).to(device).unsqueeze(0)

    history = []; best_val_rmse = float('inf'); best_path = os.path.join(out_dir, 'model.pt')
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss, tr_rmse = train_epoch(model, train_loader, optimizer, device, base_loss, args.corr_w, baseline_vec_t, tau)
        va_loss, va_rmse = eval_epoch(model, val_loader, device, base_loss, args.corr_w, baseline_vec_t, tau)
        dt = time.time() - t0
        history.append({'epoch': epoch, 'train_loss': tr_loss, 'val_loss': va_loss, 'train_rmse': tr_rmse, 'val_rmse': va_rmse, 'time_sec': dt})
        print(f"Epoch {epoch:03d} | loss {tr_loss:.5f}/{va_loss:.5f} | RMSD {tr_rmse:.4f}/{va_rmse:.4f} | {dt:.1f}s")
        if va_rmse < best_val_rmse:
            best_val_rmse = va_rmse
            torch.save({'model_state': model.state_dict(),
                        'tau': tau,
                        'args': vars(args),
                        'baseline_source': baseline_src,
                        'feat_dim': feat_dim}, best_path)

    # Save outputs
    hist_df = pd.DataFrame(history)
    hist_csv = os.path.join(out_dir, 'history.csv'); hist_df.to_csv(hist_csv, index=False)
    rmse_png   = os.path.join(out_dir, 'rmsd_curve.png')
    loss_png = os.path.join(out_dir, 'loss_curve.png')
    plot_curves(hist_df, rmse_png, loss_png)

    print(f"Saved best model: {best_path}")
    print(f"Saved history   : {hist_csv}")
    print(f"Saved plots     : {rmse_png}, {loss_png}")


if __name__ == '__main__':
    main()
