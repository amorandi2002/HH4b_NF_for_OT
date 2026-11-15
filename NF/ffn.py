"""
Feed-forward network estimating the 2b -> 4b likelihood ratio r(x) = p_4b(x) / p_2b(x).

The network output is r(x) > 0 (Softplus); it is trained as a classifier with logit = log r(x), so that
p(4b | x) = r / (1 + r). Trained in the control region (CR), it gives the event weights that reweight
2b events to the 4b kinematics; the closure is checked in the signal region (SR).
"""
import copy
import json
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import utils  # noqa: F401  (puts the repository root on sys.path)
from lib import data_loader as dl
from lib import metrics
from lib import visualization as vis


class LikelihoodRatioFFN:
    def __init__(self, input_dim, device=None, lr=1e-3, gamma=0.95, gamma_min=0.5, gamma_step=0.05,
                 gamma_patience=10, early_stop_patience=30, dropout=0.5, pos_weight=None):
        """
        Exponential LR decay lr_{e+1} = gamma * lr_e; every `gamma_patience` epochs without improvement of the
        validation loss gamma is reduced by `gamma_step` (down to gamma_min).
        pos_weight: weight of the 4b class in the BCE loss (N_2b / N_4b balances the classes).
        """
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.input_dim = input_dim
        layers, dims = [nn.BatchNorm1d(input_dim)], [input_dim, 512, 256, 128, 64]
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU(), nn.BatchNorm1d(d_out), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers, nn.Linear(dims[-1], 1), nn.Softplus()).to(self.device)

        self.optimizer = optim.AdamW(self.net.parameters(), lr=lr)
        self.gamma, self.gamma_min, self.gamma_step = gamma, gamma_min, gamma_step
        self.gamma_patience, self.early_stop_patience = gamma_patience, early_stop_patience
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        pw = None if pos_weight is None else torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.history = {'train_loss': [], 'val_loss': [], 'lr': [], 'gamma': []}
        self.best_epoch, self.best_val, self.best_state = 0, np.inf, None

    def _logit(self, x):
        return torch.log(self.net(x).squeeze(-1).clamp_min(1e-7))

    def _run_epoch(self, loader, train):
        self.net.train(train)
        total, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for x, y in tqdm(loader, desc="Train" if train else "Validation", leave=False):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                loss = self.criterion(self._logit(x), y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total += loss.item() * len(x)
                n += len(x)
        return total / n

    def fit(self, train_loader, val_loader, epochs=300):
        no_improve = 0
        for epoch in (bar := tqdm(range(1, epochs + 1), desc="FFN training")):
            tr, va = self._run_epoch(train_loader, True), self._run_epoch(val_loader, False)
            for k, v in zip(self.history, (tr, va, self.optimizer.param_groups[0]['lr'], self.gamma)):
                self.history[k].append(v)
            if va < self.best_val:
                self.best_val, self.best_epoch, no_improve = va, epoch, 0
                self.best_state = copy.deepcopy(self.net.state_dict())
            else:
                no_improve += 1
                if no_improve % self.gamma_patience == 0 and self.gamma > self.gamma_min:
                    self.gamma = max(self.gamma_min, round(self.gamma - self.gamma_step, 10))
                    self.scheduler.gamma = self.gamma
            self.scheduler.step()
            bar.set_postfix(train=f"{tr:.4f}", val=f"{va:.4f}", gamma=f"{self.gamma:.2f}", best=self.best_epoch)
            if no_improve >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (best epoch {self.best_epoch})")
                break
        self.net.load_state_dict(self.best_state)

    @torch.no_grad()
    def predict_ratio(self, X, batch_size=8192):
        """Event-by-event likelihood ratio r(x) = p_4b(x) / p_2b(x) (X already scaled)."""
        self.net.eval()
        X = torch.as_tensor(X, dtype=torch.float32)
        return torch.cat([self.net(X[i:i + batch_size].to(self.device)).squeeze(-1).cpu()
                          for i in range(0, len(X), batch_size)]).numpy()

    def save(self, weights_dir, scaler):
        os.makedirs(weights_dir, exist_ok=True)
        torch.save({'state_dict': self.net.state_dict(), 'input_dim': self.input_dim,
                    'best_epoch': self.best_epoch, 'best_val': self.best_val},
                   os.path.join(weights_dir, 'ffn_best.pt'))
        with open(os.path.join(weights_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f)
        joblib.dump(scaler, os.path.join(weights_dir, 'scaler.joblib'))

    @classmethod
    def load(cls, weights_dir, device=None):
        ckpt = torch.load(os.path.join(weights_dir, 'ffn_best.pt'), map_location='cpu')
        model = cls(ckpt['input_dim'], device=device)
        model.net.load_state_dict(ckpt['state_dict'])
        model.net.eval()
        return model, joblib.load(os.path.join(weights_dir, 'scaler.joblib'))

    def weights(self, X, scaler, n_target=None):
        """
        FFN weights of (raw) events X; events with non-finite features get weight 0.
        If n_target is given the weights are normalised so that sum(w) = n_target.
        """
        X = np.asarray(X, dtype=np.float64)
        ok = np.isfinite(X).all(axis=1)
        w = np.zeros(len(X))
        w[ok] = self.predict_ratio(scaler.transform(X[ok]).astype(np.float32))
        if n_target is not None:
            if w.sum() <= 0:
                raise ValueError("Sum of the FFN weights <= 0")
            w *= n_target / w.sum()
        return w


# =============================================================================
# Data
# =============================================================================
def load_xy(region, features, njet_min=None):
    """2b (label 0) and 4b (label 1) data of the CR or SR, raw features. njet_min selects njet >= njet_min."""
    X2, X4 = dl.load_2b_4b(region, features, njet_min)
    return np.vstack([X2, X4]), np.concatenate([np.zeros(len(X2)), np.ones(len(X4))])


# =============================================================================
# Closure
# =============================================================================
def closure(model, scaler, X, y, features, plot_dir, name):
    """Reweight the 2b events of (X, y) to 4b and compare 2b, reweighted 2b and 4b for each feature."""
    X2, X4 = X[y == 0], X[y == 1]
    w = model.weights(X2, scaler, n_target=len(X4))
    r = model.weights(X, scaler)
    print(f"[{name}] AUC = {roc_auc_score(y, r):.4f} | weights mean = {w.mean():.3f}, max = {w.max():.2f}, "
          f"Kish ESS fraction = {w.sum() ** 2 / (w ** 2).sum() / len(w):.3f}")
    vis.plot_weights(w, os.path.join(plot_dir, f'weights_{name}.png'))
    for i, ft in enumerate(features):
        edges = vis.percentile_edges(np.concatenate([X2[:, i], X4[:, i]]), max(vis.feature_bins(ft, 4) // 2, 20))
        vis.plot_reweighting_closure(ft, edges, metrics.hist_density(X2[:, i], edges),
                                     metrics.hist_density(X2[:, i], edges, w=w), metrics.hist_density(X4[:, i], edges),
                                     os.path.join(plot_dir, f'closure_{ft}_{name}.png'))
    return w


def train_ffn(features, out_dir, njet_min=None, epochs=300, batch_size=1024, num_workers=8, seed=42):
    """
    Train the FFN on the CR (2b vs 4b) and run the closure test on the CR test split and on the SR.
    Output: <out_dir>/CR/weights (model + scaler), <out_dir>/CR/plots, <out_dir>/SR/plots,
            <out_dir>/SR/weights_2b_to_4b_SR.npy
    """
    weights_dir = os.path.join(out_dir, 'CR', 'weights')
    # Stratified 70 / 20 / 10 split; the FFN uses the raw features (RobustScaler only, no log, no clipping)
    splits = dl.split(*load_xy('CR', features, njet_min), validation_size=0.2, test_size=0.1, seed=seed)
    loaders, scaler = dl.make_loaders(splits, features, log=False, clip=None, batch_size=batch_size,
                                      num_workers=num_workers)
    y_tr = splits['train'][1]

    model = LikelihoodRatioFFN(len(features), pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum())
    model.fit(loaders['train'], loaders['val'], epochs=epochs)
    model.save(weights_dir, scaler)

    cr_plots = os.path.join(out_dir, 'CR', 'plots')
    vis.plot_curves({'Train': model.history['train_loss'], 'Validation': model.history['val_loss']},
                    os.path.join(cr_plots, 'training_loss.png'), 'BCE loss')
    vis.plot_curves({'Learning rate': model.history['lr']}, os.path.join(cr_plots, 'learning_rate.png'),
                    'Learning rate', logy=True, label=False)
    closure(model, scaler, *splits['test'], features, cr_plots, 'CR_test')

    X_sr, y_sr = load_xy('SR', features, njet_min)
    w_sr = closure(model, scaler, X_sr, y_sr, features, os.path.join(out_dir, 'SR', 'plots'), 'SR')
    np.save(os.path.join(out_dir, 'SR', 'weights_2b_to_4b_SR.npy'), w_sr)
    return model, scaler
