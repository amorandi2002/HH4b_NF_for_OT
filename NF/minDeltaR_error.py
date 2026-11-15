"""
Correction of minDeltaR_Higgjj after the high-level feature reconstruction.

The reconstructed minDeltaR_Higgjj has a mis-modelled tail. A light conditional Flow Matching model
(rectified flow with 1D optimal-transport coupling) learns the velocity field that moves the reconstructed
values onto the SR ones, conditioned on all the other high-level features. SWAG gives its uncertainty.
The model and the scalers are fitted in the 2b case and re-used for 4b.
"""
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

import utils  # noqa: F401  (puts the repository root on sys.path)
from lib.features import MIN_DR
from lib import visualization as vis
from model import EarlyStopper
from swag import get_tracker, swag_predict


# =============================================================================
# Model
# =============================================================================
class FlowMatchingTailCorrector(nn.Module):
    """Velocity field v(x, t | c) of a 1D feature x, with time t in [0, 1] (0 = reco, 1 = SR)."""

    def __init__(self, context_dim, hidden_dim=128):
        super().__init__()
        self.v_net = nn.Sequential(
            nn.Linear(2 + context_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}

    def forward(self, x, t, context):
        return self.v_net(torch.cat([x, t.view(-1, 1), context], dim=-1))

    def loss(self, x_reco, x_orig, c_reco):
        """
        Flow-matching loss with 1D optimal-transport coupling: sorting both samples gives the OT pairing.
        The context follows the reco events. Path x_t = (1 - t) x_reco + t x_orig, target velocity x_orig - x_reco.
        """
        with torch.no_grad():
            i_reco = torch.argsort(x_reco.squeeze(-1))
            i_orig = torch.argsort(x_orig.squeeze(-1))
            x0, x1, c = x_reco[i_reco], x_orig[i_orig], c_reco[i_reco]
            t = torch.rand((len(x0), 1), device=x0.device)
            x_t = (1 - t) * x0 + t * x1
        return nn.functional.mse_loss(self(x_t, t, c), x1 - x0)

    def train_model(self, train, val, save_dir, epochs=300, batch_size=2048, patience=20, lr=1e-3, gamma_lr=0.98):
        """
        train / val = (x_reco, x_orig, c_reco) tensors. The best model (validation loss) is saved in save_dir.
        """
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, 'best_model_fm.pth')
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        scheduler = ExponentialLR(optimizer, gamma=gamma_lr)
        stopper = EarlyStopper(patience=patience)

        for epoch in tqdm(range(epochs), desc="Flow matching training"):
            self.train()
            perm = torch.randperm(len(train[0]))
            train_loss = []
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                optimizer.zero_grad()
                loss = self.loss(*(t[idx] for t in train))
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss.append(loss.item())

            self.eval()
            with torch.no_grad():
                val_loss = np.mean([self.loss(*(t[i:i + batch_size] for t in val)).item()
                                    for i in range(0, len(val[0]), batch_size)])
            self.history['train_loss'].append(float(np.mean(train_loss)))
            self.history['val_loss'].append(float(val_loss))
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            scheduler.step()

            if val_loss < stopper.best:
                torch.save(self.state_dict(), best_path)
            if stopper.early_stop(val_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.load_state_dict(torch.load(best_path, map_location='cpu'))
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=4)

    @torch.no_grad()
    def correct(self, x_reco, context, steps=50):
        """Euler integration of dx/dt = v(x, t | c) from t = 0 to t = 1."""
        self.eval()
        x = x_reco.clone()
        for k in range(steps):
            t = torch.full((len(x), 1), k / steps, device=x.device)
            x = x + self(x, t, context) / steps
        return x

    @torch.no_grad()
    def velocity_grid(self, x_min, x_max, context_mean, n=60):
        """Velocity on a (t, x) grid at fixed (average) context, for plotting."""
        self.eval()
        device = next(self.parameters()).device
        T, X = np.meshgrid(np.linspace(0, 1, n), np.linspace(x_min, x_max, n))
        x = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
        t = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=device)
        c = torch.as_tensor(context_mean, dtype=torch.float32, device=device).repeat(len(x), 1)
        return T, X, self(x, t, c).cpu().numpy().reshape(X.shape)


# =============================================================================
# Correction of the reconstructed feature
# =============================================================================
def correct_minDeltaR_feature(X_reco, V_reco, X_sig, features, weights_dir, plot_dir, fit=True, retrain=False,
                              epochs=300, batch_size=2048, patience=20, retrain_swag=False, swag_epochs=30,
                              swag_lr=1e-5, num_swag_samples=30, steps=50, seed=42, device=None):
    """
    Correct the column minDeltaR_Higgjj of X_reco (and set its variance in V_reco from SWAG).

    param X_reco, V_reco: reconstructed features and variances (physical space), V_reco can be None
    param X_sig: SR features (target of the correction)
    param weights_dir: where the model, the scalers and the SWAG state are stored (2b directory)
    param fit: True in the 2b case (fit scalers, train if needed); False in the 4b case (load everything)
    return: corrected (X_reco, V_reco)
    """
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    rng = np.random.default_rng(seed)
    i_mdr = features.index(MIN_DR)
    i_ctx = [i for i in range(len(features)) if i != i_mdr]
    valid = np.isfinite(X_reco).all(axis=1)
    if (~valid).any():
        print(f"{(~valid).sum()} reco events with NaN are not corrected")
    x_reco, c_reco = X_reco[valid][:, [i_mdr]], X_reco[valid][:, i_ctx]
    x_sig, c_sig = X_sig[:, [i_mdr]], X_sig[:, i_ctx]

    # --- Scalers (fitted on 2b) ---
    os.makedirs(weights_dir, exist_ok=True)
    scaler_path = os.path.join(weights_dir, 'scalers.pkl')
    if fit:
        scalers = {'mDR': StandardScaler().fit(x_sig), 'context': StandardScaler().fit(np.vstack([c_reco, c_sig]))}
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
    else:
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"{scaler_path} not found: run the 2b case first")
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)

    def tensor(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    xr, cr = tensor(scalers['mDR'].transform(x_reco)), tensor(scalers['context'].transform(c_reco))
    xs = tensor(scalers['mDR'].transform(x_sig))

    # --- Unpaired training sample: the same number of reco and SR events, 75 / 25 split ---
    n = min(len(xr), len(xs))
    i_r, i_s = rng.permutation(len(xr))[:n], rng.permutation(len(xs))[:n]
    n_val = n // 4
    train = (xr[i_r[n_val:]], xs[i_s[n_val:]], cr[i_r[n_val:]])
    val = (xr[i_r[:n_val]], xs[i_s[:n_val]], cr[i_r[:n_val]])

    # --- Model ---
    model = FlowMatchingTailCorrector(context_dim=len(i_ctx)).to(device)
    best_path = os.path.join(weights_dir, 'best_model_fm.pth')
    if fit and (retrain or not os.path.exists(best_path)):
        model.train_model(train, val, weights_dir, epochs=epochs, batch_size=batch_size, patience=patience)
    elif os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        raise FileNotFoundError(f"{best_path} not found: run the 2b case first")
    history_path = os.path.join(weights_dir, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path) as f:
            h = json.load(f)
        vis.plot_curves({'Train': h['train_loss'], 'Validation': h['val_loss']},
                        os.path.join(plot_dir, 'training_history.png'), 'MSE loss')

    # --- Nominal correction (all valid reco events) ---
    corr_scaled = model.correct(xr, cr, steps).cpu().numpy()

    # --- SWAG uncertainty ---
    state_path = os.path.join(weights_dir, f'swag_mDR_state_{swag_lr}.pth')
    if not fit and not os.path.exists(state_path):
        raise FileNotFoundError(f"{state_path} not found: run the 2b case first")
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*train), batch_size=batch_size, shuffle=True)
    tracker = get_tracker(model, state_path, lambda b: model.loss(*b), loader, swag_epochs, swag_lr,
                          retrain=fit and retrain_swag, set_train=model.train)
    _, var_scaled = swag_predict(model, tracker, lambda: model.correct(xr, cr, steps), num_swag_samples)

    corrected = scalers['mDR'].inverse_transform(corr_scaled).ravel()
    X_out = X_reco.copy()
    X_out[valid, i_mdr] = corrected
    V_out = None
    if V_reco is not None:
        V_out = V_reco.copy()
        V_out[valid, i_mdr] = var_scaled.cpu().numpy().ravel() * scalers['mDR'].scale_[0] ** 2

    # --- Plots ---
    vis.plot_correction(x_sig.ravel(), x_reco.ravel(), corrected, MIN_DR,
                        os.path.join(plot_dir, 'correction_result.png'))
    model.load_state_dict(torch.load(best_path, map_location=device))
    T, X, V = model.velocity_grid(float(xr.min()), float(xr.max()), cr.mean(0).cpu().numpy())
    vis.plot_velocity_map(T, X, V, os.path.join(plot_dir, 'velocity_map.png'))
    return X_out, V_out


def study_problematic_tail(X_sig, X_reco, features, out_dir, target=MIN_DR, threshold=1.0):
    """
    Events with target < threshold: where do they end up in the other features? (SR vs Reco SR)
    """
    i_t = features.index(target)
    tail_sig, tail_reco = X_sig[:, i_t] < threshold, X_reco[:, i_t] < threshold
    print(f"Tail {target} < {threshold}: SR {tail_sig.mean():.2%}, Reco SR {tail_reco.mean():.2%}")
    for i, ft in enumerate(tqdm(features, desc="Tail study")):
        if i == i_t:
            continue
        s, r = X_sig[:, i], X_reco[:, i]
        ok_s, ok_r = np.isfinite(s), np.isfinite(r)
        vis.plot_tail_study(ft, s[ok_s], r[ok_r], s[ok_s & tail_sig], r[ok_r & tail_reco], f'{target} < {threshold}',
                            os.path.join(out_dir, f'{ft}_tail.png'))
