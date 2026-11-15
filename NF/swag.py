"""
SWAG (Stochastic Weight Averaging - Gaussian) estimate of the model uncertainty.

Starting from the trained weights, the model is trained for a few more epochs at small learning rate,
collecting the weights after each epoch. The weight posterior is approximated with a Gaussian
(diagonal + low-rank covariance); sampling models from it gives the per-event variance of the prediction.
"""
import math
import os

import torch
from tqdm import tqdm

import utils


class SWAGTracker:
    """SWAG moments of the parameters of any object exposing .parameters() (FlowModel, nn.Module, ...)."""

    def __init__(self, model, max_num_models=20):
        flat = self._get_flat(model)
        self.device = flat.device
        self.max_num_models = max_num_models
        self.n_models = 0
        self.mean = torch.zeros_like(flat)
        self.sq_mean = torch.zeros_like(flat)
        self.cov_mat_sqrt = torch.empty((0, len(flat)), device=self.device)

    @staticmethod
    def _get_flat(model):
        return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    @staticmethod
    def _set_flat(model, flat):
        offset = 0
        for p in model.parameters():
            p.data.copy_(flat[offset:offset + p.numel()].view_as(p))
            offset += p.numel()

    def update(self, model):
        """Add the current weights to the running moments and to the deviation matrix."""
        flat = self._get_flat(model)
        n = self.n_models
        self.mean = (self.mean * n + flat) / (n + 1)
        self.sq_mean = (self.sq_mean * n + flat ** 2) / (n + 1)
        self.cov_mat_sqrt = torch.cat([self.cov_mat_sqrt, (flat - self.mean).unsqueeze(0)])[-self.max_num_models:]
        self.n_models += 1

    def sample(self, model, scale=0.5):
        """Load into `model` one set of weights sampled from the SWAG Gaussian."""
        diag_std = torch.sqrt(torch.clamp(self.sq_mean - self.mean ** 2, min=1e-8))
        z1 = torch.randn_like(self.mean)
        k = self.cov_mat_sqrt.shape[0]
        low_rank = (torch.randn(k, device=self.device) @ self.cov_mat_sqrt) / math.sqrt(k - 1) if k > 1 \
            else torch.zeros_like(self.mean)
        self._set_flat(model, self.mean + math.sqrt(scale) * (diag_std * z1 + low_rank))

    def set_mean(self, model):
        """Load the SWA mean weights into `model`."""
        self._set_flat(model, self.mean)

    def save(self, path):
        torch.save({'mean': self.mean.cpu(), 'sq_mean': self.sq_mean.cpu(),
                    'cov_mat_sqrt': self.cov_mat_sqrt.cpu(), 'n_models': self.n_models}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.mean = ckpt['mean'].to(self.device)
        self.sq_mean = ckpt['sq_mean'].to(self.device)
        self.cov_mat_sqrt = ckpt['cov_mat_sqrt'].to(self.device)
        self.n_models = ckpt['n_models']


def train_swag(model, loss_fn, loader, epochs, lr, set_train=None):
    """
    Collect the SWAG moments: `epochs` epochs of AdamW at learning rate `lr`, one snapshot per epoch.
    loss_fn(batch) -> loss tensor; set_train(bool) switches the model train / eval mode.
    """
    tracker = SWAGTracker(model, max_num_models=epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs), desc="SWAG epochs"):
        if set_train:
            set_train(True)
        total = 0.0
        for batch in tqdm(loader, desc="SWAG batches", leave=False):
            optimizer.zero_grad()
            loss = loss_fn(batch)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"SWAG epoch {epoch + 1}/{epochs} | loss {total / len(loader):.4f}")
        tracker.update(model)
    return tracker


def get_tracker(model, state_path, loss_fn, loader, epochs, lr, retrain=False, set_train=None):
    """Train the SWAG tracker (and save it) or load it from state_path."""
    if retrain or not os.path.exists(state_path):
        tracker = train_swag(model, loss_fn, loader, epochs, lr, set_train)
        tracker.save(state_path)
        print(f"SWAG state saved to {state_path}")
    else:
        tracker = SWAGTracker(model, max_num_models=epochs)
        tracker.load(state_path)
        print(f"SWAG state loaded from {state_path}")
    return tracker


@torch.no_grad()
def swag_predict(model, tracker, predict, num_samples=30):
    """
    Mean and variance of predict() over `num_samples` models sampled from SWAG, accumulated on the fly
    (Welford) so that memory does not grow with num_samples. The original weights are restored at the end.
    """
    original = SWAGTracker._get_flat(model).clone()
    mean = m2 = None
    for k in tqdm(range(1, num_samples + 1), desc="SWAG sampling"):
        tracker.sample(model)
        pred = predict().double()
        if mean is None:
            mean, m2 = torch.zeros_like(pred), torch.zeros_like(pred)
        delta = pred - mean
        mean += delta / k
        m2 += delta * (pred - mean)
    SWAGTracker._set_flat(model, original)
    var = m2 / (num_samples - 1) if num_samples > 1 else torch.zeros_like(m2)
    return mean.float(), var.float()


def run_flow_swag(model, train_loader, eval_dir, swag_lr=1e-5, swag_epochs=20, num_samples=30, retrain=False,
                  c_from=0.0, c_to=1.0):
    """
    SWAG uncertainty of the VR -> SR transport of a FlowModel.

    The SWAG state is stored next to the flow weights (swag_state_<lr>.pth). The transport is evaluated on the
    SAME VR events stored in eval_dir/transform_bkg_to_sig.h5, so that the per-event variance is aligned with the
    nominal reconstruction. Output: eval_dir/transform_bkg_to_sig_swag_<lr>.h5 with X_reco_sig (SWAG mean),
    X_reco_sig_var, z_latent_bkg (scaled space) and the checksum of the transported events.
    c_from / c_to: conditions of the transport (see FlowModel.transport). The SWAG state depends only on the
    weights, so a model shared by several VRs trains it once and reuses it for every VR.
    """
    def loss_fn(batch):
        X, c = batch
        return model.loss_function(X.to(model.device), c.to(model.device))

    state_path = os.path.join(model.weights_dir, f'swag_state_{swag_lr}.pth')
    tracker = get_tracker(model, state_path, loss_fn, train_loader, swag_epochs, swag_lr, retrain, model.set_train)

    X_bkg_np = utils.load_h5(os.path.join(eval_dir, 'transform_bkg_to_sig.h5'), ['X_bkg'])['X_bkg']
    X_bkg = torch.as_tensor(X_bkg_np, dtype=torch.float32)

    def predict():
        z, x = model.transport(X_bkg, c_from=c_from, c_to=c_to)
        return torch.cat([x, z], dim=1)

    mean, var = swag_predict(model, tracker, predict, num_samples)
    n = X_bkg.shape[1]
    out_path = os.path.join(eval_dir, swag_file(swag_lr))
    utils.save_h5(out_path, {'X_reco_sig': mean[:, :n].numpy(), 'X_reco_sig_var': var[:, :n].numpy(),
                             'z_latent_bkg': mean[:, n:].numpy(), 'bkg_checksum': utils.checksum(X_bkg_np)})
    print(f"SWAG transport saved to {out_path}")
    model.load_model()          # SWAG training modified the weights: back to the nominal model


def swag_file(swag_lr):
    return f'transform_bkg_to_sig_swag_{swag_lr}.h5'
