"""
Conditional normalizing flow P(x | c), c = 0 (validation region) or 1 (signal region).

If phi features are given, the density is factorised as
    P(x, phi | c) = P(x | phi, c) * P(phi | c)
with a Neural Circular Spline Flow (NCSF) for the periodic phi features and a Neural Spline Flow (NSF)
for the other features, conditioned on c and phi.

Transport VR -> SR: z = T(x | c=0), x_reco = T^-1(z | c=1).
"""
import math
import os

import h5py
import numpy as np
import torch
import torch.optim as optim
import zuko
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

HISTORY_KEYS = ('train_loss', 'val_loss', 'lr', 'train_log_det_J', 'val_log_det_J')


def linear_warmup_cosine_decay(optimizer, warmup_steps, total_steps):
    """LR scheduler: linear warm-up 0 -> lr in `warmup_steps`, then cosine decay lr -> 0 until `total_steps`."""
    def lr_lambda(step):
        if step <= warmup_steps:
            return step / max(1, warmup_steps)
        if step >= total_steps:
            return 1e-8
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


class EarlyStopper:
    """Stops the training when the validation loss has not improved for `patience` epochs."""
    def __init__(self, patience=30, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = np.inf

    def early_stop(self, loss):
        if loss < self.best:
            self.best = loss
            self.counter = 0
        elif loss > self.best + self.min_delta:
            self.counter += 1
        return self.counter >= self.patience


class FlowModel:
    def __init__(self, input_dim, dir_path, context_dim=1, bins=20, transforms=4, hidden_features=(256, 256),
                 phi_indices=None, phi_bins=None, phi_transforms=None, phi_hidden_features=(64, 64),
                 lr=1e-3, patience=30, device=None):
        """
        param input_dim: number of input features (phi included)
        param dir_path: output directory of the model (weights in <dir_path>/weights)
        param context_dim: dimension of the condition (class label)
        param bins, transforms, hidden_features: NSF hyper-parameters
        param phi_indices: indices of the phi features (None -> single NSF on all features)
        param phi_bins, phi_transforms, phi_hidden_features: NCSF hyper-parameters (bins / transforms default to the NSF ones)
        param lr: learning rate of AdamW
        param patience: early-stopping patience (epochs)
        """
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dir_path = dir_path
        self.weights_dir = os.path.join(dir_path, 'weights')
        self.phi_indices = list(phi_indices or [])
        self.has_phi = len(self.phi_indices) > 0
        self.non_phi_indices = [i for i in range(input_dim) if i not in self.phi_indices]

        n_phi = len(self.phi_indices)
        # P(x | phi, c): context = c + phi
        self.flow = zuko.flows.NSF(features=len(self.non_phi_indices), context=context_dim + n_phi, bins=bins,
                                   transforms=transforms, hidden_features=list(hidden_features)).to(self.device)
        # P(phi | c)
        self.phi_flow = zuko.flows.NCSF(features=n_phi, context=context_dim, bins=phi_bins or bins,
                                        transforms=phi_transforms or transforms,
                                        hidden_features=list(phi_hidden_features)).to(self.device) if self.has_phi else None

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.early_stopper = EarlyStopper(patience=patience)
        self.history = {k: [] for k in HISTORY_KEYS}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def modules(self):
        return [self.flow] + ([self.phi_flow] if self.has_phi else [])

    def parameters(self):
        return [p for m in self.modules() for p in m.parameters()]

    def set_train(self, mode=True):
        for m in self.modules():
            m.train(mode)

    @staticmethod
    def _context(c):
        return c.unsqueeze(-1) if c.dim() == 1 else c

    @staticmethod
    def context(c, n):
        """
        (n, context_dim) condition tensor from one condition value:
        scalar -> column filled with c (binary VR / SR label); 1D vector -> the vector repeated on n rows.
        """
        c = torch.as_tensor(c, dtype=torch.float32)
        if c.dim() == 0:
            return torch.full((n, 1), float(c))
        return c.reshape(1, -1).expand(n, -1).clone()

    def _blocks(self, X, c):
        """List of (distribution, x) pairs whose log-probabilities sum to log P(X | c)."""
        c = self._context(c)
        if not self.has_phi:
            return [(self.flow(c), X)]
        x_phi = X[:, self.phi_indices]
        return [(self.phi_flow(c), x_phi), (self.flow(torch.cat([c, x_phi], dim=1)), X[:, self.non_phi_indices])]

    def _merge(self, x_non_phi, x_phi):
        """Inverse of the phi / non-phi split (original feature order)."""
        X = torch.empty(x_non_phi.shape[0], x_non_phi.shape[1] + x_phi.shape[1], device=x_non_phi.device)
        X[:, self.non_phi_indices] = x_non_phi
        X[:, self.phi_indices] = x_phi
        return X

    def print_model_summary(self):
        for m in self.modules():
            print(m)
        print(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    def log_prob_tensor(self, X, c):
        return sum(dist.log_prob(x) for dist, x in self._blocks(X, c))

    def loss_function(self, X, c):
        """Negative log-likelihood."""
        return -self.log_prob_tensor(X, c).mean()

    @torch.no_grad()
    def _log_det_jacobian(self, X, c):
        """Mean log|det J| of the forward transformation x -> z (monitoring only)."""
        total = 0.0
        for dist, x in self._blocks(X, c):
            total = total + dist.transform.log_abs_det_jacobian(x, dist.transform(x))
        return total.mean().item()

    def _run_epoch(self, loader, train):
        self.set_train(train)
        loss_sum, logdet_sum = 0.0, 0.0
        with torch.set_grad_enabled(train):
            for X, c in tqdm(loader, desc="Train" if train else "Validation", leave=False):
                X, c = X.to(self.device), c.to(self.device)
                loss = self.loss_function(X, c)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                loss_sum += loss.item()
                logdet_sum += self._log_det_jacobian(X, c)
        return loss_sum / len(loader), logdet_sum / len(loader)

    def train(self, train_loader, val_loader, epochs=500, warmup_epochs=50, resume=False):
        """
        Train with linear warm-up + cosine decay. The best (validation loss) weights are saved in weights_dir.
        resume=True restarts from the saved weights / history (the scheduler is fast-forwarded).
        """
        steps = len(train_loader)
        self.scheduler = linear_warmup_cosine_decay(self.optimizer, steps * warmup_epochs, steps * epochs)
        start_epoch = 0
        if resume and os.path.exists(self.model_path()):
            self.load_model()
            self.history = self.load_history()
            start_epoch = len(self.history['train_loss'])
            for _ in range(start_epoch * steps):
                self.scheduler.step()
            self.early_stopper.best = min(self.history['val_loss'], default=np.inf)
            print(f"Resuming from epoch {start_epoch}, LR = {self.optimizer.param_groups[0]['lr']:.2e}")

        print(f"Training on {self.device}...")
        for epoch in range(start_epoch, epochs):
            train_loss, train_logdet = self._run_epoch(train_loader, train=True)
            val_loss, val_logdet = self._run_epoch(val_loader, train=False)
            lr = self.optimizer.param_groups[0]['lr']
            for k, v in zip(HISTORY_KEYS, (train_loss, val_loss, lr, train_logdet, val_logdet)):
                self.history[k].append(v)
            print(f"Epoch {epoch + 1}/{epochs} | Train {train_loss:.4f} | Val {val_loss:.4f} | LR {lr:.2e}")

            if val_loss < self.early_stopper.best:
                self.save_model()
            self.save_history()
            if self.early_stopper.early_stop(val_loss):
                print("Early stopping triggered.")
                break
        self.load_model()      # keep the best weights

    # -------------------------------------------------------------------------
    # Save / load
    # -------------------------------------------------------------------------
    def model_path(self, weights_dir=None):
        return os.path.join(weights_dir or self.weights_dir, 'flow_model.pth')

    def save_model(self):
        os.makedirs(self.weights_dir, exist_ok=True)
        torch.save(self.flow.state_dict(), self.model_path())
        if self.has_phi:
            torch.save(self.phi_flow.state_dict(), os.path.join(self.weights_dir, 'phi_flow_model.pth'))

    def load_model(self, weights_dir=None):
        """Load the weights from weights_dir (default: the model's own directory)."""
        weights_dir = weights_dir or self.weights_dir
        self.flow.load_state_dict(torch.load(self.model_path(weights_dir), map_location=self.device))
        if self.has_phi:
            self.phi_flow.load_state_dict(torch.load(os.path.join(weights_dir, 'phi_flow_model.pth'),
                                                     map_location=self.device))
        print(f"Flow loaded from {weights_dir}")

    def history_path(self):
        return os.path.join(self.weights_dir, 'training_history.h5')

    def save_history(self):
        os.makedirs(self.weights_dir, exist_ok=True)
        with h5py.File(self.history_path(), 'w') as f:
            for k in HISTORY_KEYS:
                f.create_dataset(k, data=np.array(self.history[k]))

    def load_history(self):
        if not os.path.exists(self.history_path()):
            return self.history
        with h5py.File(self.history_path(), 'r') as f:
            return {k: f[k][:].tolist() for k in HISTORY_KEYS if k in f}

    # -------------------------------------------------------------------------
    # Inference (inputs are moved to the device, outputs returned on CPU)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_loss(self, loader):
        return self._run_epoch(loader, train=False)[0]

    @torch.no_grad()
    def log_prob(self, X, c):
        """log P(X | c)."""
        self.set_train(False)
        return self.log_prob_tensor(X.to(self.device), c.to(self.device)).cpu()

    @torch.no_grad()
    def sample(self, c):
        """One sample of P(x | c) per row of c."""
        self.set_train(False)
        c = self._context(c.to(self.device))
        if not self.has_phi:
            return self.flow(c).sample().cpu()
        x_phi = self.phi_flow(c).sample()
        x_non_phi = self.flow(torch.cat([c, x_phi], dim=1)).sample()
        return self._merge(x_non_phi, x_phi).cpu()

    @torch.no_grad()
    def to_latent(self, X, c):
        """z = T(X | c), in the original feature order."""
        self.set_train(False)
        z = [dist.transform(x) for dist, x in self._blocks(X.to(self.device), c.to(self.device))]
        return (z[0] if not self.has_phi else self._merge(z[1], z[0])).cpu()

    @torch.no_grad()
    def from_latent(self, z, c):
        """X = T^-1(z | c). With phi: first phi is inverted, then used as context for the other features."""
        self.set_train(False)
        z, c = z.to(self.device), self._context(c.to(self.device))
        if not self.has_phi:
            return self.flow(c).transform.inv(z).cpu()
        x_phi = self.phi_flow(c).transform.inv(z[:, self.phi_indices])
        x_non_phi = self.flow(torch.cat([c, x_phi], dim=1)).transform.inv(z[:, self.non_phi_indices])
        return self._merge(x_non_phi, x_phi).cpu()

    @torch.no_grad()
    def transport(self, X, c_from=0.0, c_to=1.0, batch_size=4096):
        """
        Transport X from condition c_from to condition c_to through the latent space. Returns (z, X_transported).
        c_from / c_to: scalars (binary label) or 1D vectors (multi-dimensional condition), see context().
        """
        z_all, x_all = [], []
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            z = self.to_latent(xb, self.context(c_from, len(xb)))
            z_all.append(z)
            x_all.append(self.from_latent(z, self.context(c_to, len(xb))))
        return torch.cat(z_all), torch.cat(x_all)
