"""
Tests of a trained FlowModel on one region (VR = class 0, SR = class 1), inputs in the scaled space.

- classification     : flow log-likelihood ratio log P(x|SR) - log P(x|VR) as VR / SR classifier
- feature importance : |correlation| between each feature and the log-likelihood ratio
- transport          : VR -> latent -> SR, saved to transform_bkg_to_sig.h5
                       + normality of the latent space (plots/latent_space_comparison/latent_normality)

The conditions of the VR and of the SR are c_bkg / c_sig: scalars (0 / 1, one flow per VR) or vectors
(e.g. one-hot region labels of the all-VR model, or (region, 4b flag) of the 2b4b models).

Every evaluation is cached in <dir_path>/eval_data and recomputed only if requested.
"""
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

import utils
from lib import visualization as vis

TRANSPORT_FILE = 'transform_bkg_to_sig.h5'


class ModelTester:
    def __init__(self, model, X_bkg, X_sig, features, dir_path, phi_features=(), bkg_label='VR', sig_label='SR',
                 c_bkg=0.0, c_sig=1.0, feature_importance=True):
        """
        param model: trained FlowModel
        param X_bkg, X_sig: scaled input features of the VR and of the SR (numpy arrays)
        param features: names of the input features
        param dir_path: region directory (plots in plots/, cached results in eval_data/)
        param phi_features: names of the phi features (uniform latent target)
        param c_bkg, c_sig: condition of the VR and of the SR (scalar or 1D vector, see FlowModel.context)
        param feature_importance: plot the feature importance in plots/ (False when it is shown elsewhere, e.g. in the
                                  training_diagnostic/ of the all-VR model; the correlations are always saved)
        """
        self.model = model
        self.X_bkg = torch.as_tensor(X_bkg, dtype=torch.float32)
        self.X_sig = torch.as_tensor(X_sig, dtype=torch.float32)
        self.features = list(features)
        self.phi_features = list(phi_features)
        self.bkg_label, self.sig_label = bkg_label, sig_label
        self.c_bkg, self.c_sig = c_bkg, c_sig
        self.feature_importance = feature_importance
        self.plot_dir = os.path.join(dir_path, 'plots')
        self.eval_dir = os.path.join(dir_path, 'eval_data')
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

    def _cached(self, filename, compute, force):
        """Load eval_data/<filename>, or compute it (compute() returns a dict) if forced, missing or outdated."""
        path = os.path.join(self.eval_dir, filename)
        if force or not utils.is_current(path):
            utils.save_h5(path, compute())
            print(f"  Saved {path}")
        return utils.load_h5(path)

    # -------------------------------------------------------------------------
    # Log-likelihood ratio
    # -------------------------------------------------------------------------
    def _llr(self, X, batch_size=8192):
        """log P(x | SR) - log P(x | VR) for each event."""
        out = []
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            lp_sig = self.model.log_prob(xb, self.model.context(self.c_sig, len(xb)))
            lp_bkg = self.model.log_prob(xb, self.model.context(self.c_bkg, len(xb)))
            out.append(lp_sig - lp_bkg)
        return torch.cat(out).numpy()

    def _compute_llr(self):
        X = torch.cat([self.X_bkg, self.X_sig])
        y = np.concatenate([np.zeros(len(self.X_bkg)), np.ones(len(self.X_sig))])
        llr = self._llr(X)
        corr = [abs(np.corrcoef(X[:, i].numpy(), llr)[0, 1]) for i in range(len(self.features))]
        return {'y': y, 'llr': llr, 'correlations': np.array(corr)}

    def classification_analysis(self, force_recompute=False):
        """ROC, score distribution, confusion matrix and feature importance from the log-likelihood ratio."""
        data = self._cached('classification.h5', self._compute_llr, force_recompute)
        y, llr = data['y'], data['llr']
        out = os.path.join(self.plot_dir, 'classification')

        fpr, tpr, thr = roc_curve(y, llr)
        best = np.argmax(tpr - fpr)
        auc = roc_auc_score(y, llr)
        y_pred = (llr > thr[best]).astype(int)
        labels = [f'{self.bkg_label} events', f'{self.sig_label} events']
        print(f"AUC = {auc:.4f} | accuracy (optimal threshold) = {accuracy_score(y, y_pred):.4f}")
        print(classification_report(y, y_pred, target_names=labels))

        vis.plot_roc(fpr, tpr, auc, best, os.path.join(out, 'roc_curve.png'))
        vis.plot_score_distribution(llr, y, thr[best], os.path.join(out, 'llr_distribution.png'),
                                    xlabel=rf'$\log[P(x|{self.sig_label}) / P(x|{self.bkg_label})]$',
                                    bkg_label=self.bkg_label, sig_label=self.sig_label)
        vis.plot_confusion_matrix(y, y_pred, [self.bkg_label, self.sig_label],
                                  os.path.join(out, 'confusion_matrix.png'))
        if self.feature_importance:
            vis.plot_feature_importance(self.features, data['correlations'],
                                        os.path.join(self.plot_dir, 'feature_importance.png'))
        return auc

    # -------------------------------------------------------------------------
    # Transport VR -> SR
    # -------------------------------------------------------------------------
    def _compute_transport(self):
        z, X_reco = self.model.transport(self.X_bkg, c_from=self.c_bkg, c_to=self.c_sig)
        return {'X_bkg': self.X_bkg.numpy(), 'X_sig': self.X_sig.numpy(),
                'X_reco_sig': X_reco.numpy(), 'z_latent_bkg': z.numpy()}

    def transport_analysis(self, force_recompute=False, plots=True):
        """Transport the VR events to the SR; plots the normality of the latent space."""
        data = self._cached(TRANSPORT_FILE, self._compute_transport, force_recompute)
        if plots:
            vis.plot_latent_normality(data['z_latent_bkg'], self.features,
                                      os.path.join(self.plot_dir, 'latent_space_comparison'), self.phi_features)
        return data
