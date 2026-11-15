"""
All the plotting functions of the project (CMS style through mplhep).

Common conventions:
  - model / prediction : filled + step histogram (coloured)
  - data / reference   : black points with error bars
  - ratio panels       : data / model, with the 1 sigma (hatched) and 2 sigma (light) model-uncertainty bands
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import mplhep as hep
from scipy.ndimage import gaussian_filter
from scipy.stats import norm, uniform


# =============================================================================
# Constants
# =============================================================================
LUMINOSITY = 34.7          # fb^-1
SQRT_S = 13.6              # TeV
CMS_TEXT = "Work in Progress"

KL_LIMIT = 0.003
KS_LIMIT = 0.03
CHI2_LIMIT = 5

VR_COLORS = {1: 'blue', 2: 'green', 3: 'red', 4: 'purple', 5: 'orange'}
VR_MARKERS = {1: 'o', 2: 's', 3: '^', 4: 'v', 5: '*'}

# Number of bins of each feature in the 2b case (halved for 4b, which has fewer events)
BIN_XFEATURES = {
    "HT": 200,
    "higgs1_reco_pt": 200, "higgs1_reco_eta": 100, "higgs1_reco_phi": 50, "higgs1_reco_mass": 80,
    "higgs2_reco_pt": 200, "higgs2_reco_eta": 100, "higgs2_reco_phi": 50, "higgs2_reco_mass": 80,
    "hh_vec_mass": 200, "hh_vec_pt": 200, "hh_vec_eta": 100, "hh_vec_phi": 50,
    "hh_vec_DeltaR": 100, "hh_vec_DeltaPhi": 50, "hh_vec_DeltaEta": 100,
    "higgs1_reco_jet1_pt": 200, "higgs1_reco_jet1_eta": 70, "higgs1_reco_jet1_phi": 50, "higgs1_reco_jet1_mass": 200,
    "higgs1_reco_jet2_pt": 200, "higgs1_reco_jet2_eta": 70, "higgs1_reco_jet2_phi": 50, "higgs1_reco_jet2_mass": 200,
    "higgs1_DeltaPhijj": 100, "higgs1_DeltaEtajj": 100, "higgs1_DeltaRjj": 100,
    "higgs2_reco_jet1_pt": 200, "higgs2_reco_jet1_eta": 70, "higgs2_reco_jet1_phi": 50, "higgs2_reco_jet1_mass": 200,
    "higgs2_reco_jet2_pt": 200, "higgs2_reco_jet2_eta": 70, "higgs2_reco_jet2_phi": 50, "higgs2_reco_jet2_mass": 200,
    "higgs2_DeltaPhijj": 100, "higgs2_DeltaEtajj": 100, "higgs2_DeltaRjj": 100,
    "minDeltaR_Higgjj": 100, "maxDeltaR_Higgjj": 100,
    "higgs1_helicityCosTheta": 60, "higgs2_helicityCosTheta": 60, "hh_CosThetaStar_CS": 60,
    "add_jet1pt_pt": 200, "add_jet1pt_eta": 70, "add_jet1pt_phi": 50, "add_jet1pt_mass": 200,
    "add_jet1pt_Higgs1_deta": 100, "add_jet1pt_Higgs1_dphi": 50, "add_jet1pt_Higgs1_m": 200,
    "add_jet1pt_Higgs2_deta": 100, "add_jet1pt_Higgs2_dphi": 50, "add_jet1pt_Higgs2_m": 200,
}

# Labels used in the final 4b evaluation
LBL_TILDE = r'$\widetilde{SR}_{4b}$'   # VR2b -> SR2b (flow), reweighted to 4b (FFN)
LBL_HAT = r'$\widehat{SR}_{4b}$'       # VR4b -> SR4b (flow)
LBL_TRUE = r'true $SR_{4b}$'
TRUE_FILL, TRUE_HATCH, TRUE_HATCH_COLOR = '0.85', '///', '0.45'


# =============================================================================
# Helpers
# =============================================================================
def set_style():
    hep.style.use("CMS")


def cms_label(ax, lumi=LUMINOSITY, com=SQRT_S):
    """'CMS Work in Progress' + luminosity. Recent mplhep versions call the text `text`, older ones `label`."""
    try:
        hep.cms.label(ax=ax, data=True, text=CMS_TEXT, lumi=lumi, com=com, loc=0)
    except TypeError:
        hep.cms.label(ax=ax, data=True, label=CMS_TEXT, lumi=lumi, com=com, loc=0)


def save_fig(fig, filename, dpi=150):
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)


def feature_bins(feature_name, nbjet=2, default=100):
    """Number of bins of a feature: BIN_XFEATURES for 2b, half of it for 4b."""
    n = BIN_XFEATURES.get(feature_name, default)
    return n if nbjet == 2 else max(n // 2, 10)


def percentile_edges(vals, n_bins, q=(0.5, 99.5)):
    """Uniform binning covering the [q0, q1] percentile range of vals."""
    lo, hi = np.percentile(vals, q)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, n_bins + 1)


def step_band(ax, edges, lo, hi, **kw):
    """Band drawn as steps following the binning."""
    ax.fill_between(edges, np.append(lo, lo[-1]), np.append(hi, hi[-1]), step='post', **kw)


def sigma_bands(ax, edges, band, color, center=1.0):
    """1 sigma (hatched) and 2 sigma (light) bands around `center`, band = relative uncertainty."""
    step_band(ax, edges, center - 2 * band, center + 2 * band, color=color, alpha=0.1, linewidth=0,
              label=r'$2\sigma$ Unc.')
    step_band(ax, edges, center - band, center + band, color=color, alpha=0.3, hatch='////', linewidth=0,
              label=r'$1\sigma$ Unc.')


def ratio_to_model(c_data, e_data, c_model, e_model):
    """data / model ratio, its error (data only) and the relative model uncertainty band."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ok = c_model > 0
        ratio = np.where(ok, c_data / c_model, np.nan)
        ratio_err = np.where(ok, e_data / c_model, 0)
        band = np.where(ok, e_model / c_model, 0)
    return ratio, ratio_err, band


def _style_ratio_axis(axr, ylabel, xlabel=None, legend=True, fontsize=14):
    axr.axhline(1, color='black', linestyle='--', linewidth=1)
    axr.set_ylabel(ylabel, fontsize=fontsize)
    axr.set_ylim(0.5, 1.5)
    axr.grid(True, alpha=0.3, axis='y')
    if xlabel is not None:
        axr.set_xlabel(xlabel)
    if legend:
        axr.legend(loc='upper right', fontsize=10, ncol=2)


def _chunks(features, n_per_row):
    return [features[i:i + n_per_row] for i in range(0, len(features), n_per_row)]


# =============================================================================
# Preprocessing
# =============================================================================
def plot_preprocessing(raw, processed, features, out_dir):
    """
    For each feature: distributions before (left) and after (right) the preprocessing.
    raw / processed = {sample label: (N, n_features) array}, same labels in both.
    """
    set_style()
    colors = dict(zip(raw, ['green', 'black', 'blue', 'red']))
    for i, ft in enumerate(features):
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        for ax, data, title in ((axes[0], raw, 'Raw'), (axes[1], processed, 'Preprocessed')):
            for lbl, X in data.items():
                v = X[:, i][np.isfinite(X[:, i])]
                counts, edges = np.histogram(v, bins=percentile_edges(v, 100), density=True)
                hep.histplot(counts, bins=edges, ax=ax, histtype='step', color=colors[lbl], linewidth=2, label=lbl)
            ax.set_xlabel(f'{ft} ({title.lower()})')
            ax.set_ylabel('Density')
            ax.legend(loc='upper right')
        cms_label(axes[0])
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, f'{ft}_preprocessing.png'))


# =============================================================================
# Training curves
# =============================================================================
def plot_curves(curves, filename, ylabel, xlabel='Epoch', logy=False, label=True):
    """Generic line plot: curves = {label: values (one per epoch)}."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl, values in curves.items():
        ax.plot(np.arange(1, len(values) + 1), values, label=lbl, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ax.legend()
    if label:
        cms_label(ax)
    save_fig(fig, filename)


def plot_flow_history(history, plot_dir):
    """Loss, learning rate (linear and log) and log|det J| of the normalizing flow training."""
    plot_curves({'Train': history['train_loss'], 'Validation': history['val_loss']},
                os.path.join(plot_dir, 'training_validation_loss.png'), 'Negative Log Likelihood')
    for logy, name in ((False, 'learning_rate.png'), (True, 'learning_rate_logscale.png')):
        plot_curves({'Learning rate': history['lr']}, os.path.join(plot_dir, name), 'Learning Rate',
                    logy=logy, label=False)
    plot_curves({'Train': history['train_log_det_J'], 'Validation': history['val_log_det_J']},
                os.path.join(plot_dir, 'log_det_J.png'), r'$\log|\det J|$')


def plot_velocity_map(T, X, V, filename, ylabel='Scaled minDeltaR_Higgjj $x$'):
    """Heat map + stream plot of a learned 1D velocity field v(x, t)."""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 10))
    contour = ax.contourf(T, X, V, levels=50, cmap='RdBu_r')
    fig.colorbar(contour, ax=ax, label='Velocity $v(x, t)$')
    ax.streamplot(T, X, np.ones_like(V), V, color='k', density=1.2, linewidth=1, arrowsize=1.5)
    ax.set_xlabel('Time $t$ (0 = Reco, 1 = True)')
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    save_fig(fig, filename)


# =============================================================================
# Classification (VR vs SR) with the flow likelihood ratio
# =============================================================================
def plot_roc(fpr, tpr, auc, optimal_idx, filename):
    set_style()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    ax.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], s=100, c='red', zorder=5, label='Optimal threshold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    cms_label(ax)
    save_fig(fig, filename)


def plot_score_distribution(scores, y, threshold, filename, xlabel, bkg_label, sig_label, n_bins=100):
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    edges = percentile_edges(scores, n_bins, q=(0.1, 99.9))
    for cls, lbl, color in ((0, bkg_label, 'green'), (1, sig_label, 'black')):
        counts, _ = np.histogram(scores[y == cls], bins=edges)
        hep.histplot(counts, bins=edges, ax=ax, histtype='step', color=color, linewidth=2, label=lbl)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label='Optimal threshold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Events')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    cms_label(ax)
    save_fig(fig, filename)


def plot_confusion_matrix(y_true, y_pred, class_labels, filename):
    """
    Two confusion matrices side by side, both normalised to the total number of events (entries sum to 1):
    left  True vs Predicted (the classifier), right True vs True (the ideal classifier, i.e. the class fractions).
    """
    set_style()
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    n_cls = len(class_labels)
    n_tot = max(len(y_true), 1)

    def matrix(rows, cols):
        cm = np.zeros((n_cls, n_cls))
        np.add.at(cm, (rows, cols), 1)
        return cm / n_tot

    panels = ((matrix(y_true, y_pred), 'True vs Predicted', 'Predicted', 'Blues'),
              (matrix(y_true, y_true), 'True vs True', 'True', 'Oranges'))
    vmax = max(p[0].max() for p in panels)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for ax, (cm, title, xlabel, cmap) in zip(axes, panels):
        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(j, i, f'{cm[i, j]:.3f}', ha='center', va='center',
                        color='white' if cm[i, j] > 0.6 * vmax else 'black', fontsize=22)
        ax.set_xticks(range(n_cls))
        ax.set_xticklabels(class_labels)
        ax.set_yticks(range(n_cls))
        ax.set_yticklabels(class_labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('True')
        ax.set_title(title, fontsize=22, pad=40)
        cms_label(ax)
    fig.tight_layout()
    save_fig(fig, filename)


def plot_feature_importance(features, values, filename, xlabel='|corr(feature, log LR)|'):
    """Horizontal bar chart sorted by importance."""
    set_style()
    order = np.argsort(values)[::-1]
    fig, ax = plt.subplots(figsize=(14, max(8, 0.45 * len(features))))
    ax.barh(range(len(features)), np.asarray(values)[order], color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([features[i] for i in order], fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', alpha=0.3)
    cms_label(ax)
    fig.tight_layout()
    save_fig(fig, filename)


def plot_feature_importance_vrs(features, values, filename, xlabel='|corr(feature, log LR)|'):
    """
    Feature importance of all the validation regions in one figure (grouped horizontal bars),
    values = {vr: array of one value per feature}. Features sorted by the mean importance over the VRs.
    """
    set_style()
    vrs = sorted(values)
    M = np.array([values[vr] for vr in vrs], dtype=float)
    order = np.argsort(np.nanmean(M, axis=0))[::-1]
    n_f, n_v = len(features), len(vrs)
    height = 0.8 / n_v
    fig, ax = plt.subplots(figsize=(14, max(8, 0.6 * n_f)))
    y = np.arange(n_f)
    for k, vr in enumerate(vrs):
        ax.barh(y - 0.4 + (k + 0.5) * height, M[k, order], height=height, color=VR_COLORS.get(vr, 'gray'),
                edgecolor='black', linewidth=0.5, label=vr_label(vr))
    ax.set_yticks(y)
    ax.set_yticklabels([features[i] for i in order], fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='lower right', fontsize=16)
    cms_label(ax)
    fig.tight_layout()
    save_fig(fig, filename)


# =============================================================================
# Single validation region
# =============================================================================
def _model_vs_data(ax, axr, edges, model, data, model_label, data_label, color, xlim=None):
    """Distribution panel (model filled, data points) + ratio panel data / model with model bands."""
    (c_m, e_m), (c_d, e_d) = model, data
    hep.histplot(c_m, bins=edges, ax=ax, histtype='fill', color=color, alpha=0.3)
    hep.histplot(c_m, bins=edges, ax=ax, histtype='step', color=color, label=model_label, linewidth=2)
    hep.histplot(c_d, bins=edges, yerr=e_d, ax=ax, histtype='errorbar', color='black', label=data_label,
                 markersize=5, elinewidth=1.5)
    ax.set_ylabel('Density')
    ax.legend(loc='upper right', fontsize=16)
    ax.tick_params(labelbottom=False)
    cms_label(ax)

    ratio, ratio_err, band = ratio_to_model(c_d, e_d, c_m, e_m)
    sigma_bands(axr, edges, band, color)
    hep.histplot(ratio, bins=edges, yerr=ratio_err, ax=axr, histtype='errorbar', color='black',
                 markersize=5, elinewidth=1.5)
    _style_ratio_axis(axr, f'{data_label} / {model_label}')
    if xlim is not None:
        ax.set_xlim(*xlim)
        axr.set_xlim(*xlim)


def plot_bkg_sig_reco(feature_name, edges, sig, bkg, reco, filename,
                      bkg_label='VR', sig_label='SR', reco_label='Reco SR'):
    """
    2x2 figure. Left: SR vs VR (before the transport). Right: SR vs Reco SR (after the transport).
    sig / bkg / reco = (density, error).
    """
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw={'height_ratios': [3, 1]})
    _model_vs_data(axes[0, 0], axes[1, 0], edges, bkg, sig, bkg_label, sig_label, 'green')
    _model_vs_data(axes[0, 1], axes[1, 1], edges, reco, sig, reco_label, sig_label, 'blue')
    for axr in axes[1]:
        axr.set_xlabel(feature_name)
    fig.tight_layout()
    save_fig(fig, filename)


def plot_qq(ref_vals, others, filename, xlabel, ylabel, n_quantiles=100):
    """
    Q-Q plot. x = quantiles of the reference sample, y = quantiles of each other sample.
    others = {label: (values, color)}
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    lo, hi = np.percentile(ref_vals, [0.5, 99.5])
    for lbl, (vals, color) in others.items():
        n_q = min(n_quantiles, len(ref_vals), len(vals))
        if n_q < 2:
            continue
        q = np.linspace(0.01, 0.99, n_q)
        ax.scatter(np.nanquantile(ref_vals, q), np.nanquantile(vals, q), alpha=0.6, color=color, s=20, label=lbl)
        v_lo, v_hi = np.percentile(vals, [0.5, 99.5])
        lo, hi = min(lo, v_lo), max(hi, v_hi)
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=2, label='y = x')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=14)
    fig.tight_layout()
    save_fig(fig, filename)


def plot_summary_bar(values, filename, ylabel, color='purple', hline=None, sort_descending=True):
    """Bar plot of one value per feature, values = {feature: value}."""
    set_style()
    items = sorted(values.items(), key=lambda kv: (np.nan_to_num(kv[1], nan=-np.inf)), reverse=sort_descending)
    fig, ax = plt.subplots(figsize=(max(14, 0.45 * len(items)), 8))
    x = np.arange(len(items))
    ax.bar(x, [v for _, v in items], color=color, alpha=0.8, edgecolor='black', linewidth=1.2)
    if hline is not None:
        ax.axhline(hline, color='black', linestyle='--', linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels([k for k, _ in items], rotation=45, ha='right', fontsize=14)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    cms_label(ax)
    fig.tight_layout()
    save_fig(fig, filename)


def plot_latent_normality(z, features, out_dir, phi_features=()):
    """
    Histogram of each latent dimension vs its target distribution:
    N(0, 1) for standard features, U(-pi, pi) for the phi features (circular spline flow).
    Also produces the summary of the mean / sigma residuals. Returns {'mu', 'sigma'} arrays.
    """
    set_style()
    dim_dir = os.path.join(out_dir, 'latent_normality')
    mus, sigmas, targets = [], [], []
    for i, name in enumerate(features):
        data = z[:, i]
        is_phi = name in phi_features
        x = np.linspace(min(-4, data.min() - 1), max(4, data.max() + 1), 200)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.hist(data, bins=50, density=True, alpha=0.3, color='blue', label='Latent')
        if is_phi:
            mu, sigma, target = data.mean(), data.std(), np.pi / np.sqrt(3)
            ax.plot(x, uniform.pdf(x, loc=-np.pi, scale=2 * np.pi), 'k--', linewidth=2, label=r'Target $U(-\pi, \pi)$')
        else:
            mu, sigma = norm.fit(data)
            target = 1.0
            ax.plot(x, norm.pdf(x, mu, sigma), 'b-', linewidth=2, label=f'Fit N({mu:.2f}, {sigma:.2f})')
            ax.plot(x, norm.pdf(x, 0, 1), 'k--', linewidth=2, label='Target N(0, 1)')
        mus.append(mu)
        sigmas.append(sigma)
        targets.append(target)
        ax.set_xlabel(f'z ({name})')
        ax.set_ylabel('Density')
        ax.legend(fontsize=14, loc='upper right')
        ax.grid(alpha=0.3)
        save_fig(fig, os.path.join(dim_dir, f'latent_z_{name}.png'))

    mus, sigmas = np.array(mus), np.array(sigmas)
    plot_summary_bar(dict(zip(features, mus)), os.path.join(out_dir, 'latent_mean_residuals.png'),
                     'Mean - target mean', color='blue', hline=0, sort_descending=False)
    plot_summary_bar(dict(zip(features, sigmas - np.array(targets))), os.path.join(out_dir, 'latent_sigma_residuals.png'),
                     r'$\sigma$ - target $\sigma$', color='red', hline=0, sort_descending=False)
    return {'mu': mus, 'sigma': sigmas}


def plot_variance_hist(var, feature_name, n_bins, filename):
    """Distribution of the per-event SWAG variance of one feature."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(var[np.isfinite(var)], bins=n_bins, color='orange', alpha=0.7)
    ax.set_xlabel(f'Var({feature_name})')
    ax.set_ylabel('Events')
    ax.set_yscale('log')
    cms_label(ax)
    save_fig(fig, filename)


def plot_correction(true, reco, corrected, feature_name, filename, n_bins=60):
    """Flow-matching correction of one feature: true vs uncorrected vs corrected, plus residuals."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    edges = np.linspace(0, max(np.max(true), np.max(reco)), n_bins)
    c_true, _ = np.histogram(true, bins=edges, density=True)
    c_reco, _ = np.histogram(reco, bins=edges, density=True)
    c_corr, _ = np.histogram(corrected, bins=edges, density=True)
    raw_true, _ = np.histogram(true, bins=edges)

    hep.histplot(c_true, bins=edges, ax=ax1, histtype='errorbar', color='black', label='SR', markersize=5)
    hep.histplot(c_reco, bins=edges, ax=ax1, histtype='fill', color='red', alpha=0.3, label='Reco SR')
    hep.histplot(c_corr, bins=edges, ax=ax1, histtype='step', color='green', linewidth=2.5, label='FM corrected Reco SR')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper right', fontsize=16)
    cms_label(ax1)

    centers = 0.5 * (edges[:-1] + edges[1:])
    ok = c_true > 0
    err = np.where(raw_true > 0, 1 / np.sqrt(np.maximum(raw_true, 1)), 0)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.errorbar(centers[ok], (c_reco[ok] - c_true[ok]) / c_true[ok], yerr=err[ok], fmt='o', color='red',
                 markersize=4, label='Reco')
    ax2.errorbar(centers[ok], (c_corr[ok] - c_true[ok]) / c_true[ok], yerr=err[ok], fmt='s', color='green',
                 markersize=4, label='Corrected')
    ax2.set_xlabel(feature_name)
    ax2.set_ylabel('(X - SR) / SR', fontsize=14)
    ax2.set_ylim(-0.5, 0.5)
    ax2.legend(loc='upper right', fontsize=12, ncol=2)
    fig.tight_layout()
    save_fig(fig, filename)


def plot_tail_study(feature, complete_orig, complete_reco, tail_orig, tail_reco, tail_label, filename, n_bins=60):
    """Where the events of a problematic tail end up in another feature (fraction of events)."""
    set_style()
    edges = np.linspace(min(complete_orig.min(), complete_reco.min()),
                        max(complete_orig.max(), complete_reco.max()), n_bins)
    n_orig, n_reco = len(complete_orig), len(complete_reco)
    fig, ax = plt.subplots(figsize=(10, 8))
    hep.histplot(np.histogram(complete_orig, edges)[0] / n_orig, bins=edges, ax=ax, histtype='fill',
                 color='gray', alpha=0.3, label='SR')
    hep.histplot(np.histogram(complete_reco, edges)[0] / n_reco, bins=edges, ax=ax, histtype='step',
                 color='black', linestyle='--', label='Reco SR')
    hep.histplot(np.histogram(tail_orig, edges)[0] / n_orig, bins=edges, ax=ax, histtype='fill',
                 color='blue', alpha=0.3, label=f'SR, {tail_label}')
    hep.histplot(np.histogram(tail_reco, edges)[0] / n_reco, bins=edges, ax=ax, histtype='step',
                 color='red', linewidth=2.5, label=f'Reco SR, {tail_label}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Fraction of events')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=12)
    cms_label(ax)
    save_fig(fig, filename)


# =============================================================================
# All validation regions in the same plot
# =============================================================================
def plot_multi_vr_distribution(feature_name, edges, sig, bkg_dict, reco_dict, filename, legend_fs=14,
                               sig_label='SR'):
    """
    Same structure as plot_bkg_sig_reco but with all the VRs.
    Left column: VRs vs SR. Right column: Reco SR of each VR vs SR. One ratio panel per VR below each column.
    sig = (density, error), bkg_dict / reco_dict = {vr: (density, error)}.
    """
    set_style()
    vrs = sorted(set(bkg_dict) | set(reco_dict))
    n, ratio_h = len(vrs), 0.5
    fig = plt.figure(figsize=(18, 3 * (3 + ratio_h * n) + 1))
    gs = fig.add_gridspec(1 + n, 2, height_ratios=[3] + [ratio_h] * n, hspace=0.12, wspace=0.18,
                          left=0.07, right=0.98, top=0.95, bottom=0.06)
    columns = [(bkg_dict, lambda vr: f'VR{vr}'), (reco_dict, lambda vr: f'Reco SR (VR{vr})')]
    c_sig, e_sig = sig

    for col, (model, label) in enumerate(columns):
        ax = fig.add_subplot(gs[0, col])
        for vr in vrs:
            if vr in model:
                hep.histplot(model[vr][0], bins=edges, ax=ax, histtype='step', color=VR_COLORS[vr],
                             label=label(vr), linewidth=2)
        hep.histplot(c_sig, bins=edges, yerr=e_sig, ax=ax, histtype='errorbar', color='black', label=sig_label,
                     markersize=5, elinewidth=1.5)
        if col == 0:
            ax.set_ylabel('Density')
        ax.set_xlim(edges[0], edges[-1])
        ax.legend(loc='best', fontsize=legend_fs, ncol=2)
        ax.tick_params(labelbottom=False)
        cms_label(ax)

        for k, vr in enumerate(vrs):
            axr = fig.add_subplot(gs[1 + k, col], sharex=ax)
            if vr not in model:
                axr.axis('off')
                continue
            ratio, ratio_err, band = ratio_to_model(c_sig, e_sig, *model[vr])
            step_band(axr, edges, 1 - 2 * band, 1 + 2 * band, color=VR_COLORS[vr], alpha=0.1, label=r'$2\sigma$')
            step_band(axr, edges, 1 - band, 1 + band, color=VR_COLORS[vr], alpha=0.3, hatch='////', label=r'$1\sigma$')
            hep.histplot(ratio, bins=edges, yerr=ratio_err, ax=axr, histtype='errorbar', color='black',
                         markersize=4, elinewidth=1.2)
            _style_ratio_axis(axr, f'{sig_label} / {label(vr)}', legend=False, fontsize=12)
            axr.set_yticks([0.75, 1.0, 1.25])
            axr.legend(loc='upper right', fontsize=legend_fs, ncol=2, framealpha=0.7)
            if k < n - 1:
                axr.tick_params(labelbottom=False)
            else:
                axr.set_xlabel(feature_name)
    save_fig(fig, filename)


def plot_metric_summary(bars, features, filename, ylabel, limit=None, markers=None,
                        bar_label=lambda vr: f'VR{vr}', marker_label=None, n_per_row=26):
    """
    Grouped bar plot of a metric for each feature and each VR, split over rows of `n_per_row` features.
      bars    = {vr: {feature: value}}  -> one bar per VR
      markers = {vr: {feature: value}}  -> optional marker on top of the bar of the same VR
    """
    set_style()
    markers = markers or {}
    vrs = sorted(set(bars) | set(markers))
    bar_w = 0.8 / max(len(vrs), 1)
    vals = [d.get(vr, {}).get(f, np.nan) for d in (bars, markers) for vr in vrs for f in features]
    if limit is not None:
        vals.append(limit)
    ymax = np.nanmax(vals) * (1.45 if markers else 1.1) if np.any(np.isfinite(vals)) else 1.0

    rows = _chunks(features, n_per_row)
    fig, axes = plt.subplots(len(rows), 1, figsize=(max(18, 0.6 * len(features)), 8 * len(rows)))
    axes = np.atleast_1d(axes)
    for r, (ax, feats) in enumerate(zip(axes, rows)):
        x = np.arange(len(feats))
        col_bar, col_mark = [], []
        for k, vr in enumerate(vrs):
            xk = x + (k - (len(vrs) - 1) / 2) * bar_w
            b = ax.bar(xk, [bars.get(vr, {}).get(f, np.nan) for f in feats], bar_w, color=VR_COLORS[vr],
                       alpha=0.85, edgecolor='black', linewidth=0.6)
            col_bar.append((b, bar_label(vr)))
            if markers:
                m, = ax.plot(xk, [markers.get(vr, {}).get(f, np.nan) for f in feats], linestyle='none',
                             marker=VR_MARKERS.get(vr, 'o'), markersize=9, color=VR_COLORS[vr],
                             markeredgecolor='black', markeredgewidth=0.8, zorder=3)
                col_mark.append((m, marker_label(vr)))
        if limit is not None:
            ax.axhline(limit, color='black', linestyle='--', linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(feats, rotation=45, ha='right', fontsize=18)
        ax.set_xlim(-0.5, n_per_row - 0.5)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        if r == 0:
            columns = [col_bar, col_mark] if markers else [col_bar]
            columns_legend(ax, columns, loc='upper right', fontsize=18, framealpha=0.8)
            cms_label(ax)
    fig.tight_layout()
    save_fig(fig, filename)


# =============================================================================
# Corner plot
# =============================================================================
def _contour_levels(H, fracs=(0.68, 0.95)):
    """Density levels containing the fractions `fracs` of the total (increasing, as needed by contour)."""
    flat = np.sort(H.ravel())[::-1]
    if flat.sum() <= 0:
        return None
    cum = np.cumsum(flat) / flat.sum()
    levels = np.unique([flat[min(np.searchsorted(cum, f), len(flat) - 1)] for f in fracs])
    return levels if len(levels) > 0 and levels[-1] > 0 else None


def corner_plot(datasets, features, filename, colors=None, max_events=100000, n_bins_1d=50, n_bins_2d=40,
                smooth=1.0, q_range=(0.5, 99.5), fracs=(0.68, 0.95), cell_size=1.7, label_fs=11, legend_fs=60,
                seed=42):
    """
    datasets = {label: array (N, n_features)}. The FIRST one is the reference: it defines the ranges and is drawn
    with filled grey contours, the others with coloured contour lines.
    Diagonal: normalised 1D histograms. Lower triangle: 2D contours with the 68% and 95% of the events.
    """
    set_style()
    labels, n = list(datasets), len(features)
    colors = colors or ['black', 'blue', 'red', 'green', 'purple', 'orange']
    rng = np.random.default_rng(seed)
    data = {}
    for lab, arr in datasets.items():
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr).all(axis=1)]
        if len(arr) > max_events:
            arr = arr[rng.choice(len(arr), max_events, replace=False)]
        data[lab] = arr
    ranges = [tuple(percentile_edges(data[labels[0]][:, k], 1, q_range)) for k in range(n)]

    fig, axes = plt.subplots(n, n, figsize=(cell_size * n, cell_size * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue
            if i == j:
                for d, lab in enumerate(labels):
                    ax.hist(data[lab][:, i], bins=n_bins_1d, range=ranges[i], density=True, histtype='step',
                            color=colors[d], linewidth=1.8)
                ax.set_xlim(*ranges[i])
                ax.set_yticks([])
            else:
                for d, lab in enumerate(labels):
                    H, xe, ye = np.histogram2d(data[lab][:, j], data[lab][:, i], bins=n_bins_2d,
                                               range=[ranges[j], ranges[i]])
                    H = gaussian_filter(H, smooth)
                    levels = _contour_levels(H, fracs)
                    if levels is None:
                        continue
                    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
                    if d == 0 and len(levels) >= 2:
                        ax.contourf(xc, yc, H.T, levels=list(levels) + [H.max() * 1.001],
                                    colors=['0.88', '0.65'], zorder=0)
                    ax.contour(xc, yc, H.T, levels=levels, colors=colors[d],
                               linewidths=[1.0, 1.6][-len(levels):], zorder=1 + d)
                ax.set_xlim(*ranges[j])
                ax.set_ylim(*ranges[i])
            ax.tick_params(labelsize=8, length=3)
            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(features[j], fontsize=label_fs)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            if j > 0 and i != j:
                ax.set_yticklabels([])
            elif j == 0 and i > 0:
                ax.set_ylabel(features[i], fontsize=label_fs)

    handles = [Patch(facecolor='0.75', edgecolor=colors[0], label=labels[0])]
    handles += [Line2D([0], [0], color=colors[d], linewidth=3, label=lab) for d, lab in enumerate(labels) if d > 0]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.95, 0.93), fontsize=legend_fs, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.2, w_pad=0.2)
    save_fig(fig, filename, dpi=120)


# =============================================================================
# Final 4b evaluation (FFN-reweighted 2b vs 4b transport vs true SR 4b)
# =============================================================================
def vr_label(vr):
    return rf'VR$_{{{vr}}}$'


def _true_handle():
    return Patch(facecolor=TRUE_FILL, edgecolor='black', hatch=TRUE_HATCH, linewidth=1.5)


def columns_legend(ax, columns, center=None, **kw):
    """
    Legend with fixed columns: columns = [[(handle, label), ...], ...].
    Shorter columns are padded with invisible cells; `center` = index of the column to centre vertically.
    """
    nrow = max(len(c) for c in columns)
    ghost = columns[0][0][1]      # padding cells reuse a real label (same height) which is then hidden
    handles, labels, hidden = [], [], []
    for j, col in enumerate(columns):
        pad = nrow - len(col)
        top = pad // 2 if j == center else 0
        for cell in [None] * top + list(col) + [None] * (pad - top):
            if cell is None:
                hidden.append(len(labels))
                cell = (Line2D([], [], linestyle='none', marker='', alpha=0), ghost)
            handles.append(cell[0])
            labels.append(cell[1])
    leg = ax.legend(handles, labels, ncol=len(columns), **kw)
    for i in hidden:
        leg.get_texts()[i].set_alpha(0)
    return leg


def _ratio_legend(axr, rows, legend_fs):
    """Two-row legend of a ratio panel: rows = [(loc, [(handle, label) or None, ...]), ...]."""
    legs = []
    for loc, items in rows:
        items = [it for it in items if it is not None]
        if items:
            legs.append(axr.legend([h for h, _ in items], [l for _, l in items], loc=loc, ncol=len(items),
                                   fontsize=legend_fs - 3, framealpha=0.7))
    for leg in legs[:-1]:
        axr.add_artist(leg)        # otherwise the next legend replaces the previous one


def _draw_final4b_column(fig, gs, col, panel, vrs, feature_name, legend_fs, mode='model', true_points=False):
    """
    One column of the final 4b figure: main panel + one ratio panel per VR.
    panel = dict(edges, line={vr: (c, e)}, mark={vr: (c, e)}, true=(c, e) or None,
                 lbl_line=f(vr), lbl_mark=f(vr), lbl_line_short=str, lbl_mark_short=str)
    mode='model': ratio denominator = line (coloured 1/2 sigma bands = line uncertainty).
    mode='true' : ratio denominator = true SR 4b (grey bands), lines drawn with their error band.
    """
    edges, line, mark, true = panel['edges'], panel['line'], panel['mark'], panel['true']
    line_band = (mode == 'true')
    ax = fig.add_subplot(gs[0, col])

    if true is not None:
        ax.stairs(true[0], edges, fill=True, facecolor=TRUE_FILL, edgecolor=TRUE_HATCH_COLOR, hatch=TRUE_HATCH,
                  linewidth=0, zorder=0.5)
        ax.stairs(true[0], edges, color='black', linewidth=1.5, zorder=0.6)
    for vr in vrs:
        if vr in line:
            c_l, e_l = line[vr]
            if line_band:
                step_band(ax, edges, c_l - e_l, c_l + e_l, color=VR_COLORS[vr], alpha=0.3, linewidth=0)
            hep.histplot(c_l, bins=edges, ax=ax, histtype='step', color=VR_COLORS[vr], linewidth=2,
                         label=panel['lbl_line'](vr))
        if vr in mark:
            hep.histplot(mark[vr][0], bins=edges, yerr=mark[vr][1], ax=ax, histtype='errorbar', color=VR_COLORS[vr],
                         marker=VR_MARKERS.get(vr, 'o'), markersize=5, elinewidth=1.2, linestyle='none',
                         label=panel['lbl_mark'](vr))
    if col == 0:
        ax.set_ylabel('Density')
    ax.set_xlim(edges[0], edges[-1])

    # Legend in columns: lines | markers | true SR (centred)
    hmap = dict(zip(*ax.get_legend_handles_labels()[::-1]))

    def line_handle(lbl, vr):
        if not line_band:
            return hmap[lbl]
        return (Patch(facecolor=VR_COLORS[vr], alpha=0.3, linewidth=0), Line2D([], [], color=VR_COLORS[vr], linewidth=2))

    col_line = [(line_handle(panel['lbl_line'](v), v), panel['lbl_line'](v)) for v in vrs if panel['lbl_line'](v) in hmap]
    col_mark = [(hmap[l], l) for l in (panel['lbl_mark'](v) for v in vrs) if l in hmap]
    columns = [c for c in (col_line, col_mark) if c]
    if true is not None:
        columns.append([(_true_handle(), LBL_TRUE)])
    if columns:
        columns_legend(ax, columns, center=len(columns) - 1, loc='best', fontsize=legend_fs)
    ax.tick_params(labelbottom=False)
    cms_label(ax)

    for k, vr in enumerate(vrs):
        axr = fig.add_subplot(gs[1 + k, col], sharex=ax)
        has_den = (true is not None and (vr in line or vr in mark)) if mode == 'true' else (vr in line)
        if not has_den:
            axr.axis('off')
            continue
        color = VR_COLORS[vr]
        axr.axhline(1, color='black', linestyle='--', linewidth=1)
        h_mark = h_other = None

        if mode == 'true':
            c_den, e_den = true
            den_ok = c_den > 0
            band = np.where(den_ok, e_den / c_den, 0)
            step_band(axr, edges, 1 - 2 * band, 1 + 2 * band, color='0.85', linewidth=0, zorder=0.3)
            step_band(axr, edges, 1 - band, 1 + band, facecolor='0.65', edgecolor=TRUE_HATCH_COLOR, hatch=TRUE_HATCH,
                      linewidth=0, zorder=0.4)
            h2s = (Patch(facecolor='0.85', linewidth=0), rf'{LBL_TRUE} $2\sigma$')
            h1s = (Patch(facecolor='0.65', edgecolor=TRUE_HATCH_COLOR, hatch=TRUE_HATCH, linewidth=0),
                   rf'{LBL_TRUE} $1\sigma$')
            den_lbl = r'$SR_{4b}^{true}$'
            if vr in line:
                c_l, e_l = line[vr]
                r_l = np.where(den_ok, c_l / c_den, np.nan)
                re_l = np.where(den_ok, e_l / c_den, 0)
                step_band(axr, edges, r_l - re_l, r_l + re_l, color=color, alpha=0.3, linewidth=0)
                axr.stairs(r_l, edges, color=color, linewidth=1.8, baseline=None)
                h_other = ((Patch(facecolor=color, alpha=0.3, linewidth=0), Line2D([], [], color=color, linewidth=1.8)),
                           f"{panel['lbl_line_short']} / {den_lbl}")
        else:
            c_den, e_den = line[vr]
            den_ok = c_den > 0
            band = np.where(den_ok, e_den / c_den, 0)
            step_band(axr, edges, 1 - 2 * band, 1 + 2 * band, color=color, alpha=0.1, linewidth=0)
            step_band(axr, edges, 1 - band, 1 + band, color=color, alpha=0.3, hatch='////', linewidth=0)
            h2s = (Patch(facecolor=color, alpha=0.1, linewidth=0), r'$2\sigma$')
            h1s = (Patch(facecolor=color, alpha=0.3, hatch='////', linewidth=0), r'$1\sigma$')
            den_lbl = panel['lbl_line_short']
            if true is not None:
                c_t, e_t = true
                r_t = np.where(den_ok, c_t / c_den, np.nan)
                re_t = np.where(den_ok, e_t / c_den, 0)
                if true_points:
                    hep.histplot(r_t, bins=edges, yerr=re_t, ax=axr, histtype='errorbar', color='black', marker='.',
                                 markersize=6, elinewidth=1.2, linestyle='none')
                    h_other = (Line2D([], [], color='black', marker='.', markersize=8, linestyle='none'),
                               f'{LBL_TRUE} / {den_lbl}')
                else:
                    hep.histplot(r_t, bins=edges, yerr=re_t, ax=axr, histtype='step', color='black', linewidth=1.8)
                    h_other = (Line2D([], [], color='black', linewidth=1.8, marker='|', markersize=8),
                               f'{LBL_TRUE} / {den_lbl}')

        if vr in mark:
            c_m, e_m = mark[vr]
            hep.histplot(np.where(den_ok, c_m / c_den, np.nan), bins=edges, yerr=np.where(den_ok, e_m / c_den, 0),
                         ax=axr, histtype='errorbar', color=color, marker=VR_MARKERS.get(vr, 'o'), markersize=4.5,
                         elinewidth=1.2, linestyle='none')
            h_mark = (Line2D([], [], color=color, marker=VR_MARKERS.get(vr, 'o'), markersize=5, linestyle='none'),
                      f"{panel['lbl_mark_short']} / {den_lbl}")

        axr.set_ylabel(r'$X\,/\,$' + den_lbl + '\n' + vr_label(vr), fontsize=12)
        axr.set_ylim(0.5, 1.5)
        axr.set_yticks([0.75, 1.0, 1.25])
        axr.grid(True, alpha=0.3, axis='y')
        _ratio_legend(axr, [('upper left', [h2s, h_mark]), ('lower left', [h1s, h_other])], legend_fs)
        if k < len(vrs) - 1:
            axr.tick_params(labelbottom=False)
        else:
            axr.set_xlabel(feature_name)


def plot_final4b_feature(feature_name, edges, h4b, h2b, hsr4b, filename, legend_fs=14,
                         edges_init=None, h4b_init=None, h2b_init=None, hsr4b_init=None, switch=False, mode='model'):
    """
    Two-column figure:
      left  : initial distributions   -> VR 2b (lines), VR 4b (markers), true SR 4b
      right : transported distributions -> SR~4b (lines), SR^4b (markers), true SR 4b
    switch=True swaps the roles of 2b and 4b (4b as lines and ratio denominator).
    mode: 'model' ratio w.r.t. the lines | 'true' ratio w.r.t. the true SR 4b | 'ffn' like 'model' without markers.
    If the initial distributions are not available only the right column is drawn.
    """
    assert mode in ('model', 'true', 'ffn'), mode
    set_style()
    transported = {'2b': (h2b, LBL_TILDE), '4b': (h4b, LBL_HAT)}
    initial = {'2b': h2b_init or {}, '4b': h4b_init or {}}
    k_line, k_mark = ('4b', '2b') if switch else ('2b', '4b')
    no_mark = (mode == 'ffn')

    right = dict(edges=edges, line=transported[k_line][0], mark={} if no_mark else transported[k_mark][0], true=hsr4b,
                 lbl_line=lambda vr, l=transported[k_line][1]: f'{l} ({vr_label(vr)})',
                 lbl_mark=lambda vr, l=transported[k_mark][1]: f'{l} ({vr_label(vr)})',
                 lbl_line_short=transported[k_line][1], lbl_mark_short=transported[k_mark][1])
    panels = [right]
    if edges_init is not None and (initial[k_line] or (initial[k_mark] and not no_mark)):
        left = dict(edges=edges_init, line=initial[k_line], mark={} if no_mark else initial[k_mark], true=hsr4b_init,
                    lbl_line=lambda vr, l=k_line: f'{vr_label(vr)} {l}',
                    lbl_mark=lambda vr, l=k_mark: f'{vr_label(vr)} {l}',
                    lbl_line_short=k_line, lbl_mark_short=k_mark)
        panels = [left, right]

    vrs = sorted({vr for p in panels for vr in set(p['line']) | set(p['mark'])})
    n, ratio_h, ncol = len(vrs), 0.5, len(panels)
    fig = plt.figure(figsize=(10 * ncol, 3 * (3 + ratio_h * n) + 1))
    gs = fig.add_gridspec(1 + n, ncol, height_ratios=[3] + [ratio_h] * n, hspace=0.12, wspace=0.18,
                          left=0.08 if ncol == 2 else 0.12, right=0.98, top=0.95, bottom=0.06)
    with np.errstate(divide='ignore', invalid='ignore'):
        for col, panel in enumerate(panels):
            _draw_final4b_column(fig, gs, col, panel, vrs, feature_name, legend_fs,
                                 mode='true' if mode == 'true' else 'model', true_points=(mode == 'ffn'))
    save_fig(fig, filename)


# =============================================================================
# FFN 2b -> 4b reweighting
# =============================================================================
def plot_weights(w, filename, xlabel='w = r(x)'):
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    w = w[w > 0]
    ax.hist(w, bins=np.logspace(np.log10(max(w.min(), 1e-4)), np.log10(w.max()), 60), color='steelblue')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Events')
    cms_label(ax)
    save_fig(fig, filename)


def plot_reweighting_closure(feature_name, edges, raw, reweighted, target, filename,
                             raw_label='2b', rw_label='2b reweighted', target_label='4b'):
    """Closure test of the reweighting: target (points) vs raw and reweighted source; ratios target / source."""
    set_style()
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    hep.histplot(raw[0], bins=edges, ax=ax, histtype='step', color='blue', linewidth=2, label=raw_label)
    hep.histplot(reweighted[0], bins=edges, ax=ax, histtype='fill', color='green', alpha=0.3)
    hep.histplot(reweighted[0], bins=edges, ax=ax, histtype='step', color='green', linewidth=2, label=rw_label)
    hep.histplot(target[0], bins=edges, yerr=target[1], ax=ax, histtype='errorbar', color='black',
                 label=target_label, markersize=5)
    ax.set_ylabel('Density')
    ax.legend(loc='upper right', fontsize=16)
    cms_label(ax)

    for (c, e), color in ((raw, 'blue'), (reweighted, 'green')):
        r, r_err, _ = ratio_to_model(target[0], target[1], c, e)
        hep.histplot(r, bins=edges, yerr=r_err, ax=axr, histtype='errorbar', color=color, markersize=4)
    _, _, band = ratio_to_model(target[0], target[1], *reweighted)
    sigma_bands(axr, edges, band, 'green')
    _style_ratio_axis(axr, f'{target_label} / X', xlabel=feature_name)
    fig.tight_layout()
    save_fig(fig, filename)
