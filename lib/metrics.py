"""
Histogram and distance metrics shared by the evaluation scripts.

All histograms are normalised to unit area (density) inside the binning range.
"""
import numpy as np


def hist_density(vals, edges, w=None, var=None, rel_profile=None):
    """
    Normalised histogram with its uncertainty.

    err^2 = sum(w^2)                       statistical part
          + sum(w^2 * var)                 per-event model variance (if var is given)
          + (rel_interp * counts)^2        relative model uncertainty profile (if rel_profile is given)

    param vals: array, values to histogram
    param edges: array, bin edges
    param w: array or None, event weights
    param var: array or None, per-event model variance
    param rel_profile: (centers, rel_unc) or None, relative model uncertainty interpolated on the bin centers
    return: (density, error) or (None, None) if the histogram is empty
    """
    w = np.ones(len(vals), dtype=np.float64) if w is None else np.asarray(w, dtype=np.float64)
    counts, _ = np.histogram(vals, bins=edges, weights=w)
    err2, _ = np.histogram(vals, bins=edges, weights=w ** 2)
    if var is not None:
        sum_var, _ = np.histogram(vals, bins=edges, weights=(w ** 2) * var)
        err2 = err2 + sum_var
    if rel_profile is not None:
        centers = 0.5 * (edges[:-1] + edges[1:])
        rel = np.interp(centers, *rel_profile)          # outside the profile range: edge value
        err2 = err2 + (rel * counts) ** 2

    norm = counts.sum() * (edges[1] - edges[0])
    if norm <= 0:
        return None, None
    return counts / norm, np.sqrt(err2) / norm


def model_unc_profile(vals, var, edges):
    """
    Relative model uncertainty per bin, sqrt(sum of per-event variances) / counts.
    Computed in the 2b case and re-used (interpolated) for the 4b case.

    return: (centers, rel_unc) of the non-empty bins, or None if fewer than 2 bins are valid
    """
    counts, _ = np.histogram(vals, bins=edges)
    sum_var, _ = np.histogram(vals, bins=edges, weights=var)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(counts > 0, np.sqrt(sum_var) / counts, np.nan)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ok = np.isfinite(rel)
    return (centers[ok], rel[ok]) if ok.sum() >= 2 else None


def kl_div(c_p, c_q, bin_width, eps=1e-10):
    """KL(P || Q) between two density histograms with the same binning (empty bins are skipped)."""
    P, Q = c_p * bin_width, c_q * bin_width
    ok = (P > eps) & (Q > eps)
    if ok.sum() == 0:
        return np.nan
    P, Q = P[ok] / P[ok].sum(), Q[ok] / Q[ok].sum()
    return float(np.sum(P * np.log(P / Q)))


def weighted_ks(x1, x2, w1=None, w2=None):
    """Two-sample Kolmogorov-Smirnov statistic for (optionally) weighted samples."""
    w1 = np.ones(len(x1)) if w1 is None else w1
    w2 = np.ones(len(x2)) if w2 is None else w2
    allx = np.sort(np.concatenate([x1, x2]))
    o1, o2 = np.argsort(x1), np.argsort(x2)
    c1 = np.concatenate([[0], np.cumsum(w1[o1])]) / w1.sum()
    c2 = np.concatenate([[0], np.cumsum(w2[o2])]) / w2.sum()
    f1 = c1[np.searchsorted(x1[o1], allx, side='right')]
    f2 = c2[np.searchsorted(x2[o2], allx, side='right')]
    return float(np.max(np.abs(f1 - f2)))


def chi2_ndf(c_ref, e_ref, c_test):
    """Chi2 / ndf between a reference histogram (with errors) and a test histogram."""
    ok = e_ref > 0
    if ok.sum() == 0:
        return np.nan
    return float(np.sum((c_ref[ok] - c_test[ok]) ** 2 / e_ref[ok] ** 2) / ok.sum())
