"""
Evaluation of the VR -> SR transport in the physical space.

- reconstruct_hlf_region : for one VR, back-transform the transported jets, reconstruct the high-level features
                           (with SWAG variances), optionally correct minDeltaR, compare Reco SR with SR and VR
- compare_all_vrs        : all VRs in the same plots (distributions, Q-Q, KL / KS summaries)
- run_corner             : corner plots SR / Reco SR / VR

Model uncertainty: in the 2b case it comes from the per-event SWAG variances; its relative size per bin is saved
(model_unc_profile_<tag>.h5 in the 2b eval_data folder) and applied to the 4b case, where SWAG is not run.
"""
import os

import h5py
import joblib
import numpy as np
from tqdm import tqdm

import features_reconstruction as fr
import utils
from lib import data_loader as dl
from lib import features as F
from lib import metrics
from lib import visualization as vis
from minDeltaR_error import correct_minDeltaR_feature, study_problematic_tail
from swag import swag_file
from test_model import TRANSPORT_FILE


# =============================================================================
# Model-uncertainty profile (2b -> 4b)
# =============================================================================
def save_profile(unc_file, feature, profile):
    with h5py.File(unc_file, 'a') as f:
        if feature in f:
            del f[feature]
        g = f.create_group(feature)
        g.create_dataset('centers', data=profile[0])
        g.create_dataset('rel_unc', data=profile[1])


def load_profile(unc_file, feature):
    if unc_file is None or not os.path.exists(unc_file):
        return None
    with h5py.File(unc_file, 'r') as f:
        return (f[feature]['centers'][()], f[feature]['rel_unc'][()]) if feature in f else None


# =============================================================================
# One validation region
# =============================================================================
def _finite(*arrays):
    """Common finite mask of 1D arrays (None entries are ignored)."""
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        if a is not None:
            mask &= np.isfinite(a)
    return mask


def compare_distributions(X_bkg, X_sig, X_reco, features, nbjet, plot_dir, V_reco=None, unc_file=None, vr=None):
    """
    For each feature: SR vs VR and SR vs Reco SR (with ratio and model-uncertainty bands), Q-Q plot,
    chi2 / KL / KS between SR and Reco SR, and their summaries.

    nbjet == 2: Reco SR uncertainty from V_reco (per-event variances); the relative profile is saved in unc_file.
    nbjet == 4: Reco SR uncertainty from the profile stored in unc_file (statistical only if missing).
    """
    if nbjet == 4 and unc_file and not os.path.exists(unc_file):
        print(f"[WARN] {unc_file} not found (run the 2b case with SWAG first): only statistical uncertainties")
    bkg_label = f'VR{vr}' if vr else 'VR'
    results = {'chi2': {}, 'kl': {}, 'ks': {}}

    for i, ft in enumerate(tqdm(features, desc="Feature comparison")):
        s, b = X_sig[:, i], X_bkg[:, i]
        s, b = s[np.isfinite(s)], b[np.isfinite(b)]
        v = V_reco[:, i] if V_reco is not None else None
        ok = _finite(X_reco[:, i], v)
        r, v = X_reco[ok, i], (v[ok] if v is not None else None)
        if len(s) == 0 or len(r) == 0:
            continue
        edges = vis.percentile_edges(s, vis.feature_bins(ft, nbjet))

        if nbjet == 2:
            reco = metrics.hist_density(r, edges, var=v)
            profile = metrics.model_unc_profile(r, v, edges) if v is not None else None
            if unc_file and profile is not None:
                save_profile(unc_file, ft, profile)
        else:
            reco = metrics.hist_density(r, edges, rel_profile=load_profile(unc_file, ft))
        sig, bkg = metrics.hist_density(s, edges), metrics.hist_density(b, edges)
        if reco[0] is None or sig[0] is None or bkg[0] is None:
            continue

        vis.plot_bkg_sig_reco(ft, edges, sig, bkg, reco, os.path.join(plot_dir, 'distribution', f'{ft}_comparison.png'),
                              bkg_label=bkg_label)
        vis.plot_qq(s, {'Reco SR': (r, 'blue')}, os.path.join(plot_dir, 'scatter', f'{ft}_qq.png'),
                    xlabel=f'{ft} (SR)', ylabel=f'{ft} (Reco SR)')
        results['chi2'][ft] = metrics.chi2_ndf(sig[0], sig[1], reco[0])
        results['kl'][ft] = metrics.kl_div(sig[0], reco[0], edges[1] - edges[0])
        results['ks'][ft] = metrics.weighted_ks(s, r)

    for key, ylabel, color, limit in (('chi2', r'$\chi^2$ / ndf', 'purple', vis.CHI2_LIMIT),
                                      ('kl', 'KL divergence', 'green', vis.KL_LIMIT),
                                      ('ks', 'KS statistic', 'blue', vis.KS_LIMIT)):
        if results[key]:
            vis.plot_summary_bar(results[key], os.path.join(plot_dir, f'{key}_summary.png'), ylabel, color, limit)
    return results


def reconstruct_hlf_region(base_path, vr, nbjet, jet5, include_min_dr=False, use_variance=True, swag_lr=1e-5,
                           plot_latent=True, plot_hlf=True, plot_variances=False, tail_study=False, mdr_kwargs=None,
                           layout=None):
    """
    Physical-space evaluation of one region, starting from eval_data/transform_bkg_to_sig.h5 (scaled space).

    1. inverse preprocessing (2b scaler) of the transported jets and of the SWAG variances (2b only)
    2. comparison of the jet features (plots/latent_space_comparison)
    3. reconstruction of the high-level features + variance propagation
    4. optional flow-matching correction of minDeltaR_Higgjj (fitted in 2b, applied in 4b)
    5. comparison of the high-level features (plots/hlf_comparison), results saved in
       eval_data/reconstructed_features_def.h5 (+ _variance.h5)

    layout: utils.Layout of the output directories (default: one flow per VR, utils.Layout(base_path))
    """
    group = utils.group_name(jet5)
    in_feats, hlf = F.input_features(jet5), F.hlf_features(jet5, include_min_dr)
    layout = layout or utils.Layout(base_path)
    eval_dir = layout.region(group, vr, nbjet, 'eval_data')
    plot_dir = layout.region(group, vr, nbjet, 'plots')
    eval_2b = layout.region(group, vr, 2, 'eval_data')
    scaler = joblib.load(os.path.join(layout.weights(group, vr), 'scaler.pkl'))

    # --- 1. Transported jets in the physical space ---
    transport_path = os.path.join(eval_dir, TRANSPORT_FILE)
    if not utils.is_current(transport_path):
        raise RuntimeError(f"{transport_path} is missing or was written by the old code: run the 'transport' step")
    transport = utils.load_h5(transport_path, ['X_reco_sig', 'X_bkg'])
    X_reco_scaled = transport['X_reco_sig']
    V_scaled = None
    if nbjet == 2 and use_variance:
        swag_path = os.path.join(eval_2b, swag_file(swag_lr))
        swag = utils.load_h5(swag_path, ['X_reco_sig_var', 'bkg_checksum']) if os.path.exists(swag_path) else {}
        # The per-event variances must refer to exactly the transported events
        if swag.get('bkg_checksum', b'').decode() == utils.checksum(transport['X_bkg']):
            V_scaled = swag['X_reco_sig_var']
        else:
            print(f"[WARN] {swag_path} missing or computed on other events (run the 'swag' step): "
                  f"only statistical uncertainties")
        del swag
    del transport
    X_reco_in, V_in = dl.inverse_preprocessing(X_reco_scaled, in_feats, scaler, V_scaled)

    def unc_file(tag):
        return os.path.join(eval_2b, f'model_unc_profile_{tag}.h5') if (use_variance or nbjet == 4) else None

    # --- 2. Jet features ---
    if plot_latent:
        X_bkg_in, X_sig_in = utils.load_vr_sr(vr, nbjet, in_feats, jet5)
        compare_distributions(X_bkg_in, X_sig_in, X_reco_in, in_feats, nbjet,
                              os.path.join(plot_dir, 'latent_space_comparison'), V_in, unc_file('latent'), vr)

    # --- 3. High-level features ---
    feat = fr.reconstruct_hlf(X_reco_in, in_feats, include_min_dr)
    X_reco = fr.to_matrix(feat, hlf)
    V_reco = fr.to_matrix(fr.propagate_variance(X_reco_in, V_in, in_feats, include_min_dr), hlf) if V_in is not None else None
    X_bkg, X_sig = utils.load_vr_sr(vr, nbjet, hlf, jet5)

    # --- 4. minDeltaR correction ---
    if include_min_dr:
        mdr_dir = os.path.join(plot_dir, 'minDeltaR_analysis')
        if tail_study:
            study_problematic_tail(X_sig, X_reco, hlf, os.path.join(mdr_dir, 'tail_study'))
        weights_dir = os.path.join(layout.region(group, vr, 2, 'plots'), 'minDeltaR_analysis', 'weights')
        X_reco, V_reco = correct_minDeltaR_feature(X_reco, V_reco, X_sig, hlf, weights_dir, mdr_dir,
                                                   fit=(nbjet == 2), **(mdr_kwargs or {}))

    utils.save_reco(eval_dir, hlf, X_reco, V_reco)

    # --- 5. Comparison ---
    hlf_dir = os.path.join(plot_dir, 'hlf_comparison')
    if plot_hlf:
        compare_distributions(X_bkg, X_sig, X_reco, hlf, nbjet, hlf_dir, V_reco, unc_file('hlf'), vr)
    if plot_variances and V_reco is not None:
        for i, ft in enumerate(hlf):
            vis.plot_variance_hist(V_reco[:, i], ft, vis.feature_bins(ft, nbjet),
                                   os.path.join(hlf_dir, 'variances', f'{ft}_variance.png'))


# =============================================================================
# All the validation regions together
# =============================================================================
def _load_group(layout, group, features, nbjet, regions, load_variance):
    """
    {vr: dict(reco, sig, bkg, var)}: saved reconstructions + original VR events read from ROOT.
    The SR is the same for all the VRs: it is read once and stored only in the first entry.
    """
    njet_min = utils.njet_min(group == utils.GROUP_5J)
    out = {}
    for vr in regions:
        reco, var = utils.load_reco(layout.region(group, vr, nbjet, 'eval_data'), features, load_variance)
        if reco is None:
            continue
        sig = None if out else dl.load_matrix(dl.sr_path(nbjet), features, njet_min)
        bkg = dl.load_matrix(dl.vr_path(nbjet, vr), features, njet_min)
        out[vr] = dict(reco=reco, sig=sig, bkg=bkg, var=var)
    return out


def _profiles_2b(layout, group, features, regions):
    """{vr: {feature: profile}} of the relative model uncertainty from the 2b reconstruction."""
    profiles = {}
    for vr in regions:
        reco, var = utils.load_reco(layout.region(group, vr, 2, 'eval_data'), features, True)
        if reco is None or var is None:
            print(f"[WARN] No 2b variance for {group} VR{vr}: 4b with statistical uncertainty only")
            continue
        profiles[vr] = {}
        for i, ft in enumerate(features):
            ok = _finite(reco[:, i], var[:, i])
            if ok.any():
                edges = vis.percentile_edges(reco[ok, i], vis.feature_bins(ft, 2))
                profiles[vr][ft] = metrics.model_unc_profile(reco[ok, i], var[ok, i], edges)
    return profiles


def compare_all_vrs(base_path, groups, features, nbjets=(2, 4), regions=range(1, 6), exist_variance=True,
                    out_name='Total', legend_fs=14, layout=None):
    """
    All VRs in the same plots, output in <base_path>/<out_name>/<nb>b/.

    param groups: {group: features of the group}; each feature is taken from the first group that contains it
                  (e.g. {'All': ..., '5jet': ...} takes the shared features from All and the 5th-jet ones from 5jet)
    param features: features to plot
    param exist_variance: 2b -> per-event SWAG variance, 4b -> 2b relative profile; False -> statistical only
    param layout: utils.Layout of the per-region results (default utils.Layout(base_path))
    """
    layout = layout or utils.Layout(base_path)
    regions = list(regions)
    for nbjet in nbjets:
        out_dir = os.path.join(base_path, out_name, f'{nbjet}b')
        print(f"\n=== All-VR comparison {nbjet}b ===")
        data = {g: _load_group(layout, g, gf, nbjet, regions, exist_variance and nbjet == 2)
                for g, gf in groups.items()}
        profiles = {g: _profiles_2b(layout, g, gf, regions) for g, gf in groups.items()} \
            if exist_variance and nbjet == 4 else {}
        kl, ks = {vr: {} for vr in regions}, {vr: {} for vr in regions}

        for ft in tqdm(features, desc=f"All-VR comparison {nbjet}b"):
            g = next((g for g, gf in groups.items() if ft in gf and data[g]), None)
            if g is None:
                print(f"[WARN] {ft}: no data, skipped")
                continue
            idx, gdata = groups[g].index(ft), data[g]
            s = gdata[min(gdata)]['sig'][:, idx]
            s = s[np.isfinite(s)]
            edges = vis.percentile_edges(s, max(vis.feature_bins(ft, nbjet) // 2, 10))
            sig = metrics.hist_density(s, edges)

            bkg_h, reco_h, reco_raw = {}, {}, {}
            for vr, d in sorted(gdata.items()):
                b = d['bkg'][:, idx]
                bkg_h[vr] = metrics.hist_density(b[np.isfinite(b)], edges)
                v = d['var'][:, idx] if d['var'] is not None else None
                ok = _finite(d['reco'][:, idx], v)
                r = d['reco'][ok, idx]
                prof = profiles.get(g, {}).get(vr, {}).get(ft)
                reco = metrics.hist_density(r, edges, var=v[ok] if v is not None else None, rel_profile=prof)
                if reco[0] is None:
                    continue
                reco_h[vr], reco_raw[vr] = reco, r
                kl[vr][ft] = metrics.kl_div(sig[0], reco[0], edges[1] - edges[0])
                ks[vr][ft] = metrics.weighted_ks(s, r)
            if not reco_h:
                continue
            vis.plot_multi_vr_distribution(ft, edges, sig, {k: h for k, h in bkg_h.items() if h[0] is not None},
                                           reco_h, os.path.join(out_dir, 'distribution', f'{ft}_comparison.png'),
                                           legend_fs=legend_fs)
            vis.plot_qq(s, {f'VR{vr}': (r, vis.VR_COLORS[vr]) for vr, r in reco_raw.items()},
                        os.path.join(out_dir, 'scatter', f'{ft}_qq.png'), xlabel=f'{ft} (SR)', ylabel=f'{ft} (Reco SR)')

        kl, ks = {vr: v for vr, v in kl.items() if v}, {vr: v for vr, v in ks.items() if v}
        if kl:
            vis.plot_metric_summary(kl, features, os.path.join(out_dir, 'kl_summary.png'), 'KL Divergence', vis.KL_LIMIT)
            vis.plot_metric_summary(ks, features, os.path.join(out_dir, 'ks_summary.png'), 'KS Statistic', vis.KS_LIMIT)
    print(f"All-VR comparison saved in {os.path.join(base_path, out_name)}")


def run_corner(base_path, group, features, nbjets=(2, 4), regions=range(1, 6), max_events=100000, per_vr=True,
               layout=None):
    """Corner plots (SR, Reco SR, VR) for each VR and with all the VRs; output in <base_path>/corner/<group>/<nb>b."""
    for nbjet in nbjets:
        out_dir = os.path.join(base_path, 'corner', group, f'{nbjet}b')
        loaded = _load_group(layout or utils.Layout(base_path), group, features, nbjet, regions, load_variance=False)
        if not loaded:
            print(f"[WARN] corner {group} {nbjet}b: no data")
            continue
        sig = loaded[min(loaded)]['sig']
        if per_vr:
            for vr, d in loaded.items():
                vis.corner_plot({'SR': sig, 'Reco SR': d['reco'], f'VR{vr}': d['bkg']}, features,
                                os.path.join(out_dir, f'corner_VR{vr}.png'), colors=['black', '#d62728', '#1f77b4'],
                                max_events=max_events)
        datasets = {'SR': sig, **{f'Reco SR (VR{vr})': d['reco'] for vr, d in loaded.items()}}
        vis.corner_plot(datasets, features, os.path.join(out_dir, 'corner_all_VRs.png'),
                        colors=['black'] + [vis.VR_COLORS[vr] for vr in loaded], max_events=max_events)


def run_global(base_path, include_min_dr=False, regions=range(1, 6), corner=True, corner_max_events=100000,
               layout=None):
    """
    All-VR comparison (<base_path>/Total/{2b,4b}) and corner plots of BOTH groups.

    The features shared by the two models are taken from the 4-jet group ('All'), the 5th-jet features from the
    '5jet' group: the plots in Total/ contain all the high-level features whose group has results on disk,
    independently of which pipeline (4-jet or 5-jet) runs this step. Run it when both groups are done.
    """
    groups = {utils.GROUP_4J: F.hlf_features(False, include_min_dr),
              utils.GROUP_5J: F.hlf_features(True, include_min_dr)}
    compare_all_vrs(base_path, groups, F.hlf_features(True, include_min_dr), regions=regions, layout=layout)
    if corner:
        for group, feats in groups.items():
            run_corner(base_path, group, feats, regions=regions, max_events=corner_max_events, layout=layout)
