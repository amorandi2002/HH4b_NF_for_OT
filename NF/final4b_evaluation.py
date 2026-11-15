"""
Final evaluation of the 4b signal-region prediction.

Two estimates of the 4b SR are compared with the true 4b SR data, for every VR:
  SR~4b (tilde) : VR 2b transported to the SR by the flow, then reweighted 2b -> 4b with the FFN likelihood ratio
                  evaluated on the transported (SR) kinematics
  SR^4b (hat)   : VR 4b transported to the SR by the flow trained on 2b

The FFN of each group ('All': 4-jet features, all events; '5jet': 5-jet features, njet >= 5) is trained in the
control region (2b vs 4b) if requested or if its weights are missing. Each feature is taken from the 'All' group if
possible, from the '5jet' group otherwise (5th-jet features).

Output in <base_path>/Total/evaluation/:
  distribution/        2b as lines (ratio denominator), 4b as markers
  distribution_switch/ 4b as lines, 2b as markers
  distribution_true/   ratios with respect to the true SR 4b
  ffn_comparison/      2b only vs true SR 4b
  kl_summary.png, ks_summary.png, metrics.json

Example:
  python NF/final4b_evaluation.py --retrain_ffn
"""
import argparse
import json
import os

import numpy as np
from tqdm import tqdm

import utils
from ffn import LikelihoodRatioFFN, train_ffn
from lib import data_loader as dl
from lib import features as F
from lib import metrics
from lib import visualization as vis


def _hist(vals, edges, w=None, var=None):
    """Density histogram of the finite entries, (None, None) if empty."""
    ok = np.isfinite(vals)
    for a in (w, var):
        if a is not None:
            ok &= np.isfinite(a)
    return metrics.hist_density(vals[ok], edges, w=None if w is None else w[ok], var=None if var is None else var[ok])


def _load_initial(vr, nbjet, features, jet5):
    """VR distribution before the transport (ROOT file of the VR); None if not readable."""
    try:
        return dl.load_matrix(dl.vr_path(nbjet, vr), features, utils.njet_min(jet5))
    except Exception as e:      # missing branches / files: the left column is just not drawn
        print(f"[WARN] initial VR{vr} {nbjet}b not available: {e}")
        return None


def load_group(layout, group, features, jet5, ffn, regions, use_model_variance, plot_initial):
    """
    Data of one group: true SR 4b and, for each VR, transported 4b, transported 2b + variance + FFN weights.
    ffn = (model, scaler, ffn_features); the weights are evaluated on the transported 2b features.
    """
    model, scaler, ffn_features = ffn
    missing = [f for f in ffn_features if f not in features]
    if missing:
        raise ValueError(f"[{group}] FFN features not reconstructed: {missing}")
    cols = [features.index(f) for f in ffn_features]

    out = {'true': utils.load_true_sr4b(features, jet5), 'vr': {}}
    for vr in regions:
        X4 = utils.load_reco(layout.region(group, vr, 4, 'eval_data'), features)[0]
        X2, V2 = utils.load_reco(layout.region(group, vr, 2, 'eval_data'), features, use_model_variance)
        if X4 is None or X2 is None:
            print(f"[WARN] {group} VR{vr}: reconstruction missing, skipped")
            continue
        w2 = model.weights(X2[:, cols], scaler, n_target=len(X4))
        print(f"[INFO] {group} VR{vr}: N4b = {len(X4)}, N2b = {len(X2)}, "
              f"N_eff(2b) = {w2.sum() ** 2 / (w2 ** 2).sum():.0f}")
        entry = dict(x4=X4, x2=X2, v2=V2, w2=w2, x2_init=None, x4_init=None)
        if plot_initial:
            for nb in (2, 4):
                entry[f'x{nb}_init'] = _load_initial(vr, nb, features, jet5)
        out['vr'][vr] = entry
    return out


def _initial_histograms(gdata, idx, n_bins, v_true):
    """Left column: initial VR 2b / 4b distributions and true SR 4b, on a binning covering all of them."""
    init = {nb: {vr: d[f'x{nb}_init'][:, idx] for vr, d in gdata['vr'].items() if d[f'x{nb}_init'] is not None}
            for nb in (2, 4)}
    pooled = [v[np.isfinite(v)] for dv in init.values() for v in dv.values()]
    if not pooled:
        return None, {}, {}, None
    edges = vis.percentile_edges(np.concatenate(pooled + [v_true]), n_bins)
    hists = {nb: {vr: h for vr, v in dv.items() if (h := _hist(v, edges))[0] is not None} for nb, dv in init.items()}
    h_true = _hist(v_true, edges)
    return edges, hists[2], hists[4], h_true if h_true[0] is not None else None


def compare_weighted_vrs(base_path, groups, features, regions=range(1, 6), use_model_variance=True, plot_initial=True,
                         legend_fs=14, layout=None):
    """
    param groups: {group: dict(features=..., jet5=..., ffn=(model, scaler, ffn_features))}, in order of preference
    param features: features to plot
    param layout: utils.Layout of the per-region results (default: one flow per VR, utils.Layout(base_path))
    """
    layout = layout or utils.Layout(base_path)
    regions = list(regions)
    out_dir = os.path.join(base_path, 'Total', 'evaluation')

    def group_of(ft):
        return next((g for g, cfg in groups.items() if ft in cfg['features']), None)

    needed = sorted({group_of(ft) for ft in features} - {None})
    print(f"[INFO] groups used: { {g: sum(group_of(f) == g for f in features) for g in needed} }")
    data = {g: load_group(layout, g, groups[g]['features'], groups[g]['jet5'], groups[g]['ffn'], regions,
                          use_model_variance, plot_initial) for g in needed}

    kl = {k: {vr: {} for vr in regions} for k in ('tilde', 'hat')}
    ks = {k: {vr: {} for vr in regions} for k in ('tilde', 'hat')}
    for ft in tqdm(features, desc='Comparison with the true SR 4b'):
        g = group_of(ft)
        if g is None or not data[g]['vr']:
            continue
        gdata, idx = data[g], groups[g]['features'].index(ft)

        pooled = np.concatenate([d['x4'][:, idx] for d in gdata['vr'].values()])
        n_bins = max(vis.feature_bins(ft, 2) // 4, 10)
        edges = vis.percentile_edges(pooled[np.isfinite(pooled)], n_bins)
        bw = edges[1] - edges[0]
        v_true = gdata['true'][:, idx]
        v_true = v_true[np.isfinite(v_true)]
        h_true = _hist(v_true, edges)
        h_true = h_true if h_true[0] is not None else None

        h4b, h2b = {}, {}
        for vr, d in sorted(gdata['vr'].items()):
            v2 = d['v2'][:, idx] if d['v2'] is not None else None
            h4, h2 = _hist(d['x4'][:, idx], edges), _hist(d['x2'][:, idx], edges, w=d['w2'], var=v2)
            if h4[0] is None or h2[0] is None:
                continue
            h4b[vr], h2b[vr] = h4, h2
            if h_true is not None:
                x4, x2, w2 = d['x4'][:, idx], d['x2'][:, idx], d['w2']
                ok4, ok2 = np.isfinite(x4), np.isfinite(x2) & np.isfinite(w2)
                kl['tilde'][vr][ft] = metrics.kl_div(h_true[0], h2[0], bw)
                kl['hat'][vr][ft] = metrics.kl_div(h_true[0], h4[0], bw)
                ks['tilde'][vr][ft] = metrics.weighted_ks(v_true, x2[ok2], w2=w2[ok2])
                ks['hat'][vr][ft] = metrics.weighted_ks(v_true, x4[ok4])
        if not h4b:
            continue

        edges_l, h2b_init, h4b_init, h_true_l = (_initial_histograms(gdata, idx, n_bins, v_true) if plot_initial
                                                 else (None, {}, {}, None))
        kw = dict(legend_fs=legend_fs, edges_init=edges_l, h4b_init=h4b_init, h2b_init=h2b_init, hsr4b_init=h_true_l)
        for sub, extra in (('distribution', {}), ('distribution_switch', {'switch': True}),
                           ('distribution_true', {'mode': 'true'}), ('ffn_comparison', {'mode': 'ffn'})):
            vis.plot_final4b_feature(ft, edges, h4b, h2b, h_true, os.path.join(out_dir, sub, f'{ft}_comparison.png'),
                                     **kw, **extra)

    used = {name: {k: {vr: v for vr, v in res[k].items() if v} for k in res} for name, res in (('KL', kl), ('KS', ks))}
    if used['KL']['tilde'] or used['KL']['hat']:
        for name, limit in (('KL', vis.KL_LIMIT), ('KS', vis.KS_LIMIT)):
            vis.plot_metric_summary(used[name]['tilde'], features, os.path.join(out_dir, f'{name.lower()}_summary.png'),
                                    f'{name} (vs {vis.LBL_TRUE})', limit, markers=used[name]['hat'],
                                    bar_label=lambda vr: f'{vis.LBL_TILDE} ({vis.vr_label(vr)})',
                                    marker_label=lambda vr: f'{vis.LBL_HAT} ({vis.vr_label(vr)})')
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump({name: {f'{k}_vs_trueSR4b': {f'VR{vr}': v for vr, v in res[k].items()} for k in res}
                       for name, res in used.items()}, f, indent=2)
    print(f"Results saved in {out_dir}")


def main(base_path=utils.BASE_PATH, ffn_base_path=utils.FFN_BASE_PATH, regions=(1, 2, 3, 4, 5), retrain_ffn=False,
         ffn_epochs=300, batch_size=1024, num_workers=8, seed=42, include_min_dr=False, use_model_variance=True,
         plot_initial=True, legend_fs=14, layout=None):
    """
    Train (or load) the FFN of each group, then compare the weighted predictions with the true SR 4b.
    layout: utils.Layout of the NF results (default one flow per VR; utils.AllVRsLayout for NF/main_allVRs.py).
    """
    groups = {}
    for jet5 in (False, True):
        group = utils.group_name(jet5)
        ffn_features = F.hlf_features(jet5)          # FFN inputs: high-level features (without minDeltaR)
        ffn_dir = os.path.join(ffn_base_path, group)
        weights_dir = os.path.join(ffn_dir, 'CR', 'weights')
        if retrain_ffn or not os.path.exists(os.path.join(weights_dir, 'ffn_best.pt')):
            print(f"\n=== Training the FFN of group {group} ===")
            model, scaler = train_ffn(ffn_features, ffn_dir, njet_min=utils.njet_min(jet5), epochs=ffn_epochs,
                                      batch_size=batch_size, num_workers=num_workers, seed=seed)
        else:
            model, scaler = LikelihoodRatioFFN.load(weights_dir)
            print(f"FFN of group {group} loaded from {weights_dir}")
        groups[group] = dict(features=F.hlf_features(jet5, include_min_dr), jet5=jet5,
                             ffn=(model, scaler, ffn_features))

    compare_weighted_vrs(base_path, groups, F.hlf_features(True, include_min_dr), regions=regions,
                         use_model_variance=use_model_variance, plot_initial=plot_initial, legend_fs=legend_fs,
                         layout=layout)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base_path', default=utils.BASE_PATH, help='NF output directory (contains All/ and 5jet/)')
    p.add_argument('--ffn_base_path', default=utils.FFN_BASE_PATH, help='FFN directory (contains All/ and 5jet/)')
    p.add_argument('--regions', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--retrain_ffn', action='store_true', help='retrain the FFNs even if the weights exist')
    p.add_argument('--ffn_epochs', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--include_min_dr', action='store_true', help='the reconstructions include minDeltaR_Higgjj')
    p.add_argument('--no_model_variance', dest='use_model_variance', action='store_false',
                   help='statistical uncertainty only for SR~4b')
    p.add_argument('--no_plot_initial', dest='plot_initial', action='store_false',
                   help='do not draw the initial VR distributions (left column)')
    p.add_argument('--legend_fs', type=int, default=14)
    return p


if __name__ == "__main__":
    main(**vars(build_parser().parse_args()))
