"""
Reconstruction of the high-level features (Higgs candidates, HH system, angles, ...) from the jet four-vectors
transported by the flow, and first-order propagation of the per-event SWAG variances of the jet features.

Input matrices are in the physical space, with columns named as in lib.features.input_features(jet5).
"""
import numpy as np
import vector

JETS = {  # name -> prefix of the branches
    'h1j1': 'higgs1_reco_jet1', 'h1j2': 'higgs1_reco_jet2',
    'h2j1': 'higgs2_reco_jet1', 'h2j2': 'higgs2_reco_jet2',
    'j5': 'add_jet1pt',
}
COMPONENTS = ('pt', 'eta', 'phi', 'mass')
EPS = 1e-9


def _columns(X, features, prefix):
    """{'pt': ..., 'eta': ..., 'phi': ..., 'mass': ...} of one jet."""
    return {c: X[:, features.index(f'{prefix}_{c}')] for c in COMPONENTS}


def build_jets(X, features):
    """Jet four-vectors (vector.MomentumNumpy4D) present in the features: h1j1, h1j2, h2j1, h2j2 and j5 if available."""
    return {name: vector.array(_columns(X, features, prefix))
            for name, prefix in JETS.items() if f'{prefix}_pt' in features}


def _pairwise_dr(jets):
    """DeltaR of the 6 pairs of Higgs jets, shape (6, N)."""
    names = ['h1j1', 'h1j2', 'h2j1', 'h2j2']
    return np.stack([np.asarray(jets[a].deltaR(jets[b])) for i, a in enumerate(names) for b in names[i + 1:]])


def _cos_theta(particle, boost_to):
    """|cos(theta)| of `particle` in the rest frame of `boost_to`."""
    boosted = particle.boost(-boost_to.to_beta3())
    return np.abs(np.asarray(boosted.pz) / (np.asarray(boosted.p) + EPS))


def reconstruct_hlf(X, features, include_min_dr=False):
    """
    High-level features of each event.

    param X: (N, n_features) physical jet features
    param features: names of the columns of X
    param include_min_dr: also compute minDeltaR_Higgjj
    return: dict {feature name: array}
    """
    jets = build_jets(X, features)
    h1 = jets['h1j1'] + jets['h1j2']
    h2 = jets['h2j1'] + jets['h2j2']
    hh = h1 + h2
    feat = {'HT': X[:, features.index('HT')]}

    for name, prefix in JETS.items():
        if name in jets:
            for c in COMPONENTS:
                feat[f'{prefix}_{c}'] = np.asarray(getattr(jets[name], c))

    for i, (h, j1, j2) in enumerate(((h1, jets['h1j1'], jets['h1j2']), (h2, jets['h2j1'], jets['h2j2'])), start=1):
        for c in COMPONENTS:
            feat[f'higgs{i}_reco_{c}'] = np.asarray(getattr(h, c))
        feat[f'higgs{i}_DeltaRjj'] = np.asarray(j1.deltaR(j2))
        feat[f'higgs{i}_DeltaPhijj'] = np.asarray(j1.deltaphi(j2))
        feat[f'higgs{i}_DeltaEtajj'] = np.abs(np.asarray(j1.eta - j2.eta))

    for c in COMPONENTS:
        feat[f'hh_vec_{c}'] = np.asarray(getattr(hh, c))
    feat['hh_vec_DeltaR'] = np.asarray(h1.deltaR(h2))
    feat['hh_vec_DeltaPhi'] = np.asarray(h1.deltaphi(h2))
    feat['hh_vec_DeltaEta'] = np.abs(np.asarray(h1.eta - h2.eta))

    dr_all = _pairwise_dr(jets)
    feat['maxDeltaR_Higgjj'] = dr_all.max(axis=0)
    if include_min_dr:
        feat['minDeltaR_Higgjj'] = dr_all.min(axis=0)

    # Helicity angles: leading jet in the Higgs rest frame, H1 in the HH rest frame
    feat['higgs1_helicityCosTheta'] = _cos_theta(jets['h1j1'], h1)
    feat['higgs2_helicityCosTheta'] = _cos_theta(jets['h2j1'], h2)
    feat['hh_CosThetaStar_CS'] = _cos_theta(h1, hh)

    if 'j5' in jets:
        jet5 = jets['j5']
        # NOTE: (h2 + jet5) instead of (h1 + jet5) on purpose, to be consistent with a bug in the input data
        feat['add_jet1pt_Higgs1_m'] = (h2 + jet5).mass #(h1 + jet5).mass # ERROR JUSTIFIED BY A BUG IN DATA
        feat['add_jet1pt_Higgs2_m'] = (h2 + jet5).mass
        for i, h in ((1, h1), (2, h2)):
            feat[f'add_jet1pt_Higgs{i}_dphi'] = np.asarray(h.deltaphi(jet5))
            feat[f'add_jet1pt_Higgs{i}_deta'] = np.abs(np.asarray(h.eta - jet5.eta))
    return feat


# =============================================================================
# Variance propagation
# =============================================================================
def propagate_variance(X, V, features, include_min_dr=False, chunk_size=200000):
    """
    First-order propagation of the per-event variances of the jet features to the high-level features:
        Var(f) = sum_i (df/dx_i)^2 Var(x_i)
    (jet features treated as uncorrelated). The derivatives are computed with central finite differences of
    reconstruct_hlf itself, so every feature (helicity angles, Delta R, 5th-jet masses, ...) is treated exactly
    with the same definition used for its value.

    param X, V: (N, n_features) physical jet features and their variances
    param features: names of the columns
    return: dict {feature name: variance}
    """
    out = []
    for start in range(0, len(X), chunk_size):
        x, v = X[start:start + chunk_size], V[start:start + chunk_size]
        var = None
        for i in range(x.shape[1]):
            if not np.any(v[:, i] > 0):
                continue
            h = 1e-4 * (np.abs(x[:, i]) + 1)
            x_up, x_dn = x.copy(), x.copy()
            x_up[:, i] += h
            x_dn[:, i] -= h
            f_up = reconstruct_hlf(x_up, features, include_min_dr)
            f_dn = reconstruct_hlf(x_dn, features, include_min_dr)
            if var is None:
                var = {k: np.zeros(len(x)) for k in f_up}
            for k in f_up:
                diff = np.asarray(f_up[k], dtype=np.float64) - np.asarray(f_dn[k], dtype=np.float64)
                if 'phi' in k.lower():          # angles: no 2pi jumps across the boundary
                    diff = (diff + np.pi) % (2 * np.pi) - np.pi
                var[k] += (diff / (2 * h)) ** 2 * v[:, i]
        out.append(var)
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def to_matrix(feat_dict, features):
    """Stack the requested features of a dict in a (N, n_features) matrix."""
    return np.column_stack([np.asarray(feat_dict[f], dtype=np.float64) for f in features])
