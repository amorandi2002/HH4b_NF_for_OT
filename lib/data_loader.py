"""
Data access and preprocessing.

- ROOT files     : paths of the 2022 data (CR / SR, 2b / 4b) and of the VR / SR trees used to train the flow
- loading        : branches -> numpy matrices, optional selection njet >= njet_min (5-jet model)
- preprocessing  : x -> log(1 + x) for the positive long-tailed features, RobustScaler, clipping to [-5, 5]
- datasets       : background (label 0) + signal (label 1), resampled to the requested sizes, split into
                   train / validation / test and wrapped in torch DataLoaders
"""
import os

import numpy as np
import torch
import uproot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from .features import log_features
except ImportError:
    from features import log_features


# =============================================================================
# ROOT files
# =============================================================================
DATA_DIR = "/eos/cms/store/group/phys_smp/rgerosa/HH4b/analysis_2022_ParkingHH"
VR_DIR = "/eos/cms/store/user/rgerosa/HH4b/JetTransformer/data_4b_VR/original_files"
ERAS = ("2022EE", "2022preEE")
_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}


def data_paths(region, nbjet):
    """2022 data of the control region (region='CR') or signal region ('SR'), 2b or 4b, both eras."""
    name = {"CR": "control", "SR": "signal"}[region]
    return [f"{DATA_DIR}/{nbjet}b_{name}_region_only/data_{era}.root" for era in ERAS]


def vr_path(nbjet, vr):
    """Validation region vr = 1..5 (2b reweighted to 4b, or 4b)."""
    suffix = {2: "2b_rwt_to_4b_2022", 4: "4b"}[nbjet]
    return f"{VR_DIR}/validation_region_sr_{_ROMAN[vr]}_{suffix}.root"


def sr_path(nbjet):
    """Signal region tree used as target of the VR -> SR transport."""
    return {2: f"{VR_DIR}/tree_2b_signal_weighted_to_4b_2022.root",
            4: f"{VR_DIR}/tree_4b_signal_region_2022.root"}[nbjet]


# =============================================================================
# Loading
# =============================================================================
def load_matrix(paths, features, njet_min=None):
    """
    Read `features` from one or more ROOT files (tree 'tree') into a (N, n_features) float64 matrix.
    njet_min: keep only the events with njet >= njet_min.
    """
    paths = [paths] if isinstance(paths, str) else paths
    branches = list(features) + (["njet"] if njet_min else [])
    blocks = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"ROOT file not found (or not readable): {path}")
        with uproot.open(path) as f:
            arr = f["tree"].arrays(branches, library="np")
        X = np.column_stack([arr[ft] for ft in features]).astype(np.float64)
        blocks.append(X[arr["njet"] >= njet_min] if njet_min else X)
    return np.vstack(blocks)


def load_vr_sr(vr, nbjet, features, njet_min=None):
    """Events of VR_vr (background) and of the SR (signal) for 2b or 4b: returns X_vr, X_sr."""
    return load_matrix(vr_path(nbjet, vr), features, njet_min), load_matrix(sr_path(nbjet), features, njet_min)


def load_2b_4b(region, features, njet_min=None):
    """2b (background) and 4b (signal) data of the CR or SR: returns X_2b, X_4b."""
    return (load_matrix(data_paths(region, 2), features, njet_min),
            load_matrix(data_paths(region, 4), features, njet_min))


# =============================================================================
# Preprocessing
# =============================================================================
LOG_FEATURES = set(log_features)


def log_reshape(X, features):
    """x -> log(1 + x) on the columns listed in LOG_FEATURES (returns a copy)."""
    X = np.array(X, dtype=np.float64)
    for i, ft in enumerate(features):
        if ft in LOG_FEATURES:
            X[:, i] = np.log1p(X[:, i])
    return X


def fit_scaler(X, features, log=True):
    """RobustScaler fitted on the (log-reshaped) raw features."""
    return RobustScaler().fit(log_reshape(X, features) if log else X)


def preprocess(X, features, scaler, log=True, clip=5.0):
    """Raw features -> network inputs: log reshaping, scaling, clipping to [-clip, clip] (clip=None: no clipping)."""
    X = scaler.transform(log_reshape(X, features) if log else np.asarray(X, dtype=np.float64))
    return np.clip(X, -clip, clip) if clip else X


def inverse_preprocessing(X, features, scaler, var=None):
    """
    Network outputs -> physical features (inverse of preprocess with log=True; the clipping cannot be undone).
    If `var` (variance in the scaled space) is given it is propagated too:
        RobustScaler  x = u * scale + center  ->  var_x = scale^2 var_u
        log reshaping x_phys = exp(x) - 1     ->  var_phys = exp(x)^2 var_x
    """
    X = scaler.inverse_transform(np.asarray(X, dtype=np.float64))
    if var is not None:
        var = np.asarray(var, dtype=np.float64) * scaler.scale_ ** 2
    for i, ft in enumerate(features):
        if ft in LOG_FEATURES:
            expx = np.exp(X[:, i])
            X[:, i] = expx - 1
            if var is not None:
                var[:, i] *= expx ** 2
    return X, var


# =============================================================================
# Datasets and DataLoaders
# =============================================================================
def resample(X, n, rng):
    """
    n events from X: all of them if n == len(X), a random subset (no repetitions) if n < len(X),
    all of them plus len(X) - n repeated events (bootstrap) if n > len(X).
    """
    if n <= len(X):
        return X if n == len(X) else X[rng.choice(len(X), n, replace=False)]
    return np.vstack([X, X[rng.choice(len(X), n - len(X), replace=True)]])


def build_dataset(X_bkg, X_sig, coef_bkg=1.0, coef_sig=None, seed=42):
    """
    Background (label 0) + signal (label 1) sample.
    coef_bkg / coef_sig: size of each class relative to its number of events;
    coef_sig=None -> as many signal events as background events (balanced classes).
    """
    rng = np.random.default_rng(seed)
    n_bkg = int(coef_bkg * len(X_bkg))
    n_sig = n_bkg if coef_sig is None else int(coef_sig * len(X_sig))
    X = np.vstack([resample(X_bkg, n_bkg, rng), resample(X_sig, n_sig, rng)])
    return X, np.concatenate([np.zeros(n_bkg), np.ones(n_sig)])


def split(X, y, validation_size=0.25, test_size=0.0, seed=42, strata=None):
    """
    Stratified split: {'train': (X, y), 'val': (X, y), 'test': (X, y)}; the empty splits are omitted.
    y can be a label vector or a condition matrix (N, k); in the latter case `strata` (one integer per event,
    e.g. the region index) defines the stratification (default: y itself).
    """
    strata = y if strata is None else np.asarray(strata)
    splits = {}
    if test_size > 0:
        X, X_te, y, y_te, strata, _ = train_test_split(X, y, strata, test_size=test_size, stratify=strata,
                                                       random_state=seed)
        splits['test'] = (X_te, y_te)
    if validation_size > 0:
        X, X_va, y, y_va, strata, _ = train_test_split(X, y, strata, test_size=validation_size, stratify=strata,
                                                       random_state=seed)
        splits['val'] = (X_va, y_va)
    splits['train'] = (X, y)
    return splits


def torch_loader(X, y, batch_size=512, shuffle=True, num_workers=4):
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)


def make_loaders(splits, features, scaler=None, log=True, clip=5.0, batch_size=512, num_workers=4):
    """
    Preprocess the splits and wrap them in DataLoaders (only the training one is shuffled).
    The scaler is fitted on the training split if not given. Returns ({split name: DataLoader}, scaler).
    """
    if scaler is None:
        scaler = fit_scaler(splits['train'][0], features, log)
    loaders = {name: torch_loader(preprocess(X, features, scaler, log, clip), y, batch_size, name == 'train', num_workers)
               for name, (X, y) in splits.items()}
    return loaders, scaler


def vr_sr_loaders(vr, nbjet, features, njet_min=None, scaler=None, coef_vr=1.0, coef_sr=None, validation_size=0.25,
                  test_size=0.0, seed=42, batch_size=512, num_workers=4):
    """
    DataLoaders VR_vr (label 0) vs SR (label 1) for the flow training.
    By default all the VR events and as many SR events (classes balanced). Returns ({split: DataLoader}, scaler).
    """
    X, y = build_dataset(*load_vr_sr(vr, nbjet, features, njet_min), coef_vr, coef_sr, seed)
    return make_loaders(split(X, y, validation_size, test_size, seed), features, scaler,
                        batch_size=batch_size, num_workers=num_workers)


# =============================================================================
# Preprocessing plots
# =============================================================================
if __name__ == "__main__":
    import argparse
    try:
        from . import visualization as vis
        from .features import features_5j
    except ImportError:
        import visualization as vis
        from features import features_5j

    p = argparse.ArgumentParser(description="Distributions of the features before and after the preprocessing.")
    p.add_argument('--out_dir', default='/eos/user/a/amorandi/HH4b/data_preprocessing')
    p.add_argument('--regions', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--nbjet', type=int, default=4)
    args = p.parse_args()

    for vr in args.regions:
        X_vr, X_sr = load_vr_sr(vr, args.nbjet, features_5j, njet_min=5)
        scaler = fit_scaler(np.vstack([X_vr, X_sr]), features_5j)
        raw = {f'VR{vr}': X_vr, 'SR': X_sr}
        vis.plot_preprocessing(raw, {k: preprocess(X, features_5j, scaler) for k, X in raw.items()},
                               features_5j, os.path.join(args.out_dir, f'VR{vr}_{args.nbjet}b'))
