"""
Utilities of the NF package: output paths, h5 I/O of the evaluation results, data loading per group.

Importing this module also puts the repository root on sys.path, so that `lib` can be imported
when the scripts are run as `python NF/<script>.py`.
"""
import os
import sys

import hashlib

import h5py
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib import data_loader as dl                         # noqa: E402
from lib.features import MIN_DR                           # noqa: E402


# =============================================================================
# Groups and paths
# =============================================================================
BASE_PATH = '/eos/user/a/amorandi/HH4b/NF_model'
FFN_BASE_PATH = '/eos/user/a/amorandi/HH4b/Backup_def/FFN_2bvs4b'

# 'All': 4-jet model on all the events; '5jet': model with the 5th jet, on the events with njet >= 5
GROUP_4J, GROUP_5J = 'All', '5jet'
JET5_NJET_MIN = 5


def group_name(jet5):
    return GROUP_5J if jet5 else GROUP_4J


def njet_min(jet5):
    """Selection on the number of jets of a group (None = no selection)."""
    return JET5_NJET_MIN if jet5 else None


def region_dir(base_path, group, vr, nbjet, sub=None):
    """
    <base_path>/<group>/VR<vr>_vs_SR_result[/sub]   for the 2b case (training region)
    <base_path>/<group>/VR<vr>_4b[/sub]             for the 4b case
    The directory is created if it does not exist.
    """
    repo = f"VR{vr}_vs_SR_result" if nbjet == 2 else f"VR{vr}_4b"
    path = os.path.join(base_path, group, repo, *([sub] if sub else []))
    os.makedirs(path, exist_ok=True)
    return path


class Layout:
    """
    Output layout of a model with one flow per validation region (NF/main.py, NF/main_2b4b.py):
        region  : <base_path>/<group>/VR<vr>_vs_SR_result (2b) or VR<vr>_4b (4b)   [see region_dir]
        weights : weights/ of the 2b region (flow, scaler, SWAG state)
    All the directories are created on request.
    """
    def __init__(self, base_path):
        self.base_path = base_path

    def region(self, group, vr, nbjet, sub=None):
        return region_dir(self.base_path, group, vr, nbjet, sub)

    def weights(self, group, vr):
        return self.region(group, vr, 2, 'weights')


class AllVRsLayout(Layout):
    """
    Output layout of the model trained on all the validation regions together (NF/main_allVRs.py):
        region  : <base_path>/<group>/<nbjet>b/VR<vr>
        weights : <base_path>/<group>/weights (one flow, scaler and SWAG state for all the VRs)
    """
    def region(self, group, vr, nbjet, sub=None):
        path = os.path.join(self.base_path, group, f'{nbjet}b', f'VR{vr}', *([sub] if sub else []))
        os.makedirs(path, exist_ok=True)
        return path

    def diagnostic(self, group, nbjet):
        """<base_path>/<group>/<nbjet>b/training_diagnostic: plots shared by all the VRs (one training)."""
        path = os.path.join(self.base_path, group, f'{nbjet}b', 'training_diagnostic')
        os.makedirs(path, exist_ok=True)
        return path

    def weights(self, group, vr=None):
        path = os.path.join(self.base_path, group, 'weights')
        os.makedirs(path, exist_ok=True)
        return path


# =============================================================================
# h5 I/O
# =============================================================================
# Version of the files written by this code: older files (previous code) are recomputed, not reused
FORMAT_VERSION = 2


def save_h5(path, data):
    """Save a dict of arrays (strings / lists of strings are stored as bytes), tagged with FORMAT_VERSION."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.attrs['format_version'] = FORMAT_VERSION
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, str):
                value = np.bytes_(value)
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
                value = np.array(value, dtype='S')
            f.create_dataset(key, data=np.asarray(value))


def load_h5(path, keys=None):
    with h5py.File(path, 'r') as f:
        keys = keys or list(f.keys())
        return {k: f[k][()] for k in keys if k in f}


def is_current(path):
    """True if the file exists and was written by this version of the code."""
    if not os.path.exists(path):
        return False
    with h5py.File(path, 'r') as f:
        return f.attrs.get('format_version', 0) == FORMAT_VERSION


def checksum(X):
    """Fingerprint of an event matrix, used to check that two files refer to the same events."""
    return hashlib.sha1(np.ascontiguousarray(X, dtype=np.float32).tobytes()).hexdigest()


RECO_FILE = 'reconstructed_features_def.h5'
RECO_VAR_FILE = 'reconstructed_features_variance.h5'


def save_reco(eval_dir, features, X_reco, V_reco=None):
    """
    Save the reconstructed high-level features (and their variance) of one region.
    The original SR / VR events are not duplicated here: they are read again from the ROOT files when needed.
    """
    save_h5(os.path.join(eval_dir, RECO_FILE), {'features': features, 'X_reco_sig': X_reco})
    if V_reco is not None:
        save_h5(os.path.join(eval_dir, RECO_VAR_FILE), {'features': features, 'X_reco_sig_var': V_reco})


def _select_columns(data, key, features):
    """Columns `features` of data[key] (matrix) using the stored names, or the per-feature datasets (old format)."""
    if key in data:
        X = data[key]
        if 'features' in data:
            names = [n.decode() if isinstance(n, bytes) else n for n in data['features']]
            return X[:, [names.index(f) for f in features]]
        if X.shape[1] == len(features) + 1 and MIN_DR not in features:
            # Old files without names that also contain minDeltaR_Higgjj (just before maxDeltaR_Higgjj)
            X = np.delete(X, features.index("maxDeltaR_Higgjj"), axis=1)
        if X.shape[1] != len(features):
            raise ValueError(f"{key}: {X.shape[1]} columns, {len(features)} features expected")
        return X
    if all(f in data for f in features):
        return np.column_stack([np.asarray(data[f]).reshape(-1) for f in features])
    return None


def load_reco(eval_dir, features, load_variance=False):
    """
    Load (X_reco, V_reco) of one region (V_reco is None if not requested or not available; both None if the
    region has not been reconstructed). Also reads the files written by the previous code.
    """
    f_def, f_var = os.path.join(eval_dir, RECO_FILE), os.path.join(eval_dir, RECO_VAR_FILE)
    if not os.path.exists(f_def):
        print(f"[WARN] Missing file: {f_def}")
        return None, None
    data = load_h5(f_def)
    X_reco = _select_columns(data, 'X_reco_sig', features)
    V_reco = None
    if load_variance:
        if os.path.exists(f_var):
            V_reco = _select_columns(load_h5(f_var), 'X_reco_sig_var', features)
        elif 'X_reco_sig_var' in data:
            V_reco = _select_columns(data, 'X_reco_sig_var', features)
    return X_reco, V_reco


# =============================================================================
# Data of a group
# =============================================================================
def load_vr_sr(vr, nbjet, features, jet5):
    """Original VR and SR events of the group (njet >= 5 for the 5-jet one): returns X_bkg (VR), X_sig (SR)."""
    return dl.load_vr_sr(vr, nbjet, features, njet_min(jet5))


def load_true_sr4b(features, jet5):
    """True 4b signal region data (2022EE + 2022preEE), target of the final evaluation."""
    return dl.load_matrix(dl.data_paths('SR', 4), features, njet_min(jet5))
