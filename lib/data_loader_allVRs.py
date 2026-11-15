"""
Data for the model trained on all the validation regions at the same time (NF/main_allVRs.py).

The condition is a one-hot region label with N_REGIONS = 6 entries:
    VR1 ... VR5 -> e_1 ... e_5,    SR -> e_6
The training sample contains all the events of VR1..VR5 and as many SR events as the total number of VR events
(subset or bootstrap of the SR), so that SR and VR are balanced as in the one-flow-per-VR training.
"""
import numpy as np

try:
    from . import data_loader as dl
except ImportError:
    import data_loader as dl

VR_REGIONS = (1, 2, 3, 4, 5)
N_REGIONS = len(VR_REGIONS) + 1
SR_INDEX = N_REGIONS - 1


def region_condition(index):
    """One-hot condition of the region with index 0..4 (VR1..VR5) or SR_INDEX (SR)."""
    return np.eye(N_REGIONS, dtype=np.float32)[index]


def vr_condition(vr):
    """One-hot condition of the validation region vr = 1..5."""
    return region_condition(vr - 1)


SR_CONDITION = region_condition(SR_INDEX)


def allvrs_dataset(nbjet, features, njet_min=None, regions=VR_REGIONS, coef_vr=1.0, coef_sr=None, seed=42):
    """
    Events of all the VRs and of the SR.
    coef_vr: size of each VR relative to its number of events; coef_sr: size of the SR relative to its number of
    events (None -> as many SR events as the total number of VR events).
    Returns X (N, n_features), C (N, N_REGIONS) one-hot conditions, region index (N,) used for the stratification.
    """
    rng = np.random.default_rng(seed)
    blocks, index = [], []
    for vr in regions:
        X_vr = dl.load_matrix(dl.vr_path(nbjet, vr), features, njet_min)
        X_vr = dl.resample(X_vr, int(coef_vr * len(X_vr)), rng)
        blocks.append(X_vr)
        index.append(np.full(len(X_vr), vr - 1))
        print(f"  VR{vr} ({nbjet}b): {len(X_vr)} events")
    n_vr = sum(len(b) for b in blocks)
    X_sr = dl.load_matrix(dl.sr_path(nbjet), features, njet_min)
    n_sr = n_vr if coef_sr is None else int(coef_sr * len(X_sr))
    blocks.append(dl.resample(X_sr, n_sr, rng))
    index.append(np.full(n_sr, SR_INDEX))
    print(f"  SR ({nbjet}b): {len(X_sr)} events -> {n_sr} used")
    index = np.concatenate(index)
    return np.vstack(blocks), np.eye(N_REGIONS, dtype=np.float32)[index], index


def allvrs_loaders(nbjet, features, njet_min=None, scaler=None, regions=VR_REGIONS, coef_vr=1.0, coef_sr=None,
                   validation_size=0.25, test_size=0.0, seed=42, batch_size=512, num_workers=4):
    """DataLoaders (X, one-hot condition) of all the VRs + SR. Returns ({split: DataLoader}, scaler)."""
    X, C, index = allvrs_dataset(nbjet, features, njet_min, regions, coef_vr, coef_sr, seed)
    return dl.make_loaders(dl.split(X, C, validation_size, test_size, seed, strata=index), features, scaler,
                           batch_size=batch_size, num_workers=num_workers)
