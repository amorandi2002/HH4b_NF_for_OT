"""
Feature lists of the analysis (names of the branches of the ROOT trees).

- jet-level features : inputs of the normalizing flow (four-vectors of the Higgs jets + HT [+ 5th jet])
- high-level features: quantities reconstructed from the jets, used for the evaluation and by the FFN
The 5-jet lists are the 4-jet ones followed by the 5th-jet features (the order matters for the saved models).
"""

JET_COMPONENTS = ("pt", "eta", "phi", "mass")
HIGGS_JETS = ("higgs1_reco_jet1", "higgs1_reco_jet2", "higgs2_reco_jet1", "higgs2_reco_jet2")
JET5 = "add_jet1pt"
MIN_DR = "minDeltaR_Higgjj"


def _four_vector(prefix):
    return [f"{prefix}_{c}" for c in JET_COMPONENTS]


# --- Inputs of the flow ---
features_jet = [f for jet in HIGGS_JETS for f in _four_vector(jet)] + ["HT"]
features_5jet = features_jet + _four_vector(JET5)

# --- High-level features ---
features = [
    "HT",
    *_four_vector("higgs1_reco"),
    *_four_vector("higgs2_reco"),
    "hh_vec_mass", "hh_vec_pt", "hh_vec_eta", "hh_vec_phi",
    "hh_vec_DeltaR", "hh_vec_DeltaPhi", "hh_vec_DeltaEta",
    *_four_vector("higgs1_reco_jet1"), *_four_vector("higgs1_reco_jet2"),
    "higgs1_DeltaPhijj", "higgs1_DeltaEtajj", "higgs1_DeltaRjj",
    *_four_vector("higgs2_reco_jet1"), *_four_vector("higgs2_reco_jet2"),
    "higgs2_DeltaPhijj", "higgs2_DeltaEtajj", "higgs2_DeltaRjj",
    "maxDeltaR_Higgjj",                       # minDeltaR_Higgjj is added on request, see hlf_features()
    "higgs1_helicityCosTheta", "higgs2_helicityCosTheta",
    "hh_CosThetaStar_CS",
]
features_5j = features + _four_vector(JET5) + [
    "add_jet1pt_Higgs1_deta", "add_jet1pt_Higgs1_dphi", "add_jet1pt_Higgs1_m",
    "add_jet1pt_Higgs2_deta", "add_jet1pt_Higgs2_dphi", "add_jet1pt_Higgs2_m",
]

# --- Features transformed with x -> log(1 + x) before the scaling (positive, long-tailed) ---
log_features = [
    "HT",
    "sigma_over_higgs1_reco_mass", "higgs1_reco_pt",
    "sigma_over_higgs2_reco_mass", "higgs2_reco_pt",
    "hh_vec_mass", "hh_vec_pt",
    "higgs1_reco_jet1_pt", "higgs1_reco_jet1_mass", "higgs1_reco_jet2_pt", "higgs1_reco_jet2_mass",
    "higgs1_DeltaRjj",
    "higgs2_reco_jet1_pt", "higgs2_reco_jet1_mass", "higgs2_reco_jet2_pt", "higgs2_reco_jet2_mass",
    MIN_DR,
    "add_jet1pt_pt", "add_jet1pt_mass",
]


def input_features(jet5=False):
    """Jet-level inputs of the flow: 4 Higgs jets + HT (+ 5th jet)."""
    return list(features_5jet if jet5 else features_jet)


def phi_features(jet5=False):
    """Periodic inputs, modelled by the circular spline flow."""
    return [f"{jet}_phi" for jet in HIGGS_JETS] + ([f"{JET5}_phi"] if jet5 else [])


def hlf_features(jet5=False, include_min_dr=False):
    """High-level features (+ minDeltaR_Higgjj, just before maxDeltaR_Higgjj, if include_min_dr)."""
    feats = list(features_5j if jet5 else features)
    if include_min_dr:
        feats.insert(feats.index("maxDeltaR_Higgjj"), MIN_DR)
    return feats
