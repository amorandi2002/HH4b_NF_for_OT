"""
General library of the HH4b analysis (independent of the model).

- data_loader   : ROOT paths and loading, preprocessing (log reshaping, RobustScaler, clipping), DataLoaders
- data_loader_allVRs / data_loader_2b4b : training samples of the model variants
- features      : feature lists (flow inputs, high-level features, log-reshaped features)
- metrics       : histograms with uncertainties and distance metrics (KL, KS, chi2)
- visualization : all plotting functions (CMS style)
"""
