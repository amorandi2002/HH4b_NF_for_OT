"""
Normalizing Flow pipeline with ONE flow trained on all the validation regions at the same time.

The flow learns P(x | c) with a one-hot region condition c (VR1..VR5 = e_1..e_5, SR = e_6), trained on the 2b
events of the five VRs and of the SR (as many SR events as VR events in total). The VR_i -> SR transport uses
c = e_i -> e_6; the 4b results use the same 2b-trained flow.
With --with_4b the flow is trained on 2b AND 4b events (4b VRs and SR bootstrapped x --coef_4b) with the
condition (one-hot region, 4b flag), see lib/data_loader_2b4b.py; default output NF_model_2b4b_allVRs.
The steps are the ones of NF/main.py:
  train     : train the flow once on all the 2b VRs + SR (always all five VRs, --regions only selects the evaluation)
  tests     : classification VR_i vs SR with the likelihood ratio log P(x|SR) - log P(x|VR_i), feature importance
  transport : VR_i -> SR transport in 2b and 4b, latent space checks
  swag      : SWAG uncertainty of the 2b transport (one SWAG state for all the VRs)
  hlf       : high-level features reconstruction and comparison with the SR
  global    : all-VR comparison (Total/, both groups) and corner plots      } run when both the 4-jet and
  final4b   : final 4b evaluation with the FFN reweight (Total/evaluation/)  } the 5-jet pipelines are done

Output layout:
  <base_path>/<group>/weights/                       flow, scaler, training history, SWAG state
  <base_path>/<group>/{2b,4b}/training_diagnostic/   plots shared by all the VRs (one training): loss, log|det J|,
                                                     learning rate (lin / log), feature importance of every VR
  <base_path>/<group>/{2b,4b}/VR<i>/{plots,eval_data}
  <base_path>/Total/{2b,4b,evaluation}/              all-VR comparison and final 4b evaluation

Example:
  python NF/main_allVRs.py --steps train tests transport
  python NF/main_allVRs.py --jet5
"""
import os

import joblib
import torch
import torch.multiprocessing

import utils
import main as nf_main
from evaluation import reconstruct_hlf_region
from lib import data_loader as dl
from lib import data_loader_2b4b as dl24
from lib import data_loader_allVRs as dla
from lib import features as F
from lib import visualization as vis
from model_allVRs import FlowModelAllVRs
from swag import run_flow_swag
from test_model import ModelTester

BASE_PATH = '/eos/user/a/amorandi/HH4b/NF_model_allVRs'
"""Base path for the output directories (contains All/ and 5jet/)."""
BASE_PATH_2B4B = '/eos/user/a/amorandi/HH4b/NF_model_2b4b_allVRs'
"""Default base path of the model trained on 2b and 4b events (--with_4b)."""

EPOCHS = 500
"""Number of epochs for the flow training."""


def train_loaders(jet5, features, scaler, validation_size, batch_size, num_workers, seed, with_4b=False,
                  coef_4b=dl24.BOOTSTRAP_4B):
    """
    Loaders of all the VRs + SR: 2b only with one-hot conditions, or (with_4b) 2b + 4b with (one-hot, 4b flag).
    Returns ({split: DataLoader}, scaler).
    """
    if with_4b:
        return dl24.allvrs_loaders(features, utils.njet_min(jet5), scaler, coef_4b=coef_4b,
                                   validation_size=validation_size, seed=seed, batch_size=batch_size,
                                   num_workers=num_workers)
    return dla.allvrs_loaders(2, features, utils.njet_min(jet5), scaler, validation_size=validation_size, seed=seed,
                              batch_size=batch_size, num_workers=num_workers)


def plot_training_diagnostic(layout, group, history, features, regions=dla.VR_REGIONS):
    """
    Plots shared by all the VRs in <group>/{2b,4b}/training_diagnostic: training curves (one training for all the
    VRs) and the feature importance of every VR in one figure (from the cached VR_i/eval_data/classification.h5).
    """
    for nbjet in (2, 4):
        out = layout.diagnostic(group, nbjet)
        vis.plot_flow_history(history, out)
        corr = {}
        for vr in regions:
            # path built without layout.region(), which would create the folders of VRs not evaluated yet
            path = os.path.join(layout.base_path, group, f'{nbjet}b', f'VR{vr}', 'eval_data', 'classification.h5')
            if utils.is_current(path):
                corr[vr] = utils.load_h5(path, ['correlations'])['correlations']
        if corr:
            vis.plot_feature_importance_vrs(features, corr, os.path.join(out, 'feature_importance.png'))
        print(f"Training diagnostic ({nbjet}b) saved in {out}")


def main(base_path=BASE_PATH, regions=nf_main.REGIONS, jet5=nf_main.JET5, include_min_dr=nf_main.INCLUDE_MIN_DR,
         steps=nf_main.STEPS, bins=nf_main.BINS, transforms=nf_main.TRANSFORMS, hidden_features=nf_main.HIDDEN_FEATURES,
         phi_bins=nf_main.PHI_BINS, phi_transforms=nf_main.PHI_TRANSFORMS, phi_hidden_features=nf_main.PHI_HIDDEN_FEATURES,
         epochs=EPOCHS, warmup_epochs=nf_main.WARMUP_EPOCHS, patience=nf_main.PATIENCE, lr=nf_main.LR_initial,
         batch_size=nf_main.BATCH_SIZE, num_workers=nf_main.NUM_WORKERS, validation_size=nf_main.VALIDATION_SIZE,
         seed=nf_main.SEED, retrain=nf_main.RETRAIN, resume=nf_main.RESUME, recompute_eval=nf_main.RECOMPUTE_EVAL,
         plot_latent=nf_main.PLOT_LATENT, plot_variances=nf_main.PLOT_VARIANCES, swag_epochs=nf_main.SWAG_EPOCHS,
         swag_lr=nf_main.SWAG_LR, swag_samples=nf_main.SWAG_SAMPLES, retrain_swag=nf_main.RETRAIN_SWAG,
         mdr_epochs=nf_main.MDR_EPOCHS, mdr_patience=nf_main.MDR_PATIENCE, mdr_swag_epochs=nf_main.MDR_SWAG_EPOCHS,
         mdr_swag_lr=nf_main.MDR_SWAG_LR, mdr_swag_samples=nf_main.MDR_SWAG_SAMPLES, mdr_steps=nf_main.MDR_STEPS,
         retrain_mdr=nf_main.RETRAIN_MDR, tail_study=nf_main.TAIL_STUDY, corner=nf_main.CORNER,
         corner_max_events=nf_main.CORNER_MAX_EVENTS, with_4b=False, coef_4b=dl24.BOOTSTRAP_4B):

    torch.multiprocessing.set_sharing_strategy('file_system')
    layout = utils.AllVRsLayout(base_path)
    group = utils.group_name(jet5)
    features, phi = F.input_features(jet5), F.phi_features(jet5)
    hlf = F.hlf_features(jet5, include_min_dr)
    print(f"All-VR model | group: {group} | {len(features)} input features | training on "
          f"{'2b + 4b (x' + str(coef_4b) + ')' if with_4b else '2b'} | steps: {list(steps)}")

    weights_dir = layout.weights(group)
    scaler_path = os.path.join(weights_dir, 'scaler.pkl')
    # dir_path/weights == layout.weights(group)
    model = FlowModelAllVRs(input_dim=len(features), dir_path=os.path.join(base_path, group), with_4b=with_4b, bins=bins,
                            transforms=transforms, hidden_features=hidden_features,
                            phi_indices=[features.index(p) for p in phi], phi_bins=phi_bins,
                            phi_transforms=phi_transforms, phi_hidden_features=phi_hidden_features, lr=lr,
                            patience=patience)
    model.print_model_summary()

    # --- Training (2b, all the VRs + SR) ---
    trained = os.path.exists(model.model_path()) and os.path.exists(scaler_path)
    if 'train' in steps and (retrain or resume or not trained):
        scaler = joblib.load(scaler_path) if (resume and os.path.exists(scaler_path)) else None
        loaders, scaler = train_loaders(jet5, features, scaler, validation_size, batch_size, num_workers, seed,
                                        with_4b, coef_4b)
        joblib.dump(scaler, scaler_path)
        if 'val' not in loaders:
            raise ValueError("The training needs a validation set: use validation_size > 0")
        print(f"Training events: {len(loaders['train'].dataset)}, validation events: {len(loaders['val'].dataset)}")
        model.train(loaders['train'], loaders['val'], epochs=epochs, warmup_epochs=warmup_epochs, resume=resume)
        del loaders
    elif not trained:
        raise FileNotFoundError(f"No trained flow in {weights_dir}: run with the 'train' step")
    else:
        model.load_model()
    scaler = joblib.load(scaler_path)
    history = model.load_history()
    plot_training_diagnostic(layout, group, history, features)

    swag_loader = None
    for k, vr in enumerate(regions):
        print(f"\n{'=' * 80}\nVALIDATION REGION {vr} ({group}, all-VR model)\n{'=' * 80}")

        # --- Tests and transport (2b and 4b with the 2b-trained flow) ---
        if 'tests' in steps or 'transport' in steps:
            for nbjet in (2, 4):
                c_bkg, c_sig = model.conditions(vr, nbjet)
                X_bkg, X_sig = utils.load_vr_sr(vr, nbjet, features, jet5)
                tester = ModelTester(model, dl.preprocess(X_bkg, features, scaler),
                                     dl.preprocess(X_sig, features, scaler), features,
                                     layout.region(group, vr, nbjet), phi_features=phi,
                                     bkg_label=f'VR{vr}', sig_label='SR', c_bkg=c_bkg, c_sig=c_sig,
                                     feature_importance=False)
                if 'tests' in steps:
                    tester.classification_analysis(force_recompute=recompute_eval)
                if 'transport' in steps:
                    tester.transport_analysis(force_recompute=recompute_eval)

        # --- SWAG (2b): the SWAG state is shared by all the VRs, it is (re)trained only for the first one ---
        if 'swag' in steps:
            if swag_loader is None:
                swag_loader = train_loaders(jet5, features, scaler, 0.0, batch_size, num_workers, seed,
                                            with_4b, coef_4b)[0]['train']
            c_bkg, c_sig = model.conditions(vr, 2)
            run_flow_swag(model, swag_loader, layout.region(group, vr, 2, 'eval_data'), swag_lr=swag_lr,
                          swag_epochs=swag_epochs, num_samples=swag_samples, retrain=(retrain_swag and k == 0),
                          c_from=c_bkg, c_to=c_sig)

        # --- High-level features (2b first: minDeltaR correction and uncertainty profiles) ---
        if 'hlf' in steps:
            mdr_kwargs = dict(retrain=retrain_mdr, epochs=mdr_epochs, batch_size=batch_size, patience=mdr_patience,
                              retrain_swag=retrain_mdr, swag_epochs=mdr_swag_epochs, swag_lr=mdr_swag_lr,
                              num_swag_samples=mdr_swag_samples, steps=mdr_steps, seed=seed)
            for nbjet in (2, 4):
                reconstruct_hlf_region(base_path, vr, nbjet, jet5, include_min_dr=include_min_dr, swag_lr=swag_lr,
                                       plot_latent=plot_latent, plot_variances=plot_variances, tail_study=tail_study,
                                       mdr_kwargs=mdr_kwargs, layout=layout)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Feature importance of all the VRs (classification.h5 written by the 'tests' step) ---
    if 'tests' in steps:
        plot_training_diagnostic(layout, group, history, features)

    # --- Steps combining the two groups: Total/ (all-VR comparison, corner) and Total/evaluation/ (final 4b) ---
    nf_main.run_final_steps(base_path, steps, regions, include_min_dr, corner, corner_max_events, num_workers, seed,
                            layout=layout)


def build_parser():
    p = nf_main.build_parser(description=__doc__, base_path=None, epochs=EPOCHS)
    p.add_argument('--with_4b', action='store_true',
                   help=f'train on 2b and 4b events (default base path {BASE_PATH_2B4B})')
    p.add_argument('--coef_4b', type=float, default=dl24.BOOTSTRAP_4B,
                   help='bootstrap coefficient of the 4b VR and SR events in the training (with --with_4b)')
    return p


if __name__ == "__main__":
    # 4-jet model:  python NF/main_allVRs.py            (2b training)     -> NF_model_allVRs
    # 5-jet model:  python NF/main_allVRs.py --jet5
    # 2b + 4b:      python NF/main_allVRs.py --with_4b  (2b + 4b training) -> NF_model_2b4b_allVRs
    args = vars(build_parser().parse_args())
    if args['base_path'] is None:
        args['base_path'] = BASE_PATH_2B4B if args['with_4b'] else BASE_PATH
    main(**args)
