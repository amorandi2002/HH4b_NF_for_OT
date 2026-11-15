"""
Full Normalizing Flow pipeline, 4-jet inputs (group 'All', all the events).
The same pipeline with the 5th jet (group '5jet', njet >= 5) runs with --jet5.

For each validation region VR_i:
  train   : train the flow on 2b VR_i (c=0) vs 2b SR (c=1), or load the existing weights; loss plots
  tests   : classification VR vs SR with the flow likelihood ratio, feature importance (2b and 4b)
  transport: VR -> SR transport in 2b and, with the 2b-trained flow, in 4b; latent space checks
  swag    : SWAG model uncertainty of the 2b transport
  hlf     : high-level features reconstruction and comparison with the SR (2b, then 4b)
After the loop (run them when BOTH the 4-jet and the 5-jet pipelines are done):
  global  : all-VR comparison plots (Total/, all the features of both groups) and corner plots of both groups
  final4b : final 4b evaluation (Total/evaluation/): flow + FFN 2b->4b reweight vs true SR 4b, see final4b_evaluation.py

Example:
  python NF/main.py --regions 1 2 3 --steps train transport swag hlf global
"""
import argparse
import os

import joblib
import torch
import torch.multiprocessing

import utils
import final4b_evaluation
from evaluation import reconstruct_hlf_region, run_global
from lib import data_loader as dl
from lib import features as F
from lib import visualization as vis
from model import FlowModel
from swag import run_flow_swag
from test_model import ModelTester

BASE_PATH = utils.BASE_PATH
"""Base path for the output directories (contains All/ and 5jet/)."""
REGIONS = (1, 2, 3, 4, 5)
"""Validation regions to run the pipeline on."""
JET5 = False
"""Whether to run the 5-jet pipeline (njet >= 5) instead of the 4-jet one."""
INCLUDE_MIN_DR = False
"""Whether to include the minimum delta-R feature."""
STEPS = ('train', 'tests', 'transport', 'swag', 'hlf', 'global')
"""Pipeline steps run by default (in order)."""
STEP_CHOICES = STEPS + ('final4b',)
"""All the available steps."""

# flow architecture
BINS = 20
"""Number of bins for the flow."""
TRANSFORMS = 4
"""Number of transforms for the flow."""
HIDDEN_FEATURES = (256, 256)
"""Hidden features for the flow."""
PHI_BINS = 20
"""Number of bins for the phi flow."""
PHI_TRANSFORMS = 4
"""Number of transforms for the phi flow."""
PHI_HIDDEN_FEATURES = (64, 64)
"""Hidden features for the phi flow."""

# training
EPOCHS = 100
"""Number of epochs for training the flow."""
WARMUP_EPOCHS = 10
"""Number of warmup epochs for training the flow."""
PATIENCE = 100
"""Early-stopping patience for training the flow."""
LR_initial = 1e-3
"""Initial learning rate for training the flow."""
BATCH_SIZE = 2048
"""Batch size for training the flow."""
NUM_WORKERS = 8
"""Number of workers for training the flow."""
VALIDATION_SIZE = 0.25
"""Size of the validation set."""   
SEED = 42
"""Random seed for reproducibility."""
RETRAIN = False
"""Whether to retrain the flow even if weights exist."""
RESUME = False
"""Whether to resume training from existing weights."""

# evaluation
RECOMPUTE_EVAL = False
"""Whether to recompute the evaluation metrics."""
PLOT_LATENT = True
"""Whether to plot the latent space comparison."""
PLOT_VARIANCES = False
"""Whether to plot the per-event variances."""

# SWAG
SWAG_EPOCHS = 20
"""Number of epochs for SWAG training."""
SWAG_LR = 1e-5
"""Learning rate for SWAG training."""  
SWAG_SAMPLES = 30
"""Number of samples for SWAG evaluation."""
RETRAIN_SWAG = False
"""Whether to retrain the SWAG model even if weights exist."""

# minDeltaR flow-matching correction
MDR_EPOCHS = 300
"""Number of epochs for the minDeltaR flow-matching correction."""
MDR_PATIENCE = 20
"""Early-stopping patience for the minDeltaR flow-matching correction."""
MDR_SWAG_EPOCHS = 30
"""Number of epochs for SWAG training of the minDeltaR flow-matching correction."""
MDR_SWAG_LR = 1e-5
"""Learning rate for SWAG training of the minDeltaR flow-matching correction."""
MDR_SWAG_SAMPLES = 30
"""Number of samples for SWAG evaluation of the minDeltaR flow-matching correction."""
MDR_STEPS = 50
"""Number of steps for the minDeltaR flow-matching correction."""
RETRAIN_MDR = False
"""Whether to retrain the minDeltaR flow-matching correction even if weights exist."""
TAIL_STUDY = False
"""Whether to perform a tail study."""

# global plots
CORNER = True
"""Whether to run the corner plots."""
CORNER_MAX_EVENTS = 100000
"""Maximum number of events for the corner plots."""


def train_loaders(vr, jet5, features, scaler, validation_size, batch_size, num_workers, seed):
    """2b loaders VR_vr (c=0) vs SR (c=1), as many SR as VR events. Returns ({split: DataLoader}, scaler)."""
    return dl.vr_sr_loaders(vr, 2, features, utils.njet_min(jet5), scaler, validation_size=validation_size, seed=seed,
                            batch_size=batch_size, num_workers=num_workers)


def binary_conditions(vr, nbjet):
    """Conditions (VR, SR) of the transport for the binary flow: c = 0 (VR) and c = 1 (SR), for 2b and 4b."""
    return 0.0, 1.0


def main(base_path=BASE_PATH, regions=REGIONS, jet5=JET5, include_min_dr=INCLUDE_MIN_DR, steps=STEPS, bins=BINS, 
         transforms=TRANSFORMS, hidden_features=HIDDEN_FEATURES, phi_bins=PHI_BINS, phi_transforms=PHI_TRANSFORMS, 
         phi_hidden_features=PHI_HIDDEN_FEATURES, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS, patience=PATIENCE, lr=LR_initial, 
         batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, validation_size=VALIDATION_SIZE, seed=SEED, retrain=RETRAIN, resume=RESUME,
         recompute_eval=RECOMPUTE_EVAL, plot_latent=PLOT_LATENT, plot_variances=PLOT_VARIANCES, swag_epochs=SWAG_EPOCHS, 
         swag_lr=SWAG_LR, swag_samples=SWAG_SAMPLES, retrain_swag=RETRAIN_SWAG, mdr_epochs=MDR_EPOCHS, mdr_patience=MDR_PATIENCE, 
         mdr_swag_epochs=MDR_SWAG_EPOCHS, mdr_swag_lr=MDR_SWAG_LR, mdr_swag_samples=MDR_SWAG_SAMPLES, mdr_steps=MDR_STEPS,
         retrain_mdr=RETRAIN_MDR, tail_study=TAIL_STUDY, corner=CORNER, corner_max_events=CORNER_MAX_EVENTS,
         loaders_fn=train_loaders, model_cls=FlowModel, context_dim=1, conditions=binary_conditions):
    """
    Pipeline with one flow per validation region. The last arguments let the variants of the model reuse it
    (they are not command-line options):
      loaders_fn(vr, jet5, features, scaler, validation_size, batch_size, num_workers, seed) -> (loaders, scaler)
      model_cls / context_dim : flow class and dimension of its condition
      conditions(vr, nbjet) -> (c_bkg, c_sig): conditions of the VR and of the SR used in the tests / transport
    """
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    group = utils.group_name(jet5)
    features, phi = F.input_features(jet5), F.phi_features(jet5)
    hlf = F.hlf_features(jet5, include_min_dr)
    print(f"Group: {group} | {len(features)} input features | steps: {list(steps)}")

    for vr in regions:
        print(f"\n{'=' * 80}\nVALIDATION REGION {vr} ({group})\n{'=' * 80}")
        dir_2b = utils.region_dir(base_path, group, vr, 2)
        weights_dir = utils.region_dir(base_path, group, vr, 2, 'weights')
        scaler_path = os.path.join(weights_dir, 'scaler.pkl')

        model = model_cls(input_dim=len(features), dir_path=dir_2b, context_dim=context_dim, bins=bins, transforms=transforms,
                          hidden_features=hidden_features, phi_indices=[features.index(p) for p in phi],
                          phi_bins=phi_bins, phi_transforms=phi_transforms, phi_hidden_features=phi_hidden_features,
                          lr=lr, patience=patience)
        model.print_model_summary()

        # --- Training (2b) ---
        trained = os.path.exists(model.model_path()) and os.path.exists(scaler_path)
        if 'train' in steps and (retrain or resume or not trained):
            scaler = joblib.load(scaler_path) if (resume and os.path.exists(scaler_path)) else None
            loaders, scaler = loaders_fn(vr, jet5, features, scaler, validation_size, batch_size, num_workers, seed)
            joblib.dump(scaler, scaler_path)
            if 'val' not in loaders:
                raise ValueError("The training needs a validation set: use validation_size > 0")
            print(f"Training events: {len(loaders['train'].dataset)}, validation events: {len(loaders['val'].dataset)}")
            model.train(loaders['train'], loaders['val'], epochs=epochs, warmup_epochs=warmup_epochs, resume=resume)
        elif not trained:
            raise FileNotFoundError(f"No trained flow in {weights_dir}: run with the 'train' step")
        else:
            model.load_model()
        history = model.load_history()
        for nbjet in (2, 4):      # training curves in the plots/ of both the 2b and the 4b results
            vis.plot_flow_history(history, utils.region_dir(base_path, group, vr, nbjet, 'plots'))
        scaler = joblib.load(scaler_path)

        # --- Tests and transport (2b with its own events, 4b with the 2b-trained flow) ---
        if 'tests' in steps or 'transport' in steps:
            for nbjet in (2, 4):
                X_bkg, X_sig = utils.load_vr_sr(vr, nbjet, features, jet5)
                c_bkg, c_sig = conditions(vr, nbjet)
                tester = ModelTester(model, dl.preprocess(X_bkg, features, scaler),
                                     dl.preprocess(X_sig, features, scaler), features,
                                     utils.region_dir(base_path, group, vr, nbjet), phi_features=phi,
                                     bkg_label=f'VR{vr}', sig_label='SR', c_bkg=c_bkg, c_sig=c_sig)
                if 'tests' in steps:
                    tester.classification_analysis(force_recompute=recompute_eval)
                if 'transport' in steps:
                    tester.transport_analysis(force_recompute=recompute_eval)

        # --- SWAG (2b) ---
        if 'swag' in steps:
            loaders, _ = loaders_fn(vr, jet5, features, scaler, 0.0, batch_size, num_workers, seed)
            c_bkg, c_sig = conditions(vr, 2)
            run_flow_swag(model, loaders['train'], utils.region_dir(base_path, group, vr, 2, 'eval_data'),
                          swag_lr=swag_lr, swag_epochs=swag_epochs, num_samples=swag_samples, retrain=retrain_swag,
                          c_from=c_bkg, c_to=c_sig)

        # --- High-level features (2b first: it fits the minDeltaR correction and the uncertainty profiles) ---
        if 'hlf' in steps:
            mdr_kwargs = dict(retrain=retrain_mdr, epochs=mdr_epochs, batch_size=batch_size, patience=mdr_patience,
                              retrain_swag=retrain_mdr, swag_epochs=mdr_swag_epochs, swag_lr=mdr_swag_lr,
                              num_swag_samples=mdr_swag_samples, steps=mdr_steps, seed=seed)
            for nbjet in (2, 4):
                reconstruct_hlf_region(base_path, vr, nbjet, jet5, include_min_dr=include_min_dr, swag_lr=swag_lr,
                                       plot_latent=plot_latent, plot_variances=plot_variances, tail_study=tail_study,
                                       mdr_kwargs=mdr_kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_final_steps(base_path, steps, regions, include_min_dr, corner, corner_max_events, num_workers, seed)


def run_final_steps(base_path, steps, regions, include_min_dr, corner, corner_max_events, num_workers, seed,
                    layout=None):
    """Steps that combine the two groups: 'global' (Total/ + corner plots) and 'final4b' (Total/evaluation/)."""
    if 'global' in steps:
        run_global(base_path, include_min_dr, regions, corner, corner_max_events, layout=layout)
    if 'final4b' in steps:
        final4b_evaluation.main(base_path=base_path, regions=regions, include_min_dr=include_min_dr,
                                num_workers=num_workers, seed=seed, layout=layout)


def build_parser(description=__doc__, **defaults):
    """Command-line interface of main(); `defaults` override the default values."""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a('--base_path', default=BASE_PATH, help='output directory (contains All/ and 5jet/)')
    a('--regions', type=int, nargs='+', default=REGIONS, help='validation regions')
    a('--jet5', action='store_true', default=JET5,
      help="5-jet model (group '5jet', events with njet >= 5) instead of the 4-jet one (group 'All')")
    a('--include_min_dr', action='store_true', default=INCLUDE_MIN_DR,
      help='reconstruct minDeltaR_Higgjj and correct it with flow matching')
    a('--steps', nargs='+', choices=STEP_CHOICES, default=list(STEPS), help='pipeline steps to run')

    g = p.add_argument_group('flow architecture')
    g.add_argument('--bins', type=int, default=BINS)
    g.add_argument('--transforms', type=int, default=TRANSFORMS)
    g.add_argument('--hidden_features', type=int, nargs='+', default=HIDDEN_FEATURES)
    g.add_argument('--phi_bins', type=int, default=PHI_BINS)
    g.add_argument('--phi_transforms', type=int, default=PHI_TRANSFORMS)
    g.add_argument('--phi_hidden_features', type=int, nargs='+', default=PHI_HIDDEN_FEATURES)

    g = p.add_argument_group('training')
    g.add_argument('--epochs', type=int, default=EPOCHS)
    g.add_argument('--warmup_epochs', type=int, default=WARMUP_EPOCHS)
    g.add_argument('--patience', type=int, default=PATIENCE, help='early-stopping patience')
    g.add_argument('--lr', type=float, default=LR_initial, help='initial learning rate')
    g.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    g.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    g.add_argument('--validation_size', type=float, default=VALIDATION_SIZE)
    g.add_argument('--seed', type=int, default=SEED)
    g.add_argument('--retrain', action='store_true', default=RETRAIN, help='retrain even if weights exist')
    g.add_argument('--resume', action='store_true', default=RESUME, help='continue a previous training')

    g = p.add_argument_group('evaluation')
    g.add_argument('--recompute_eval', action='store_true', default=RECOMPUTE_EVAL, help='recompute the cached evaluations')
    g.add_argument('--no_plot_latent', dest='plot_latent', action='store_false', default=PLOT_LATENT,
                   help='skip the comparison of the transported jet features')
    g.add_argument('--plot_variances', action='store_true', default=PLOT_VARIANCES, help='plot the per-event variances')
    g.add_argument('--no_corner', dest='corner', action='store_false', default=CORNER)
    g.add_argument('--corner_max_events', type=int, default=CORNER_MAX_EVENTS)

    g = p.add_argument_group('SWAG')
    g.add_argument('--swag_epochs', type=int, default=SWAG_EPOCHS)
    g.add_argument('--swag_lr', type=float, default=SWAG_LR)
    g.add_argument('--swag_samples', type=int, default=SWAG_SAMPLES)
    g.add_argument('--retrain_swag', action='store_true', default=RETRAIN_SWAG)

    g = p.add_argument_group('minDeltaR correction (with --include_min_dr)')
    g.add_argument('--mdr_epochs', type=int, default=MDR_EPOCHS)
    g.add_argument('--mdr_patience', type=int, default=MDR_PATIENCE)
    g.add_argument('--mdr_swag_epochs', type=int, default=MDR_SWAG_EPOCHS)
    g.add_argument('--mdr_swag_lr', type=float, default=MDR_SWAG_LR)
    g.add_argument('--mdr_swag_samples', type=int, default=MDR_SWAG_SAMPLES)
    g.add_argument('--mdr_steps', type=int, default=MDR_STEPS, help='Euler steps of the ODE')
    g.add_argument('--retrain_mdr', action='store_true', default=RETRAIN_MDR, help='retrain the flow-matching model and its SWAG')
    g.add_argument('--tail_study', action='store_true', default=TAIL_STUDY, help='plots of the events in the minDeltaR tail')

    p.set_defaults(**defaults)
    return p


if __name__ == "__main__":
    # 4-jet pipeline:  python NF/main.py
    # 5-jet pipeline:  python NF/main.py --jet5
    main(**vars(build_parser().parse_args()))
