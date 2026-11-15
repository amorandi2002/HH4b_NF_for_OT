"""
Conditional normalizing flow trained on all the validation regions at the same time.

Same architecture as model.FlowModel (NCSF for phi x NSF for the other features); the condition is a one-hot
region label (VR1..VR5 = e_1..e_5, SR = e_6, see lib/data_loader_allVRs.py):
    with_4b=False : c in R^6, trained on 2b events only (applied unchanged to 4b)
    with_4b=True  : c = (one-hot region, 4b flag) in R^7, trained on 2b and 4b events (lib/data_loader_2b4b.py)
Transport VR_i -> SR:  z = T(x | c = e_i [, f]),  x_reco = T^-1(z | c = e_6 [, f]),  f = 0 (2b) / 1 (4b).
"""
import utils  # noqa: F401  (puts the repository root on sys.path)
from lib import data_loader_2b4b as dl24
from lib.data_loader_allVRs import N_REGIONS, SR_CONDITION, SR_INDEX, vr_condition
from model import FlowModel


class FlowModelAllVRs(FlowModel):
    def __init__(self, input_dim, dir_path, with_4b=False, **kwargs):
        kwargs.pop('context_dim', None)
        self.with_4b = with_4b
        super().__init__(input_dim, dir_path, context_dim=N_REGIONS + int(with_4b), **kwargs)

    def conditions(self, vr, nbjet=2):
        """Conditions (VR_vr, SR) of the transport of the validation region vr with nbjet b-tagged jets."""
        if not self.with_4b:
            return vr_condition(vr), SR_CONDITION
        return dl24.allvrs_condition(vr - 1, nbjet), dl24.allvrs_condition(SR_INDEX, nbjet)
