"""Panel computations of Extended Data Figures 7 and 8 (Figure 5 reads the readout tables directly).

Every panel writes its result to results/panels/<name>.pkl; the notebooks only read these files. The trained networks are reused;
nothing is retrained.

    python -m mtrnn.panels <name> [<name> ...]     one or more of the names below, or a group: cpu, simulate, post, silencing, all

    name                    figure panel   needs
    performance             ED 7a          results/performance of both main runs
    index_reliability       ED 7b          features
    tail_size               ED 7c          features
    softplus                ED 7d, e       features (main_softplus)
    pct_ti_within_type      ED 7f          features
    noise_sweep             ED 7g          features, reference/three_point.csv
    pursuit_dial            ED 7g right    results/measured of the five pursuit_dial runs
    training_trajectory     ED 7h          simulate_training_trajectory (checkpoints of main_relu; GPU advised)
    silencing               ED 8a          the networks (GPU advised, ~40 min), task_decoding
    unit_removal            ED 8b          features
    task_decoding           ED 8c          simulate_task_decoding (GPU advised), noise_sweep
    stim_vs_response        ED 8d          features
    rule_weights            ED 8e          the networks
    recurrent_weights       ED 8f          the networks

"features" = results/readout/<run>/features.npz, written by `python -m mtrnn.readout features <run>`.
"""
import sys
import time

from .common import *                                            # noqa: F401,F403
from .common import PAPER_WINDOW, _simulate                      # noqa: F401
from .robustness import *                                        # noqa: F401,F403
from .roles import *                                             # noqa: F401,F403
from .robustness import (panel_performance, panel_index_reliability, panel_tail_size, panel_softplus, panel_pct_ti_within_type,
                         panel_noise_sweep, panel_noise_sweep_retarget, panel_pursuit_dial, simulate_training_trajectory,
                         panel_training_trajectory)
from .roles import (panel_unit_removal, panel_stim_vs_response, panel_rule_weights, panel_recurrent_weights, simulate_task_decoding,
                    panel_task_decoding, panel_silencing)


def _noise_sweep():
    panel_noise_sweep(); panel_noise_sweep_retarget(PAPER_WINDOW)


PANELS = dict(performance=panel_performance, index_reliability=panel_index_reliability, tail_size=panel_tail_size, softplus=panel_softplus,
              pct_ti_within_type=panel_pct_ti_within_type, noise_sweep=_noise_sweep, pursuit_dial=lambda: panel_pursuit_dial(PAPER_WINDOW),
              unit_removal=panel_unit_removal, stim_vs_response=panel_stim_vs_response, rule_weights=panel_rule_weights,
              recurrent_weights=panel_recurrent_weights, simulate_task_decoding=simulate_task_decoding,
              simulate_training_trajectory=simulate_training_trajectory, task_decoding=panel_task_decoding,
              training_trajectory=panel_training_trajectory, silencing=panel_silencing)
GROUPS = dict(cpu=['performance', 'stim_vs_response', 'rule_weights', 'noise_sweep', 'pursuit_dial', 'tail_size', 'softplus',
                   'pct_ti_within_type', 'unit_removal', 'index_reliability', 'recurrent_weights'],
              simulate=['simulate_task_decoding', 'simulate_training_trajectory'], post=['task_decoding', 'training_trajectory'],
              silencing=['silencing'])
GROUPS['all'] = [*GROUPS['cpu'], *GROUPS['simulate'], *GROUPS['post'], *GROUPS['silencing']]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__); return
    todo = [p for a in argv for p in (GROUPS.get(a) or [a])]
    for p in todo:
        t0 = time.time(); print(f'== {p}', flush=True); PANELS[p](); print(f'== {p} done in {time.time() - t0:.0f}s', flush=True)
