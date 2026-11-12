"""Where everything lives, and which device to run on.

The repository root is found from this file (src/mtrnn/paths.py -> two levels up); set MTRNN_ROOT to use another location
(for example when the package is installed without `-e`).

    models/<run>/                    trained networks: config.json, history.json, seed<k>/model.pt (+ ckpt_<iter>.pt)
    results/performance/<run>/       evaluate.py: performance per task and network, provenance check, clusters
    results/readout/<run>/           readout.py, heldout.py, silencing.py: unit and task-pair tables (+ features.npz)
    results/measured/<run>__<pair>/  measured.py: the two analog pairs read like the recordings
    results/panels/                  panels: one .pkl per figure panel (what the notebooks read) + large intermediates
    reference/                       the recordings' values the networks are compared with
"""
import os

ROOT = os.environ.get('MTRNN_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
MODELS = os.path.join(ROOT, 'models')
RESULTS = os.path.join(ROOT, 'results')
PERFORMANCE = os.path.join(RESULTS, 'performance')
READOUT = os.path.join(RESULTS, 'readout')
MEASURED = os.path.join(RESULTS, 'measured')
PANELS = os.path.join(RESULTS, 'panels')
REFERENCE = os.path.join(ROOT, 'reference')


def pick_device(device=None):
    """'cuda' if available, else 'mps' (Apple GPU), else 'cpu'; a device given explicitly is returned unchanged."""
    if device:
        return str(device)
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
