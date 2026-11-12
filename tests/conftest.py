import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from mtrnn import paths  # noqa: E402

RUNS = ['main_relu', 'main_softplus'] + [f'pursuit_dial_g{g:03d}' for g in (0, 25, 50, 75, 100)]


def pytest_addoption(parser):
    parser.addoption('--slow', action='store_true', help='also run the tests that recompute results from the networks')


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--slow'):
        skip = pytest.mark.skip(reason='needs --slow')
        for item in items:
            if 'slow' in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope='session')
def results():
    """Skip when the stored results have not been downloaded (scripts/download_results.py)."""
    if not os.path.exists(os.path.join(paths.PANELS, 'performance.pkl')):
        pytest.skip('results/ not present: run scripts/download_results.py')
    return paths.RESULTS


@pytest.fixture(scope='session')
def features():
    if not os.path.exists(os.path.join(paths.READOUT, 'main_relu', 'features.npz')):
        pytest.skip('unit features not present: run scripts/download_results.py --features or python -m mtrnn.readout features main_relu')
    return paths.READOUT
