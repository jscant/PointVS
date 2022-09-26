# This is higher than usual, because for infinite Lie groups, there are several
# approximations (MC sampling, for example) which reduce the true invariance
# to approximate invariance.
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from point_vs.preprocessing.data_loaders import get_data_loader, \
    PygPointCloudDataset
from point_vs.preprocessing.preprocessing import uniform_random_rotation
from point_vs.preprocessing.pyg_single_item_dataset import \
    get_pyg_single_graph_for_inference
from point_vs.utils import to_numpy, _set_precision


def setup():
    """Ensure we can write to temporary test location."""
    _set_precision('float')

    # Tests should be repeatable
    torch.random.manual_seed(2)
    np.random.seed(2)

    # Check if we can write to /tmp/; if not, write to test directory
    dump_path = Path('/tmp/pointvs_test')
    try:
        with open(dump_path / 'probe', 'w'):  # pylint: disable=unspecified-encoding
            pass
    except IOError:
        dump_path = Path('test/dump_path')
    return dump_path


_test_dl = get_data_loader(
    Path('test/resources'),
    dataset_class=PygPointCloudDataset,
    batch_size=1, compact=True, radius=4,
    use_atomic_numbers=False, rot=False,
    augmented_actives=0,
    min_aug_angle=0,
    polar_hydrogens=False, receptors=None, mode='val',
    types_fname=Path('test/resources/test.types'),
    fname_suffix='.parquet',
    edge_radius=4,
    estimate_bonds=True,
)

DATALOADER_KWARGS = {
    'compact': True, 'receptors': None,
    'augmented_active_count': 0,
    'augmented_active_min_angle': 0,
    'polar_hydrogens': False,
    'max_active_rms_distance': None,
    'min_inactive_rms_distance': None,
    'use_atomic_numbers': False,
    'fname_suffix': 'parquet',
    'types_fname': 'test/resources/test.types',
    'edge_radius': 6,
    'estimate_bonds': True,
    'prune': True
}

ORIGINAL_GRAPH = list(_test_dl)[0]

rotated_coords = torch.from_numpy(
    uniform_random_rotation(to_numpy(ORIGINAL_GRAPH.pos)))
ROTATED_GRAPH = get_pyg_single_graph_for_inference(Data(
    x=ORIGINAL_GRAPH.x,
    edge_index=ORIGINAL_GRAPH.edge_index,
    edge_attr=ORIGINAL_GRAPH.edge_attr,
    pos=rotated_coords,
))

EGNN_EPS = 3e-5
LIFT_EPS = 3e-2
MODEL_KWARGS = {
    'act': 'relu',
    'bn': True,
    'cache': False,
    'k': 32,
    'mean': True,
    'nbhd': 32,
    'num_layers': 6,
    'pool': True,
    'dropout': 0,
    'dim_input': 12,
    'dim_output': 1,
    'dim_hidden': 32,  # == 32
    'mc_samples': 1,
    'attention_fn': 'softmax',
    'feature_embed_dim': None,
    'max_sample_norm': None,
    'lie_algebra_nonlinearity': None,
    'pooling_only': True,
    'linear_gap': True,
    'graphnorm': True,
    'update_coords': True,
    'node_attention': True,
    'residual': True
}

N_SAMPLES = 10
ORIGINAL_COORDS = torch.rand(1, 100, 3).to(_device)
ROTATED_COORDS = [
    torch.from_numpy(
        uniform_random_rotation(
            to_numpy(
                ORIGINAL_COORDS.reshape(
                    (100, 3))))).reshape(1, 100, 3).float().to(_device)
    for _ in range(N_SAMPLES)]
FEATS = torch.rand(1, 100, 12).to(_device)
MASK = torch.ones(1, 100).bool().to(_device)
