# This is higher than usual, because for infinite Lie groups, there are several
# approximations (MC sampling, for example) which reduce the true invariance
# to approximate invariance.
from pathlib import Path

import numpy as np
import torch
from lie_conv.lieGroups import SE3

from point_vs.preprocessing.data_loaders import random_rotation
from point_vs.utils import to_numpy, _set_precision


def setup():
    _set_precision('float')

    # Tests should be repeatable
    torch.random.manual_seed(2)
    np.random.seed(2)

    # Check if we can write to /tmp/; if not, write to test directory
    dump_path = Path('/tmp/pointvs_test')
    try:
        open(dump_path / 'probe')
    except IOError:
        dump_path = Path('test/dump_path')
    return dump_path


EPS = 3e-2
MODEL_KWARGS = {
    'act': 'relu',
    'bn': True,
    'cache': False,
    'ds_frac': 0.75,
    'fill': 0.75,
    'group': SE3(0.2),
    'k': 32,
    'knn': False,
    'liftsamples': 1,
    'mean': True,
    'nbhd': 32,
    'num_layers': 6,
    'pool': True,
    'dropout': 0,
    'dim_input': 12,
    'dim_output': 1,
    'dim_hidden': 32,  # == 32
    'num_heads': 8,
    'global_pool': True,
    'global_pool_mean': True,
    'block_norm': "layer_pre",
    'output_norm': "none",
    'kernel_norm': "none",
    'kernel_type': 'mlp',
    'kernel_dim': 16,
    'kernel_act': 'relu',
    'mc_samples': 1,
    'attention_fn': 'softmax',
    'feature_embed_dim': None,
    'max_sample_norm': None,
    'lie_algebra_nonlinearity': None,
    'pooling_only': True
}

N_SAMPLES = 10
ORIGINAL_COORDS = torch.rand(1, 100, 3).cuda()
ROTATED_COORDS = [
    torch.from_numpy(random_rotation(to_numpy(ORIGINAL_COORDS))).float().cuda()
    for _ in range(N_SAMPLES)]
FEATS = torch.rand(1, 100, 12).cuda()
MASK = torch.ones(1, 100).bool().cuda()
