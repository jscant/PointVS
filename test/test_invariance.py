from pathlib import Path

import numpy as np
import pytest
import torch
from lie_conv.lieGroups import SE3

from point_vs.models.lie_conv import LieResNet
from point_vs.models.lie_transformer import EquivariantTransformer
from point_vs.models.point_neural_network import to_numpy
from point_vs.preprocessing.data_loaders import random_rotation

# This is higher than usual, because for infinite Lie groups, there are several
# approximations (MC sampling, for example) which reduce the true invariance
# to approximate invariance.
EPS = 5e-3
MODEL_KWARGS = {
    'act': 'relu',
    'bn': True,
    'cache': False,
    'ds_frac': 1.0,
    'fill': 1.0,
    'group': SE3(0.2),
    'k': 16,
    'knn': False,
    'liftsamples': 4,
    'mean': True,
    'nbhd': 128,
    'num_layers': 2,
    'pool': True,
    'dropout': 0,
    'dim_input': 12,
    'dim_output': 1,
    'dim_hidden': 12,  # == 32
    'num_heads': 4,
    'global_pool': True,
    'global_pool_mean': True,
    'block_norm': "layer_pre",
    'output_norm': "none",
    'kernel_norm': "none",
    'kernel_type': 'mlp',
    'kernel_dim': 16,
    'kernel_act': 'relu',
    'mc_samples': 4,
    'attention_fn': 'dot_product',
    'feature_embed_dim': None,
    'max_sample_norm': None,
    'lie_algebra_nonlinearity': None,
}

torch.random.manual_seed(1)
np.random.seed(1)


def _set_precision(precision):
    if precision == 'double':
        torch.set_default_dtype(torch.float64)
        torch.set_default_tensor_type(torch.DoubleTensor)
    else:
        torch.set_default_dtype(torch.float32)
        torch.set_default_tensor_type(torch.FloatTensor)


def test_lie_conv_double():
    _set_precision('double')
    original_coords = torch.rand(1, 100, 3)
    rotated_coords = torch.from_numpy(
        random_rotation(to_numpy(original_coords))).double()

    feats = torch.rand(1, 100, 12)
    mask = torch.ones(1, 100).bool()
    model = LieResNet(Path('test/data_dump'), 0, 0, None, None, **MODEL_KWARGS)

    unrotated_result = float((model((original_coords, feats, mask))))
    rotated_result = float(to_numpy(model((rotated_coords, feats, mask))))

    assert unrotated_result == pytest.approx(rotated_result, abs=EPS)


def test_lie_conv_float():
    _set_precision('float')
    original_coords = torch.rand(1, 100, 3)
    rotated_coords = torch.from_numpy(
        random_rotation(to_numpy(original_coords))).float()

    feats = torch.rand(1, 100, 12)
    mask = torch.ones(1, 100).bool()
    model = LieResNet(Path('test/data_dump'), 0, 0, None, None, **MODEL_KWARGS)

    unrotated_result = float(to_numpy(model((original_coords, feats, mask))))
    rotated_result = float((model((rotated_coords, feats, mask))))

    assert unrotated_result == pytest.approx(rotated_result, abs=EPS)


def test_lie_transformer_double():
    _set_precision('double')
    original_coords = torch.rand(1, 100, 3).cuda()
    rotated_coords = torch.from_numpy(
        random_rotation(to_numpy(original_coords))).double().cuda()

    feats = torch.rand(1, 100, 12).cuda()
    mask = torch.ones(1, 100).bool().cuda()
    model = EquivariantTransformer(
        Path('test/data_dump'), 0, 0, None, None, **MODEL_KWARGS).cuda()

    unrotated_result = float(to_numpy(model((original_coords, feats, mask))))
    rotated_result = float((model((rotated_coords, feats, mask))))

    assert unrotated_result == pytest.approx(rotated_result, abs=EPS)


def test_lie_transformer_float():
    _set_precision('float')
    original_coords = torch.rand(1, 100, 3).cuda()
    rotated_coords = torch.from_numpy(
        random_rotation(to_numpy(original_coords))).float().cuda()

    feats = torch.rand(1, 100, 12).cuda()
    mask = torch.ones(1, 100).bool().cuda()
    model = EquivariantTransformer(
        Path('test/data_dump'), 0, 0, None, None, **MODEL_KWARGS).cuda()

    unrotated_result = float(to_numpy(model((original_coords, feats, mask))))
    rotated_result = float((model((rotated_coords, feats, mask))))

    assert unrotated_result == pytest.approx(rotated_result, abs=EPS)
