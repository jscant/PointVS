"""Check if important classes are correctly instantiated."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from point_vs.models.geometric.egnn_lucid import PygLucidEGNN
from point_vs.models.geometric.egnn_satorras import SartorrasEGNN
from point_vs.preprocessing.data_loaders import PointCloudDataset, \
    PygPointCloudDataset
from .setup_and_params import MODEL_KWARGS, DATALOADER_KWARGS


# Tests should be repeatable
torch.random.manual_seed(2)
np.random.seed(2)


def test_sartorras_egnn_instantiation():
    """Check if satorras-egnn is instantiated."""
    with TemporaryDirectory() as tmpdir:
        model = SartorrasEGNN(Path(tmpdir), 0, 0, None, None, **MODEL_KWARGS)
        assert str(model.save_path) == tmpdir


def test_lucid_egnn_instantiation():
    """Check if lucid-egnn is instantiated."""
    with TemporaryDirectory() as tmpdir:
        model = PygLucidEGNN(Path(tmpdir), 0, 0, None, None, **MODEL_KWARGS)
        assert str(model.save_path) == tmpdir


def test_vanilla_data_loader_instantiation():
    """Check if data_loader is instantiated."""
    with TemporaryDirectory() as tmpdir:
        test_dl = PygPointCloudDataset(Path(tmpdir), **DATALOADER_KWARGS)
        assert len(test_dl) == 2


def test_pyg_data_loader_instantiation():
    """Check is pyg data_loader is instantiated."""
    with TemporaryDirectory() as tmpdir:
        test_dl = PointCloudDataset(Path(tmpdir), **DATALOADER_KWARGS)
        assert len(test_dl) == 2
