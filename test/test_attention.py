"""Tests for attention mechanism."""
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import numpy as np
import torch

from point_vs.models.geometric.egnn_satorras import SartorrasEGNN
from point_vs.global_objects import DEVICE
from point_vs.utils import to_numpy

from .setup_and_params import MODEL_KWARGS
from .setup_and_params import ORIGINAL_GRAPH_TWO_ITEMS


# Tests should be repeatable
torch.random.manual_seed(2)
np.random.seed(2)


def test_satorras_egnn_attention():
    """Check that satorras-egnn-based softmax attention sums to one for each node."""
    with TemporaryDirectory() as tmpdir:
        model = SartorrasEGNN(
            Path(tmpdir), 0, 0, None, None, **MODEL_KWARGS).to(DEVICE).eval()
        model(ORIGINAL_GRAPH_TWO_ITEMS)
        edges = to_numpy(ORIGINAL_GRAPH_TWO_ITEMS.edge_index)
        scatter_indices = edges[0, :]
        has_checked_attention = False
        for layer in model.layers:
            if hasattr(layer, 'att_val'):
                has_checked_attention = True
                node_atn_sums = np.zeros(
                    (max(scatter_indices) + 1,))

                # Scatter sum attention values using edge indices as index
                np.add.at(
                    node_atn_sums, scatter_indices, layer.att_val.squeeze())

                np.testing.assert_allclose(
                    node_atn_sums,
                    np.ones_like(node_atn_sums),
                    atol=1e-6)
        if not has_checked_attention:
            pytest.fail('No tests run (make sure test model uses attention!)')
