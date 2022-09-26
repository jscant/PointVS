import pytest
import torch

from point_vs.models.geometric.egnn_lucid import PygLucidEGNN
from point_vs.models.geometric.egnn_satorras import SartorrasEGNN

from .setup_and_params import MODEL_KWARGS
from .setup_and_params import ORIGINAL_GRAPH
from .setup_and_params import EGNN_EPS
from .setup_and_params import ROTATED_GRAPH
from .setup_and_params import setup
from .setup_and_params import DEVICE

dump_path = setup()


def test_lucid_egnn_invariance():
    """Check that lucidrains-based network is invariant to SE(3) transformations."""
    model = PygLucidEGNN(
        dump_path, 0, 0, None, None, **MODEL_KWARGS).to(DEVICE).eval()

    unrotated_result = float(torch.sigmoid(model(ORIGINAL_GRAPH)))
    rotated_result = float(torch.sigmoid(model(ROTATED_GRAPH)))
    assert unrotated_result == pytest.approx(rotated_result, abs=EGNN_EPS)


def test_sartorras_egnn_invariance():
    """Check that satorras-egnn-based network is invariant to SE(3) transformations."""
    model = SartorrasEGNN(
        dump_path, 0, 0, None, None, **MODEL_KWARGS).to(DEVICE).eval()

    unrotated_result = float(torch.sigmoid(model(ORIGINAL_GRAPH)))
    rotated_result = float(torch.sigmoid(model(ROTATED_GRAPH)))
    assert unrotated_result == pytest.approx(rotated_result, abs=EGNN_EPS)
