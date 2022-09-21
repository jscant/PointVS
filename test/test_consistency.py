"""Tests to ensure that same inputs give same outputs."""
import pytest
import torch

from point_vs.models.geometric.egnn_lucid import PygLucidEGNN
from point_vs.models.geometric.egnn_satorras import SartorrasEGNN
from .setup_and_params import MODEL_KWARGS
from .setup_and_params import EGNN_EPS
from .setup_and_params import N_SAMPLES
from .setup_and_params import ORIGINAL_GRAPH
from .setup_and_params import setup

dump_path = setup()


def test_sartorras_egnn_consistency():
    """Check that satorras satorras-egnn-based model is consistent."""
    model = SartorrasEGNN(
        dump_path, 0, 0, None, None, **MODEL_KWARGS).cuda().eval()
    unrotated_result = float(torch.sigmoid(model(ORIGINAL_GRAPH)))

    assert unrotated_result != pytest.approx(0, abs=1e-5)
    for _ in range(N_SAMPLES):
        assert float(torch.sigmoid(model(ORIGINAL_GRAPH))) == pytest.approx(
            unrotated_result, abs=EGNN_EPS)


def test_lucid_egnn_consistency():
    """Check that lucid lucidrains-egnn-based model is consistent."""
    model = PygLucidEGNN(
        dump_path, 0, 0, None, None, **MODEL_KWARGS).cuda().eval()
    unrotated_result = float(torch.sigmoid(model(ORIGINAL_GRAPH)))

    assert unrotated_result != pytest.approx(0, abs=1e-5)
    for _ in range(N_SAMPLES):
        assert float(torch.sigmoid(model(ORIGINAL_GRAPH))) == pytest.approx(
            unrotated_result, abs=EGNN_EPS)
