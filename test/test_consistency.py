import pytest
from torch import sigmoid

from point_vs.models.egnn_lucid import PygLucidEGNN
from point_vs.models.egnn_satorras import SartorrasEGNN
from point_vs.models.lie_conv import LieResNet
from point_vs.models.lie_transformer import EquivariantTransformer
from .setup_and_params import MODEL_KWARGS, ORIGINAL_COORDS, FEATS, MASK, \
    EGNN_EPS, LIFT_EPS, N_SAMPLES, setup, ORIGINAL_GRAPH

dump_path = setup()


def test_sartorras_egnn_consistency():
    model = SartorrasEGNN(
        dump_path, 0, 0, None, None, **MODEL_KWARGS).cuda().eval()
    unrotated_result = float(sigmoid(model(ORIGINAL_GRAPH)))

    assert unrotated_result != pytest.approx(0, abs=1e-5)
    for _ in range(N_SAMPLES):
        assert float(sigmoid(model(ORIGINAL_GRAPH))) == pytest.approx(
            unrotated_result, abs=EGNN_EPS)


def test_lucid_egnn_consistency():
    model = PygLucidEGNN(
        dump_path, 0, 0, None, None, **MODEL_KWARGS).cuda().eval()
    unrotated_result = float(sigmoid(model(ORIGINAL_GRAPH)))

    assert unrotated_result != pytest.approx(0, abs=1e-5)
    for _ in range(N_SAMPLES):
        assert float(sigmoid(model(ORIGINAL_GRAPH))) == pytest.approx(
            unrotated_result, abs=EGNN_EPS)


def test_lie_transformer_consistency():
    model = EquivariantTransformer(
        dump_path, 0, 0, None, None, **MODEL_KWARGS).cuda().eval()
    unrotated_result = float(sigmoid(model((ORIGINAL_COORDS, FEATS, MASK))))

    assert unrotated_result != pytest.approx(0, abs=1e-5)
    for _ in range(N_SAMPLES):
        assert float(
            sigmoid(model((ORIGINAL_COORDS, FEATS, MASK)))) == pytest.approx(
            unrotated_result, abs=LIFT_EPS)


def test_lie_conv_consistency():
    model = LieResNet(dump_path, 0, 0, None, None, **MODEL_KWARGS).eval()
    unrotated_result = float(sigmoid(model((ORIGINAL_COORDS, FEATS, MASK))))

    assert unrotated_result != pytest.approx(0, abs=1e-5)
    for _ in range(N_SAMPLES):
        assert float(
            sigmoid(model((ORIGINAL_COORDS, FEATS, MASK)))) == pytest.approx(
            unrotated_result, abs=LIFT_EPS)
