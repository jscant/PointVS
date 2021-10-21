import pytest
from torch import sigmoid

from point_vs.models.egnn_lucid import PygLucidEGNN
from point_vs.models.egnn_satorras import SartorrasEGNN
from point_vs.models.lie_conv import LieResNet
from point_vs.models.lie_transformer import EquivariantTransformer
from .setup_and_params import MODEL_KWARGS, ORIGINAL_COORDS, FEATS, MASK, \
    EGNN_EPS, LIFT_EPS, ROTATED_COORDS, ORIGINAL_GRAPH, setup, ROTATED_GRAPH

dump_path = setup()


def test_lucid_egnn_invariance():
    model = PygLucidEGNN(
        dump_path, 0, 0, None, None, **MODEL_KWARGS).cuda().eval()

    unrotated_result = float(sigmoid(model(ORIGINAL_GRAPH)))
    rotated_result = float(sigmoid(model(ROTATED_GRAPH)))
    assert unrotated_result == pytest.approx(rotated_result, abs=EGNN_EPS)


def test_sartorras_egnn_invariance():
    model = SartorrasEGNN(
        dump_path, 0, 0, None, None, **MODEL_KWARGS).cuda().eval()

    unrotated_result = float(sigmoid(model(ORIGINAL_GRAPH)))
    rotated_result = float(sigmoid(model(ROTATED_GRAPH)))
    assert unrotated_result == pytest.approx(rotated_result, abs=EGNN_EPS)


def test_lie_transformer_invariance():
    model = EquivariantTransformer(
        dump_path, 0, 0, None, None, **MODEL_KWARGS).cuda().eval()

    unrotated_result = float(sigmoid(model((ORIGINAL_COORDS, FEATS, MASK))))
    assert unrotated_result != pytest.approx(0, abs=1e-5)

    for rotated_coords in ROTATED_COORDS:
        rotated_result = float(sigmoid(model((rotated_coords, FEATS, MASK))))

        assert unrotated_result != pytest.approx(0, abs=1e-5)
        assert unrotated_result == pytest.approx(rotated_result, abs=LIFT_EPS)


def test_lie_conv_invariance():
    model = LieResNet(dump_path, 0, 0, None, None, **MODEL_KWARGS).eval()

    unrotated_result = float(sigmoid(model((ORIGINAL_COORDS, FEATS, MASK))))
    assert unrotated_result != pytest.approx(0, abs=1e-5)

    for rotated_coords in ROTATED_COORDS:
        rotated_result = float(sigmoid(model((rotated_coords, FEATS, MASK))))

        assert unrotated_result == pytest.approx(rotated_result, abs=LIFT_EPS)
