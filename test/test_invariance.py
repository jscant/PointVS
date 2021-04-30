import pytest
from torch import sigmoid

from point_vs.models.egnn_network import EGNN
from point_vs.models.lie_conv import LieResNet
from point_vs.models.lie_transformer import EquivariantTransformer
from test.setup_and_params import MODEL_KWARGS, ORIGINAL_COORDS, FEATS, MASK, \
    EPS, ROTATED_COORDS, setup

dump_path = setup()


def test_egnn_invariance():
    model = EGNN(dump_path, 0, 0, None, None, **MODEL_KWARGS).cuda().eval()

    unrotated_result = float(sigmoid(model((ORIGINAL_COORDS, FEATS, MASK))))
    assert unrotated_result != pytest.approx(0, abs=1e-5)

    for rotated_coords in ROTATED_COORDS:
        rotated_result = float(sigmoid(model((rotated_coords, FEATS, MASK))))

        assert unrotated_result == pytest.approx(rotated_result, abs=EPS)


def test_lie_transformer_invariance():
    model = EquivariantTransformer(
        dump_path, 0, 0, None, None, **MODEL_KWARGS).cuda().eval()

    unrotated_result = float(sigmoid(model((ORIGINAL_COORDS, FEATS, MASK))))
    assert unrotated_result != pytest.approx(0, abs=1e-5)

    for rotated_coords in ROTATED_COORDS:
        rotated_result = float(sigmoid(model((rotated_coords, FEATS, MASK))))

        assert unrotated_result != pytest.approx(0, abs=1e-5)
        assert unrotated_result == pytest.approx(rotated_result, abs=EPS)


def test_lie_conv_invariance():
    model = LieResNet(dump_path, 0, 0, None, None, **MODEL_KWARGS).eval()

    unrotated_result = float(sigmoid(model((ORIGINAL_COORDS, FEATS, MASK))))
    assert unrotated_result != pytest.approx(0, abs=1e-5)

    for rotated_coords in ROTATED_COORDS:
        rotated_result = float(sigmoid(model((rotated_coords, FEATS, MASK))))

        assert unrotated_result == pytest.approx(rotated_result, abs=EPS)
