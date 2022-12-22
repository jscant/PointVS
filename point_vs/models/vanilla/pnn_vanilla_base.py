"""Base class for non-pyg models."""

from abc import ABC

from point_vs.models.point_neural_network_base import PointNeuralNetworkBase


class PNNVanillaBase(PointNeuralNetworkBase, ABC):
    """Base (abstract) class for non-geometric point cloud classifiers."""

    def unpack_input_data_and_predict(self, input_data):
        feats, y_true, ligands, receptors = input_data
        feats = self.prepare_input(feats)
        y_pred = self(feats).reshape(-1, )
        return y_pred, y_true, ligands, receptors
