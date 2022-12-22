from abc import ABC

from point_vs.models.point_neural_network_base import PointNeuralNetworkBase


class PNNVanillaBase(PointNeuralNetworkBase, ABC):
    """Base (abstract) class for non-geometric point cloud classifiers."""

    def unpack_input_data_and_predict(self, inputs):
        x, y_true, ligands, receptors = inputs
        x = self.prepare_input(x)
        y_pred = self(x).reshape(-1, )
        return y_pred, y_true, ligands, receptors
