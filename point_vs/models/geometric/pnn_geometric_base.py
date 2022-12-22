from abc import abstractmethod

import torch
from torch import nn

from torch_geometric.nn import global_mean_pool

from point_vs import logging
from point_vs.global_objects import DEVICE
from point_vs.models.point_neural_network_base import PointNeuralNetworkBase
from point_vs.utils import to_numpy


LOG = logging.get_logger('PointVS')


class PNNGeometricBase(PointNeuralNetworkBase):
    """Base (abstract) class for all pytorch geometric point neural networks."""

    @abstractmethod
    def get_embeddings(self, feats, edges, coords, edge_attributes, batch):
        """Implement code to go from input features to final node embeddings."""

    def forward(self, x):
        # torch.max seems to be bugged for integers, must specify batch size to
        # global_mean_pool (https://github.com/pytorch/pytorch/issues/90273)
        batch_size = torch.max(x.batch.int()).long() + 1

        def pooling_op(x):
            """global_mean_pool bugged for size=1 on MPS."""
            if batch_size == 1:
                return torch.mean(x, axis=0)
            return global_mean_pool(x, batch, size=batch_size)

        feats, edges, coords, edge_attributes, batch = self.unpack_graph(x)
        feats, _ = self.get_embeddings(
            feats, edges, coords, edge_attributes, batch)
        if self.feats_linear_layers is not None:
            feats = pooling_op(feats)  # (total_nodes, k)
            feats = self.feats_linear_layers(feats)  # (bs, k)
        return feats

    def unpack_input_data_and_predict(self, input_data):
        """See base class."""
        y_true = input_data.y
        try:
            y_true = y_true.float()
        except (AttributeError, TypeError):
            pass
        y_pred = self(input_data).reshape(-1, )
        ligands = input_data.lig_fname
        receptors = input_data.rec_fname
        return y_pred, y_true, ligands, receptors

    def unpack_graph(self, graph):
        return (graph.x.float().to(DEVICE), graph.edge_index.to(DEVICE),
                graph.pos.float().to(DEVICE), graph.edge_attr.to(DEVICE),
                graph.batch.to(DEVICE))


class PygLinearPass(nn.Module):
    """Helper class for neater forward passes.

    Gives a linear layer with the same semantic behaviour as the E_GCL and
    EGNN_Sparse layers.

    Arguments:
        module: nn.Module (usually a linear layer)
        feats_appended_to_coords: does the input include coordinates in the
            first three columns of the node feature vector
        return_coords_and_edges: return a tuple containing the node features,
            the coords and the edges rather than just the node features
    """

    def __init__(self, module, feats_appended_to_coords=False,
                 return_coords_and_edges=False):
        super().__init__()
        self.m = module
        self.feats_appended_to_coords = feats_appended_to_coords
        self.return_coords_and_edges = return_coords_and_edges
        self.intermediate_coords = None

    def forward(self, h, **kwargs):
        if self.feats_appended_to_coords:
            self.intermediate_coords = to_numpy(h[:, :3])
            feats = h[:, 3:]
            res = torch.hstack([h[:, :3], self.m(feats)])
        else:
            self.intermediate_coords = to_numpy(kwargs['coord'])
            res = self.m(h)
        if self.return_coords_and_edges:
            return res, kwargs['coord'], kwargs['edge_attr'], kwargs.get(
                'edge_messages', None)
        return res
