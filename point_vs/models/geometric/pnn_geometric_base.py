from abc import abstractmethod

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

from point_vs.models.point_neural_network_base import PointNeuralNetworkBase


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

    def forward(self, x, *args, **kwargs):
        if self.feats_appended_to_coords:
            feats = x[:, 3:]
            res = torch.hstack([x[:, :3], self.m(feats)])
        else:
            res = self.m(x)
        if self.return_coords_and_edges:
            return res, kwargs['coord'], kwargs['edge_attr']
        return res


class PNNGeometricBase(PointNeuralNetworkBase):
    """Base (abstract) class for all pytorch geometric point neural networks."""

    def forward(self, graph):
        feats, edges, coords, edge_attributes, batch = self.unpack_graph(graph)
        feats = self.get_embeddings(
            feats, edges, coords, edge_attributes, batch)
        if self.linear_gap:
            feats = self.layers[-1](feats, edges, edge_attributes, batch)
            feats = global_mean_pool(feats, graph.batch.cuda())
        else:
            feats = global_mean_pool(feats, graph.batch.cuda())
            feats = self.layers[-1](feats, edges, edge_attributes, batch)
        return feats

    def process_graph(self, graph):
        y_true = graph.y.float()
        y_pred = self(graph).reshape(-1, )
        ligands = graph.lig_fname
        receptors = graph.rec_fname
        return y_pred, y_true, ligands, receptors

    @abstractmethod
    def get_embeddings(self, feats, edges, coords, edge_attributes, batch):
        """Implement code to go from input features to final node embeddings."""
        pass

    def prepare_input(self, x):
        return x.cuda()

    @staticmethod
    def unpack_graph(graph):
        return (graph.x.float().cuda(), graph.edge_index.cuda(),
                graph.pos.float().cuda(), graph.edge_attr.cuda(),
                graph.batch.cuda())
