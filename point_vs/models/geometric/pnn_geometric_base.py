from abc import abstractmethod

import torch
from torch import nn

from torch_geometric.nn import global_mean_pool

from point_vs.global_objects import DEVICE
from point_vs.models.point_neural_network_base import PointNeuralNetworkBase
from point_vs.utils import to_numpy


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


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

    def forward(self, h, *args, **kwargs):
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


class GlobalAveragePooling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, graph_sizes=None):
        assert len(
            x.shape) == 3, 'input to GAP should be shape (bs, max_len, k)'
        if graph_sizes is None:
            return torch.mean(x, dim=2)
        output = torch.zeros(x.shape[0], x.shape[2]).to(DEVICE)
        for i in range(x.size(0)):
            output[i] = torch.mean(x[i, :graph_sizes[i], :], dim=0)
        return output


class PNNGeometricBase(PointNeuralNetworkBase):
    """Base (abstract) class for all pytorch geometric point neural networks."""

    def forward(self, graph):
        # torch.max seems to be bugged for integers, must specify batch size to
        # global_mean_pool (https://github.com/pytorch/pytorch/issues/90273)
        batch_size = torch.max(graph.batch.float().long()) + 1

        def pooling_op(x):
            """global_mean_pool bugged for size=1 on MPS."""
            if batch_size == 1:
                return torch.mean(x, axis=0)
            return global_mean_pool(x, batch, size=batch_size)

        if self.include_strain_info:
            feats, edges, coords, edge_attributes, batch, dE, rmsd = \
                self.unpack_graph(
                    graph)
        else:
            feats, edges, coords, edge_attributes, batch = self.unpack_graph(
                graph)
            dE, _ = None, None
        feats, messages = self.get_embeddings(
            feats, edges, coords, edge_attributes, batch)
        total_nodes, _ = feats.shape
        row, _ = edges
        if self.linear_gap:
            if self.feats_linear_layers is not None:
                feats = self.feats_linear_layers(feats)
                feats = pooling_op(feats)
            if self.edges_linear_layers is not None:
                agg = unsorted_segment_sum(
                    messages, row, num_segments=total_nodes)
                messages = self.edges_linear_layers(agg)
                messages = pooling_op(messages)
        else:
            if self.feats_linear_layers is not None:
                feats = pooling_op(feats)  # (total_nodes, k)
                if self.include_strain_info:
                    feats = torch.cat((feats, dE), dim=1)
                feats = self.feats_linear_layers(feats)  # (bs, k)
            if self.edges_linear_layers is not None:
                agg = unsorted_segment_sum(
                    messages, row, num_segments=total_nodes)
                messages = pooling_op(messages)
                messages = self.edges_linear_layers(messages)
        if self.feats_linear_layers is not None and \
                self.edges_linear_layers is not None:
            return torch.add(feats.squeeze(), messages.squeeze())
        elif self.feats_linear_layers is not None:
            return feats
        elif self.edges_linear_layers is not None:
            return messages
        raise RuntimeError(
            'We must either classify on feats, edges or both.')

    def process_graph(self, graph):
        y_true = graph.y
        try:
            y_true = y_true.float()
        except (AttributeError, TypeError):
            pass
        y_pred = self(graph).reshape(-1, )
        ligands = graph.lig_fname
        receptors = graph.rec_fname
        return y_pred, y_true, ligands, receptors

    @abstractmethod
    def get_embeddings(self, feats, edges, coords, edge_attributes, batch):
        """Implement code to go from input features to final node embeddings."""
        pass

    def prepare_input(self, x):
        return x.to(DEVICE)

    def unpack_graph(self, graph):
        objs = [graph.x.float().to(DEVICE), graph.edge_index.to(DEVICE),
                graph.pos.float().to(DEVICE), graph.edge_attr.to(DEVICE),
                graph.batch.to(DEVICE)]
        if self.include_strain_info:
            objs += [graph.dE.reshape(-1, 1).float().to(DEVICE),
                     graph.rmsd.reshape(-1, 1).float().to(DEVICE)]
        return tuple(objs)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
