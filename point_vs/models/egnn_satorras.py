"""Directly copied from https://github.com/vgsatorras/egnn/blob
/3c079e7267dad0aa6443813ac1a12425c3717558/models/egnn_clean/egnn_clean.py"""

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

from point_vs.models.point_neural_network_pyg import PygPointNeuralNetwork


class EGNNPass(nn.Module):
    def __init__(self, egnn):
        super().__init__()
        self.egnn = egnn

    def forward(self, x):
        if len(x) == 2:
            coors, feats = x
            mask = None
        else:
            coors, feats, mask = x
        feats, coors = self.egnn(h=feats, coors=coors, mask=mask)
        return coors, feats, mask


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,
                 act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, m_ij):
        row, col = edge_index

        agg = unsorted_segment_sum(m_ij, row, num_segments=x.size(0))
        agg = torch.cat([x, agg], dim=1)

        # Eq. 6: h_i = phi_h(h_i, m_i)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord += agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat)

        return h, coord, edge_attr


class SartorrasEGNN(PygPointNeuralNetwork):
    def build_net(self, dim_input, k, dim_output,
                  act_fn=nn.SiLU(), num_layers=4, residual=True,
                  attention=False, normalize=True, tanh=True,
                  **kwargs):
        """
        Arguments:
            dim_input: Number of features for 'h' at the input
            k: Number of hidden features
            dim_output: Number of features for 'h' at the output
            act_fn: Non-linearity
            num_layers: Number of layer for the EGNN
            residual: Use residual connections, we recommend not changing
                this one
            attention: Whether using attention or not
            normalize: Normalizes the coordinates messages such that:
                instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(
                m_ij)/||x_i - x_j||
                We noticed it may help in the stability or generalization.
            tanh: Sets a tanh activation function at the output of phi_x(
                m_ij). I.e. it bounds the output of phi_x(m_ij) which definitely
                improves in stability but it may decrease in accuracy.
        """
        layers = [nn.Linear(dim_input, k)]
        self.n_layers = num_layers
        for i in range(0, num_layers):
            layers.append(E_GCL(k, k, k,
                                edges_in_d=3,
                                act_fn=act_fn,
                                residual=residual,
                                attention=attention,
                                normalize=normalize,
                                tanh=tanh))
        layers.append(nn.Linear(k, dim_output))
        for idx, layer in enumerate(layers):
            self.add_module(str(idx) + '_', layer)
        return nn.Sequential(*layers)

    def forward(self, graph):
        feats = graph.x.float().cuda()
        edges = graph.edge_index.cuda()
        coords = graph.pos.float().cuda()
        edge_attributes = graph.edge_attr.cuda()

        feats = self._modules['0_'](feats)
        for i in range(1, self.n_layers + 1):
            feats, coords, edge_attributes = self._modules[str(i) + '_'](
                feats, edges, coords, edge_attr=edge_attributes)
        feats = self._modules[str(self.n_layers + 1) + '_'](feats)
        feats = global_mean_pool(feats, graph.batch.cuda())
        return feats


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
