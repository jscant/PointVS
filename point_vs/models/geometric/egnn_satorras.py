import torch
from torch import nn
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import dropout_adj

from point_vs.models.geometric.pnn_geometric_base import PNNGeometricBase, \
    PygLinearPass


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
    """Directly copied from https://github.com/vgsatorras/egnn"""

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,
                 act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False,
                 graphnorm=False, update_coords=True):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.use_coords = update_coords
        edge_coords_nf = 1 if update_coords else 0

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            GraphNorm(hidden_nf) if graphnorm else nn.Identity(),
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
        inp = [source, target]
        if self.use_coords:
            inp.append(radial)
        if edge_attr is not None:
            inp.append(edge_attr)
        out = torch.cat(inp, dim=1)
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
        if not self.use_coords:
            return coord
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

    def forward(self, x, edge_index, coord, edge_attr=None):
        row, col = edge_index
        if self.use_coords:
            radial, coord_diff = self.coord2radial(edge_index, coord)
        else:
            radial, coord_diff = None, None

        edge_feat = self.edge_model(x[row], x[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        x, agg = self.node_model(x, edge_index, edge_feat)

        return x, coord, edge_attr


class SartorrasEGNN(PNNGeometricBase):
    def build_net(self, dim_input, k, dim_output,
                  act_fn=nn.SiLU(), num_layers=4, residual=True,
                  attention=False, normalize=True, tanh=True, dropout=0,
                  graphnorm=True, **kwargs):
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
        layers = [PygLinearPass(nn.Linear(dim_input, k),
                                return_coords_and_edges=True)]
        self.n_layers = num_layers
        self.dropout_p = dropout
        for i in range(0, num_layers):
            layers.append(E_GCL(k, k, k,
                                edges_in_d=3,
                                act_fn=act_fn,
                                residual=residual,
                                attention=attention,
                                normalize=normalize,
                                graphnorm=graphnorm,
                                tanh=tanh))
        layers.append(PygLinearPass(nn.Linear(k, dim_output),
                                    return_coords_and_edges=False))
        return nn.Sequential(*layers)

    def get_embeddings(self, feats, edges, coords, edge_attributes, batch):
        if self.dropout_p > 0:
            edges, edge_attributes = dropout_adj(
                edges, edge_attributes, self.dropout_p, force_undirected=True,
                training=self.training)
        for i in self.layers[:-1]:
            feats, coords, edge_attributes = i(
                x=feats, edge_index=edges, coord=coords,
                edge_attr=edge_attributes)
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
