import torch
from egnn_pytorch.egnn_pytorch_geometric import EGNN_Sparse
from torch import nn
from torch.nn import SiLU
from torch_geometric.nn import global_mean_pool

from point_vs.models.point_neural_network_pyg import PygPointNeuralNetwork


class PygLucidEGNN(PygPointNeuralNetwork):
    """Pytorch geometric version of LucidRains' EGNN implementation."""

    def build_net(self, dim_input, k, dim_output, num_layers=4, dropout=0.0,
                  norm_coords=True, norm_feats=True, fourier_features=0,
                  attention=False, tanh=True, **kwargs):
        layers = [nn.Linear(dim_input, k)]
        for i in range(0, num_layers):
            layer = EGNN_Sparse(
                k,
                pos_dim=3,
                edge_attr_dim=3,
                m_dim=k,
                fourier_features=fourier_features,
                soft_edge=int(attention),
                norm_feats=norm_feats,
                norm_coors=norm_coords,
                norm_coors_scale_init=1e-2,
                update_feats=True,
                update_coors=True,
                dropout=dropout,
                coor_weights_clamp_value=2.0,
                aggr='add',
            )
            if tanh:
                # Add a Tanh for stability
                layer.coors_mlp = nn.Sequential(
                    nn.Linear(k, k * 4),
                    layer.dropout,
                    SiLU(),
                    nn.Linear(layer.mdim * 4, 1),
                    nn.Tanh()
                )
                layers.append(layer)
        layers.append(nn.Linear(k, dim_output))
        for idx, layer in enumerate(layers):
            self.add_module(str(idx) + '_', layer)
        return nn.Sequential(*layers)

    def forward(self, graph):
        feats = graph.x.float().cuda()
        edges = graph.edge_index.cuda()
        coords = graph.pos.float().cuda()
        edge_attributes = graph.edge_attr.cuda()
        batch = graph.batch.cuda()

        feats = self._modules['0_'](feats)
        x = torch.cat([coords, feats], dim=-1).cuda()
        for i in range(1, self.n_layers + 1):
            x = self._modules[str(i) + '_'](
                x=x, edge_index=edges, edge_attr=edge_attributes, batch=batch)
        feats = self._modules[str(self.n_layers + 1) + '_'](x[:, 3:])
        feats = global_mean_pool(feats, graph.batch.cuda())
        return feats
