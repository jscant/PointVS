import torch
from egnn_pytorch.egnn_pytorch_geometric import EGNN_Sparse
from torch import nn
from torch.nn import SiLU

from point_vs.models.point_neural_network_pyg import PygPointNeuralNetwork, \
    PygLinearPass


class PygLucidEGNN(PygPointNeuralNetwork):
    """Pytorch geometric version of LucidRains' EGNN implementation."""

    def build_net(self, dim_input, k, dim_output, num_layers=4, dropout=0.0,
                  norm_coords=True, norm_feats=True, fourier_features=0,
                  attention=False, tanh=True, linear_gap=False, **kwargs):
        layers = [PygLinearPass(
            nn.Linear(dim_input, k), feats_appended_to_coords=True)]
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
                aggr='mean',
            )
            if tanh:
                # Add a Tanh for stability
                layer.coors_mlp = nn.Sequential(
                    nn.Linear(k, k * 4),
                    layer.dropout,
                    SiLU(),
                    nn.Linear(layer.m_dim * 4, 1),
                    nn.Tanh()
                )
            layers.append(layer)
        layers.append(PygLinearPass(nn.Linear(k, dim_output)))
        return nn.Sequential(*layers)

    def get_embeddings(self, feats, edges, coords, edge_attributes, batch):
        x = torch.cat([coords, feats], dim=-1).cuda()
        for i in self.layers[:-1]:
            x = i(x=x, edge_index=edges, edge_attr=edge_attributes, batch=batch)
        return x[:, 3:]
