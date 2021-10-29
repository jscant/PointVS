import torch
from egnn_pytorch.egnn_pytorch_geometric import EGNN_Sparse
from torch import nn
from torch.nn import SiLU
from torch_geometric.nn import GraphNorm

from point_vs.models.geometric.pnn_geometric_base import PNNGeometricBase, \
    PygLinearPass


class PygLucidEGNN(PNNGeometricBase):
    """Pytorch geometric version of LucidRains' EGNN implementation."""

    def build_net(self, dim_input, k, dim_output, num_layers=4, dropout=0.0,
                  norm_coords=True, norm_feats=True, fourier_features=0,
                  attention=False, tanh=True, update_coords=True,
                  linear_gap=False, thick_attention=False, graphnorm=False,
                  thin_mlps=False, node_final_act=False, **kwargs):
        self.linear_gap = linear_gap
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
                update_coors=update_coords,
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
                    nn.Tanh() if tanh else nn.Identity()
                )
            if thick_attention:
                layer.edge_weight = nn.Sequential(
                    nn.Linear(k, k),
                    nn.SiLU(),
                    nn.Linear(k, 1),
                    nn.Sigmoid()
                )
            if thin_mlps:
                layer.node_mlp = nn.Sequential(
                    nn.Linear(k + k, k),
                    layer.dropout,
                    GraphNorm(k * 2) if graphnorm else nn.Identity(),
                    nn.SiLU() if node_final_act else nn.Identity()
                )
                layer.coors_mlp = nn.Sequential(
                    nn.Linear(k, 1),
                    layer.dropout,
                    nn.Tanh() if tanh else nn.Identity()
                )
            else:
                layer.node_mlp = nn.Sequential(
                    nn.Linear(k + k, k * 2),
                    layer.dropout,
                    GraphNorm(k * 2) if graphnorm else nn.Identity(),
                    SiLU(),
                    nn.Linear(k * 2, k),
                    nn.SiLU() if node_final_act else nn.Identity()
                )
                layer.coors_mlp = nn.Sequential(
                    nn.Linear(k, k * 4),
                    layer.dropout,
                    SiLU(),
                    nn.Linear(layer.m_dim * 4, 1),
                    nn.Tanh() if tanh else nn.Identity()
                )
                layer.edge_mlp = nn.Sequential(
                    nn.Linear(layer.edge_input_dim, layer.edge_input_dim * 2),
                    layer.dropout,
                    SiLU(),
                    nn.Linear(layer.edge_input_dim * 2, k),
                    SiLU()
                )
            layers.append(layer)
        self.feats_linear_layers = nn.Sequential(nn.Linear(k, dim_output))
        return nn.Sequential(*layers)

    def get_embeddings(self, feats, edges, coords, edge_attributes, batch):
        x = torch.cat([coords, feats], dim=-1).cuda()
        for i in self.layers:
            x = i(x=x, edge_index=edges, edge_attr=edge_attributes, batch=batch)
        return x[:, 3:], None
