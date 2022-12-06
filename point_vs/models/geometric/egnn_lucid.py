from typing import List

import torch
import torch_geometric
from egnn_pytorch.egnn_pytorch import CoorsNorm, fourier_encode_dist, exists
from einops import rearrange
from torch import nn, Tensor
from torch.nn import SiLU
from torch_geometric.nn import GraphNorm, MessagePassing
from torch_geometric.typing import Adj, OptTensor, Size

from point_vs.global_objects import DEVICE
from point_vs.models.geometric.pnn_geometric_base import PNNGeometricBase, \
    PygLinearPass

    
class EGNN_Sparse(MessagePassing):
    """ Different from the above since it separates the edge assignment
        from the computation (this allows for great reduction in time and
        computations when the graph is locally or sparse connected).
        * aggr: one of ["add", "mean", "max"]
    """

    def __init__(
            self,
            feats_dim,
            pos_dim=3,
            edge_attr_dim=0,
            m_dim=16,
            fourier_features=0,
            soft_edge=0,
            norm_feats=False,
            norm_coors=False,
            norm_coors_scale_init=1e-2,
            update_feats=True,
            update_coors=True,
            dropout=0.,
            coor_weights_clamp_value=None,
            aggr="add",
            **kwargs
    ):
        assert aggr in {'add', 'sum', 'max',
                        'mean'}, 'pool method must be a valid option'
        assert update_feats or update_coors, 'you must update either ' \
                                             'features, coordinates, or both'
        kwargs.setdefault('aggr', aggr)
        super(EGNN_Sparse, self).__init__(**kwargs)
        # model params
        self.fourier_features = fourier_features
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_coors = update_coors
        self.update_feats = update_feats
        self.coor_weights_clamp_value = None
        self.att_val = None

        self.edge_input_dim = (fourier_features * 2) + edge_attr_dim + 1 + (
                feats_dim * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        #  EDGES
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            SiLU()
        )

        self.edge_weight = nn.Sequential(nn.Linear(m_dim, 1),
                                         nn.Sigmoid()
                                         ) if soft_edge else None

        # NODES - can't do identity in node_norm bc pyg expects 2 inputs,
        # but identity expects 1.
        self.node_norm = torch_geometric.nn.norm.LayerNorm(
            feats_dim) if norm_feats else None
        self.coors_norm = CoorsNorm(
            scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        ) if update_feats else None

        #  COORS
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            self.dropout,
            SiLU(),
            nn.Linear(self.m_dim * 4, 1)
        ) if update_coors else None

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN
            # with greater depths
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, h: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, batch: Adj = None,
                angle_data: List = None, size: Size = None) -> Tensor:
        """ Inputs:
            * x: (n_points, d) where d is pos_dims + feat_dims
            * edge_index: (n_edges, 2)
            * edge_attr: tensor (n_edges, n_feats) excluding basic distance
            feats.
            * batch: (n_points,) long tensor. specifies xloud belonging for
            each point
            * angle_data: list of tensors (levels, n_edges_i, n_length_path)
            long tensor.
            * size: None
        """
        coors, feats = h[:, :self.pos_dim], h[:, self.pos_dim:]

        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist,
                                           num_encodings=self.fourier_features)
            rel_dist = rearrange(rel_dist, 'n () d -> n d')

        if exists(edge_attr):
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist

        hidden_out, coors_out = self.propagate(edge_index, x=feats,
                                               edge_attr=edge_attr_feats,
                                               coors=coors, rel_coors=rel_coors,
                                               batch=batch)
        return torch.cat([coors_out, hidden_out], dim=-1)

    def message(self, x_i, x_j, edge_attr) -> Tensor:
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """The initial call to start propagating messages.
            Args:
            `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional) if none, the size will be inferred
                and assumed to be quadratic.
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)

        #  get messages
        m_ij = self.message(**msg_kwargs)

        # update coors if specified
        if self.update_coors:
            coor_wij = self.coors_mlp(m_ij)
            # clamp if arg is set
            if self.coor_weights_clamp_value:
                coor_weights_clamp_value = self.coor_weights_clamp_value
                coor_wij.clamp_(
                    min=-coor_weights_clamp_value, max=coor_weights_clamp_value)

            # normalize if needed
            kwargs["rel_coors"] = self.coors_norm(kwargs["rel_coors"])

            mhat_i = self.aggregate(coor_wij * kwargs["rel_coors"],
                                    **aggr_kwargs)
            coors_out = kwargs["coors"] + mhat_i
        else:
            coors_out = kwargs["coors"]

        # update feats if specified
        if self.update_feats:
            # weight the edges if arg is passed
            if self.soft_edge:
                self.att_val = self.edge_weight(m_ij)
                m_ij = m_ij * self.att_val
            m_i = self.aggregate(m_ij, **aggr_kwargs)

            hidden_feats = self.node_norm(kwargs["x"], kwargs[
                "batch"]) if self.node_norm else kwargs["x"]
            hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
            hidden_out = kwargs["x"] + hidden_out
        else:
            hidden_out = kwargs["x"]

        # return tuple
        return self.update((hidden_out, coors_out), **update_kwargs)

    def __repr__(self):
        return "E(n)-GNN Layer for Graphs " + str(self.__dict__)


class PygLucidEGNN(PNNGeometricBase):
    """Pytorch geometric version of LucidRains' EGNN implementation."""

    def build_net(self, dim_input, k, dim_output, num_layers=4, dropout=0.0,
                  norm_coords=True, norm_feats=True, fourier_features=0,
                  attention=False, tanh=True, update_coords=True,
                  thick_attention=False, graphnorm=False,
                  thin_mlps=False, node_final_act=False, **kwargs):
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
                    GraphNorm(k) if graphnorm else nn.Identity(),
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
        h = torch.cat([coords, feats], dim=-1).to(DEVICE)
        for i in self.layers:
            h = i(h=h, edge_index=edges, edge_attr=edge_attributes, batch=batch)
        return h[:, 3:], None
