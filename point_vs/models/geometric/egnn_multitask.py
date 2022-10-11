"""Equivariant graph neural network class with two heads."""
from torch import nn
from torch_geometric.nn import global_mean_pool

from point_vs.models.geometric.pnn_geometric_base import PygLinearPass
from point_vs.models.geometric.egnn_satorras import E_GCL
from point_vs.models.geometric.egnn_satorras import SartorrasEGNN


class MultitaskSatorrasEGNN(SartorrasEGNN):
    """Equivariant network based on the E_GCL layer."""
    # pylint: disable = R, W0201, W0613, W0211
    def build_net(
        self,
        dim_input: int,
        k: int,
        dim_output: int,
        act_fn: nn.Module = nn.SiLU(),
        num_layers: int = 4,
        residual: bool = True,
        edge_residual: bool =False,
        edge_attention: bool = False,
        normalize: bool = True,
        tanh: bool = True,
        dropout: float = 0.0,
        graphnorm: bool = True,
        update_coords: bool = True,
        permutation_invariance: bool = False,
        attention_activation_fn: str = 'sigmoid',
        node_attention: bool = False,
        node_attention_final_only: bool = False,
        edge_attention_final_only: bool = False,
        node_attention_first_only: bool = False,
        edge_attention_first_only: bool = False,
        gated_residual: bool = False,
        rezero: bool = False,
        model_task: str = 'classification',
        final_softplus: bool = False,
        **kwargs
    ):
        """
        Arguments:
            dim_input: Number of features for 'h' at the input
            k: Number of hidden features
            dim_output: Number of features for 'h' at the output
            act_fn: Non-linearity
            num_layers: Number of layer for the EGNN
            residual: Use residual connections, we recommend not changing
                this one
            edge_residual: Use residual connections for individual messages
            edge_attention: Whether using attention or not
            normalize: Normalizes the coordinates messages such that:
                instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(
                m_ij)/||x_i - x_j||
                We noticed it may help in the stability or generalization.
            tanh: Sets a tanh activation function at the output of phi_x(
                m_ij). I.e. it bounds the output of phi_x(m_ij) which definitely
                improves in stability but it may decrease in accuracy.
            dropout: dropout rate for edge/node/coordinate MLPs.
            graphnorm: apply graph normalisation.
            multi_fc: multiple fully connected layers at the end.
            update_coords: coordinates update at each layer.
            permutation_invariance: edges embeddings are summed rather than
                concatenated.
            attention_activation_fn: activation function for attention MLPs.
            node_attention: apply attention MLPs to node embeddings.
            node_attention_final_only: only apply attention to nodes in final
                layer.
            edge_attention_final_only: only apply attention to edges in final
                layer.
            node_attention_first_only: only apply attention to nodes in first
                layer.
            edge_attention_first_only: only apply attention to nodes in first
                layer.
            gated_residual: residual connections are gated by learnable scalar
                value.
            rezero: apply ReZero normalisation.
            model_task: 'classification' or 'regression'.
        """
        embedding_layers = [PygLinearPass(nn.Linear(dim_input, k),
                            return_coords_and_edges=True)]
        self.n_layers = num_layers
        self.dropout_p = dropout
        self.residual = residual
        self.edge_residual = edge_residual
        self.gated_residual = gated_residual
        self.rezero = rezero
        self.model_task = model_task

        assert not (gated_residual and rezero), \
            'gated_residual and rezero are incompatible'
        for i in range(0, num_layers):
            # apply node/edge attention or not?
            if node_attention:
                if not node_attention_first_only and not \
                        node_attention_final_only:
                    apply_node_attention = True
                elif node_attention_first_only and i == 0:
                    apply_node_attention = True
                elif node_attention_final_only and i == num_layers - 1:
                    apply_node_attention = True
                else:
                    apply_node_attention = False
            else:
                apply_node_attention = False

            if edge_attention:
                if not edge_attention_first_only and not \
                        edge_attention_final_only:
                    apply_edge_attention = True
                elif edge_attention_first_only and i == 0:
                    apply_edge_attention = True
                elif edge_attention_final_only and i == num_layers - 1:
                    apply_edge_attention = True
                else:
                    apply_edge_attention = False
            else:
                apply_edge_attention = False

            embedding_layers.append(
                E_GCL(k, k, k,
                      edges_in_d=3,
                      act_fn=act_fn,
                      residual=residual,
                      edge_attention=apply_edge_attention,
                      normalize=normalize,
                      graphnorm=graphnorm,
                      tanh=tanh, update_coords=update_coords,
                      permutation_invariance=permutation_invariance,
                      attention_activation_fn=attention_activation_fn,
                      node_attention=apply_node_attention,
                      edge_residual=edge_residual,
                      gated_residual=gated_residual,
                      rezero=rezero))

        feats_linear_layers_pose = [nn.Linear(k, 1)]
        feats_linear_layers_affinity = [nn.Linear(k, dim_output)]
        feats_linear_layers_affinity.append(
            nn.Softplus() if final_softplus else nn.ReLU())
        self.feats_linear_layers_pose = nn.Sequential(*feats_linear_layers_pose)
        self.feats_linear_layers_affinity = nn.Sequential(
            *feats_linear_layers_affinity)
        return nn.Sequential(*embedding_layers)


    def forward(self, graph):
        feats, edges, coords, edge_attributes, batch = self.unpack_graph(
            graph)
        feats, _ = self.get_embeddings(
            feats, edges, coords, edge_attributes, batch)
        feats = global_mean_pool(feats, batch)  # (total_nodes, k)
        if 'classification' in self.model_task:
            feats = self.feats_linear_layers_pose(feats)  # (bs, k)
        else:
            feats = self.feats_linear_layers_affinity(feats)
        return feats
