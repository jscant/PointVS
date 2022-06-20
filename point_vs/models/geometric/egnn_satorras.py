import torch
from torch import nn
from torch.nn import functional as F, TransformerEncoder
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import dropout_adj

from point_vs.models.geometric.pnn_geometric_base import PNNGeometricBase, \
    PygLinearPass, unsorted_segment_sum, unsorted_segment_mean
from point_vs.utils import to_numpy


class E_GCL(nn.Module):
    """Modified from https://github.com/vgsatorras/egnn"""

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,
                 act_fn=nn.SiLU(), residual=True, edge_residual=False,
                 edge_attention=False, normalize=False, coords_agg='mean',
                 tanh=False, graphnorm=False, update_coords=True,
                 permutation_invariance=False, node_attention=False,
                 attention_activation_fn='sigmoid',
                 gated_residual=False, rezero=False):
        assert not (gated_residual and rezero), 'gated_residual and rezero ' \
                                                'are incompatible'
        super(E_GCL, self).__init__()
        input_edge = input_nf if permutation_invariance else input_nf * 2
        self.gated_residual = gated_residual
        self.rezero = rezero
        self.residual = residual
        self.edge_residual = edge_residual
        self.edge_attention = edge_attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.use_coords = update_coords
        self.permutation_invariance = permutation_invariance
        self.att_val = None
        self.node_attention = node_attention
        self.node_att_val = None
        self.intermediate_coords = None
        self.hidden_nf = hidden_nf
        attention_activation = {
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'silu': nn.SiLU
        }[attention_activation_fn]
        self.attention_activation = attention_activation
        edge_coords_nf = 1

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

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
            nn.Tanh() if tanh else nn.Identity()
        )

        if self.edge_attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                attention_activation())

        if self.node_attention:
            self.node_att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                attention_activation())

        if self.rezero:
            if self.edge_residual:
                self.edge_gate_parameter = nn.parameter.Parameter(
                    torch.zeros(1, ), requires_grad=True)
            if self.residual:
                self.node_gate_parameter = nn.parameter.Parameter(
                    torch.zeros(1, ), requires_grad=True)
        elif self.gated_residual:
            if self.edge_residual:
                self.edge_gate_parameter = nn.parameter.Parameter(
                    0.5 * torch.ones(1, ), requires_grad=True)
            if self.residual:
                self.node_gate_parameter = nn.parameter.Parameter(
                    0.5 * torch.ones(1, ), requires_grad=True)

    def edge_model(self, source, target, radial, edge_attr):
        if self.permutation_invariance:
            inp = [torch.add(source, target), radial]
        else:
            inp = [source, target, radial]
        if edge_attr is not None:
            inp.append(edge_attr)
        out = torch.cat(inp, dim=1)
        out = self.edge_mlp(out)
        return out

    def node_model(self, x, edge_index, m_ij):
        row, col = edge_index

        if self.edge_attention:
            att_val = self.att_mlp(m_ij)
            self.att_val = to_numpy(att_val)
            agg = unsorted_segment_sum(
                att_val * m_ij, row, num_segments=x.size(0))
        else:
            agg = unsorted_segment_sum(m_ij, row, num_segments=x.size(0))

        agg = torch.cat([x, agg], dim=1)

        # Eq. 6: h_i = phi_h(h_i, m_i)
        out = self.node_mlp(agg)
        if self.node_attention:
            att_val = self.node_att_mlp(out)
            out = out * att_val
            self.node_att_val = to_numpy(att_val)
        if self.residual:
            if self.rezero:
                out = x + self.node_gate_parameter * out
            elif self.gated_residual:
                gate_val = F.relu(self.node_gate_parameter)
                out = gate_val * out + (1 - gate_val) * x
            else:
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
        self.intermediate_coords = to_numpy(coord)
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, edge_messages=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        if self.edge_residual and edge_messages is not None:
            if self.rezero:
                edge_feat = edge_messages + self.edge_gate_parameter * edge_feat
            elif self.gated_residual:
                gate_val = F.relu(self.edge_gate_parameter)
                edge_feat = gate_val * edge_feat + (
                        1 - gate_val) * edge_messages
            else:
                edge_feat = edge_feat + edge_messages
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat)

        return h, coord, edge_attr, edge_feat

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class SartorrasEGNN(PNNGeometricBase):
    def build_net(self, dim_input, k, dim_output,
                  act_fn=nn.SiLU(), num_layers=4, residual=True,
                  edge_residual=False, edge_attention=False, normalize=True,
                  tanh=True, dropout=0, graphnorm=True, classify_on_edges=False,
                  classify_on_feats=True, multi_fc=False, update_coords=True,
                  permutation_invariance=False,
                  attention_activation_fn='sigmoid',
                  node_attention=False, node_attention_final_only=False,
                  edge_attention_final_only=False,
                  node_attention_first_only=False,
                  edge_attention_first_only=False,
                  gated_residual=False, rezero=False,
                  model_task='classification',
                  include_strain_info=False,
                  final_softplus=False,
                  transformer_encoder=False,
                  d_model=512,
                  dim_feedforward=2048,
                  n_heads=8,
                  transformer_at_end=True,
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
            dropout:
            graphnorm:
            classify_on_edges:
            classify_on_feats:
            multi_fc:
            update_coords:
            permutation_invariance:
            attention_activation_fn:
            node_attention:
            node_attention_final_only:
            edge_attention_final_only:
            node_attention_first_only:
            edge_attention_first_only:
            gated_residual:
            rezero:
            model_task:
        """
        if transformer_encoder and not transformer_at_end:
            layers = [PygLinearPass(nn.Linear(dim_input, d_model),
                                    return_coords_and_edges=True)]
            transformer_encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                dropout=0.0)
            transformer_encoder_block = torch.nn.TransformerEncoder(
                transformer_encoder_layer, 6)
            layers.append(transformer_encoder_block)
            layers.append(nn.Linear(d_model, k))
            layers = nn.Sequential(*layers)
            layers.apply(init_weights)
        else:
            layers = [PygLinearPass(nn.Linear(dim_input, k),
                                    return_coords_and_edges=True)]
        self.n_layers = num_layers
        self.dropout_p = dropout
        self.residual = residual
        self.edge_residual = edge_residual
        self.gated_residual = gated_residual
        self.rezero = rezero
        self.model_task = model_task
        self.include_strain_info = include_strain_info
        self.transformer_encoder = transformer_encoder
        self.transformer_at_end = transformer_at_end

        assert classify_on_feats or classify_on_edges, \
            'We must use either or both of classify_on_feats and ' \
            'classify_on_edges'
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

            layers.append(E_GCL(k, k, k,
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

        if include_strain_info:
            k += 1
        if multi_fc:  # check the order of these..??!
            fc_layer_dims = ((k, 32), (32, 16), (16, dim_output))
        else:
            fc_layer_dims = ((k, dim_output),)

        if transformer_encoder and False:
            transformer_encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                dropout=0.1)
            transformer_encoder_block = torch.nn.TransformerEncoder(
                transformer_encoder_layer, 6)
            self.project_to_d_model = nn.Linear(k, d_model)
            self.attention_block = transformer_encoder_block
            self.feats_linear_layers = nn.Linear(d_model, dim_output)
        elif transformer_encoder and transformer_at_end:
            transformer_encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                dropout=0.0)
            transformer_encoder_block = torch.nn.TransformerEncoder(
                transformer_encoder_layer, 6)
            transformer_encoder_block.apply(init_weights)
            self.feats_linear_layers = nn.Sequential(
                nn.Linear(k, d_model),
                transformer_encoder_block,
                nn.Linear(d_model, dim_output)
            )
        else:
            feats_linear_layers = []
            edges_linear_layers = []
            for idx, (in_dim, out_dim) in enumerate(fc_layer_dims):
                feats_linear_layers.append(nn.Linear(in_dim, out_dim))
                edges_linear_layers.append(nn.Linear(in_dim, out_dim))
                if idx < len(fc_layer_dims) - 1:
                    feats_linear_layers.append(nn.SiLU())
                    edges_linear_layers.append(nn.SiLU())
            if final_softplus:
                feats_linear_layers.append(nn.ReLU())
            if classify_on_feats:
                self.feats_linear_layers = nn.Sequential(*feats_linear_layers)
            if classify_on_edges:
                self.edges_linear_layers = nn.Sequential(*edges_linear_layers)
        return nn.Sequential(*layers)

    def get_embeddings(self, feats, edges, coords, edge_attributes, batch):
        if self.dropout_p > 0:
            edges, edge_attributes = dropout_adj(
                edges, edge_attributes, self.dropout_p, force_undirected=True,
                training=self.training)
        edge_messages = None
        next_linear = False
        for i in self.layers:
            if not self.transformer_at_end and (
                    isinstance(i, torch.nn.TransformerEncoder) or next_linear):
                feats = i(feats)
                next_linear = True
                if not isinstance(i, torch.nn.TransformerEncoder):
                    next_linear = False
            else:
                x = i(
                    h=feats, edge_index=edges, coord=coords,
                    edge_attr=edge_attributes, edge_messages=edge_messages)
                feats, coords, edge_attributes, edge_messages = i(
                    h=feats, edge_index=edges, coord=coords,
                    edge_attr=edge_attributes, edge_messages=edge_messages)
        return feats, edge_messages
