import torch
from egnn_pytorch import EGNN as EGNNLayer
from egnn_pytorch.egnn_pytorch import SiLU, exists
from eqv_transformer.utils import GlobalPool
from lie_conv.utils import Pass
from torch import nn

from point_vs.models.point_neural_network import PointNeuralNetwork
from point_vs.utils import get_layer_shapes


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
        feats, coors = self.egnn(feats=feats, coors=coors, mask=mask)
        return coors, feats, mask


class ThinEGNNLayer(EGNNLayer):
    def __init__(self, **kwargs):
        """The parent class (EGNN) does not store much of the information
        required to reconstruct the layer with single layer models for node,
        coordinate and edge networks - so some sneaky workarounds are
        necessary if we don't want to rewrite this layer every time the parent
        layer is updated."""

        def _find_dropout_p(model):
            """Find dropout value (if any) of first dropout layer in model."""
            for layer in model:
                if isinstance(layer, nn.Dropout):
                    return layer.p
            return 0

        super().__init__(**kwargs)

        p = _find_dropout_p(self.edge_mlp)
        edge_shapes = get_layer_shapes(self.edge_mlp)
        edge_inp_dim = edge_shapes[0][1]
        edge_out_dim = edge_shapes[-1][0]
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_inp_dim, edge_out_dim),
            nn.Dropout(p) if p > 0 else nn.Identity(),
            SiLU(),
        )

        if exists(self.node_mlp):
            p = _find_dropout_p(self.node_mlp)
            edge_shapes = get_layer_shapes(self.node_mlp)
            node_inp_dim = edge_shapes[0][1]
            node_out_dim = edge_shapes[-1][0]
            self.node_mlp = nn.Sequential(
                nn.Linear(node_inp_dim, node_out_dim),
                nn.Dropout(p) if p > 0 else nn.Identity(),
                SiLU(),
            )

        if exists(self.coors_mlp):
            p = _find_dropout_p(self.coors_mlp)
            edge_shapes = get_layer_shapes(self.coors_mlp)
            coors_inp_dim = edge_shapes[0][1]
            coors_out_dim = 1
            self.coors_mlp = nn.Sequential(
                nn.Linear(coors_inp_dim, coors_out_dim),
                nn.Dropout(p) if p > 0 else nn.Identity(),
                SiLU(),
            )

        self.apply(self.init_)


class EGNN(PointNeuralNetwork):

    # We have our own initialisation methods for EGNN
    @staticmethod
    def xavier_init(m):
        pass

    def _get_y_true(self, y):
        return y.cuda()

    def _process_inputs(self, x):
        return [i.cuda() for i in x]

    def build_net(self, dim_input, dim_output=1, k=12, nbhd=0,
                  dropout=0.0, num_layers=6, fourier_features=16,
                  norm_coords=True, norm_feats=False, thin_mlps=False,
                  **kwargs):
        m_dim = 12
        layer_class = ThinEGNNLayer if thin_mlps else EGNNLayer
        egnn = lambda: layer_class(
            dim=k, m_dim=m_dim, norm_coors=norm_coords, norm_feats=norm_feats,
            dropout=dropout, fourier_features=fourier_features,
            num_nearest_neighbors=nbhd, init_eps=1e-2)

        return nn.Sequential(
            Pass(nn.Linear(dim_input, k), dim=1),
            *[EGNNPass(egnn()) for _ in range(num_layers)],
            GlobalPool(mean=True),
            nn.Linear(k, dim_output)
        )

    def forward(self, x):
        return self.layers(x)

    def _get_min_max(self):
        """For debugging: print min and max value in each layer."""
        for i in range(3):
            print()
        network = self.layers
        res = ''
        min_ = 1e7
        max_ = -1e7
        min_abs = 1e7
        for layer in network:
            if isinstance(layer, nn.Linear):
                min_ = min(min_, float(torch.min(layer.weight.grad)))
                max_ = max(max_, float(torch.max(layer.weight.grad)))
                min_abs = min(min_abs,
                              float(torch.min(torch.abs(layer.weight.grad))))
                res += 'Linear: {0}, {1} {2}\n'.format(
                    float(torch.min(layer.weight.grad)),
                    float(torch.max(layer.weight.grad)),
                    float(torch.min(torch.abs(layer.weight.grad))))
            if isinstance(layer, Pass):
                if isinstance(layer.module, nn.Linear):
                    min_ = min(min_, float(torch.min(layer.module.weight.grad)))
                    max_ = max(max_, float(torch.max(layer.module.weight.grad)))
                    min_abs = min(min_abs, float(
                        torch.min(torch.abs(layer.module.weight.grad))))
                    res += 'Linear:{0} {1} {2}\n'.format(
                        float(torch.min(layer.module.weight.grad)),
                        float(torch.max(layer.module.weight.grad)),
                        float(torch.min(torch.abs(layer.module.weight.grad))))
            elif isinstance(layer, EGNNPass):
                layer = layer.egnn
                for network_type, network_name in zip(
                        (layer.edge_mlp, layer.node_mlp, layer.coors_mlp),
                        ('EGNN-edge', 'EGNN-node', 'EGNN-coors')):
                    for sublayer in network_type:
                        if isinstance(sublayer, nn.Linear):
                            try:
                                max_ = max(max_, float(
                                    torch.max(sublayer.weight.grad)))
                                min_ = min(min_, float(
                                    torch.min(sublayer.weight.grad)))
                                min_abs = min(min_abs, float(
                                    torch.min(torch.abs(sublayer.weight.grad))))
                                res += network_name + ': {0} {1} {2}\n'.format(
                                    float(torch.min(sublayer.weight.grad)),
                                    float(torch.max(sublayer.weight.grad)),
                                    float(
                                        torch.min(
                                            torch.abs(sublayer.weight.grad))))
                            except TypeError:
                                res += 'NONE_LAYER\n'
        return res[:-1], min_, max_, min_abs
