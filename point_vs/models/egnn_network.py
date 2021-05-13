from egnn_pytorch import EGNN as EGNNLayer
from egnn_pytorch.egnn_pytorch import SiLU
from eqv_transformer.utils import GlobalPool
from lie_conv.utils import Pass
from torch import nn

from point_vs.models.point_neural_network import PointNeuralNetwork


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
        egnn = lambda: EGNNLayer(
            dim=k, m_dim=m_dim, norm_coors=norm_coords, norm_feats=norm_feats,
            dropout=dropout, fourier_features=fourier_features,
            num_nearest_neighbors=nbhd, init_eps=1e-2)

        edge_input_dim = (fourier_features * 2) + (k * 2) + 1

        egnn_layers = [EGNNPass(egnn()) for _ in range(num_layers)]

        # Thinner MLPs for updating coords, edges and nodes
        if thin_mlps:
            for layer in egnn_layers:
                dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
                layer.edge_mlp = nn.Sequential(
                    nn.Linear(edge_input_dim, m_dim),
                    dropout,
                    SiLU(),
                )
                layer.node_mlp = nn.Sequential(
                    nn.Linear(k + m_dim, k),
                    dropout,
                    SiLU(),
                )
                layer.coors_mlp = nn.Sequential(
                    nn.Linear(m_dim, 1),
                    dropout,
                    SiLU(),
                )

        return nn.Sequential(
            Pass(nn.Linear(dim_input, k), dim=1),
            *egnn_layers,
            GlobalPool(mean=True),
            nn.Linear(k, dim_output)
        )

    def forward(self, x):
        return self.layers(x)
