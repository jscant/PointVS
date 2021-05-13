from egnn_pytorch import EGNN as EGNNLayer
from egnn_pytorch.egnn_pytorch import SiLU, CoorsNorm
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


class ThinEGNNLayer(EGNNLayer):
    def __init__(
            self,
            dim,
            edge_dim=0,
            m_dim=16,
            fourier_features=0,
            num_nearest_neighbors=0,
            dropout=0.0,
            init_eps=1e-3,
            norm_feats=False,
            norm_coors=False,
            update_feats=True,
            update_coors=True,
            only_sparse_neighbors=False,
            valid_radius=float('inf'),
            m_pool_method='sum',
    ):
        super(EGNNLayer, self).__init__()
        assert m_pool_method in {'sum', 'mean'}, \
            'pool method must be either sum or mean'
        assert update_feats or update_coors, \
            'you must update either features, coordinates, or both'

        self.fourier_features = fourier_features

        edge_input_dim = (fourier_features * 2) + (dim * 2) + edge_dim + 1
        dropout_fn = lambda: nn.Dropout(
            dropout) if dropout > 0 else nn.Identity()

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, m_dim),
            dropout_fn(),
            SiLU(),
        )

        self.node_norm = nn.LayerNorm(dim) if norm_feats else nn.Identity()
        self.coors_norm = CoorsNorm() if norm_coors else nn.Identity()

        self.m_pool_method = m_pool_method

        self.node_mlp = nn.Sequential(
            nn.Linear(dim + m_dim, dim),
            dropout_fn(),
            SiLU(),
        ) if update_feats else None

        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, 1),
            dropout_fn(),
            SiLU(),
        ) if update_coors else None

        self.num_nearest_neighbors = num_nearest_neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_radius = valid_radius

        self.init_eps = init_eps
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
