import torch
from eqv_transformer.eqv_attention import GlobalPool, \
    EquivariantTransformerBlock
from eqv_transformer.utils import Swish
from lie_conv.lieGroups import SE3
from lie_conv.utils import Pass
from torch import nn

from point_vs.models.point_neural_network import PointNeuralNetwork


class EquivariantTransformer(PointNeuralNetwork):
    """Adapted from https://github.com/anonymous-code-0/lie-transformer"""

    def _get_y_true(self, y):
        return y.cuda()

    def _process_inputs(self, x):
        return tuple([ten.cuda() for ten in x])

    def build_net(self, dim_input, dim_output, dim_hidden, num_layers,
                  num_heads, act='relu', global_pool=True,
                  global_pool_mean=True,
                  group=SE3(0.2), liftsamples=1, block_norm="layer_pre",
                  output_norm="none", kernel_norm="none", kernel_type="mlp",
                  kernel_dim=16, kernel_act="swish", nbhd=0, fill=1.0,
                  attention_fn="norm_exp", feature_embed_dim=None,
                  max_sample_norm=None, lie_algebra_nonlinearity=None,
                  pooling_only=False,
                  dropout=0, **kwargs):

        if isinstance(dim_hidden, int):
            dim_hidden = [dim_hidden] * (num_layers + 1)

        if isinstance(num_heads, int):
            num_heads = [num_heads] * num_layers

        attention_block = lambda dim, n_head: EquivariantTransformerBlock(
            dim, n_head, group, block_norm=block_norm, kernel_norm=kernel_norm,
            kernel_type=kernel_type, kernel_dim=kernel_dim,
            kernel_act=kernel_act, mc_samples=nbhd, fill=fill,
            attention_fn=attention_fn, feature_embed_dim=feature_embed_dim,
        )

        activation_fns = {
            'swish': Swish,
            'relu': nn.ReLU,
            'softplus': nn.Softplus,
        }
        activation_fn = activation_fns[act]

        if output_norm == 'batch':
            norm1 = nn.BatchNorm1d(dim_hidden[-1])
            norm2 = nn.BatchNorm1d(dim_hidden[-1])
            norm3 = nn.BatchNorm1d(dim_hidden[-1])
        elif output_norm == 'layer':
            norm1 = nn.LayerNorm(dim_hidden[-1])
            norm2 = nn.LayerNorm(dim_hidden[-1])
            norm3 = nn.LayerNorm(dim_hidden[-1])
        elif output_norm == 'none':
            norm1 = nn.Sequential()
            norm2 = nn.Sequential()
            norm3 = nn.Sequential()
        else:
            raise ValueError('{} is not a valid norm type.'.format(output_norm))

        layers = nn.Sequential(
            Pass(nn.Linear(dim_input, dim_hidden[0]), dim=1),
            *[
                attention_block(dim_hidden[i], num_heads[i])
                for i in range(num_layers)
            ],
            GlobalPool(mean=global_pool_mean),
            nn.Linear(dim_hidden[-1], dim_output)
        )

        self.group = group
        self.liftsamples = liftsamples
        self.max_sample_norm = max_sample_norm

        self.lie_algebra_nonlinearity = lie_algebra_nonlinearity
        if lie_algebra_nonlinearity is not None:
            if lie_algebra_nonlinearity == 'tanh':
                self.lie_algebra_nonlinearity = nn.Tanh()
            else:
                raise ValueError('{} is not a supported nonlinearity'.format(
                    lie_algebra_nonlinearity))

        return layers

    def forward(self, x):
        if self.max_sample_norm is None:
            lifted_data = self.group.lift(x, self.liftsamples)
        else:
            lifted_data = [
                torch.tensor(self.max_sample_norm * 2, device=x[0].device),
                0,
                0,
            ]
            while lifted_data[0].norm(dim=-1).max() > self.max_sample_norm:
                lifted_data = self.group.lift(x, self.liftsamples)

        if self.lie_algebra_nonlinearity is not None:
            lifted_data = list(lifted_data)
            pairs_norm = lifted_data[0].norm(dim=-1) + 1e-6
            lifted_data[0] = lifted_data[0] * (
                    self.lie_algebra_nonlinearity(pairs_norm / 7) / pairs_norm
            ).unsqueeze(-1)

        return self.layers(lifted_data)
