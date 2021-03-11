import time
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from egnn_pytorch import EGNN
from egnn_pytorch.egnn_pytorch import fourier_encode_dist, exists
from einops import rearrange, repeat
from eqv_transformer.eqv_attention import GlobalPool, \
    EquivairantMultiheadAttention
from eqv_transformer.multihead_neural import (
    MLP, )
from eqv_transformer.utils import Swish
from lie_conv.lieConv import BottleBlock, LieConv
from lie_conv.lieGroups import SE3
from lie_conv.masked_batchnorm import MaskBatchNormNd
from lie_conv.utils import Expression, Pass
from torch import nn, einsum
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.nn.init import _no_grad_normal_

from acs import utils
from acs.model import ReparamFullDense, LocalReparamDense
from layers import EGNNBatchNorm, EGNNGlobalPool
from lieconv_utils import get_eta, format_time, print_with_overwrite


class PointNeuralNetwork(nn.Module):

    def __init__(self, save_path, learning_rate, weight_decay=None,
                 wandb_project=None, wandb_run=None, **model_kwargs):
        super().__init__()
        self.batch = 0
        self.epoch = 0
        self.losses = []
        self.final_activation = nn.CrossEntropyLoss()
        self.save_path = Path(save_path).expanduser()
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.save_path / 'predictions.txt'

        self.loss_plot_file = self.save_path / 'loss.png'

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.translated_actives = model_kwargs.get('translated_actives', None)
        self.n_translated_actives = model_kwargs.get('n_translated_actives', 0)

        self.loss_log_file = self.save_path / 'loss.log'

        self.cross_entropy = nn.CrossEntropyLoss()

        self.wandb_project = wandb_project
        self.wandb_path = self.save_path / 'wandb_{}'.format(wandb_project)
        self.wandb_run = wandb_run

        self.build_net(**model_kwargs)
        self.optimiser = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=weight_decay)

        with open(save_path / 'model_kwargs.yaml', 'w') as f:
            yaml.dump(model_kwargs, f)

        self.apply(self.xavier_init)
        self.cuda()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @abstractmethod
    def _get_y_true(self, y):
        pass

    def _process_inputs(self, x):
        return x.cuda()

    def _get_loss(self, y_true, y_pred, scale=None):
        loss = self.cross_entropy(y_pred, y_true.long())
        self.bce_loss = loss
        return loss

    def forward_pass(self, x):
        return self.forward(x)

    def optimise(self, data_loader, epochs=1, opt_cycle=-1):
        """Train the network.

        Trains the neural network. Displays training information and plots the
        loss. All figures and logs are saved to save_path.

        Arguments:
            data_loader: pytorch DataLoader object for training
            epochs: number of complete training cycles
            opt_cycle: (for active learning): active learning cycle
        """
        start_time = time.time()
        total_iters = epochs * len(data_loader)
        log_interval = 10
        global_iter = 0
        self.train()
        if data_loader.batch_size == 1:
            aggrigation_interval = 32
        else:
            aggrigation_interval = 1
        decoy_mean_pred, active_mean_pred = [], []
        loss = 0.0
        bce_loss = 0.0
        for self.epoch in range(epochs):
            for self.batch, (x, y_true, ligands, receptors) in enumerate(
                    data_loader):

                x = self._process_inputs(x)
                y_true = self._get_y_true(y_true).cuda()
                y_pred = self.forward_pass(x)

                y_true_np = y_true.cpu().detach().numpy()
                y_pred_np = nn.Softmax(dim=1)(y_pred).cpu().detach().numpy()

                active_idx = (np.where(y_true_np > 0.5), 1)
                decoy_idx = (np.where(y_true_np < 0.5), 1)

                scale = len(y_true) / len(data_loader)
                loss += self._get_loss(y_true, y_pred, scale)
                bce_loss += float(self.bce_loss)

                if len(active_idx[0][0]):
                    active_mean_pred.append(np.mean(y_pred_np[active_idx]))
                if len(decoy_idx[0][0]):
                    decoy_mean_pred.append(np.mean(y_pred_np[decoy_idx]))

                if not (self.batch + 1) % aggrigation_interval:
                    self.optimiser.zero_grad()
                    loss /= aggrigation_interval
                    bce_loss /= aggrigation_interval
                    reported_batch = (self.batch + 1) // aggrigation_interval
                    loss.backward()
                    self.optimiser.step()
                    self.losses.append(bce_loss)

                    if not (reported_batch + 1) % log_interval or \
                            self.batch == total_iters - 1:
                        self.save_loss(log_interval)
                    global_iter += 1

                    eta = get_eta(start_time, global_iter, total_iters)
                    time_elapsed = format_time(time.time() - start_time)

                    if opt_cycle >= 0:
                        suffix = '(train, cycle {})'.format(opt_cycle)
                    else:
                        suffix = '(train)'
                    wandb_update_dict = {
                        'Time remaining ' + suffix: eta,
                        'Binary crossentropy ' + suffix: (
                            bce_loss),
                        'Batch ' + suffix:
                            (self.epoch * len(data_loader) + reported_batch),
                        'Mean decoy prediction (train)': np.mean(
                            decoy_mean_pred),
                        'Mean active prediction (train)': np.mean(
                            active_mean_pred),
                    }
                    try:
                        wandb.log(wandb_update_dict)
                    except wandb.errors.error.Error:
                        pass  # wandb has not been initialised so ignore

                    print_with_overwrite(
                        (
                            'Epoch:',
                            '{0}/{1}'.format(self.epoch + 1, epochs),
                            '|', 'Batch:', '{0}/{1}'.format(
                                reported_batch, len(data_loader))),
                        ('Time elapsed:', time_elapsed, '|',
                         'Time remaining:', eta),
                        ('Loss: {0:.4f}'.format(bce_loss), '|',
                         'Mean active: {0:.4f}'.format(np.mean(
                             active_mean_pred)), '|',
                         'Mean decoy: {0:.4f}'.format(np.mean(
                             decoy_mean_pred)))
                    )

                    bce_loss = 0.0
                    loss = 0.0
                    decoy_mean_pred, active_mean_pred = [], []

            # save after each epoch
            self.save()

    def test(self, data_loader, predictions_file=None):
        """Use trained network to perform inference on the test set.

        Uses the neural network (in Session.network), to perform predictions
        on the structures found in <test_data_root>, and saves this output
        to <save_path>/predictions_<test_data_root.name>.txt.

        Arguments:
            data_loader:
            predictions_file:
        """
        self.cuda()
        start_time = time.time()
        log_interval = 10
        decoy_mean_pred, active_mean_pred = 0.5, 0.5
        predictions = ''
        if predictions_file is None:
            predictions_file = self.predictions_file
        predictions_file = Path(predictions_file).expanduser()
        if predictions_file.is_file():
            predictions_file.unlink()
        self.eval()
        with torch.no_grad():
            for self.batch, (x, y_true, ligands, receptors) in enumerate(
                    data_loader):

                x = self._process_inputs(x)
                y_true = self._get_y_true(y_true).cuda()
                y_pred = self.forward_pass(x)
                y_true_np = y_true.cpu().detach().numpy()
                y_pred_np = nn.Softmax(dim=1)(y_pred).cpu().detach().numpy()

                active_idx = (np.where(y_true_np > 0.5), 1)
                decoy_idx = (np.where(y_true_np < 0.5), 1)

                scale = len(y_true) / len(data_loader)
                _ = self._get_loss(y_true, y_pred, scale)

                eta = get_eta(start_time, self.batch, len(data_loader))
                time_elapsed = format_time(time.time() - start_time)

                wandb_update_dict = {
                    'Time remaining (validation)': eta,
                    'Binary crossentropy (validation)': self.bce_loss,
                    'Batch': self.batch + 1
                }

                if len(active_idx[0][0]):
                    active_mean_pred = np.mean(y_pred_np[active_idx])
                    wandb_update_dict.update({
                        'Mean active prediction (validation)': active_mean_pred
                    })
                if len(decoy_idx[0][0]):
                    decoy_mean_pred = np.mean(y_pred_np[decoy_idx])
                    wandb_update_dict.update({
                        'Mean decoy prediction (validation)': decoy_mean_pred,
                    })

                try:
                    wandb.log(wandb_update_dict)
                except wandb.errors.error.Error:
                    pass  # wandb has not been initialised so ignore

                print_with_overwrite(
                    ('Inference on: {}'.format(data_loader.dataset.base_path),
                     '|', 'Iteration:', '{0}/{1}'.format(
                        self.batch + 1, len(data_loader))),
                    ('Time elapsed:', time_elapsed, '|',
                     'Time remaining:', eta),
                    ('Loss: {0:.4f}'.format(self.bce_loss), '|',
                     'Mean active: {0:.4f}'.format(active_mean_pred), '|',
                     'Mean decoy: {0:.4f}'.format(decoy_mean_pred))
                )

                predictions += '\n'.join(['{0} | {1:.7f} {2} {3}'.format(
                    int(y_true_np[i]),
                    y_pred_np[i, 1],
                    receptors[i],
                    ligands[i]) for i in range(len(receptors))]) + '\n'

                # Periodically write predictions to disk
                if not (self.batch + 1) % log_interval or self.batch == len(
                        data_loader) - 1:
                    with open(predictions_file, 'a') as f:
                        f.write(predictions)
                        predictions = ''

    def save(self, save_path=None):
        """Save all network attributes, including internal states."""

        if save_path is None:
            fname = 'ckpt_epoch_{}.pt'.format(self.epoch + 1)
            save_path = self.save_path / 'checkpoints' / fname

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'learning_rate': self.lr,
            'weight_decay': self.weight_decay,
            'epoch': self.epoch,
            'losses': self.losses,
            'bce_loss': self.bce_loss,
            'model_state_dict': self.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict()
        }, save_path)

    def save_loss(self, save_interval):
        """Save the loss information to disk.

        Arguments:
            save_interval: how often the loss is being recorded (in batches).
        """
        log_file = self.save_path / 'loss.log'
        start_idx = save_interval * (self.batch // save_interval)
        with open(log_file, 'a') as f:
            f.write('\n'.join(
                [str(idx + start_idx + 1) + ' ' + str(loss) for idx, loss in
                 enumerate(self.losses[-save_interval:])]) + '\n')

    @property
    def param_count(self):
        return sum(
            [torch.numel(t) for t in self.parameters() if t.requires_grad])

    @staticmethod
    def xavier_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)


class LieResNet(PointNeuralNetwork):
    """Generic ResNet architecture from https://arxiv.org/abs/2002.12880"""

    def _get_y_true(self, y):
        return y.cuda()

    def _process_inputs(self, x):
        return tuple([inp.cuda() for inp in x])

    def build_net(self, chin, ds_frac=1, k=1536, nbhd=np.inf, act="swish",
                  bn=True, num_layers=6, mean=True, pool=True, liftsamples=1,
                  fill=1 / 4, group=SE3, knn=False, cache=False, **kwargs):
        """
        Arguments:
            chin: number of input channels: 1 for MNIST, 3 for RGB images, other
                for non images
            ds_frac: total downsampling to perform throughout the layers of the
                net. In (0,1)
            k: channel width for the network. Can be int (same for all) or array
                to specify individually.
            nbhd: number of samples to use for Monte Carlo estimation (p)
            act:
            bn: whether or not to use batch normalization. Recommended in al
                cases except dynamical systems.
            num_layers: number of BottleNeck Block layers in the network
            mean:
            pool:
            liftsamples: number of samples to use in lifting. 1 for all groups
                with trivial stabilizer. Otherwise 2+
            fill: specifies the fraction of the input which is included in local
                neighborhood. (can be array to specify a different value for
                each layer)
            group: group to be equivariant to
            knn:
            cache:
        """
        if isinstance(fill, (float, int)):
            fill = [fill] * num_layers
        if isinstance(k, int):
            k = [k] * (num_layers + 1)
        conv = lambda ki, ko, fill: LieConv(
            ki, ko, mc_samples=nbhd, ds_frac=ds_frac, bn=bn, act=act, mean=mean,
            group=group, fill=fill, cache=cache, knn=knn)
        self.layers = nn.ModuleList([
            Pass(nn.Linear(chin, k[0]), dim=1),
            *[BottleBlock(k[i], k[i + 1], conv, bn=bn, act=act, fill=fill[i])
              for i in range(num_layers)],
            MaskBatchNormNd(k[-1]) if bn else nn.Sequential(),
            Pass(nn.ReLU(), dim=1),
            Pass(nn.Linear(k[-1], 2), dim=1),
            GlobalPool(mean=mean) if pool else Expression(lambda x: x[1])
        ])
        self.group = group
        self.liftsamples = liftsamples

    def forward(self, x):
        x = tuple([ten.cuda() for ten in self.group.lift(x, self.liftsamples)])
        for layer in self.layers:
            x = layer(x)
        return x


class EnTransformerBlock(EGNN):
    def forward(self, x):
        if len(x) == 3:
            coors, feats, mask = x
            edges = None
        else:
            coors, feats, mask, edges = x
        b, n, d, fourier_features = *feats.shape, self.fourier_features

        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(
            coors, 'b j d -> b () j d')
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        if fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist,
                                           num_encodings=fourier_features)
            rel_dist = rearrange(rel_dist, 'b i j () d -> b i j d')

        feats_i = repeat(feats, 'b i d -> b i n d', n=n)
        feats_j = repeat(feats, 'b j d -> b n j d', n=n)
        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim=-1)

        if exists(edges):
            edge_input = torch.cat((edge_input, edges), dim=-1)

        m_ij = self.edge_mlp(edge_input)

        coor_weights = self.coors_mlp(m_ij)
        coor_weights = rearrange(coor_weights, 'b i j () -> b i j')

        coors_out = einsum('b i j, b i j c -> b i c', coor_weights,
                           rel_coors) + coors

        m_i = m_ij.sum(dim=-2)

        node_mlp_input = torch.cat((feats, m_i), dim=-1)
        node_out = self.node_mlp(node_mlp_input) + feats

        return coors_out, node_out, mask


class EnResBlock(nn.Module):

    def __init__(self, chin, chout, conv, bn=False, act='swish'):
        super().__init__()
        nonlinearity = Swish if act == 'swish' else nn.ReLU
        self.conv = conv
        self.net = nn.ModuleList([
            MaskBatchNormNd(chin) if bn else nn.Sequential(),
            Pass(nonlinearity(), dim=1),
            Pass(nn.Linear(chin, chin // 4), dim=1),
            MaskBatchNormNd(chin // 4) if bn else nn.Sequential(),
            Pass(nonlinearity(), dim=1),
            self.conv,
            MaskBatchNormNd(chout // 4) if bn else nn.Sequential(),
            Pass(nonlinearity(), dim=1),
            Pass(nn.Linear(chout // 4, chout), dim=1),
        ])
        self.chin = chin

    def forward(self, inp):
        sub_coords, sub_values, mask = inp
        for layer in self.net:
            inp = layer(inp)
        new_coords, new_values, mask = inp
        new_values[..., :self.chin] += sub_values
        return new_coords, new_values, mask


class EquivariantTransformerBlock(nn.Module):

    def __init__(
            self,
            dim,
            n_heads,
            group,
            block_norm="layer_pre",
            kernel_norm="none",
            kernel_type="mlp",
            kernel_dim=16,
            kernel_act="swish",
            hidden_dim_factor=1,
            mc_samples=0,
            fill=1.0,
            attention_fn="softmax",
            feature_embed_dim=None,
    ):

        super().__init__()
        self.ema = EquivairantMultiheadAttention(
            dim,
            dim,
            n_heads,
            group,
            kernel_type=kernel_type,
            kernel_dim=kernel_dim,
            act=kernel_act,
            bn=kernel_norm == "batch",
            mc_samples=mc_samples,
            fill=fill,
            attention_fn=attention_fn,
            feature_embed_dim=feature_embed_dim,
        )

        self.mlp = MLP(dim, dim, dim, 2, kernel_act, kernel_norm == "batch")

        if block_norm == "none":
            self.attention_function = self._lambda_none_ema
            self.mlp_function = self._lambda_none_mlp
        elif block_norm == "layer_pre":
            self.ln_ema = nn.LayerNorm(dim)
            self.ln_mlp = nn.LayerNorm(dim)

            self.attention_function = self._lambda_layer_pre_ema
            self.mlp_function = self._lambda_layer_pre_mlp
        elif block_norm == "layer_post":
            self.ln_ema = nn.LayerNorm(dim)
            self.ln_mlp = nn.LayerNorm(dim)

            self.attention_function = self._lambda_layer_post_ema
            self.mlp_function = self._lambda_layer_post_mlp
        elif block_norm == "batch_pre":
            self.bn_ema = MaskBatchNormNd(dim)
            self.bn_mlp = MaskBatchNormNd(dim)

            self.attention_function = self._lambda_batch_pre_ema
            self.mlp_function = self._lambda_batch_pre_mlp
        elif block_norm == "batch_post":
            self.bn_ema = MaskBatchNormNd(dim)
            self.bn_mlp = MaskBatchNormNd(dim)

            self.attention_function = self._lambda_batch_post_ema
            self.mlp_function = self._lambda_batch_post_mlp
        else:
            raise ValueError(f"{block_norm} is invalid block norm type.")

    def forward(self, inpt):
        inpt[1] = self.attention_function(inpt)
        inpt[1] = self.mlp_function(inpt)

        return inpt

    def _lambda_none_ema(self, inpt):
        # lambda inpt: inpt[1] + self.ema(inpt)[1]
        return inpt[1] + self.ema(inpt)[1]

    def _lambda_none_mlp(self, inpt):
        # lambda inpt: inpt[1] + self.mlp(inpt)[1]
        return inpt[1] + self.mlp(inpt)[1]

    def _lambda_layer_pre_ema(self, inpt):
        # lambda inpt:
        # inpt[1] + self.ema((inpt[0], self.ln_ema(inpt[1]), inpt[2]))[1]
        return inpt[1] + self.ema((inpt[0], self.ln_ema(inpt[1]), inpt[2]))[1]

    def _lambda_layer_pre_mlp(self, inpt):
        # lambda inpt:
        # inpt[1] + self.mlp((inpt[0], self.ln_mlp(inpt[1]), inpt[2]))[1]
        return inpt[1] + self.mlp((inpt[0], self.ln_mlp(inpt[1]), inpt[2]))[1]

    def _lambda_layer_post_ema(self, inpt):
        # lambda inpt: inpt[1] + self.ln_ema(self.ema(inpt)[1])
        return inpt[1] + self.ln_ema(self.ema(inpt)[1])

    def _lambda_layer_post_mlp(self, inpt):
        # lambda inpt: inpt[1] + self.ln_mlp(self.mlp(inpt)[1])
        return inpt[1] + self.ln_mlp(self.mlp(inpt)[1])

    def _lambda_batch_pre_ema(self, inpt):
        # lambda inpt: inpt[1] + self.ema(self.bn_ema(inpt))[1]
        return inpt[1] + self.ema(self.bn_ema(inpt))[1]

    def _lambda_batch_pre_mlp(self, inpt):
        # lambda inpt: inpt[1] + self.mlp(self.bn_mlp(inpt))[1]
        return inpt[1] + self.mlp(self.bn_mlp(inpt))[1]

    def _lambda_batch_post_ema(self, inpt):
        # lambda inpt: inpt[1] + self.bn_ema(self.ema(inpt))[1]
        return inpt[1] + self.bn_ema(self.ema(inpt))[1]

    def _lambda_batch_post_mlp(self, inpt):
        # lambda inpt: inpt[1] + self.bn_mlp(self.mlp(inpt))[1]
        return inpt[1] + self.bn_mlp(self.mlp(inpt))[1]


class EquivariantTransformer(PointNeuralNetwork):
    """Adapted from https://github.com/anonymous-code-0/lie-transformer"""

    def _get_y_true(self, y):
        return y.cuda()

    def _process_inputs(self, x):
        return tuple([ten.cuda() for ten in x])

    def build_net(self, dim_input, dim_output, dim_hidden, num_layers,
                  num_heads, global_pool=True, global_pool_mean=True,
                  group=SE3(0.2), liftsamples=1, block_norm="layer_pre",
                  output_norm="none", kernel_norm="none", kernel_type="mlp",
                  kernel_dim=16, kernel_act="swish", mc_samples=0, fill=1.0,
                  attention_fn="norm_exp", feature_embed_dim=None,
                  max_sample_norm=None, lie_algebra_nonlinearity=None):

        if isinstance(dim_hidden, int):
            dim_hidden = [dim_hidden] * (num_layers + 1)

        if isinstance(num_heads, int):
            num_heads = [num_heads] * num_layers

        attention_block = lambda dim, n_head: EquivariantTransformerBlock(
            dim, n_head, group, block_norm=block_norm, kernel_norm=kernel_norm,
            kernel_type=kernel_type, kernel_dim=kernel_dim,
            kernel_act=kernel_act, mc_samples=mc_samples, fill=fill,
            attention_fn=attention_fn, feature_embed_dim=feature_embed_dim,
        )

        activation_fn = {
            "swish": Swish,
            "relu": nn.ReLU,
            "softplus": nn.Softplus,
        }

        if output_norm == "batch":
            norm1 = nn.BatchNorm1d(dim_hidden[-1])
            norm2 = nn.BatchNorm1d(dim_hidden[-1])
            norm3 = nn.BatchNorm1d(dim_hidden[-1])
        elif output_norm == "layer":
            norm1 = nn.LayerNorm(dim_hidden[-1])
            norm2 = nn.LayerNorm(dim_hidden[-1])
            norm3 = nn.LayerNorm(dim_hidden[-1])
        elif output_norm == "none":
            norm1 = nn.Sequential()
            norm2 = nn.Sequential()
            norm3 = nn.Sequential()
        else:
            raise ValueError(f"{output_norm} is not a valid norm type.")

        self.net = nn.Sequential(
            Pass(nn.Linear(dim_input, dim_hidden[0]), dim=1),
            *[
                attention_block(dim_hidden[i], num_heads[i])
                for i in range(num_layers)
            ],
            GlobalPool(mean=global_pool_mean) if global_pool else \
                Expression(self._lambda_index_1),
            nn.Sequential(
                norm1,
                activation_fn[kernel_act](),
                nn.Linear(dim_hidden[-1], dim_hidden[-1]),
                norm2,
                activation_fn[kernel_act](),
                nn.Linear(dim_hidden[-1], dim_hidden[-1]),
                norm3,
                activation_fn[kernel_act](),
                nn.Linear(dim_hidden[-1], dim_output),
            ),
        )

        self.group = group
        self.liftsamples = liftsamples
        self.max_sample_norm = max_sample_norm

        self.lie_algebra_nonlinearity = lie_algebra_nonlinearity
        if lie_algebra_nonlinearity is not None:
            if lie_algebra_nonlinearity == "tanh":
                self.lie_algebra_nonlinearity = nn.Tanh()
            else:
                raise ValueError(
                    f"{lie_algebra_nonlinearity} is not a supported nonlinearity"
                )

    @staticmethod
    def _lambda_index_1(x):
        return x[1]

    def forward(self, input):
        if self.max_sample_norm is None:
            lifted_data = self.group.lift(input, self.liftsamples)
        else:
            lifted_data = [
                torch.tensor(self.max_sample_norm * 2, device=input[0].device),
                0,
                0,
            ]
            while lifted_data[0].norm(dim=-1).max() > self.max_sample_norm:
                lifted_data = self.group.lift(input, self.liftsamples)

        if self.lie_algebra_nonlinearity is not None:
            lifted_data = list(lifted_data)
            pairs_norm = lifted_data[0].norm(dim=-1) + 1e-6
            lifted_data[0] = lifted_data[0] * (
                    self.lie_algebra_nonlinearity(pairs_norm / 7) / pairs_norm
            ).unsqueeze(-1)

        return self.net(lifted_data)


class EGNNStack(PointNeuralNetwork):

    @staticmethod
    def xavier_init(m):
        pass

    def _get_y_true(self, y):
        return y.cuda()

    def _process_inputs(self, x):
        return tuple([ten.cuda() for ten in x])

    def build_net(self, chin, output_dim=2, k=12, act="swish", bn=True,
                  dropout=0.0, num_layers=6, mean=False, pool=True, feats_idx=0,
                  **kwargs):
        egnn = lambda: EGNN(dim=chin, m_dim=k, norm_rel_coors=True,
                            norm_coor_weights=False, dropout=dropout)
        if bn:
            bn = lambda: EGNNBatchNorm(12)
            eggn_layers = [(egnn(), bn()) for _ in range(num_layers)]
        else:
            eggn_layers = [(egnn(),) for _ in range(num_layers)]
        if act == 'swish':
            activation_class = Swish
        elif act == 'relu':
            activation_class = nn.ReLU
        else:
            raise NotImplementedError('{} not a recognised activation'.format(
                act))
        self.layers = nn.ModuleList([
            *[a for b in eggn_layers for a in b],
            Pass(nn.Linear(chin, chin * 2), dim=feats_idx),
            Pass(activation_class(), dim=feats_idx),
            Pass(nn.Linear(chin * 2, chin), dim=feats_idx),
            EGNNGlobalPool(
                dim=feats_idx, tensor_dim=1,
                mean=mean) if pool else nn.Sequential(),
            Pass(nn.Linear(chin, chin * 2), dim=feats_idx),
            Pass(activation_class(), dim=feats_idx),
            Pass(nn.Linear(chin * 2, output_dim), dim=feats_idx),
        ])

    def forward(self, x):
        coords, feats, mask = x
        for layer in self.layers:
            if isinstance(layer, EGNN):
                feats, coords = layer(feats, coords, mask=mask)
            else:
                x = layer([feats, coords])
                if isinstance(x, (tuple, list)):
                    feats, coords = x
                else:
                    feats = x
        return feats

    @staticmethod
    def get_min_max(network):
        min_val, max_val = np.inf, -np.inf
        for layer in network:
            if isinstance(layer, nn.Linear):
                min_val = min(float(torch.min(layer.weight)), min_val)
                max_val = max(float(torch.max(layer.weight)), max_val)
        return min_val, max_val


class EnResNet_(PointNeuralNetwork):
    """ResNet for E(n)Transformer"""

    def _get_y_true(self, y):
        return y.cuda()

    def _process_inputs(self, x):
        return tuple([ten.cuda() for ten in x])

    def build_net(self, chin, k=12, act="swish", bn=True, num_layers=6,
                  mean=True, pool=True, **kwargs):
        """
        Arguments:
            chin: number of input channels: 1 for MNIST, 3 for RGB images, other
                for non images
            k: channel width for the network. Can be int (same for all) or array
                to specify individually.
            act:
            bn: whether or not to use batch normalization. Recommended in al
                cases except dynamical systems.
            num_layers: number of BottleNeck Block layers in the network
            mean:
            pool:
        """
        self.layers = nn.ModuleList([
            Pass(nn.Linear(12, k), dim=1),
            *[EnResBlock(k, k, EnTransformerBlock(
                k // 4, m_dim=16), bn=bn, act=act)
              for _ in range(num_layers)],
            MaskBatchNormNd(k) if bn else nn.Sequential(),
            Pass(nn.ReLU(), dim=1),
            Pass(nn.Linear(k, 2), dim=1),
            GlobalPool(mean=mean) if pool else Expression(lambda x: x[1])
        ])

        self.apply(self.xavier_init)
        print('Params:', self.param_count)


class EnFeatureExtractor(nn.Module):
    """ResNet feature extractor for E(n)Transformer"""

    def __init__(self, chin, k=12, act="swish", bn=True, num_layers=6,
                 mean=True, pool=None, fc_in_features=512, **kwargs):
        """
        Arguments:
            chin: number of input channels: 1 for MNIST, 3 for RGB images, other
                for non images
            ds_frac: total downsampling to perform throughout the layers of the
                net. In (0,1)
            k: channel width for the network. Can be int (same for all) or array
                to specify individually.
            nbhd: number of samples to use for Monte Carlo estimation (p)
            act:
            bn: whether or not to use batch normalization. Recommended in al
                cases except dynamical systems.
            num_layers: number of BottleNeck Block layers in the network
            mean:
            pool:
            liftsamples: number of samples to use in lifting. 1 for all groups
                with trivial stabilizer. Otherwise 2+
            fill: specifies the fraction of the input which is included in local
                neighborhood. (can be array to specify a different value for
                each layer)
            group: group to be equivariant to
            knn:
            cache:
        """
        super().__init__()
        utils.set_gpu_mode(True)
        self.pretrained = False
        self.net = nn.Sequential(
            Pass(nn.Linear(12, k), dim=1),
            *[EnResBlock(k, k, EnTransformerBlock(
                k // 4, m_dim=16), bn=bn, act=act)
              for _ in range(num_layers)],
            MaskBatchNormNd(k) if bn else nn.Sequential(),
            Pass(nn.Linear(k, fc_in_features), dim=1),
            GlobalPool(mean=mean) if pool else Expression(lambda x: x[1]),
        )

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)


class BayesianPointNN(PointNeuralNetwork):

    def forward_pass(self, x):
        y_pred_samples = self.forward(x, num_samples=100)
        y_pred = self._compute_predictive_posterior(y_pred_samples)
        return y_pred

    def _get_y_true(self, y):
        return y

    def _get_y_pred(self, x):
        return self.forward(x)

    def _get_loss(self, y_true, y_pred, scale=None):
        step_loss, kl = self._compute_loss(
            y_true, y_pred, scale)
        self.bce_loss = step_loss - kl
        return step_loss

    def _process_inputs(self, x):
        return tuple([inp.cuda() for inp in x])

    def build_net(self, feature_extractor=None, fc_in_features=512,
                  fc_out_features=256, full_cov=False, cov_rank=2):
        """Neural Linear model for multi-class classification.

        Arguments:
            feature_extractor: (nn.Module) Feature extractor to generate
                representations
            fc_in_features:
            fc_out_features: (int) Dimensionality of final feature representation
            full_cov: (bool) Use (low-rank approximation to) full covariance
                matrix for last layer distribution
            cov_rank: (int) Optional, if using low-rank approximation, specify
                rank
        """
        self.feature_extractor = feature_extractor
        self.num_features = fc_out_features

        self.fc1 = nn.Linear(
            in_features=fc_in_features, out_features=self.num_features,
            bias=True).cuda()

        nn.init.xavier_normal_(self.fc1.weight)
        for layer in feature_extractor.parameters():
            if len(layer.shape) > 1:
                nn.init.xavier_normal_(layer)

        if full_cov:
            self.linear = ReparamFullDense(
                [self.num_features, self.num_classes], rank=cov_rank).cuda()
        else:
            self.linear = LocalReparamDense(
                [self.num_features, 2]).cuda()

    def forward(self, x, num_samples=1):
        """Make prediction with model

        Arguments:
            x: (torch.tensor) Inputs
            num_samples: (int) Number of samples to use in forward pass

        Returns:
             (torch.tensor) Predictive distribution (may be tuple)
        """
        enc = self.encode(x)
        return self.linear(enc, num_samples=num_samples).squeeze()

    def encode(self, x):
        """
        Use feature extractor to get features from inputs
        :param x: (torch.tensor) Inputs
        :return: (torch.tensor) Feature representation of inputs
        """
        x = self.feature_extractor(x)
        x = self.fc1(x)
        return x

    def get_projections(
            self, dataloader, J, projection='two', gamma=0, opt_cycle=-1,
            **kwargs):
        """Get projections for ACS approximate procedure.

        Arguments:
            dataloader: (Object) Data object to get projections for
            J: (int) Number of projections to use
            projection: (str) Type of projection to use (currently only 'two'
                supported)

        Returns:
            (torch.tensor) Projections
        """
        ent = lambda py: torch.distributions.Categorical(probs=py).entropy()
        projections = []
        feat_x = []
        with torch.no_grad():
            mean, cov = self.linear._compute_posterior()
            jitter = utils.to_gpu(torch.eye(len(cov)) * 1e-6)
            theta_samples = MVN(mean, cov + jitter).sample(
                torch.Size([J])).view(J, -1, self.linear.out_features)

            start_time = time.time()
            total_iters = len(dataloader)
            global_iter = 0
            for batch, (x, _, _, _) in enumerate(dataloader):
                x = tuple(utils.to_gpu(*x))
                feat_x.append(self.encode(x))
                eta = get_eta(start_time, global_iter, total_iters)

                if opt_cycle >= 0:
                    suffix = '(projections, cycle {})'.format(opt_cycle)
                else:
                    suffix = ('(projections)')
                wandb_update_dict = {
                    'Time remaining ' + suffix: eta,
                    'Batch ' + suffix: batch
                }
                try:
                    wandb.log(wandb_update_dict)
                except wandb.errors.error.Error:
                    pass
                print_with_overwrite(
                    ('Getting projections for batch {0} of {1}'.format(
                        batch, len(dataloader)),))

            feat_x = torch.cat(feat_x)
            py = self._compute_predictive_posterior(
                self.linear(feat_x, num_samples=100), logits=False)
            ent_x = ent(py)
            if projection == 'two':
                for theta_sample in theta_samples:
                    projections.append(
                        self._compute_expected_ll(feat_x, theta_sample,
                                                  py) + gamma * ent_x[:, None])
            else:
                raise NotImplementedError

        return utils.to_gpu(
            torch.sqrt(1 / torch.FloatTensor([J]))) \
               * torch.cat(projections, dim=1), ent_x

    def _compute_log_likelihood(self, y, y_pred):
        """
        Compute log-likelihood of predictions
        :param y: (torch.tensor) Observations
        :param y_pred: (torch.tensor) Predictions
        :return: (torch.tensor) Log-likelihood of predictions
        """
        log_pred_samples = y_pred
        ll_samples = -self.cross_entropy(
            log_pred_samples, y.squeeze().long())
        return ll_samples

    def _compute_predictive_posterior(self, y_pred, logits=True):
        """
        Return posterior predictive evaluated at x
        :param x: (torch.tensor) Inputs
        :return: (torch.tensor) Probit regression posterior predictive
        """
        log_pred_samples = y_pred
        L = utils.to_gpu(torch.FloatTensor([log_pred_samples.shape[0]]))
        preds = torch.logsumexp(log_pred_samples, dim=0) - torch.log(L)
        if not logits:
            preds = torch.softmax(preds, dim=-1)

        return preds

    def _compute_loss(self, y, y_pred, kl_scale=None):
        """
        Compute loss function for variational training
        :param y: (torch.tensor) Observations
        :param y_pred: (torch.tensor) Model predictions
        :param kl_scale: (float) Scaling factor for KL-term
        :return: (torch.scalar) Loss evaluation
        """
        # The objective is 1/n * (\sum_i log_like_i - KL)
        log_likelihood = self._compute_log_likelihood(y, y_pred)
        kl = self.linear.compute_kl() * kl_scale
        elbo = log_likelihood - kl
        return -elbo, kl

    def _compute_expected_ll(self, x, theta, py):
        """
        Compute expected log-likelihood for data
        :param x: (torch.tensor) Inputs to compute likelihood for
        :param theta: (torch.tensor) Theta parameter to use in likelihood computations
        :return: (torch.tensor) Expected log-likelihood of inputs
        """
        classes = self.linear.out_features
        logits = x @ theta
        ys = torch.ones_like(logits).type(
            torch.LongTensor) * torch.arange(self.linear.out_features)[None, :]
        ys = utils.to_gpu(ys).t()

        ce = nn.CrossEntropyLoss(reduction='none')
        loglik = torch.stack([-ce(logits, y.long()) for y in ys]).t()

        if classes > 1:
            return torch.sum(py * loglik, dim=-1, keepdim=True)
        return py * loglik


class LieFeatureExtractor(nn.Module):

    def __init__(self, chin, ds_frac=1, k=64, nbhd=np.inf,
                 act="swish", bn=True, num_layers=6, mean=True, per_point=True,
                 liftsamples=1, fill=1 / 4, group=SE3, knn=False, cache=False,
                 num_outputs=None, pool=None, fc_in_features=512,
                 **kwargs):
        super().__init__()
        utils.set_gpu_mode(True)
        self.pretrained = False
        if isinstance(fill, (float, int)):
            fill = [fill] * num_layers
        if isinstance(k, int):
            k = [k] * (num_layers + 1)
        conv = lambda ki, ko, fill: LieConv(ki, ko, mc_samples=nbhd,
                                            ds_frac=ds_frac, bn=bn, act=act,
                                            mean=mean,
                                            group=group, fill=fill, cache=cache,
                                            knn=knn, **kwargs)
        self.net = nn.Sequential(
            Pass(nn.Linear(chin, k[0]), dim=1),  # embedding layer
            *[BottleBlock(k[i], k[i + 1], conv, bn=bn, act=act, fill=fill[i])
              for i in range(num_layers)],
            MaskBatchNormNd(k[-1]) if bn else nn.Sequential(),
            Pass(nn.Linear(k[-1], fc_in_features), dim=1),
            GlobalPool(mean=mean) if pool else Expression(lambda x: x[1]),
            # Expression(lambda x: x[1]),
        )
        self.liftsamples = liftsamples
        self.per_point = per_point
        self.group = group

    def forward(self, x):
        lifted_x = self.group.lift(x, self.liftsamples)
        return self.net(lifted_x)
