"""Session class handles training and hyperparameters.
TODO: rewrite this mess."""

import time
from pathlib import Path, PosixPath

import numpy as np
import torch
import torch.nn as nn
from equivariant_attention.modules import get_basis_and_r

try:
    import wandb
except ImportError:
    print('Library wandb not available. --wandb and --run flags should not be '
          'used.')
    wandb = None
from experiments.qm9.models import SE3Transformer
from lie_conv.lieConv import LieResNet

from data_loaders import SE3TransformerLoader, LieConvLoader, \
    multiple_source_dataset
from lieconv_utils import format_time, print_with_overwrite, get_eta


class LieResNetSigmoid(LieResNet):
    """We need all of our networks to finish with a sigmoid activation."""

    def forward(self, x):
        lifted_x = self.group.lift(x, self.liftsamples)
        return torch.sigmoid(self.net(lifted_x))


class SE3TransformerSigmoid(SE3Transformer):
    """We need all of our networks to finish with a sigmoid activation."""

    def forward(self, g):
        basis, r = get_basis_and_r(g, self.num_degrees - 1)
        h = {'0': g.ndata['f']}
        for layer in self.Gblock:
            h = layer(h, G=g, r=r, basis=basis)

        for layer in self.FCblock:
            h = layer(h)

        if np.prod(h.shape) == 1:
            x = torch.reshape(h, (1,))
        else:
            x = torch.squeeze(h)
        return torch.sigmoid(x)


class EvidentialLieResNet(LieResNet):

    def forward(self, x):
        lifted_x = self.group.lift(x, self.liftsamples)
        logits = self.net(lifted_x)
        evidence = torch.exp(logits)
        alpha = torch.add(evidence, 1.0)

        uncertainties = 2 / torch.sum(alpha, dim=1, keepdim=True)

        probabilities = alpha / torch.sum(alpha, dim=1, keepdim=True)
        return probabilities, uncertainties, alpha


def KL(alpha, device):
    dim = 1
    beta = torch.ones((1, 2), dtype=torch.float).to(device)
    S_alpha = torch.sum(alpha, dim=dim, keepdim=True).to(device)
    S_beta = torch.sum(beta, dim=dim, keepdim=True).to(device)
    lnB = torch.lgamma(S_alpha) - torch.sum(
        torch.lgamma(alpha), dim=dim, keepdim=True).to(device)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=dim,
                        keepdim=True) - torch.lgamma(
        S_beta).to(device)
    dg0 = torch.digamma(S_alpha).to(device)
    dg1 = torch.digamma(alpha).to(device)

    kl = torch.sum(
        (alpha - beta) * (dg1 - dg0), dim=dim, keepdim=True) + lnB + lnB_uni
    return kl.to(device)


def mse_loss(p, alpha, global_step, annealing_step, device):
    dim = 1
    S = torch.sum(alpha, dim=dim, keepdim=True)
    E = alpha - 1
    m = alpha / S

    A = torch.sum((p - m) ** 2, dim=dim, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=dim,
                  keepdim=True)

    annealing_coef = torch.min(
        torch.tensor([1.0, global_step / annealing_step])).float()

    alp = E * (1 - p) + 1
    C = annealing_coef * KL(alp, device)
    return (A + B) + C


def ce_loss(p, alpha, global_step, annealing_step, device):
    dim = 1
    f = torch.digamma
    S = torch.sum(alpha, dim=dim, keepdim=True)
    E = alpha - 1

    A = torch.sum(p * (f(S) - f(alpha)), dim=dim, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor([1.0, global_step / annealing_step])).float()

    alp = E * (1 - p) + 1
    B = annealing_coef * KL(alp, device)

    return A + B


class Session:
    """Handles training of point cloud-based models."""

    def __init__(self, network, train_data_root, save_path, batch_size,
                 test_data_root=None, train_receptors=None, test_receptors=None,
                 save_interval=-1, learning_rate=0.01, epochs=1, radius=12,
                 wandb=None, run=None, **kwargs):
        """Initialise session.

        Arguments:
            network: pytorch object inheriting from torch.nn.Module.
            train_data_root: path containing the 'receptors' and 'ligands'
                training directories, which in turn contain <rec_name>.parquets
                files and folders called <rec_name>_[active|decoy] which in turn
                contain <ligand_name>.parquets files. All parquets files from
                this directory are recursively loaded into the dataset.
            save_path: directory in which experiment outputs are stored.
            batch_size: number of training examples in each batch
            test_data_root: like train_root, but for the test set.
            train_receptors: iterable of strings denoting receptors to include
                in the dataset. if None, all receptors found in base_path are
                included.
            test_receptors: like train_receptors, but for the test set.
            save_interval: save model checkpoint every <save_interval> batches.
            radius: radius of bounding box (receptor atoms to include).
            wandb: name of wandb project (None = no logging).
            run: name of wandb project run (None = default)
            kwargs: translated actives keyword arguments
        """
        self.train_data_root = Path(train_data_root).expanduser()
        if test_data_root is not None:
            self.test_data_root = Path(test_data_root).expanduser()
        else:
            self.test_data_root = None
        self.save_path = Path(save_path).expanduser()
        self.loss_plot_file = self.save_path / 'loss.png'
        self.network = network
        self.network.apply(self.weights_init)
        self.batch_size = batch_size
        self.losses = []
        self.epoch = 0
        self.batch = 0
        self.save_interval = save_interval
        self.lr = learning_rate
        self.epochs = epochs
        self.radius = radius
        self.translated_actives = kwargs.get('translated_actives', None)
        self.n_translated_actives = kwargs.get('n_translated_actives', 0)

        if isinstance(train_receptors, str):
            train_receptors = tuple([train_receptors])
        if isinstance(test_receptors, str):
            test_receptors = tuple([test_receptors])

        dataset_kwargs = {
            'receptors': train_receptors,
            'radius': self.radius,
        }

        if isinstance(self.network, SE3Transformer):
            dataset_class = SE3TransformerLoader
            dataset_kwargs.update({
                'mode': 'interaction_edges',
                'interaction_dist': 4
            })
        elif isinstance(self.network, LieResNet):
            dataset_class = LieConvLoader
        else:
            raise NotImplementedError(
                'Unrecognised network class {}'.format(self.network.__class__))

        self.train_dataset = multiple_source_dataset(
            dataset_class, self.train_data_root, self.translated_actives,
            **dataset_kwargs)

        data_loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 0,
            'sampler': self.train_dataset.sampler,
            'collate_fn': self.train_dataset.collate
        }

        self.train_data_loader = torch.utils.data.DataLoader(
            self.train_dataset, **data_loader_kwargs
        )

        self.loss_log_file = self.save_path / 'loss.log'

        if self.test_data_root is not None:
            del data_loader_kwargs['sampler']
            dataset_kwargs.update({'receptors': test_receptors})
            self.test_dataset = dataset_class(
                self.test_data_root, **dataset_kwargs
            )
            self.test_data_loader = torch.utils.data.DataLoader(
                self.test_dataset, **data_loader_kwargs
            )
            self.predictions_file = Path(
                self.save_path, 'predictions_{}.txt'.format(
                    self.test_data_root.name))
        else:
            self.test_dataset = None
            self.test_data_loader = None
            self.predictions_file = None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.criterion = nn.BCELoss().to(device)
        self.optimiser = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.device = device

        self.wandb = wandb
        self.wandb_path = self.save_path / 'wandb_{}'.format(wandb)
        if run is not None:
            self.run_name = run

    def _setup_training_session(self):
        """Puts network on GPU and sets up directory structure."""
        self.network.to(self.device)
        self.save_path.mkdir(parents=True, exist_ok=True)
        for output_file in [self.predictions_file, self.loss_log_file]:
            try:
                output_file.unlink()
            except FileNotFoundError:
                pass
            except AttributeError:
                pass
        torch.cuda.empty_cache()
        print('Using device:', self.device, '\n\n')
        return time.time()

    def train(self):
        """Train the network.

        Trains the neural network (in Session.network), displays training
        information and plots the loss. All figures and logs are saved to
        <Session.save_path>.
        """
        start_time = self._setup_training_session()
        total_iters = self.epochs * len(self.train_data_loader)
        log_interval = 10
        global_iter = 0
        self.network.train()
        if self.wandb is not None:
            if hasattr(self, 'run_name'):
                wandb.run.name = self.run_name
            wandb.watch(self.network)
        for self.epoch in range(self.epochs):
            decoy_mean_pred, active_mean_pred = -1, -1
            for self.batch, (x, y_true, ligands, receptors) in enumerate(
                    self.train_data_loader):
                self.optimiser.zero_grad()
                if len(x) > 1:
                    x = tuple([inp.to(self.device) for inp in x])
                else:
                    x = x[0].to(self.device)

                y_true = y_true.to(self.device).squeeze()
                y_true_np = y_true.cpu().detach().numpy()
                active_idx = np.where(y_true_np > 0.5)
                decoy_idx = np.where(y_true_np < 0.5)
                y_pred = self.network(x)
                if isinstance(self.network, EvidentialLieResNet):
                    y_true = torch.nn.functional.one_hot(y_true, num_classes=2)
                    y_pred, uncertainty, alpha = y_pred
                    y_pred = y_pred.squeeze()
                    loss = ce_loss(y_true, alpha, global_iter, total_iters,
                                   self.device).mean()
                else:
                    y_pred = y_pred.squeeze()
                    loss = self.criterion(y_pred.float(), y_true.float())

                loss.backward()
                self.optimiser.step()

                eta = get_eta(start_time, global_iter, total_iters)
                time_elapsed = format_time(time.time() - start_time)

                y_pred_np = y_pred.cpu().detach().numpy()

                wandb_update_dict = {
                    'Binary crossentropy (train)': loss,
                    'Batch': self.epoch * len(
                        self.train_data_loader) + self.batch + 1
                }

                if isinstance(self.network, EvidentialLieResNet):
                    active_idx = (active_idx, 1)
                    decoy_idx = (decoy_idx, 1)
                    wandb_update_dict.update(
                        {'Mean uncertainty (train)': float(uncertainty.mean())})
                if len(active_idx[0]):
                    active_mean_pred = np.mean(
                        y_pred_np[active_idx])
                    wandb_update_dict.update({
                        'Mean active prediction (train)': active_mean_pred
                    })
                if len(decoy_idx[0]):
                    decoy_mean_pred = np.mean(
                        y_pred_np[decoy_idx])
                    wandb_update_dict.update({
                        'Mean decoy prediction (train)': decoy_mean_pred,
                    })

                if self.wandb:
                    wandb.log(wandb_update_dict)

                if isinstance(self.network, EvidentialLieResNet):
                    print_with_overwrite(
                        (
                            'Epoch:',
                            '{0}/{1}'.format(self.epoch + 1, self.epochs),
                            '|', 'Iteration:', '{0}/{1}'.format(
                                self.batch + 1, len(self.train_data_loader))),
                        ('Time elapsed:', time_elapsed, '|',
                         'Time remaining:', eta),
                        ('Loss: {0:.4f}'.format(loss), '|',
                         'Mean active: {0:.4f}'.format(active_mean_pred), '|',
                         'Mean decoy: {0:.4f}'.format(decoy_mean_pred)),
                        ('Mean uncertainty: {0:.4f}'.format(
                            float(uncertainty.mean())), ' ')
                    )
                else:
                    print_with_overwrite(
                        (
                            'Epoch:',
                            '{0}/{1}'.format(self.epoch + 1, self.epochs),
                            '|', 'Iteration:', '{0}/{1}'.format(
                                self.batch + 1, len(self.train_data_loader))),
                        ('Time elapsed:', time_elapsed, '|',
                         'Time remaining:', eta),
                        ('Loss: {0:.4f}'.format(loss), '|',
                         'Mean active: {0:.4f}'.format(active_mean_pred), '|',
                         'Mean decoy: {0:.4f}'.format(decoy_mean_pred))
                    )

                self.losses.append(float(loss))
                if not (self.batch + 1) % log_interval or \
                        self.batch == total_iters - 1:
                    self.save_loss(log_interval)

                if (self.save_interval > 0 and
                    not (global_iter + 1) % self.save_interval) or \
                        global_iter == total_iters - 1:
                    self.save()

                global_iter += 1

    def test(self):
        """Use trained network to perform inference on the test set.

        Uses the neural network (in Session.network), to perform predictions
        on the structures found in <test_data_root>, and saves this output
        to <save_path>/predictions_<test_data_root.name>.txt.
        """
        self.network.to(self.device)
        start_time = time.time()
        log_interval = 10
        decoy_mean_pred, active_mean_pred = -1, -1
        predictions = ''
        self.network.eval()
        with torch.no_grad():
            for self.batch, (x, y_true, ligands, receptors) in enumerate(
                    self.test_data_loader):
                if len(x) > 1:
                    x = tuple([inp.to(self.device) for inp in x])
                else:
                    x = x[0].to(self.device)

                y_true = y_true.to(self.device).squeeze()
                y_true_np = y_true.cpu().detach().numpy()
                active_idx = np.where(y_true_np > 0.5)
                decoy_idx = np.where(y_true_np < 0.5)
                y_pred = self.network(x)
                if isinstance(self.network, EvidentialLieResNet):
                    y_true = torch.nn.functional.one_hot(y_true, num_classes=2)
                    y_pred, uncertainty, alpha = y_pred
                    y_pred = y_pred.squeeze()
                    loss = ce_loss(y_true, alpha, 1, 1, self.device).mean()
                else:
                    y_pred = y_pred.squeeze()
                    loss = self.criterion(y_pred.float(), y_true.float())

                eta = get_eta(
                    start_time, self.batch, len(self.test_data_loader))
                time_elapsed = format_time(time.time() - start_time)

                y_pred_np = y_pred.cpu().detach().numpy()

                wandb_update_dict = {
                    'Binary crossentropy (validation)': loss,
                    'Batch': self.batch + 1
                }

                if isinstance(self.network, EvidentialLieResNet):
                    active_idx = (*active_idx, 1)
                    decoy_idx = (*decoy_idx, 1)
                    wandb_update_dict.update({
                        'Mean uncertainty (validation)':
                            float(uncertainty.mean())})
                if len(active_idx[0]):
                    active_mean_pred = np.mean(
                        y_pred_np[active_idx])
                    wandb_update_dict.update({
                        'Mean active prediction (validation)': active_mean_pred
                    })
                if len(decoy_idx[0]):
                    decoy_mean_pred = np.mean(
                        y_pred_np[decoy_idx])
                    wandb_update_dict.update({
                        'Mean decoy prediction (validation)': decoy_mean_pred,
                    })

                if self.wandb:
                    wandb.log(wandb_update_dict)

                if isinstance(self.network, EvidentialLieResNet):
                    print_with_overwrite(
                        ('Inference on: {}'.format(self.test_data_root), '|',
                         'Iteration:', '{0}/{1}'.format(
                            self.batch + 1, len(self.test_data_loader))),
                        ('Time elapsed:', time_elapsed, '|',
                         'Time remaining:', eta),
                        ('Loss: {0:.4f}'.format(loss), '|',
                         'Mean active: {0:.4f}'.format(active_mean_pred), '|',
                         'Mean decoy: {0:.4f}'.format(decoy_mean_pred)),
                        ('Mean uncertainty: {0:.4f}'.format(
                            float(uncertainty.mean())), ' ')
                    )
                else:
                    print_with_overwrite(
                        ('Inference on: {}'.format(self.test_data_root), '|',
                         'Iteration:', '{0}/{1}'.format(
                            self.batch + 1, len(self.test_data_loader))),
                        ('Time elapsed:', time_elapsed, '|',
                         'Time remaining:', eta),
                        ('Loss: {0:.4f}'.format(loss), '|',
                         'Mean active: {0:.4f}'.format(active_mean_pred), '|',
                         'Mean decoy: {0:.4f}'.format(decoy_mean_pred))
                    )

                uncertainty_np = uncertainty.cpu().detach().numpy().squeeze()
                if isinstance(self.network, EvidentialLieResNet):
                    predictions += '\n'.join(
                        ['{0} | {1:.7f} {2} {3} {4:.4f}'.format(
                            int(y_true_np[i]),
                            y_pred_np[i, 1],
                            receptors[i],
                            ligands[i],
                            uncertainty_np[i])
                            for i in range(len(receptors))]) + '\n'
                else:
                    predictions += '\n'.join(['{0} | {1:.7f} {2} {3}'.format(
                        int(y_true_np[i]),
                        y_pred_np[i],
                        receptors[i],
                        ligands[i]) for i in range(len(receptors))]) + '\n'

                # Periodically write predictions to disk
                if not (self.batch + 1) % log_interval or self.batch == len(
                        self.test_data_loader) - 1:
                    with open(self.predictions_file, 'a') as f:
                        f.write(predictions)
                        predictions = ''

    def save(self):
        """Save all session attributes, including internal states."""
        attributes = {}
        accounted_for = set('network')
        for attr in dir(self.network):
            if attr != '__class__' and \
                    'state_dict' in dir(getattr(self.network, attr)):
                attributes.update({
                    attr + '_state_dict': getattr(
                        getattr(self.network, attr), 'state_dict')()})
                accounted_for.add(attr)
        for attr in dir(self):
            if 'state_dict' in dir(getattr(self, attr)):
                attributes.update({
                    attr + '_state_dict': getattr(
                        getattr(self, attr), 'state_dict')()})
                accounted_for.add(attr)
        for var, val in [(varname, getattr(self, varname)) for varname in
                         vars(self) if varname not in accounted_for]:
            attributes.update({var: val})

        Path(self.save_path, 'checkpoints').mkdir(exist_ok=True, parents=True)
        torch.save(attributes, Path(
            self.save_path, 'checkpoints',
            'ckpt_epoch_{}_batch_{}.pt'.format(self.epoch + 1, self.batch)))

    def load(self, checkpoint_path):
        """Fully automatic loading of models saved with self.save.

        Arguments:
            checkpoint_path: directory containing saved model
        """
        checkpoint_path = Path(checkpoint_path).expanduser()
        load_dict = torch.load(checkpoint_path)
        states = {
            key.replace('_state_dict', ''): value for key, value in
            load_dict.items() if key.endswith('_state_dict')}
        attributes = {
            key: value for key, value in
            load_dict.items() if not key.endswith('_state_dict')}

        for attr_name, value in attributes.items():
            if not isinstance(value, PosixPath):
                setattr(self, attr_name, value)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.network.load_state_dict(states['network'])
        if isinstance(self.network, LieResNet):
            self.network.net.load_state_dict(states['net'])
        self.network.to(device)
        self.optimiser = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.optimiser.load_state_dict(states['optimiser'])

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

    @staticmethod
    def weights_init(m):
        """Initialise weights of network using xavier initialisation."""
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

    @staticmethod
    def sigmoid(x):
        """Sigmoid function for reporting."""
        return 1 / (1 + np.exp(-x))

    @property
    def param_count(self):
        if isinstance(self.network, LieResNet):
            return sum([torch.numel(t) for t in self.network.net.parameters()
                        if t.requires_grad])
        return sum([torch.numel(t) for t in self.network.parameters()
                    if t.requires_grad])
