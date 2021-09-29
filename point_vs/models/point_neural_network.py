import math
import math
import time
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch import nn

from point_vs.utils import get_eta, format_time, print_with_overwrite, mkdir, \
    to_numpy


class PointNeuralNetwork(nn.Module):
    """Base (abstract) class for all point cloud based binary classifiers."""

    def __init__(self, save_path, learning_rate, weight_decay=None,
                 wandb_project=None, wandb_run=None, silent=False,
                 use_1cycle=False, **model_kwargs):
        super().__init__()
        self.batch = 0
        self.epoch = 0
        self.losses = []
        self.final_activation = nn.CrossEntropyLoss()
        self.save_path = Path(save_path).expanduser()
        if not silent:
            mkdir(self.save_path)
        self.predictions_file = self.save_path / 'predictions.txt'

        self.loss_plot_file = self.save_path / 'loss.png'

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.translated_actives = model_kwargs.get('translated_actives', None)
        self.n_translated_actives = model_kwargs.get('n_translated_actives', 0)

        self.loss_log_file = self.save_path / 'loss.log'

        self.cross_entropy = nn.BCEWithLogitsLoss()

        self.wandb_project = wandb_project
        self.wandb_path = self.save_path / 'wandb_{}'.format(wandb_project)
        self.wandb_run = wandb_run

        self.layers = self.build_net(**model_kwargs)
        self.optimiser = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=weight_decay, eps=1e-3)

        self.use_1cycle = use_1cycle

        if not silent:
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

    def _get_loss(self, y_pred, y_true):
        return self.cross_entropy(y_pred, y_true)

    def forward_pass(self, x):
        return self.forward(x).squeeze()

    def optimise(self, data_loader, epochs=1, epoch_end_validation_set=None):
        """Train the network.

        Trains the neural network. Displays training information and plots the
        loss. All figures and logs are saved to save_path.

        Arguments:
            data_loader: pytorch DataLoader object for training
            epochs: number of complete training cycles
            epoch_end_validation_set: DataLoader on which to perform inference
                at the end of each epoch (if supplied)
        """
        start_time = time.time()
        total_iters = epochs * len(data_loader)
        log_interval = 10
        global_iter = 0
        self.train()
        print()
        print()
        decoy_mean_pred, active_mean_pred = [], []
        if self.use_1cycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimiser, max_lr=self.lr,
                steps_per_epoch=epochs * len(data_loader), epochs=1)
        else:
            scheduler = None
        reported_decoy_pred = reported_active_pred = 0.5
        init_epoch = self.epoch
        for self.epoch in range(init_epoch, epochs):
            for self.batch, (x, y_true, ligands, receptors) in enumerate(
                    data_loader):

                x = self._process_inputs(x)
                y_true = self._get_y_true(y_true).reshape(-1).cuda()
                y_pred = self.forward_pass(x).reshape(-1)

                y_true_np = to_numpy(y_true).reshape((-1,))
                y_pred_np = to_numpy(nn.Sigmoid()(y_pred)).reshape((-1,))

                active_idx = (np.where(y_true_np > 0.5),)
                decoy_idx = (np.where(y_true_np < 0.5),)

                loss = self._get_loss(y_pred, y_true)

                is_actives = bool(sum(y_true_np))
                is_decoys = not bool(np.product(y_true_np))

                if is_actives:
                    active_mean_pred.append(np.mean(y_pred_np[active_idx]))
                if is_decoys:
                    decoy_mean_pred.append(np.mean(y_pred_np[decoy_idx]))

                self.optimiser.zero_grad()
                reported_batch = self.batch + 1
                loss.backward()
                loss_ = float(loss.detach())
                del loss
                if math.isnan(loss_):
                    if hasattr(self, '_get_min_max'):
                        print(self._get_min_max())
                    raise RuntimeError('We have hit a NaN loss value.')
                torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)
                self.optimiser.step()
                self.losses.append(loss_)
                lr = self.optimiser.param_groups[0]['lr']

                if not reported_batch % log_interval or \
                        self.batch == total_iters - 1:
                    self.save_loss(log_interval)
                global_iter += 1

                eta = get_eta(start_time, global_iter, total_iters)
                time_elapsed = format_time(time.time() - start_time)

                if len(active_mean_pred):
                    reported_active_pred = np.mean(active_mean_pred)
                if len(decoy_mean_pred):
                    reported_decoy_pred = np.mean(decoy_mean_pred)

                wandb_update_dict = {
                    'Time remaining (train)': eta,
                    'Binary crossentropy (train)': loss_,
                    'Batch (train)':
                        (self.epoch * len(data_loader) + reported_batch),
                    'Mean decoy prediction (train)': reported_decoy_pred,
                    'Mean active prediction (train)': reported_active_pred,
                    'Examples seen (train)':
                        self.epoch * len(
                            data_loader) * data_loader.batch_size +
                        data_loader.batch_size * self.batch,
                    'Learning rate (train)': lr
                }
                try:
                    try:
                        wandb.log(wandb_update_dict)
                    except wandb.errors.error.Error:
                        pass  # wandb has not been initialised so ignore
                except AttributeError:
                    # New versions of wandb have different structure
                    pass

                if scheduler is not None:
                    scheduler.step()

                print_with_overwrite(
                    (
                        'Epoch:',
                        '{0}/{1}'.format(self.epoch + 1, epochs),
                        '|', 'Batch:', '{0}/{1}'.format(
                            reported_batch, len(data_loader)),
                        'LR:', '{0:.3e}'.format(lr)),
                    ('Time elapsed:', time_elapsed, '|',
                     'Time remaining:', eta),
                    ('Loss: {0:.4f}'.format(loss_), '|',
                     'Mean active: {0:.4f}'.format(reported_active_pred),
                     '|', 'Mean decoy: {0:.4f}'.format(reported_decoy_pred))
                )

            # save after each epoch
            self.save()

            # end of epoch validation if requested
            if epoch_end_validation_set is not None and self.epoch < epochs - 1:
                epoch_end_predictions_fname = Path(
                    self.predictions_file.parent,
                    'predictions_epoch_{}.txt'.format(self.epoch + 1))
                self.test(
                    epoch_end_validation_set,
                    predictions_file=epoch_end_predictions_fname)
            self.train()

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
                y_true = self._get_y_true(y_true).reshape(-1).cuda()
                y_pred = self.forward_pass(x).reshape(-1)

                y_true_np = to_numpy(y_true).reshape((-1,))
                y_pred_np = to_numpy(nn.Sigmoid()(y_pred)).reshape((-1,))

                is_actives = bool(sum(y_true_np))
                is_decoys = not bool(np.product(y_true_np))

                active_idx = (np.where(y_true_np > 0.5),)
                decoy_idx = (np.where(y_true_np < 0.5),)

                eta = get_eta(start_time, self.batch, len(data_loader))
                time_elapsed = format_time(time.time() - start_time)

                wandb_update_dict = {
                    'Time remaining (validation)': eta,
                    'Batch': self.batch + 1
                }

                if is_actives:
                    active_mean_pred = np.mean(y_pred_np[active_idx])
                    wandb_update_dict.update({
                        'Mean active prediction (validation)': active_mean_pred
                    })
                if is_decoys:
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
                    ('Mean active: {0:.4f}'.format(active_mean_pred), '|',
                     'Mean decoy: {0:.4f}'.format(decoy_mean_pred))
                )

                predictions += '\n'.join(['{0} | {1:.7f} {2} {3}'.format(
                    int(y_true_np[i]),
                    y_pred_np[i],
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

        mkdir(save_path.parent)
        torch.save({
            'learning_rate': self.lr,
            'weight_decay': self.weight_decay,
            'epoch': self.epoch + 1,
            'losses': self.losses,
            'model_state_dict': self.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict()
        }, save_path)

    def load_weights(self, checkpoint_file):
        checkpoint = torch.load(str(Path(checkpoint_file).expanduser()))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        self.epoch = checkpoint['epoch']
        if not self.epoch:
            self.epoch += 1
        self.losses = checkpoint['losses']
        print('Sucesfully loaded weights from', checkpoint_file)

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
