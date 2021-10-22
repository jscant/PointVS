import math
import time
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

from point_vs.analysis.top_n import top_n
from point_vs.utils import get_eta, format_time, print_with_overwrite, mkdir, \
    to_numpy


class PointNeuralNetworkBase(nn.Module):
    """Base (abstract) class for all point cloud based binary classifiers."""

    def __init__(self, save_path, learning_rate, weight_decay=None,
                 wandb_project=None, wandb_run=None, silent=False,
                 use_1cycle=False, warm_restarts=False, **model_kwargs):
        super().__init__()
        self.batch = 0
        self.epoch = 0
        self.losses = []
        self.final_activation = nn.CrossEntropyLoss()
        self.save_path = Path(save_path).expanduser()
        self.linear_gap = model_kwargs.get('linear_gap', True)
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

        self.n_layers = model_kwargs.get('num_layers', 12)
        self.layers = self.build_net(**model_kwargs)
        self.optimiser = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=weight_decay)

        assert not (use_1cycle and warm_restarts), '1cycle nad warm restarts ' \
                                                   'are mutually exclusive'

        self.use_1cycle = use_1cycle
        self.warm_restarts = warm_restarts

        self.global_iter = 0
        self.decoy_mean_pred, self.active_mean_pred = 0.5, 0.5
        self.log_interval = 10
        self.scheduler = None  # will change this in training preamble

        if not silent:
            with open(save_path / 'model_kwargs.yaml', 'w') as f:
                yaml.dump(model_kwargs, f)

        pc = self.param_count
        print('Model parameters:', pc)
        if self.wandb_project is not None:
            wandb.log({'Parameters', pc})

        self.cuda()

    @abstractmethod
    def prepare_input(self, x):
        pass

    @abstractmethod
    def process_graph(self, graph):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def train_model(self, data_loader, epochs=1, epoch_end_validation_set=None,
                    top1_on_end=False):
        """Train the network.

        Trains the neural network. Displays training information and plots the
        loss. All figures and logs are saved to save_path.

        Arguments:
            data_loader: pytorch DataLoader object for training
            epochs: number of complete training cycles
            epoch_end_validation_set: DataLoader on which to perform inference
                at the end of each epoch (if supplied)
            top1_on_end:
        """
        start_time = self.training_setup(data_loader=data_loader, epochs=epochs)
        for self.epoch in range(self.init_epoch, epochs):
            for self.batch, graph in enumerate(data_loader):
                y_pred, y_true, ligands, receptors = self.process_graph(graph)
                self.get_mean_preds(y_true, y_pred)
                loss_ = self.backprop(y_true, y_pred)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.record_and_display_info(
                    start_time=start_time,
                    epochs=epochs,
                    data_loader=data_loader,
                    loss=loss_,
                    record_type='train'
                )
            self.on_epoch_end(
                epoch_end_validation_set=epoch_end_validation_set,
                epochs=epochs,
                top1_on_end=top1_on_end)

    def val(self, data_loader, predictions_file=None, top1_on_end=False):
        """Use trained network to perform inference on the test set.

        Uses the neural network (in Session.network), to perform predictions
        on the structures found in <test_data_root>, and saves this output
        to <save_path>/predictions_<test_data_root.name>.txt.

        Arguments:
            data_loader:
            predictions_file:
            top1_on_end:
        """
        start_time = time.time()
        if predictions_file is None:
            predictions_file = self.predictions_file
        predictions_file = Path(predictions_file).expanduser()
        if predictions_file.is_file():
            predictions_file.unlink()
        predictions = ''
        with torch.no_grad():
            for self.batch, graph in enumerate(
                    data_loader):
                y_pred, y_true, ligands, receptors = self.process_graph(graph)

                y_true_np = to_numpy(y_true).reshape((-1,))
                y_pred_np = to_numpy(nn.Sigmoid()(y_pred)).reshape((-1,))

                self.get_mean_preds(y_true, y_pred)
                self.record_and_display_info(
                    start_time, None, data_loader, None, record_type='test')

                predictions += '\n'.join(['{0} | {1:.7f} {2} {3}'.format(
                    int(y_true_np[i]),
                    y_pred_np[i],
                    receptors[i],
                    ligands[i]) for i in range(len(receptors))]) + '\n'

                predictions = self.write_predictions(
                    predictions,
                    predictions_file,
                    data_loader
                )

        if top1_on_end:
            try:
                wandb.log({
                    'Validation Top1 at end of epoch {}'.format(self.epoch + 1):
                        top_n(predictions_file)
                })
            except Exception:
                pass  # wandb has not been initialised so ignore

    def get_loss(self, y_true, y_pred):
        return self.cross_entropy(y_pred, y_true.cuda())

    def training_setup(self, data_loader, epochs):
        start_time = time.time()
        self.train()
        if self.use_1cycle:
            print('Using 1cycle')
            self.scheduler = OneCycleLR(
                self.optimiser, max_lr=self.lr,
                steps_per_epoch=epochs * len(data_loader), epochs=1)
            print('Using 1cycle')
        elif self.warm_restarts:
            print('Using CosineAnnealingWarmRestarts')
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimiser, T_0=len(data_loader), T_mult=1, eta_min=0)
        else:
            print('Using a flat learning rate')
        print()
        print()
        self.init_epoch = self.epoch
        self.total_iters = epochs * len(data_loader)
        return start_time

    def get_mean_preds(self, y_true, y_pred):
        y_true_np = to_numpy(y_true).reshape((-1,))
        y_pred_np = to_numpy(nn.Sigmoid()(y_pred)).reshape((-1,))

        active_idx = (np.where(y_true_np > 0.5),)
        decoy_idx = (np.where(y_true_np < 0.5),)

        is_actives = bool(sum(y_true_np))
        is_decoys = not bool(np.product(y_true_np))

        if is_actives:
            self.active_mean_pred = np.mean(y_pred_np[active_idx])
        if is_decoys:
            self.decoy_mean_pred = np.mean(y_pred_np[decoy_idx])

    def backprop(self, y_true, y_pred):
        loss = self.get_loss(y_true, y_pred)
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)
        self.optimiser.step()
        loss_ = float(to_numpy(loss))
        if math.isnan(loss_):
            if hasattr(self, '_get_min_max'):
                print(self._get_min_max())
            raise RuntimeError('We have hit a NaN loss value.')
        self.losses.append(loss_)
        return loss_

    def record_and_display_info(
            self, start_time, epochs, data_loader, loss, record_type='train'):
        lr = self.optimiser.param_groups[0]['lr']
        if not (self.batch + 1) % self.log_interval or \
                self.batch == self.total_iters - 1:
            self.save_loss(self.log_interval)
        self.global_iter += 1

        eta = get_eta(start_time, self.global_iter,
                      self.total_iters - (len(data_loader) * self.init_epoch))
        time_elapsed = format_time(time.time() - start_time)

        if record_type == 'train':
            wandb_update_dict = {
                'Time remaining (train)': eta,
                'Binary crossentropy (train)': loss,
                'Batch (train)':
                    (self.epoch * len(data_loader) + self.batch + 1),
                'Mean active prediction (train)': self.active_mean_pred,
                'Mean decoy prediction (train)': self.decoy_mean_pred,
                'Examples seen (train)':
                    self.epoch * len(
                        data_loader) * data_loader.batch_size +
                    data_loader.batch_size * self.batch,
                'Learning rate (train)': lr
            }
            print_with_overwrite(
                (
                    'Epoch:',
                    '{0}/{1}'.format(self.epoch + 1, epochs),
                    '|', 'Batch:', '{0}/{1}'.format(
                        self.batch + 1, len(data_loader)),
                    'LR:', '{0:.3e}'.format(lr)),
                ('Time elapsed:', time_elapsed, '|',
                 'Time remaining:', eta),
                ('Loss: {0:.4f}'.format(loss), '|',
                 'Mean active: {0:.4f}'.format(self.active_mean_pred),
                 '|', 'Mean decoy: {0:.4f}'.format(self.decoy_mean_pred))
            )
        else:
            wandb_update_dict = {
                'Time remaining (validation)': eta,
                'Batch': self.batch + 1,
                'Mean active prediction (validation)':
                    self.active_mean_pred,
                'Mean decoy prediction (validation)':
                    self.decoy_mean_pred,
            }
            print_with_overwrite(
                ('Inference on: {}'.format(data_loader.dataset.base_path),
                 '|', 'Iteration:', '{0}/{1}'.format(
                    self.batch + 1, len(data_loader))),
                ('Time elapsed:', time_elapsed, '|',
                 'Time remaining:', eta),
                ('Mean active: {0:.4f}'.format(self.active_mean_pred), '|',
                 'Mean decoy: {0:.4f}'.format(self.decoy_mean_pred))
            )
        try:
            try:
                wandb.log(wandb_update_dict)
            except wandb.errors.error.Error:
                pass  # wandb has not been initialised so ignore
        except AttributeError:
            # New versions of wandb have different structure
            pass

    def on_epoch_end(self, epoch_end_validation_set, epochs, top1_on_end):
        # save after each epoch
        self.save()

        # end of epoch validation if requested
        if epoch_end_validation_set is not None and self.epoch < epochs - 1:
            epoch_end_predictions_fname = Path(
                self.predictions_file.parent,
                'predictions_epoch_{}.txt'.format(self.epoch + 1))
            self.val(
                epoch_end_validation_set,
                predictions_file=epoch_end_predictions_fname,
                top1_on_end=top1_on_end)
        self.train()

    def write_predictions(self, predictions_str, predictions_file, data_loader):
        # Periodically write predictions to disk
        if not (self.batch + 1) % self.log_interval or self.batch == len(
                data_loader) - 1:
            with open(predictions_file, 'a') as f:
                f.write(predictions_str)
                return ''
        return predictions_str

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
