import math
import time
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

from point_vs.analysis.top_n import top_n
from point_vs.utils import get_eta, format_time, print_with_overwrite, mkdir, \
    to_numpy, expand_path, load_yaml, get_regression_pearson, \
    find_latest_checkpoint

_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PointNeuralNetworkBase(nn.Module):
    """Base (abstract) class for all point cloud based binary classifiers."""

    def __init__(self, save_path, learning_rate, weight_decay=None,
                 wandb_project=None, wandb_run=None, silent=False,
                 use_1cycle=False, warm_restarts=False,
                 only_save_best_models=False, optimiser='adam',
                 **model_kwargs):
        super().__init__()
        self.model_task = model_kwargs.get('model_task', 'classification')
        self.include_strain_info = False
        self.batch = 0
        self.epoch = 0
        self.losses = []
        self.feats_linear_layers = None
        self.edges_linear_layers = None
        self.transformer_encoder = None
        self.save_path = Path(save_path).expanduser()
        self.linear_gap = model_kwargs.get('linear_gap', True)
        self.only_save_best_models = only_save_best_models
        if not silent:
            mkdir(self.save_path)
        self.predictions_file = self.save_path / 'predictions.txt'

        self.loss_plot_file = self.save_path / 'loss.png'

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.translated_actives = model_kwargs.get('translated_actives', None)
        self.n_translated_actives = model_kwargs.get('n_translated_actives', 0)

        self.loss_log_file = self.save_path / 'loss.log'

        if self.model_task == 'classification':
            self.loss_function = nn.BCEWithLogitsLoss()
            self.model_task_string = 'Binary crossentropy'
        elif self.model_task.endswith('regression'):
            self.loss_function = nn.MSELoss()
            self.model_task_string = 'Mean squared error'
        else:
            raise RuntimeError(
                'model_task must be either classification or regression')

        self.wandb_project = wandb_project
        self.wandb_path = self.save_path / 'wandb_{}'.format(wandb_project)
        self.wandb_run = wandb_run

        self.n_layers = model_kwargs.get('num_layers', 12)
        self.layers = self.build_net(**model_kwargs)
        if optimiser == 'adam':
            self.optimiser = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=weight_decay)
        elif optimiser == 'sgd':
            self.optimiser = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True)
        else:
            raise NotImplementedError('{} not recognised optimiser.'.format(
                optimiser))

        assert not (use_1cycle and warm_restarts), '1cycle nad warm restarts ' \
                                                   'are mutually exclusive'

        self.use_1cycle = use_1cycle
        self.warm_restarts = warm_restarts

        self.global_iter = 0
        self.val_iter = 0
        self.decoy_mean_pred, self.active_mean_pred = 0.5, 0.5
        self.log_interval = 10
        self.scheduler = None  # will change this in training preamble
        self.test_metric = 0

        if not silent:
            with open(save_path / 'model_kwargs.yaml', 'w') as f:
                yaml.dump(model_kwargs, f)

        pc = self.param_count
        if not silent:
            print('Model parameters:', pc)
        if self.wandb_project is not None:
            wandb.log({'Parameters': pc})

        self.to(_device)

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
            self.train()
            for self.batch, graph in enumerate(data_loader):
                y_pred, y_true, ligands, receptors = self.process_graph(graph)
                self.get_mean_preds(y_true, y_pred)
                loss_ = self.backprop(y_true, y_pred)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.global_iter += 1
                self.record_and_display_info(
                    start_time=start_time,
                    epochs=epochs,
                    data_loader=data_loader,
                    loss=loss_,
                    record_type='train'
                )
            self.eval()
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
        if predictions_file is None:
            predictions_file = self.predictions_file
        predictions_file = Path(predictions_file).expanduser()
        if predictions_file.is_file():
            predictions_file.unlink()
        predictions = ''
        self.total_iters = len(data_loader)
        self.eval()
        self.val_iter = 0
        val_start_time = time.time()

        with torch.no_grad():
            for self.batch, graph in enumerate(
                    data_loader):

                self.val_iter += 1
                y_pred, y_true, ligands, receptors = self.process_graph(graph)


                if self.model_task == 'classification':
                    is_label = y_true is not None
                    if is_label:
                        y_true_np = to_numpy(y_true).reshape((-1,))
                    y_pred_np = to_numpy(nn.Sigmoid()(y_pred)).reshape((-1,))
                    num_type = int
                elif self.model_task == 'multi_regression':
                    is_label = y_true[0][0] is not None
                    y_pred_np = to_numpy(y_pred).reshape((-1, 3))
                    if is_label:
                        y_true_np = to_numpy(y_true).reshape((-1, 3))
                        metrics = np.array(
                            [['pki', 'pkd', 'ic50'] for _ in
                             range(len(ligands))])
                        metrics = list(metrics[np.where(y_true_np > -0.5)])
                        y_pred_np = y_pred_np[np.where(y_true_np > -0.5)]
                        y_true_np = y_true_np[np.where(y_true_np > -0.5)]
                    num_type = float
                else:
                    is_label = y_true is not None
                    if is_label:
                        y_true_np = to_numpy(y_true).reshape((-1,))
                    y_pred_np = to_numpy(y_pred).reshape((-1,))
                    num_type = float

                self.get_mean_preds(y_true, y_pred, is_label=is_label)
                self.record_and_display_info(
                    val_start_time, None, data_loader, None, record_type='test')

                if self.model_task == 'multi_regression':
                    if is_label:
                        predictions += '\n'.join(
                            ['{0:.3f} | {1:.3f} {2} {3} | {4}'.format(
                                num_type(y_true_np[i]),
                                y_pred_np[i],
                                receptors[i],
                                ligands[i],
                                metrics[i]) for i in range(len(receptors))]
                        ) + '\n'
                    else:
                        predictions += '\n'.join(
                            ['{0:.3f} {1:.3f} {2:.3f} | {3} {4}'.format(
                                *y_pred_np[i],
                                receptors[i],
                                ligands[i]) for i in range(len(receptors))]
                        ) + '\n'
                else:
                    if is_label:
                        predictions += '\n'.join(
                            ['{0:.3f} | {1:.3f} {2} {3}'.format(
                                num_type(y_true_np[i]),
                                y_pred_np[i],
                                receptors[i],
                                ligands[i]) for i in range(len(receptors))]
                        ) + '\n'
                    else:
                        predictions += '\n'.join(
                            ['{0:.3f} | {1} {2}'.format(
                                y_pred_np[i],
                                receptors[i],
                                ligands[i]) for i in range(len(receptors))]
                        ) + '\n'

                predictions = self.write_predictions(
                    predictions,
                    predictions_file,
                    data_loader
                )

        if top1_on_end:
            if self.model_task == 'classification':
                top_1 = top_n(predictions_file)
                if top_1 > self.test_metric:
                    self.test_metric = top_1
                    best = True
                else:
                    best = False
                try:
                    wandb.log({
                        'Validation Top1': top_1,
                        'Best validation Top1': self.test_metric,
                        'Epoch': self.epoch + 1
                    })
                except Exception:
                    pass  # wandb has not been initialised so ignore
            else:
                r, p = get_regression_pearson(predictions_file)
                if p < 0.05 and r > self.test_metric:
                    self.test_metric = r
                    best = True
                else:
                    best = False
                wandb.log({
                    'Pearson''s correlation coefficient': r,
                    'Best PCC': self.test_metric,
                    'Epoch': self.epoch + 1
                })
            if self.only_save_best_models and not best:
                return False
        return True

    def get_loss(self, y_true, y_pred):
        if self.model_task != 'multi_regression':
            return self.loss_function(y_pred, y_true.to(_device))
        y_pred[torch.where(y_true == -1)] = -1
        return 3 * self.loss_function(y_pred, y_true.to(_device))

    def training_setup(self, data_loader, epochs):
        start_time = time.time()
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

    def get_mean_preds(self, y_true, y_pred, is_label=True):
        y_pred_np = to_numpy(nn.Sigmoid()(y_pred)).reshape((-1,))
        if is_label:
            y_true_np = to_numpy(y_true).reshape((-1,))
        else:
            self.active_mean_pred = np.mean(y_pred_np)
            return

        if self.model_task == 'classification':
            active_idx = (np.where(y_true_np > 0.5),)
            decoy_idx = (np.where(y_true_np < 0.5),)
            is_actives = bool(sum(y_true_np))
            is_decoys = not bool(np.product(y_true_np))
        else:
            active_idx = (np.where(y_true_np > -np.inf),)
            decoy_idx = (np.where(y_true_np < np.inf),)
            is_actives = True
            is_decoys = False

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

        if record_type == 'train':
            eta = get_eta(start_time, self.global_iter, self.total_iters)
        else:
            eta = get_eta(start_time, self.val_iter, len(data_loader))

        time_elapsed = format_time(time.time() - start_time)

        if record_type == 'train':
            wandb_update_dict = {
                'Time remaining (train)': eta,
                '{} (train)'.format(self.model_task_string): loss,
                'Batch (train)':
                    (self.epoch * len(data_loader) + self.batch + 1),
                'Examples seen (train)':
                    self.epoch * len(
                        data_loader) * data_loader.batch_size +
                    data_loader.batch_size * self.batch,
                'Learning rate (train)': lr
            }
            if self.model_task == 'classification':
                wandb_update_dict.update({
                    'Mean active prediction (train)': self.active_mean_pred,
                    'Mean inactive prediction (train)': self.decoy_mean_pred
                })
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
                print_with_overwrite(
                    (
                        'Epoch:',
                        '{0}/{1}'.format(self.epoch + 1, epochs),
                        '|', 'Batch:', '{0}/{1}'.format(
                            self.batch + 1, len(data_loader)),
                        'LR:', '{0:.3e}'.format(lr)),
                    ('Time elapsed:', time_elapsed, '|',
                     'Time remaining:', eta),
                    ('Loss: {0:.4f}'.format(loss),)
                )
        else:
            wandb_update_dict = {
                'Time remaining (validation)': eta,
                'Examples seen (validation)':
                    self.epoch * len(
                        data_loader) * data_loader.batch_size +
                    data_loader.batch_size * self.batch,
            }
            if self.model_task == 'classification':
                wandb_update_dict.update({
                    'Mean active prediction (validation)':
                        self.active_mean_pred,
                    'Mean decoy prediction (validation)':
                        self.decoy_mean_pred,
                })
                print_with_overwrite(
                    ('Inference on: {}'.format(data_loader.dataset.base_path),
                     '|', 'Iteration:', '{0}/{1}'.format(
                        self.batch + 1, len(data_loader))),
                    ('Time elapsed:', time_elapsed, '|',
                     'Time remaining:', eta),
                    ('Mean active: {0:.4f}'.format(self.active_mean_pred), '|',
                     'Mean decoy: {0:.4f}'.format(self.decoy_mean_pred))
                )
            else:
                print_with_overwrite(
                    ('Inference on: {}'.format(data_loader.dataset.base_path),
                     '|', 'Iteration:', '{0}/{1}'.format(
                        self.batch + 1, len(data_loader))),
                    ('Time elapsed:', time_elapsed, '|',
                     'Time remaining:', eta)
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
        if not self.only_save_best_models:
            self.save()
        # end of epoch validation if requested
        if epoch_end_validation_set is not None and self.epoch < epochs - 1:
            epoch_end_predictions_fname = Path(
                self.predictions_file.parent,
                'predictions_epoch_{}.txt'.format(self.epoch + 1))
            best = self.val(
                epoch_end_validation_set,
                predictions_file=epoch_end_predictions_fname,
                top1_on_end=top1_on_end)
            if self.only_save_best_models and best:
                self.save()

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

    @staticmethod
    def _transform_names(d):
        """For backwards compatability with some older trained models."""
        new_d = OrderedDict()
        for key, value in d.items():
            new_d[key.replace('edge_attention_mlp', 'att_mlp').replace(
                'node_attention_mlp', 'node_att_mlp')] = value
        return new_d

    def load_weights(self, checkpoint_file, silent=False):
        """All this crap is required because I renamed some things ages ago."""
        checkpoint_file = expand_path(checkpoint_file)
        if checkpoint_file.is_dir():
            checkpoint_file = find_latest_checkpoint(checkpoint_file)
        checkpoint = torch.load(str(checkpoint_file), map_location=_device)
        if self.model_task == load_yaml(
                expand_path(checkpoint_file).parents[1] /
                'model_kwargs.yaml').get('model_task', 'classification'):
            rename = False
            try:
                self.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError:
                for layer in self.layers:
                    if hasattr(layer, 'att_mlp'):
                        layer.att_mlp = nn.Sequential(
                            nn.Identity(),  # Compatability
                            nn.Identity(),  # Compatability
                            nn.Linear(layer.hidden_nf, 1),
                            layer.attention_activation())
                try:
                    self.load_state_dict(checkpoint['model_state_dict'])
                except RuntimeError:
                    rename = True
                    self.load_state_dict(self._transform_names(
                        checkpoint['model_state_dict']))
            self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
            self.epoch = checkpoint['epoch']
            if not self.epoch:
                self.epoch += 1
            self.losses = checkpoint['losses']
            if rename:
                for layer in self.layers:
                    if hasattr(layer, 'att_mlp'):
                        layer.att_mlp = layer.att_mlp
        else:
            own_state = self.state_dict()
            for name, param in checkpoint['model_state_dict'].items():
                own_state[name].copy_(param)
        if not silent:
            try:
                pth = Path(checkpoint_file).relative_to(expand_path(Path('.')))
            except ValueError:
                pth = Path(checkpoint_file)
            print('Sucesfully loaded weights from', pth)

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
