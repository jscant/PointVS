"""
Base class for group-invariant networks for pose classificatio and affinity
prediction.
"""

import math
import time
from abc import abstractmethod
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

from rich.console import Group
from rich.live import Live
from rich.progress import Progress, TimeElapsedColumn

from point_vs import logging
from point_vs.global_objects import DEVICE
from point_vs.analysis.top_n import top_n
from point_vs.utils import flatten_nested_iterables
from point_vs.utils import mkdir
from point_vs.utils import to_numpy
from point_vs.utils import expand_path
from point_vs.utils import format_time
from point_vs.utils import load_yaml
from point_vs.utils import get_regression_pearson
from point_vs.utils import find_latest_checkpoint


LOG = logging.get_logger('PointVS')


def _get_progress_ctx():
    return Progress(
        *Progress.get_default_columns(), TimeElapsedColumn(), transient=False,
        refresh_per_second=2)


class PointNeuralNetworkBase(nn.Module):
    """Base (abstract) class for all point cloud based binary classifiers."""

    def __init__(self, save_path, learning_rate, weight_decay=None,
                 wandb_project=None, wandb_run=None, silent=False,
                 use_1cycle=False, warm_restarts=False,
                 only_save_best_models=False, optimiser='adam',
                 regression_loss='mse',
                 **model_kwargs):
        super().__init__()
        self.set_task(model_kwargs.get('model_task', 'classification'))
        self.include_strain_info = False
        self.batch = 0
        self.p_epoch = 0
        self.a_epoch = 0
        self.transformer_encoder = None
        self.save_path = Path(save_path).expanduser()
        self.only_save_best_models = only_save_best_models
        if not silent:
            mkdir(self.save_path)

        self.predictions_file = Path(self.save_path, 'predictions.txt')

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.translated_actives = model_kwargs.get('translated_actives', None)
        self.n_translated_actives = model_kwargs.get('n_translated_actives', 0)

        self.bce = nn.BCEWithLogitsLoss()
        self.regression_loss = nn.MSELoss() if regression_loss == 'mse' else nn.HuberLoss()

        self.wandb_project = wandb_project
        self.wandb_path = self.save_path / f'wandb_{wandb_project}'
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
            raise NotImplementedError(f'{optimiser} not recognised optimiser.')

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
            with open(
                save_path / 'model_kwargs.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(model_kwargs, f)

        pc = self.param_count
        if not silent:
            LOG.info(f'Model parameters: {pc}')
        if self.wandb_project is not None:
            wandb.log({'Parameters': pc})

        self.to(DEVICE)
        self.total_progress = None
        self.epoch_progress = None
        self.validation_progress = None

    @abstractmethod
    def prepare_input(self, x):
        """(Abstract method) make sure inputs are in the correct format."""

    @abstractmethod
    def process_graph(self, graph):
        """(Abstract method) Unpack the graph into tensors."""

    @abstractmethod
    def forward(self, x):
        """(Abstract method) Forward pass for the neural network."""

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
        init_epoch, _ = self.training_setup(data_loader=data_loader, epochs=epochs)
        task_desc_word = 'regression' if 'reg' in self.model_task else 'classification'
        total_desc = '[green]Epoch           {0:7d}' + f'/{epochs-init_epoch:7d} ({task_desc_word})'
        epoch_desc = '[white]Batch           {0:7d}' + f'/{len(data_loader):7d} ({task_desc_word})'
        if epoch_end_validation_set:
            infer_desc = '[pink]Inference       {0:7d}' + f'/{len(epoch_end_validation_set):7d} ({task_desc_word})'
        with _get_progress_ctx() as progress:
            self.total_progress = progress.add_task(
                total_desc.format(1),
                total=len(data_loader) * (epochs-init_epoch))
            self.epoch_progress = progress.add_task(
                epoch_desc.format(1),
                total=len(data_loader))
            if epoch_end_validation_set:
                self.validation_progress = progress.add_task(
                    infer_desc.format(0),
                    total=len(epoch_end_validation_set),
                    start=False)
            epoch = 0
            for _ in range(init_epoch, epochs):
                self.train()
                epoch += 1
                if epoch_end_validation_set:
                    progress.update(self.validation_progress, visible=False)
                    progress.reset(self.validation_progress, start=False)
                progress.reset(self.epoch_progress)
                for self.batch, graph in enumerate(data_loader):
                    progress.refresh()
                    y_pred, y_true, _, _ = self.process_graph(graph)
                    self.get_mean_preds(y_true, y_pred)
                    loss_ = self.backprop(y_true, y_pred)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.global_iter += 1
                    
                    try:  # Used for wandb reporting.
                        self.eta = format_time(
                            progress._tasks[self.total_progress].time_remaining)
                    except KeyError: pass

                    self.record_and_display_info(
                        data_loader=data_loader,
                        loss=loss_,
                        record_type='train')

                    progress.update(
                        self.epoch_progress, advance=1.0,
                        description=epoch_desc.format(self.batch + 1))
                    progress.update(self.total_progress, advance=1.0,
                                    description=total_desc.format(epoch))
                self.eval()
                self.on_epoch_end(
                    epoch_end_validation_set=epoch_end_validation_set,
                    epochs=epochs,
                    top1_on_end=top1_on_end,
                    rich_ctx=progress)


    def val(self, data_loader, predictions_file=None, top1_on_end=False, rich_ctx=None):
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
        predictions_fname = f'{self.model_task_for_fnames}_' + predictions_file.name
        predictions_file = predictions_file.parent / predictions_fname
        predictions_file = Path(predictions_file).expanduser()
        if predictions_file.is_file():
            predictions_file.unlink()
        predictions = ''
        self.total_iters = len(data_loader)
        self.eval()
        self.val_iter = 0
        make_new_val_task_id = rich_ctx is None
        task_desc_word = 'regression' if 'reg' in self.model_task else 'classification'
        infer_desc = '[cyan]Epoch Inference {0:7d}' + f'/{len(data_loader):7d} ({task_desc_word})'
        with _get_progress_ctx() if rich_ctx is None else nullcontext(rich_ctx) as progress:
            if make_new_val_task_id:
                infer_desc = '[red]Final Inference {0:7d}' + f'/{len(data_loader):7d} ({task_desc_word})'
                self.validation_progress = progress.add_task(
                    infer_desc.format(1), total=len(data_loader))
            else:
                progress.update(self.validation_progress, visible=True,
                )
                progress.reset(self.validation_progress)
            progress.update(self.validation_progress, description=infer_desc.format(1))
            
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
                        check_val = y_true[0][0] if isinstance(
                            y_true, (tuple, list)) else y_true[0]
                        is_label = check_val is not None
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
                    try:
                        self.eta = format_time(
                            progress._tasks[self.validation_progress].time_remaining)
                    except KeyError: pass
                    self.record_and_display_info(
                        data_loader, None, record_type='test')

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
                    try: progress.update(self.validation_progress, advance=1.0,
                                         description=infer_desc.format(self.batch + 1))
                    except KeyError: pass

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
                        'Epoch (pose)': self.p_epoch
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
                    'Epoch (affinity)': self.a_epoch
                })
            if self.only_save_best_models and not best:
                return False
        return True

    def get_loss(self, y_true, y_pred):
        """Either bce or mse depending on model task."""
        if self.model_task == 'classification':
            return self.bce(y_pred, y_true.to(DEVICE))
        if self.model_task == 'regression':
            return self.regression_loss(y_pred, y_true.to(DEVICE))
        y_pred[torch.where(y_true == -1)] = -1
        # True loss is only one one, so reverse the mean operation over all 3.
        return 3 * self.regression_loss(y_pred, y_true.to(DEVICE))

    def training_setup(self, data_loader, epochs, model_task=None):
        start_time = time.time()
        if self.use_1cycle:
            self.scheduler = OneCycleLR(
                self.optimiser, max_lr=self.lr,
                steps_per_epoch=epochs * len(data_loader), epochs=1)
        elif self.warm_restarts:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimiser, T_0=len(data_loader), T_mult=1, eta_min=0)
        if model_task is not None:
            self.set_task(model_task)
        init_epoch = self.a_epoch if 'regression' in self.model_task else self.p_epoch
        self.total_iters = (epochs - init_epoch) * len(data_loader)
        return init_epoch, start_time

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
            active_idx = flatten_nested_iterables(
                active_idx, unpack_arrays=True)
            decoy_idx = flatten_nested_iterables(decoy_idx, unpack_arrays=True)
        else:
            y_true_np = y_true_np[y_true_np >= 0]
            is_actives = True
            is_decoys = False
            if self.model_task == 'multi_regression':
                y_pred_np = y_pred_np.reshape(-1, 3)
                y_pred_np = np.mean(y_pred_np, axis=1)
            active_idx = np.arange(len(y_true_np))

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
                LOG.error(self._get_min_max())
            LOG.error('We have hit a NaN loss value.')
            exit(1)
        return loss_

    def record_and_display_info(
            self, data_loader, loss, record_type='train'):
        lr = self.optimiser.param_groups[0]['lr']
        eta = self.eta
        epoch = self.a_epoch if 'regression' in self.model_task else self.p_epoch
        train_val = 'train' if record_type == 'train' else 'validation'
        wandb_update_dict = {
            f'Time remaining ({train_val}, {self.model_task_for_fnames})': eta,
            f'Examples seen ({train_val}, {self.model_task_for_fnames})': 
                epoch * len(data_loader) * data_loader.batch_size + 
                data_loader.batch_size * self.batch,
            
        }
        if record_type == 'train':
            wandb_update_dict.update({
                f'Time remaining (train, {self.model_task_for_fnames})': eta,
                f'{self.model_task_string} (train)': loss,
                f'Batch (train, {self.model_task_for_fnames})':
                    (epoch * len(data_loader) + self.batch + 1),
                f'Examples seen (train, {self.model_task_for_fnames})':
                    epoch * len(
                        data_loader) * data_loader.batch_size +
                    data_loader.batch_size * self.batch,
                f'Learning rate (train, {self.model_task_for_fnames})': lr
            })
            if self.model_task == 'classification':
                wandb_update_dict.update({
                    'Mean active prediction (train)': self.active_mean_pred,
                    'Mean inactive prediction (train)': self.decoy_mean_pred
                })
        elif self.model_task == 'classification':
                wandb_update_dict.update({
                    'Mean active prediction (validation)':
                        self.active_mean_pred,
                    'Mean decoy prediction (validation)':
                        self.decoy_mean_pred,
                })
        try:
            try:
                wandb.log(wandb_update_dict)
            except wandb.errors.error.Error:
                pass  # wandb has not been initialised so ignore
        except AttributeError:
            # New versions of wandb have different structure
            pass

    def on_epoch_end(self, epoch_end_validation_set, epochs, top1_on_end, rich_ctx=None):
        # save after each epoch
        if 'regression' in self.model_task:
            self.a_epoch += 1
            epoch = self.a_epoch
        else:
            self.p_epoch += 1
            epoch = self.p_epoch
        if not self.only_save_best_models:
            self.save()
        # end of epoch validation if requested
        if epoch_end_validation_set is not None and epoch < epochs:
            epoch_end_predictions_fname = Path(
                self.predictions_file.parent,
                f'predictions_epoch_{epoch}.txt')
            best = self.val(
                epoch_end_validation_set,
                predictions_file=epoch_end_predictions_fname,
                top1_on_end=top1_on_end, rich_ctx=rich_ctx)
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

        epoch = self.a_epoch if 'regression' in self.model_task else self.p_epoch
        if save_path is None:
            fname = f'{self.model_task_for_fnames}_ckpt_epoch_{epoch}.pt'
            save_path = self.save_path / 'checkpoints' / fname

        mkdir(save_path.parent)
        torch.save({
            'learning_rate': self.lr,
            'weight_decay': self.weight_decay,
            'p_epoch': self.p_epoch,
            'a_epoch': self.a_epoch,
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
        checkpoint = torch.load(str(checkpoint_file), map_location=DEVICE)
        if self.model_task == load_yaml(
                expand_path(checkpoint_file).parents[1] /
                'model_kwargs.yaml').get('model_task', 'classification'):
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
            self.p_epoch = checkpoint.get('p_epoch', checkpoint.get('epoch', 0))
            self.a_epoch = checkpoint.get('a_epoch', 0)
        else:
            own_state = self.state_dict()
            for name, param in checkpoint['model_state_dict'].items():
                own_state[name].copy_(param)
        if not silent:
            try:
                pth = Path(checkpoint_file).relative_to(expand_path(Path('.')))
            except ValueError:
                pth = Path(checkpoint_file)
            LOG.info(f'Sucesfully loaded weights from {pth}')

    @property
    def param_count(self):
        return sum(
            [torch.numel(t) for t in self.parameters() if t.requires_grad])

    def set_task(self, task):
        if task not in ('classification', 'regression', 'multi_regression'):
            raise ValueError('Argument for set_task must be one of '
                             'classification, regression or multi_regression')
        self.model_task = task
        if 'regression' in task:
            self.model_task_for_fnames = 'affinity'
            self.model_task_string = 'Mean squared error'
        else:
            self.model_task_for_fnames = 'pose'
            self.model_task_string = 'Binary crossentropy'
