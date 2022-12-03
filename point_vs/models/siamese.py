"""For Ruben (unused by @jscant)"""
import time
from abc import ABC
from pathlib import Path

import torch
import wandb
import yaml
from torch import nn
from torch.nn.functional import silu

from point_vs import logging
from point_vs.global_objects import DEVICE
from point_vs.analysis.top_n import top_n
from point_vs.models.geometric.pnn_geometric_base import PNNGeometricBase
from point_vs.models.point_neural_network_base import PointNeuralNetworkBase
from point_vs.utils import mkdir, to_numpy


LOG = logging.get_logger('PointVS')


class SiameseNeuralNetwork(PNNGeometricBase, ABC):
    """Siamese type point cloud neural network."""
    def __init__(self, gnn_class, save_path, learning_rate, weight_decay=None,
                 wandb_project=None, wandb_run=None, silent=False,
                 use_1cycle=False, warm_restarts=False, **model_kwargs):
        super(PointNeuralNetworkBase, self).__init__()

        model_kwargs['dim_output'] = 128
        embed_size = model_kwargs['dim_output']

        self.rec_nn = gnn_class(save_path, learning_rate, weight_decay,
                                silent=True, **model_kwargs)

        model_kwargs['dim_output'] = 64
        embed_size += model_kwargs['dim_output']

        model_kwargs['update_coords'] = False
        self.lig_nn = gnn_class(save_path, learning_rate, weight_decay,
                                silent=True, **model_kwargs)

        self.batch = 0
        self.epoch = 0
        self.losses = []

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

        self.linear_layers = [
            nn.Linear(embed_size, 64).to(DEVICE), nn.Linear(64, 32).to(DEVICE),
            nn.Linear(32, 1).to(DEVICE)]
        self.activation_layers = [nn.SiLU().to(DEVICE), nn.SiLU().to(DEVICE),
                                  nn.Identity().to(DEVICE)]

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
            with open(save_path / 'model_kwargs.yaml', 'w') as f:  #pylint: disable=unspecified-encoding
                yaml.dump(model_kwargs, f)

        pc = self.param_count
        LOG.info('Model parameters:', pc)
        if self.wandb_project is not None:
            wandb.log({'Parameters': pc})

        self.to(DEVICE)

    def forward(self, rec_graph, lig_graph):
        rec_embedding = self.rec_nn(rec_graph)
        lig_embedding = self.lig_nn(lig_graph)
        embedding = torch.cat([rec_embedding, lig_embedding], dim=1)
        x = silu(embedding)
        for linear, act in zip(self.linear_layers, self.activation_layers):
            x = act(linear(x))
        return x

    def process_graph(self, rec_graph, lig_graph):
        y_true = rec_graph.y.float()
        y_pred = self(rec_graph, lig_graph).reshape(-1, )
        ligands = lig_graph.lig_fname
        receptors = rec_graph.rec_fname
        return y_pred, y_true, ligands, receptors

    def train_model(self, data_loaders, epochs=1, epoch_end_validation_set=None,
                    top1_on_end=False):
        """Train the network.

        Trains the neural network. Displays training information and plots the
        loss. All figures and logs are saved to save_path.

        Arguments:
            data_loaders: tuple containing the receptor and ligand data loaders
            epochs: number of complete training cycles
            epoch_end_validation_set: DataLoader on which to perform inference
                at the end of each epoch (if supplied)
            top1_on_end: calculate the top1 at the end of each epoch
        """
        rec_dl, lig_dl = data_loaders
        start_time = self.training_setup(data_loader=rec_dl, epochs=epochs)
        for self.epoch in range(self.init_epoch, epochs):
            for self.batch, (rec_graph, lig_graph) in enumerate(
                    zip(rec_dl, lig_dl)):
                y_pred, y_true, ligands, receptors = self.process_graph(
                    rec_graph, lig_graph)
                self.get_mean_preds(y_true, y_pred)
                loss_ = self.backprop(y_true, y_pred)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.record_and_display_info(
                    start_time=start_time,
                    epochs=epochs,
                    data_loader=rec_dl,
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
            for self.batch, (rec_graph, lig_graph) in enumerate(
                    data_loader):
                y_pred, y_true, ligands, receptors = self.process_graph(
                    rec_graph, lig_graph)

                y_true_np = to_numpy(y_true).reshape((-1,))
                y_pred_np = to_numpy(nn.Sigmoid()(y_pred)).reshape((-1,))

                self.get_mean_preds(y_true, y_pred)
                self.record_and_display_info(
                    start_time, None, data_loader[0], None, record_type='test')

                predictions += '\n'.join(['{0} | {1:.7f} {2} {3}'.format(
                    int(y_true_np[i]),
                    y_pred_np[i],
                    receptors[i],
                    ligands[i]) for i in range(len(receptors))]) + '\n'

                predictions = self.write_predictions(
                    predictions,
                    predictions_file,
                    data_loader[0]
                )

        if top1_on_end:
            try:
                top_1 = top_n(predictions_file)
                wandb.log({
                    'Validation Top1 at end of epoch {}'.format(self.epoch + 1):
                        top_1,
                    'Validation Top1'.format(self.epoch + 1):
                        top_1,
                    'Epoch': self.epoch + 1
                })
            except Exception:
                pass  # wandb has not been initialised so ignore
