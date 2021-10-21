import math
import time
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import nn
from torch_geometric.nn import global_mean_pool

from point_vs.analysis.top1 import top_1
from point_vs.models.point_neural_network import PointNeuralNetwork
from point_vs.utils import get_eta, format_time, print_with_overwrite, to_numpy


class PygLinearPass(nn.Module):
    """Helper class for neater forward passes.

    Gives a linear layer with the same semantic behaviour as the E_GCL and
    EGNN_Sparse layers.

    Arguments:
        module: nn.Module (usually a linear layer)
        feats_appended_to_coords: does the input include coordinates in the
            first three columns of the node feature vector
        return_coords_and_edges: return a tuple containing the node features,
            the coords and the edges rather than just the node features
    """

    def __init__(self, module, feats_appended_to_coords=False,
                 return_coords_and_edges=False):
        super().__init__()
        self.m = module
        self.feats_appended_to_coords = feats_appended_to_coords
        self.return_coords_and_edges = return_coords_and_edges

    def forward(self, x, *args, **kwargs):
        if self.feats_appended_to_coords:
            feats = x[:, 3:]
            res = torch.hstack([x[:, :3], self.m(feats)])
        else:
            res = self.m(x)
        if self.return_coords_and_edges:
            return res, kwargs['coord'], kwargs['edge_attr']
        return res


class PygPointNeuralNetwork(PointNeuralNetwork):
    """Base (abstract) class for all point cloud based binary classifiers."""

    def optimise(self, data_loader, epochs=1, epoch_end_validation_set=None,
                 top1_on_end=False):
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
        self.train()
        print()
        print()
        decoy_mean_pred, active_mean_pred = [], []
        if self.use_1cycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimiser, max_lr=self.lr,
                steps_per_epoch=epochs * len(data_loader), epochs=1)
            print('Using 1cycle')
        elif self.warm_restarts:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimiser, T_0=len(data_loader), T_mult=1, eta_min=0)
            print('Using CosineAnnealingWarmRestarts')
        else:
            print('Using a flat learning rate')
            scheduler = None
        reported_decoy_pred = reported_active_pred = 0.5
        init_epoch = self.epoch
        global_iter = 0
        for self.epoch in range(init_epoch, epochs):
            for self.batch, graph in enumerate(
                    data_loader):

                y_true = graph.y.float()
                y_pred = self.forward_pass(graph)

                y_true_np = np.array(y_true).reshape((-1,))
                y_pred_np = to_numpy(nn.Sigmoid()(y_pred)).reshape((-1,))

                active_idx = (np.where(y_true_np > 0.5),)
                decoy_idx = (np.where(y_true_np < 0.5),)

                loss = self._get_loss(y_pred, y_true.cuda())

                is_actives = bool(sum(y_true_np))
                is_decoys = not bool(np.product(y_true_np))

                if is_actives:
                    active_mean_pred.append(np.mean(y_pred_np[active_idx]))
                if is_decoys:
                    decoy_mean_pred.append(np.mean(y_pred_np[decoy_idx]))

                max_data = max([np.amax(np.abs(to_numpy(p.data))) for p in
                                self.parameters()])
                min_data = min([np.amin(np.abs(to_numpy(p.data))) for p in
                                self.parameters()])
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

                eta = get_eta(start_time, global_iter,
                              total_iters - (len(data_loader) * init_epoch))
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
                    'Learning rate (train)': lr,
                    'Maximum parameter': max_data,
                    'Minimum parameter': min_data,
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
                    predictions_file=epoch_end_predictions_fname,
                    top1_on_end=top1_on_end)
            self.train()

    def test(self, data_loader, predictions_file=None, top1_on_end=False):
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
            for self.batch, graph in enumerate(
                    data_loader):

                y_true = graph.y.float()
                y_pred = self.forward_pass(graph)

                y_true_np = np.array(y_true).reshape((-1,))
                y_pred_np = to_numpy(nn.Sigmoid()(y_pred)).reshape((-1,))

                ligands = graph.lig_fname
                receptors = graph.rec_fname

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
        if top1_on_end:
            try:
                wandb.log({
                    'Validation Top1 at end of epoch {}'.format(self.epoch + 1):
                        top_1(predictions_file)
                })
            except Exception:
                pass  # wandb has not been initialised so ignore

    @abstractmethod
    def get_embeddings(self, feats, edges, coords, edge_attributes, batch):
        """Implement code to go from input features to final node embeddings."""
        pass

    def forward(self, graph):
        feats = graph.x.float().cuda()
        edges = graph.edge_index.cuda()
        coords = graph.pos.float().cuda()
        edge_attributes = graph.edge_attr.cuda()
        batch = graph.batch.cuda()
        feats = self.get_embeddings(
            feats, edges, coords, edge_attributes, batch)
        if self.linear_gap:
            feats = self.layers[-1](feats, edges, edge_attributes, batch)
            feats = global_mean_pool(feats, graph.batch.cuda())
        else:
            feats = global_mean_pool(feats, graph.batch.cuda())
            feats = self.layers[-1](feats, edges, edge_attributes, batch)
        return feats
