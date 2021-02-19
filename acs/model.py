"""Modified from https://github.com/rpinsler/active-bayesian-coresets"""

import time

import gtimer as gt
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

import acs.utils as utils
from acs.al_data_set import Dataset
from lieconv_utils import print_with_overwrite, format_time, get_eta


def collate(batch):
    """Processing of inputs which takes place after batch is selected.

    LieConv networks take tuples of torch tensors (p, v, m), which are:
        p, (batch_size, n_atoms, 3): coordinates of each atom
        v, (batch_size, n_atoms, n_features): features for each atom
        m, (batch_size, n_atoms): mask for each coordinate slot

    Note that n_atoms is the largest number of atoms in a structure in
    each batch.

    Arguments:
        batch: iterable of individual inputs.

    Returns:
        Tuple of feature vectors ready for input into a LieConv network.
    """
    max_len = max([b[0][-1] for b in batch])
    batch_size = len(batch)
    p_batch = torch.zeros(batch_size, max_len, 3)
    v_batch = torch.zeros(batch_size, max_len, 12)
    m_batch = torch.zeros(batch_size, max_len)
    label_batch = torch.zeros(batch_size, 1)
    ligands, receptors = [], []
    for batch_index, ((p, v, m, _), ligand, receptor, label) in enumerate(
            batch):
        p_batch[batch_index, :p.shape[1], :] = p
        v_batch[batch_index, :v.shape[1], :] = v
        m_batch[batch_index, :m.shape[1]] = m
        label_batch[batch_index] = label
        ligands.append(ligand)
        receptors.append(receptor)
    return (p_batch.float(), v_batch.float(),
            m_batch.bool()), label_batch.long(), ligands, receptors


class LinearVariance(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        """Helper module for computing the variance given a linear layer.

        Arguments:
            in_features: (int) Number of input features to layer.

        Returns:
            out_features: (int) Number of output features from layer.
        """
        super().__init__(in_features, out_features, bias)
        self.softplus = nn.Softplus()

    @property
    def w_var(self):
        """Computes variance from log std parameter.

        Returns:
            (torch.tensor) Variance
        """
        return self.softplus(self.weight) ** 2

    def forward(self, x):
        """Computes a forward pass through the layer with the squared values of
        the inputs.

        Arguments:
            x: (torch.tensor) Inputs

        Returns:
            (torch.tensor) Variance of predictions
        """
        return torch.nn.functional.linear(x ** 2, self.w_var, bias=self.bias)


class LocalReparamDense(nn.Module):
    def __init__(self, shape):
        """
        A wrapper module for functional dense layer that performs local
        reparametrization.

        Arguments:
            shape: ((int, int) tuple) Number of input / output features to layer
        """
        super().__init__()
        self.in_features, self.out_features = shape
        self.mean = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=True
        )

        self.var = LinearVariance(self.in_features, self.out_features,
                                  bias=False)

        nn.init.normal_(self.mean.weight, 0., 0.05)
        nn.init.normal_(self.var.weight, -4., 0.05)

    def forward(self, x, num_samples=1, squeeze=False):
        """
        Computes a forward pass through the layer.
        :param x: (torch.tensor) Inputs.
        :param num_samples: (int) Number of samples to take.
        :param squeeze: (bool) Squeeze unnecessary dimensions.
        :return: (torch.tensor) Reparametrized sample from the layer.
        """
        mean, var = self.mean(x), self.var(x)
        return utils.sample_normal(mean, var, num_samples, squeeze)

    def compute_kl(self):
        """Computes the KL divergence w.r.t. a standard Normal prior.

        Returns:
             (torch.tensor) KL divergence value.
        """
        mean, cov = self._compute_posterior()
        scale = 2. / self.mean.weight.shape[0]
        # scale = 1.
        return utils.gaussian_kl_diag(mean, torch.diag(cov),
                                      torch.zeros_like(mean),
                                      scale * torch.ones_like(mean))

    def _compute_posterior(self):
        """Return the approximate posterior over the weights.

        Returns:
            (torch.tensor, torch.tensor) Posterior mean and covariance for layer
            weights.
        """
        return self.mean.weight.flatten(), torch.diag(self.var.w_var.flatten())


class ReparamFullDense(nn.Module):
    def __init__(self, shape, bias=True, rank=None):
        """Reparameterization module for dense covariance layer.

        Arguments:
            shape: ((int, int) tuple) Number of input / output features.
            bias: (bool) Use a bias term in the layer.
            rank: (int) Rank of covariance matrix approximation.
        """
        super().__init__()
        self.in_features, self.out_features = shape
        self.mean = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias
        )

        # Initialize (possibly low-rank) covariance matrix
        covariance_shape = np.prod(shape)
        rank = covariance_shape if rank is None else rank
        self.F = torch.nn.Parameter(torch.zeros(covariance_shape, rank))
        self.log_std = torch.nn.Parameter(torch.zeros(covariance_shape))
        nn.init.normal_(self.mean.weight, 0., 0.05)
        nn.init.normal_(self.log_std, -4., 0.05)

    @property
    def variance(self):
        """Computes variance from log std parameter.

        Returns:
            (torch.tensor) Variance
        """
        return torch.exp(self.log_std) ** 2

    @property
    def cov(self):
        """Computes covariance matrix from matrix F and variance terms.

        Returns:
             (torch.tensor) Covariance matrix.
        """
        return self.F @ self.F.t() + torch.diag(self.variance)

    def forward(self, x, num_samples=1):
        """Computes a forward pass through the layer.

        Arguments:
            x: (torch.tensor) Inputs.
            num_samples: (int) Number of samples to take.

        Returns:
            (torch.tensor) Reparametrized sample from the layer.
        """
        mean = self.mean.weight  # need un-flattened
        post_sample = utils.sample_lr_gaussian(mean.view(1, -1), self.F,
                                               self.variance, num_samples,
                                               squeeze=True)
        post_sample = post_sample.squeeze(dim=1).view(num_samples, *mean.shape)

        return (post_sample[:, None, :, :] @ x[:, :, None].repeat(num_samples,
                                                                  1, 1,
                                                                  1)).squeeze(
            -1) + self.mean.bias

    def compute_kl(self):
        """Computes the KL divergence w.r.t. a standard Normal prior.

        Returns:
             (torch.tensor) KL divergence value.
        """
        mean, cov = self._compute_posterior()
        # scale = 1.
        scale = 2. / self.mean.weight.shape[0]
        return utils.smart_gaussian_kl(mean, cov, torch.zeros_like(mean),
                                       torch.diag(
                                           scale * torch.ones_like(mean)))

    def _compute_posterior(self):
        """Returns the approximate posterior over the weights.

        Returns:
            (torch.tensor, torch.tensor) Posterior mean and covariance for layer
            weights.
        """
        return self.mean.weight.flatten(), self.cov


### MODELS ###

class NeuralClassification(nn.Module):

    def __init__(self, feature_extractor=None, metric='Acc',
                 num_features=256, full_cov=False, cov_rank=2):
        """Neural Linear model for multi-class classification.

        Arguments:
            feature_extractor: (nn.Module) Feature extractor to generate
                representations
            metric: (str) Metric to use for evaluating model
            num_features: (int) Dimensionality of final feature representation
            full_cov: (bool) Use (low-rank approximation to) full covariance
                matrix for last layer distribution
            cov_rank: (int) Optional, if using low-rank approximation, specify
                rank
        """
        super().__init__()
        self.num_classes = 2
        self.feature_extractor = feature_extractor
        if self.feature_extractor.pretrained:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.num_features = num_features
        else:
            self.num_features = num_features
        self.fc1 = nn.Linear(in_features=256, out_features=self.num_features,
                             bias=True)
        self.fc2 = nn.Linear(in_features=self.num_features,
                             out_features=self.num_features, bias=True)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        if full_cov:
            self.linear = ReparamFullDense(
                [self.num_features, self.num_classes], rank=cov_rank)
        else:
            self.linear = LocalReparamDense(
                [self.num_features, self.num_classes])

        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.metric = metric

    def forward(self, x, num_samples=1):
        """Make prediction with model

        Arguments:
            x: (torch.tensor) Inputs
            num_samples: (int) Number of samples to use in forward pass

        Returns:
             (torch.tensor) Predictive distribution (may be tuple)
        """
        return self.linear(
            self.encode(x), num_samples=num_samples).squeeze()

    def encode(self, x):
        """
        Use feature extractor to get features from inputs
        :param x: (torch.tensor) Inputs
        :return: (torch.tensor) Feature representation of inputs
        """
        x = self.feature_extractor(x)
        x = self.fc1(x)
        return x

    def optimize(self, data, num_epochs=1000, batch_size=64, initial_lr=1e-2,
                 freq_summary=100,
                 weight_decay=1e-1, weight_decay_theta=None,
                 train_transform=None, val_transform=None, opt_cycle=0,
                 **kwargs):
        """
        Internal functionality to train model
        :param data: (Object) Training data
        :param num_epochs: (int) Number of epochs to train for
        :param batch_size: (int) Batch-size for training
        :param initial_lr: (float) Initial learning rate
        :param weight_decay: (float) Weight-decay parameter for deterministic weights
        :param weight_decay_theta: (float) Weight-decay parameter for non-deterministic weights
        :param train_transform: (torchvision.transform) Transform procedure for training data
        :param val_transform: (torchvision.transform) Transform procedure for validation data
        :param kwargs: (dict) Optional additional arguments for optimization
        :return: None
        """
        weight_decay_theta = weight_decay if weight_decay_theta is None else weight_decay_theta
        weights = [v for k, v in self.named_parameters() if
                   (not k.startswith('linear')) and k.endswith('weight')]
        weights_theta = [v for k, v in self.named_parameters() if
                         k.startswith('linear') and k.endswith('weight')]
        other = [v for k, v in self.named_parameters() if
                 not k.endswith('weight')]
        optimizer = torch.optim.Adam([
            {'params': weights, 'weight_decay': weight_decay},
            {'params': weights_theta, 'weight_decay': weight_decay_theta},
            {'params': other},
        ], lr=initial_lr)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5)

        if isinstance(data, torch.utils.data.Dataset):
            dataloader = DataLoader(
                dataset=data,
                shuffle=True,
                drop_last=True,
                batch_size=batch_size,
                collate_fn=collate
            )
        elif isinstance(data, torch.utils.data.DataLoader):
            dataloader = data
        else:
            dataloader = DataLoader(
                dataset=Dataset(data, 'train', transform=train_transform),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=4
            )

        start_time = time.time()
        total_iters = num_epochs * len(dataloader)
        global_iter = 0
        active_mean_pred, decoy_mean_pred = 0.5, 0.5
        for epoch in range(num_epochs):
            # scheduler.step()
            losses, kls, performances = [], [], []
            for batch, (x, y, _, _) in enumerate(dataloader):
                optimizer.zero_grad()
                x = tuple(utils.to_gpu(*x))
                y = utils.to_gpu(y)
                y_true_np = y.cpu().detach().numpy()  # (bs, )

                y_pred = self.forward(x)

                step_loss, kl = self._compute_loss(
                    y, y_pred, len(x) / len(data))
                ce_loss = step_loss - kl
                step_loss.backward()
                optimizer.step()

                eta = get_eta(start_time, global_iter, total_iters)
                time_elapsed = format_time(time.time() - start_time)

                wandb_update_dict = {
                    'Time remaining (train)': eta,
                    'Binary crossentropy (train, cycle {})'.format(
                        opt_cycle): ce_loss,
                    'Batch': epoch * len(dataloader) + batch
                }

                y_pred_np = nn.Softmax(dim=1)(y_pred).cpu().detach().numpy()
                active_idx = np.where(y_true_np > 0.5)  # (active_count, )
                decoy_idx = np.where(y_true_np < 0.5)  # (decoy_count, )

                if len(active_idx[0]):
                    active_mean_pred = np.mean(y_pred_np[active_idx, 1])
                    wandb_update_dict.update({
                        'Mean active prediction (train, cycle {})'.format(
                            copt_cycle): active_mean_pred
                    })
                if len(decoy_idx[0]):
                    decoy_mean_pred = np.mean(y_pred_np[decoy_idx, 1])
                    wandb_update_dict.update({
                        'Mean decoy prediction (train, cycle {})'.format(
                            opt_cycle): decoy_mean_pred,
                    })

                try:
                    wandb.log(wandb_update_dict)
                except wandb.Error:
                    pass

                print_with_overwrite(
                    (
                        'Epoch:',
                        '{0}/{1}'.format(epoch + 1, num_epochs),
                        '|', 'Iteration:', '{0}/{1}'.format(
                            batch + 1, len(dataloader))),
                    ('Time elapsed:', time_elapsed, '|',
                     'Time remaining:', eta),
                    ('Loss: {0:.4f}'.format(ce_loss), '|',
                     'KL: {0:.4f}'.format(kl), '|'
                                               'Mean active: {0:.4f}'.format(
                        active_mean_pred), '|',
                     'Mean decoy: {0:.4f}'.format(decoy_mean_pred))
                )
                global_iter += 1

                # performance = self._evaluate_performance(y, y_pred)
                losses.append(step_loss.cpu().item())
                kls.append(kl.cpu().item())
                # performances.append(performance.cpu().item())

            # if epoch % freq_summary == 0 or epoch == num_epochs - 1:
            #    val_bsz = 1024
            #    val_losses, val_performances = self._evaluate(data, val_bsz, 'val', transform=val_transform, **kwargs)
            # print('#{} loss: {:.4f} (val: {:.4f}), kl: {:.4f}, {}: {:.4f} (val: {:.4f})'.format(
            #    epoch, np.mean(losses), np.mean(val_losses), np.mean(kls),
            #    self.metric, np.mean(performances), np.mean(val_performances)))

    def get_projections(self, data, J, projection='two', gamma=0,
                        transform=None, **kwargs):
        """
        Get projections for ACS approximate procedure
        :param data: (Object) Data object to get projections for
        :param J: (int) Number of projections to use
        :param projection: (str) Type of projection to use (currently only 'two' supported)
        :return: (torch.tensor) Projections
        """
        ent = lambda py: torch.distributions.Categorical(probs=py).entropy()
        projections = []
        feat_x = []
        with torch.no_grad():
            mean, cov = self.linear._compute_posterior()
            jitter = utils.to_gpu(torch.eye(len(cov)) * 1e-6)
            theta_samples = MVN(mean, cov + jitter).sample(
                torch.Size([J])).view(J, -1, self.linear.out_features)
            if isinstance(data, DataLoader):
                dataloader = data
            else:
                dataloader = DataLoader(
                    Dataset(data, 'unlabeled', transform=transform),
                    batch_size=256, shuffle=False)

            for batch, (x, _, _, _) in enumerate(dataloader):
                x = tuple(utils.to_gpu(*x))
                feat_x.append(self.encode(x))
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

    def test(self, data, **kwargs):
        """
        Test model
        :param data: (Object) Data to use for testing
        :param kwargs: (dict) Optional additional arguments for testing
        :return: (np.array) Performance metrics evaluated for testing
        """
        print("Testing...")

        # test_bsz = len(data.index['test'])
        test_bsz = 1024
        losses, performances = self._evaluate(data, test_bsz, 'test', **kwargs)
        print("predictive ll: {:.4f}, N: {}, {}: {:.4f}".format(
            -np.mean(losses), len(data.index['train']), self.metric,
            np.mean(performances)))
        return np.hstack(losses), np.hstack(performances)

    def _compute_log_likelihood(self, y, y_pred):
        """
        Compute log-likelihood of predictions
        :param y: (torch.tensor) Observations
        :param y_pred: (torch.tensor) Predictions
        :return: (torch.tensor) Log-likelihood of predictions
        """
        log_pred_samples = y_pred
        ll_samples = -self.cross_entropy(
            log_pred_samples.float(), y.squeeze().long())
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

    def _evaluate_performance(self, y, y_pred):
        """
        Evaluate performance metric for model
        """
        log_pred_samples = y_pred
        y2 = self._compute_predictive_posterior(log_pred_samples)
        return torch.mean((y == torch.argmax(y2, dim=-1)).float())

    def _evaluate(self, data, batch_size, data_type='test', transform=None):
        """
        Evaluate model with data
        :param data: (Object) Data to use for evaluation
        :param batch_size: (int) Batch-size for evaluation procedure (memory issues)
        :param data_type: (str) Data split to use for evaluation
        :param transform: (torchvision.transform) Tranform procedure applied to data during training / validation
        :return: (np.arrays) Performance metrics for model
        """
        # assert data_type in ['val', 'test']
        losses, performances = [], []

        # if data_type == 'val' and len(data.index['val']) == 0:
        #    return losses, performances

        gt.pause()
        with torch.no_grad():
            if isinstance(data, DataLoader):
                dataloader = data
            else:
                dataloader = DataLoader(
                    dataset=Dataset(data, data_type, transform=transform),
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=4
                )
            for (x, y, _, _) in dataloader:
                x = tuple(utils.to_gpu(*x))
                # y = utils.to_gpu(torch.nn.functional.one_hot(y, num_classes=2))
                # x, y = utils.to_gpu(x, y.type(torch.LongTensor).squeeze())
                y_pred_samples = self.forward(x, num_samples=100).squeeze()
                y_pred = self._compute_predictive_posterior(y_pred_samples)[
                         None, :, :].squeeze()
                loss = self._compute_log_likelihood(y.squeeze(),
                                                    y_pred.squeeze())  # use predictive at test time
                avg_loss = loss / len(x)
                # performance = self._evaluate_performance(y, y_pred_samples)
                losses.append(avg_loss.cpu().item())
                # performances.append(performance.cpu().item())

        gt.resume()
        return losses, performances
