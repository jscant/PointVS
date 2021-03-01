"""Modified from https://github.com/rpinsler/active-bayesian-coresets"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

import acs.utils as utils


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
        ).cuda()

        self.var = LinearVariance(self.in_features, self.out_features,
                                  bias=False).cuda()

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
        mean, var = self.mean(x).cuda(), self.var(x).cuda()
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
