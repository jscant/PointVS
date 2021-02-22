"""
Active learning for PointVS; method uses (modified versions of) scripts
found at https://github.com/rpinsler/active-bayesian-coresets, from the paper
Bayesian Batch Active Learning as Sparse Subset Approximation (Pinsler et al.),
which can be found at https://arxiv.org/abs/1908.02144 .
"""

import numpy as np
import torch
import wandb

from acs.coresets import ProjectedFrankWolfe as coreset
from acs.model import NeuralClassification
from data_loaders import WeightedSubsetRandomSampler, SubsetSequentialSampler


def active_learning(session, initial_labelled_size=10000, next_pool_size=5000,
                    mode='active', projections=64, wandb_project=None,
                    wandb_run=None, ms={}, network_class=None,
                    num_features=256):
    """Trains and tests a neural network using active learning.

    Active learning is the process of selecting data to train on (and label)
    based on some heuristic, such that the model becomes more accurate over
    less training data. The paper found at https://arxiv.org/abs/1908.02144 is
    one such method, and the code found in the acs directory of this repo is
    a modified version of that.

    Arguments:
        session: instantiated Session object
        initial_labelled_size: size of starting pool (randomly selected from
            training data)
        next_pool_size: amount of data to be added to labelled training set at
            each active learning iteration
        mode: one of `control` or `active`, whether to use the data selection
            mentioned above (active) or random selection at each AL iteration
            (control)
        projections: number of projections to use for Frank-Wolfe optimisation
            (see paper)
        wandb_project: name of weights and biases project
        wandb_run: name of weights and biases run
    """

    def random_sampler_from_labelled():
        """Generate random sampler from subset of labels, weighted by class.

        This will only index from samples in the labelled_indices variable, and
        class weighting is taken from this subset rather than statistics on
        the entire (unlabelled) training set.
        """
        n_labelled_actives = np.sum(labels[labelled_indices])
        n_labelled_decoys = len(labelled_indices) - n_labelled_actives

        class_sample_count = np.array([n_labelled_decoys, n_labelled_actives])
        print('n_labelled_actives', n_labelled_actives)
        print('n_labelled_decoys', n_labelled_decoys)
        print()
        if not n_labelled_actives * n_labelled_decoys:  # All same class
            return None
        else:
            weights = 1. / class_sample_count
            sample_weights = torch.from_numpy(
                np.array([weights[i] for i in labels])).float()
            sample_weights[np.setdiff1d(indices, labelled_indices)] = 0
            return WeightedSubsetRandomSampler(
                sample_weights, labelled_indices)

    def test_fname_generator(stem):
        """Generate sequential filenames for validation outputs."""
        x = 0
        while True:
            yield '{0}_{1}.txt'.format(stem, x)
            x += 1

    # Some kwargs for later
    cs_kwargs = {'gamma': 0.7}
    optim_params = {'num_epochs': 1,
                    'batch_size': 32, 'initial_lr': 0.002,
                    'weight_decay': 5e-4,
                    'weight_decay_theta': 5e-4,
                    'train_transform': None,
                    'val_transform': None}
    train_data_loader_kwargs = {
        'batch_size': 32,
        'num_workers': 0,
        'collate_fn': session.train_dataset.collate,
        'drop_last': True,
    }
    bayes_data_loader_kwargs = {
        'batch_size': 32,
        'num_workers': 0,
        'collate_fn': session.train_dataset.collate,
        'drop_last': True,
    }

    # Global indices and labels - pool from which we will choose data to label
    indices = np.arange(len(session.train_dataset))
    labels = session.train_dataset.labels

    # Initial (random) labelled dataset
    labelled_indices = np.random.choice(
        indices, initial_labelled_size, replace=False)
    while sum(labels[labelled_indices]) == 0:  # ensure at least one active
        labelled_indices = np.random.choice(
            indices, initial_labelled_size, replace=False)

    print('Initial selection indices:', labelled_indices)
    print('Initial datset size:', len(labelled_indices))
    print('AL batch size', next_pool_size)

    # Setup for prediction output filenames
    predictions_file_base = str(
        session.predictions_file.parent / session.predictions_file.stem)
    test_fnames = test_fname_generator(predictions_file_base)

    al_cycle = 0

    # Setup weights & biases if we are using this (recommended!)
    if wandb_project is not None:
        if wandb_run is not None:
            wandb.run.name = wandb_run
        wandb.watch(session.network)
    while len(labelled_indices < len(indices)):

        # Construct training data loader from labelled subset of indices
        session.network = NeuralClassification(
            network_class(**ms), num_features=num_features).cuda()
        session.network.apply(session.weights_init)
        session.network.train()

        enumerate(session.train_dataset)
        train_data_loader_kwargs['sampler'] = random_sampler_from_labelled()
        training_loader = torch.utils.data.DataLoader(
            session.train_dataset, **train_data_loader_kwargs)

        # we want to train from scratch each time to avoid correlations
        print('Training on labelled dataset')
        session.network.optimize(training_loader, wandb_project=wandb_project,
                                 wandb_run=wandb_run, opt_cycle=al_cycle,
                                 **optim_params)
        print('Finished training')
        session.save(
            session.save_path / 'checkpoints' / '{}.pt'.format(al_cycle))


        # Perform inference on validation set
        print('Testing...')
        np.savetxt(str(
            session.predictions_file).replace(
            '.txt', '_labelled_indices.txt'),
            labelled_indices, fmt='%d', delimiter=',')
        session.predictions_file = next(test_fnames)
        session.network.eval()
        session.test()

        # Select next batch of unlabelled data to label
        remaining = len(indices) - len(labelled_indices)
        unlabelled_indices = np.setdiff1d(indices, labelled_indices)
        al_cycle += 1
        if len(unlabelled_indices) <= next_pool_size:  # label all that remains
            print('Labelling final data points')
            labelled_indices = indices
            break
        if mode == 'active':  # Bayes Batch AL
            print('Calculating projections for Bayesian core-set batch '
                  'selection')
            pool = np.random.choice(
                unlabelled_indices, min(remaining, next_pool_size * 10),
                replace=False)
            bayes_data_loader_kwargs['sampler'] = SubsetSequentialSampler(pool)
            enumerate(session.train_dataset)
            bayes_loader = torch.utils.data.DataLoader(
                session.train_dataset, **bayes_data_loader_kwargs)
            cs = coreset(
                session.network, bayes_loader, projections, **cs_kwargs)
            print('Projections calculated')
            batch_indices_local = cs.build(next_pool_size)
            batch_indices_global = pool[batch_indices_local]
        elif mode == 'control':  # random selection
            print('Generating random learning batch')
            batch_indices_global = np.random.choice(
                unlabelled_indices, next_pool_size, replace=False)
        else:
            raise NotImplementedError(
                'active learning mode must be one of `control` or `active`')
        print()
        print('Newly labelled data count:', len(batch_indices_global))
        print()
        labelled_indices = np.array(
            list(labelled_indices) + list(batch_indices_global))

    # Final model training and inference
    enumerate(session.train_dataset)
    session.network.apply(session.weights_init)
    print('Training on labelled dataset')
    session.network.optimize(training_loader, wandb_project=wandb_project,
                             wandb_run=wandb_run, opt_cycle=al_cycle,
                             **optim_params)
    print('Finished training')
    session.save(
        session.save_path / 'checkpoints' / '{}.pt'.format(al_cycle))
    print('Testing...')
    session.predictions_file = next(test_fnames)
    session.test()
