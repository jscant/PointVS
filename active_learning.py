import numpy as np
import torch

from acs.coresets import ProjectedFrankWolfe as coreset
from data_loaders import WeightedSubsetRandomSampler, SubsetSequentialSampler


def active_learning(session, initial_labelled_size=10000, next_pool_size=5000,
                    features=256, wandb_project=None, wandb_run=None):
    def generate_sampler():
        n_labelled_actives = np.sum(labels[labelled_indices])
        n_labelled_decoys = len(labelled_indices) - n_labelled_actives

        class_sample_count = np.array([n_labelled_decoys, n_labelled_actives])
        print('n_labelled_actives', n_labelled_actives)
        print('n_labelled_decoys', n_labelled_decoys)
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
        x = 0
        while True:
            yield '{0}_{1}.txt'.format(stem, x)
            x += 1

    cs_kwargs = {'gamma': 0.7}
    optim_params = {'num_epochs': 1,
                    'batch_size': 32, 'initial_lr': 0.002,
                    'weight_decay': 5e-4,
                    'weight_decay_theta': 5e-4,
                    'train_transform': None,
                    'val_transform': None}

    indices = np.arange(len(session.train_dataset))
    labels = session.train_dataset.labels

    labelled_indices = np.random.choice(
        indices, initial_labelled_size, replace=False)
    while sum(labels[labelled_indices]) == 0:
        labelled_indices = np.random.choice(
            indices, initial_labelled_size, replace=False)

    sampler = generate_sampler()

    data_loader_kwargs = {
        'batch_size': 32,
        'num_workers': 0,
        'sampler': sampler,
        'collate_fn': session.train_dataset.collate,
        'drop_last': True,
    }
    training_loader = torch.utils.data.DataLoader(
        session.train_dataset, **data_loader_kwargs)
    print('len trainig loader', len(training_loader))

    print('Initial selection indices:', labelled_indices)
    print('Initial datset size:', len(labelled_indices))
    print('AL batch size', next_pool_size)

    predictions_file_base = str(
        session.predictions_file.parent / session.predictions_file.stem)
    test_fnames = test_fname_generator(predictions_file_base)
    opt_cycle = 0
    while True:
        enumerate(session.train_dataset)
        session.network.apply(session.weights_init)
        print('Training on labelled dataset')
        session.network.optimize(training_loader, wandb_project=wandb_project,
                                 wandb_run=wandb_run, opt_cycle=opt_cycle,
                                 **optim_params)
        print('Finished training')
        print('Testing...')
        session.predictions_file = next(test_fnames)
        session.test()

        remaining = len(indices) - len(labelled_indices)
        pool = np.random.choice(
            np.setdiff1d(indices, labelled_indices),
            min(remaining, next_pool_size),
            replace=False)
        data_loader_kwargs['sampler'] = SubsetSequentialSampler(pool)
        enumerate(session.train_dataset)
        training_loader = torch.utils.data.DataLoader(
            session.train_dataset, **data_loader_kwargs)
        cs = coreset(session.network, training_loader, 10, **cs_kwargs)
        print('Projections calculated')
        batch_indices_local = cs.build(next_pool_size)
        batch_indices_global = pool[batch_indices_local]
        print('Newly labelled data count:', len(batch_indices_global))
        labelled_indices = np.array(
            list(labelled_indices) + list(batch_indices_global))
        data_loader_kwargs['sampler'] = generate_sampler()
