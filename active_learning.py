import numpy as np
import torch
import wandb

from acs.coresets import ProjectedFrankWolfe as coreset
from data_loaders import WeightedSubsetRandomSampler, SubsetSequentialSampler


def active_learning(session, initial_labelled_size=10000, next_pool_size=5000,
                    mode='active', projections=64, wandb_project=None,
                    wandb_run=None):
    def generate_sampler():
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

    print('Initial selection indices:', labelled_indices)
    print('Initial datset size:', len(labelled_indices))
    print('AL batch size', next_pool_size)

    predictions_file_base = str(
        session.predictions_file.parent / session.predictions_file.stem)
    test_fnames = test_fname_generator(predictions_file_base)
    opt_cycle = 0
    if wandb_project is not None:
        if wandb_run is not None:
            wandb.run.name = wandb_run
        wandb.watch(session.network)
    while len(labelled_indices < len(indices)):

        # Construct data loader and train on labelled data
        enumerate(session.train_dataset)
        train_data_loader_kwargs['sampler'] = generate_sampler()
        training_loader = torch.utils.data.DataLoader(
            session.train_dataset, **train_data_loader_kwargs)
        session.network.apply(session.weights_init)
        print('Training on labelled dataset')
        session.network.optimize(training_loader, wandb_project=wandb_project,
                                 wandb_run=wandb_run, opt_cycle=opt_cycle,
                                 **optim_params)
        print('Finished training')
        session.save(
            session.save_path / 'checkpoints' / '{}.pt'.format(opt_cycle))

        # Perform inference on validation set
        print('Testing...')
        session.predictions_file = next(test_fnames)
        session.test()

        # Select next batch of unlabelled data to label
        remaining = len(indices) - len(labelled_indices)
        unlabelled_indices = np.setdiff1d(indices, labelled_indices)
        opt_cycle += 1
        if len(unlabelled_indices) <= next_pool_size:
            print('Labelling final data points')
            labelled_indices = indices
            break
        if mode == 'active':
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
        elif mode == 'control':
            print('Generating random learning batch')
            batch_indices_global = np.random.choice(
                unlabelled_indices, next_pool_size, replace=False)
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
                             wandb_run=wandb_run, opt_cycle=opt_cycle,
                             **optim_params)
    print('Finished training')
    session.save(
        session.save_path / 'checkpoints' / '{}.pt'.format(opt_cycle))
    print('Testing...')
    session.predictions_file = next(test_fnames)
    session.test()
