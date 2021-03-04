"""
PointVS uses the the LieConv layer to perform virtual screening on
protein-ligand complexes. This is the main script, and can be used like so:

python3 lieconv_vs.py <model> <data_root> <save_path> --batch_size int
    --receptors [str]

for example:
python3 lieconv_vs.py lieconv data/small_chembl_test ~/test_output

Specific receptors can be specified as a list for the final argument:
python3 lieconv_vs.py resnet data/small_chembl_test ~/test_output -r 20014 28

<model> can be either of gnina or restnet.
"""

import argparse
import warnings
from pathlib import PosixPath

import torch
import yaml
# from lie_conv.lieGroups import SE3
from lie_transformer_pytorch.se3 import SE3
from torch.utils.data import DataLoader

from acs import utils
from active_learning import active_learning
from data_loaders import LieConvDataset, SE3TransformerDataset, \
    multiple_source_dataset
from models import LieResNet, BayesianPointNN, LieFeatureExtractor, EnResNet, \
    EnFeatureExtractor, EquivariantTransformer

try:
    import wandb
except ImportError:
    print('Library wandb not available. --wandb and --run flags should not be '
          'used.')
    wandb = None

warnings.filterwarnings("ignore", category=UserWarning)


def get_data_loader(ds_class, *data_roots, receptors=None, batch_size=32,
                    radius=6, rot=True, mask=True, mode='train'):
    ds_kwargs = {
        'receptors': receptors,
        'radius': radius,
    }
    if ds_class == LieConvDataset:
        ds_kwargs.update({
            'rot': rot
        })
    elif ds_class == SE3TransformerDataset:
        ds_kwargs.update({
            'mode': 'interaction_edges',
            'interaction_dist': 4
        })
    ds = multiple_source_dataset(
        ds_class, *data_roots, balanced=True, **ds_kwargs)
    collate = ds.collate if mask else ds.collate_no_masking
    sampler = ds.sampler if mode == 'train' else None
    return DataLoader(
        ds, batch_size, False, sampler=sampler, num_workers=0,
        collate_fn=collate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Type of point cloud network to'
                                                ' use: se3trans or lieconv')
    parser.add_argument('train_data_root', type=PosixPath,
                        help='Location of structure training *.parquets files. '
                             'Receptors should be in a directory named '
                             'receptors, with ligands located in their '
                             'specific receptor subdirectory under the '
                             'ligands directory.')
    parser.add_argument('save_path', type=PosixPath,
                        help='Directory in which experiment outputs are '
                             'stored.')
    parser.add_argument('--load_weights', '-l', type=PosixPath, required=False,
                        help='Load a model.')
    parser.add_argument('--test_data_root', '-t', type=PosixPath,
                        required=False,
                        help='Location of structure test *.parquets files. '
                             'Receptors should be in a directory named '
                             'receptors, with ligands located in their '
                             'specific receptor subdirectory under the '
                             'ligands directory.')
    parser.add_argument('--translated_actives', type=PosixPath,
                        help='Directory in which translated actives are stored.'
                             ' If unspecified, no translated actives will be '
                             'used. The use of translated actives are is '
                             'discussed in https://pubs.acs.org/doi/10.1021/ac'
                             's.jcim.0c00263')
    parser.add_argument('--batch_size', '-b', type=int, required=False,
                        default=32,
                        help='Number of examples to include in each batch for '
                             'training.')
    parser.add_argument('--epochs', '-e', type=int, required=False,
                        default=1,
                        help='Number of times to iterate through training set.')
    parser.add_argument('--channels', '-k', type=int, default=32,
                        help='Channels for feature vectors')
    parser.add_argument('--train_receptors', '-r', type=str, nargs='*',
                        help='Names of specific receptors for training. If '
                             'specified, other structures will be ignored.')
    parser.add_argument('--test_receptors', '-q', type=str, nargs='*',
                        help='Names of specific receptors for testing. If '
                             'specified, other structures will be ignored.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.002,
                        help='Learning rate for gradient descent')
    parser.add_argument('--wandb_project', type=str,
                        help='Name of wandb project. If left blank, wandb '
                             'logging will not be used.')
    parser.add_argument('--wandb_run', type=str,
                        help='Name of run for wandb logging.')
    parser.add_argument('--layers', type=int, default=6,
                        help='Number of layers in LieResNet')
    parser.add_argument('--channels_in', '-chin', type=int, default=12,
                        help='Input channels')
    parser.add_argument('--liftsamples', type=int, default=1,
                        help='liftsamples parameter in LieConv')
    parser.add_argument('--radius', type=int, default=6,
                        help='Maximum distance from a ligand atom for a '
                             'receptor atom to be included in input')
    parser.add_argument('--nbhd', type=int, default=25,
                        help='Number of monte carlo samples for integral')
    parser.add_argument('--al_batch_size', '-albs', type=int, default=-1,
                        help='Number of batches to increase training pool size '
                             'at each iteration of active learning. If '
                             'unspecified, active learning will not be used.')
    parser.add_argument('--al_initial_pool_size', '-alips', type=int,
                        default=-1,
                        help='Size of initial pool size for active learning.')
    parser.add_argument('--al_control', action='store_true',
                        help='Active learning with random data selection (used '
                             'as a control).')
    parser.add_argument('--al_projections', type=int, default=64,
                        help='Number of projections for bayes active learning '
                             '(J in the paper)')
    parser.add_argument('--al_features', type=int, default=256,
                        help='Size of finalature embedding for active learning')
    parser.add_argument('--al_fc_in_features', type=int, default=512,
                        help='Size of input to embedding layer for active '
                             'learning')
    parser.add_argument('--load_args', type=PosixPath,
                        help='Load yaml file with command line args. Any args '
                             'specified in the file will overwrite other args '
                             'specified on the command line.')
    parser.add_argument('--double', action='store_true',
                        help='Use 64-bit floating point precision')
    args = parser.parse_args()

    if args.double:
        torch.set_default_dtype(torch.float64)
        torch.set_default_tensor_type(torch.DoubleTensor)
    else:
        torch.set_default_dtype(torch.float32)
        torch.set_default_tensor_type(torch.FloatTensor)

    if args.load_args is not None:
        with open(args.load_args.expanduser(), 'r') as f:
            loaded_args = yaml.full_load(f)
        for key, value in loaded_args.items():
            if hasattr(args, key):
                setattr(args, key, value)

    utils.set_gpu_mode(True)
    save_path = args.save_path.expanduser()
    save_path.mkdir(parents=True, exist_ok=True)

    if args.model in (
            'lieconv', 'al_lieconv', 'entransformer', 'lietransformer'):
        ds_class = LieConvDataset
    elif args.model in ('entransformer', 'al_entransformer'):
        ds_class = LieConvDataset
    elif args.model == 'se3transformer':
        raise NotImplementedError('se3transformer has been removed')
    else:
        raise NotImplementedError(
            'Only lieconv and al_lieconv models supported')

    if isinstance(args.train_receptors, str):
        train_receptors = tuple([args.train_receptors])
    else:
        train_receptors = args.train_receptors

    if isinstance(args.test_receptors, str):
        test_receptors = tuple([args.test_receptors])
    else:
        test_receptors = args.test_receptors

    mask = False if args.model == 'entransformer' else True
    train_dl = get_data_loader(
        ds_class, args.train_data_root, args.translated_actives,
        batch_size=args.batch_size,
        receptors=train_receptors, radius=args.radius, rot=True, mask=mask,
        mode='train')

    lieconv_model_kwargs = {
        'act': 'relu',
        'bn': True,
        'cache': False,
        'chin': args.channels_in,
        'ds_frac': 1.0,
        'fill': 1.0,
        'group': SE3(),
        'k': args.channels,
        'knn': False,
        'liftsamples': 1,
        'mean': True,
        'nbhd': args.nbhd,
        'num_layers': args.layers,
        'pool': True
    }

    lietransformer_model_kwargs = {
        'dim_input': 12,
        'dim_output': 2,
        'dim_hidden': args.channels,  # == 32
        'num_layers': args.layers,  # == 6
        'num_heads': 8,
        'global_pool': True,
        'global_pool_mean': True,
        'group': SE3(0.2),
        'liftsamples': args.liftsamples,  # == 1
        'block_norm': "layer_pre",
        'output_norm': "none",
        'kernel_norm': "none",
        'kernel_type': "mlp",
        'kernel_dim': 16,
        'kernel_act': "swish",
        'mc_samples': 4,
        'fill': 1.0,
        'attention_fn': "norm_exp",
        'feature_embed_dim': None,
        'max_sample_norm': None,
        'lie_algebra_nonlinearity': None,
    }

    with open(save_path / 'model_args.yaml', 'w') as f:
        yaml.dump(lieconv_model_kwargs, f)
    with open(save_path / 'cmd_args.yaml', 'w') as f:
        yaml.dump(vars(args), f)

    if args.test_data_root is not None:
        test_dl = get_data_loader(
            ds_class, args.test_data_root, receptors=test_receptors,
            batch_size=args.batch_size,
            radius=args.radius, rot=False, mode='val')
    else:
        test_dl = None

    wandb_init_kwargs = {
        'project': args.wandb_project, 'allow_val_change': True,
        'config': vars(args)
    }

    if args.model == 'lietransformer':
        model = EquivariantTransformer(
            save_path, args.learning_rate, **lietransformer_model_kwargs)
        if args.load_weights is not None:
            model.load(args.load_weights.expanduser())
        model.save_path = save_path
        if args.wandb_project is not None:
            wandb.init(**wandb_init_kwargs)
            if args.wandb_run is not None:
                wandb.run.name = args.wandb_run
            wandb.watch(model)
        model.optimise(train_dl, epochs=args.epochs)
        if test_dl is not None:
            model.test(test_dl)

    if args.model == 'lieconv':
        model = LieResNet(save_path, args.learning_rate, **lieconv_model_kwargs)
        if args.load_weights is not None:
            model.load(args.load_weights.expanduser())
        model.save_path = save_path
        if args.wandb_project is not None:
            wandb.init(**wandb_init_kwargs)
            if args.wandb_run is not None:
                wandb.run.name = args.wandb_run
            wandb.watch(model)
        model.optimise(train_dl, epochs=args.epochs)
        if test_dl is not None:
            model.test(test_dl)
    elif args.model == 'al_lieconv':
        train_ds = train_dl.dataset
        feature_extractor = LieFeatureExtractor(
            fc_in_features=args.al_fc_in_features, **lieconv_model_kwargs)
        bayesian_point_nn_kwargs = {
            'feature_extractor': feature_extractor,
            'fc_in_features': args.al_fc_in_features,
            'fc_out_features': args.al_features,
            'full_cov': False,
            'cov_rank': 2
        }
        model = BayesianPointNN(save_path, args.learning_rate, 'scratch', 0.002,
                                **bayesian_point_nn_kwargs)
        mode = 'control' if args.al_control else 'active'

        if args.wandb_project is not None:
            wandb.init(**wandb_init_kwargs)
            if args.wandb_run is not None:
                wandb.run.name = args.run_name
        active_learning(model, train_ds, test_dl, args.al_initial_pool_size,
                        args.al_batch_size, mode=mode,
                        wandb_project=args.wandb_project,
                        wandb_run=args.wandb_run,
                        projections=args.al_projections)
    elif args.model == 'se3transformer':
        raise NotImplementedError('se3transformer has been removed')
    elif args.model == 'entransformer':
        model = EnResNet(save_path, args.learning_rate, **lieconv_model_kwargs)
        if args.load_weights is not None:
            model.load(args.load_weights.expanduser())
        model.save_path = save_path
        if args.wandb_project is not None:
            wandb.init(**wandb_init_kwargs)
            if args.wandb_run is not None:
                wandb.run.name = args.wandb_run
            wandb.watch(model)
        model.optimise(train_dl, epochs=args.epochs)
        if test_dl is not None:
            model.test(test_dl)
    elif args.model == 'al_entransformer':
        train_ds = train_dl.dataset
        feature_extractor = EnFeatureExtractor(
            fc_in_features=args.al_fc_in_features, **lieconv_model_kwargs)
        bayesian_point_nn_kwargs = {
            'feature_extractor': feature_extractor,
            'fc_in_features': args.al_fc_in_features,
            'fc_out_features': args.al_features,
            'full_cov': False,
            'cov_rank': 2
        }
        model = BayesianPointNN(save_path, args.learning_rate, 'scratch', 0.002,
                                **bayesian_point_nn_kwargs)
        mode = 'control' if args.al_control else 'active'

        if args.wandb_project is not None:
            wandb.init(**wandb_init_kwargs)
            if args.wandb_run is not None:
                wandb.run.name = args.wandb_run
        active_learning(model, train_ds, test_dl, args.al_initial_pool_size,
                        args.al_batch_size, mode=mode,
                        wandb_project=args.wandb_project,
                        wandb_run=args.wandb_run,
                        projections=args.al_projections)
    else:
        raise NotImplementedError('model must be either lieconv or al_lieconv')
