"""
PointVS uses the the various group-equivariant layers to perform virtual
screening on protein-ligand complexes. This is the main script, and can be used
like so:

python3 point_vs.py <model> <data_root> <save_path> --batch_size int
    --receptors [str]

for example:
python3 point_vs.py lieconv data/small_chembl_test ~/test_output

Specific receptors can be specified as a list for the final argument:
python3 point_vs.py resnet data/small_chembl_test ~/test_output -r 20014 28

<model> can be either of gnina or restnet.
"""

import warnings

import torch
import yaml
from lie_conv.lieGroups import SE3
from torch.utils.data import DataLoader

from acs import utils
from active_learning import active_learning
from data_loaders import LieConvDataset, SE3TransformerDataset, \
    multiple_source_dataset
from models import LieResNet, BayesianPointNN, LieFeatureExtractor, EnResNet, \
    EnFeatureExtractor, EquivariantTransformer
from parse_args import parse_args

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
    args = parse_args()

    # This is a lot slower so only use if precision is an issue
    if args.double:
        torch.set_default_dtype(torch.float64)
        torch.set_default_tensor_type(torch.DoubleTensor)
    else:
        torch.set_default_dtype(torch.float32)
        torch.set_default_tensor_type(torch.FloatTensor)

    # Load a yaml if required
    if args.load_args is not None:
        with open(args.load_args.expanduser(), 'r') as f:
            loaded_args = yaml.full_load(f)
        for key, value in loaded_args.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # No point even attempting any of this without a GPU
    utils.set_gpu_mode(True)
    save_path = args.save_path.expanduser()
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / 'cmd_args.yaml', 'w') as f:
        yaml.dump(vars(args), f)

    allowed_models = ('lieconv', 'al_lieconv', 'egnn',
                      'lietransformer', 'al_egnn')

    # Dataset class
    if args.model in allowed_models:
        ds_class = LieConvDataset
    else:
        raise NotImplementedError(
            'model must be one of ' + ', '.join(allowed_models))

    if isinstance(args.train_receptors, str):
        train_receptors = tuple([args.train_receptors])
    else:
        train_receptors = args.train_receptors

    if isinstance(args.test_receptors, str):
        test_receptors = tuple([args.test_receptors])
    else:
        test_receptors = args.test_receptors

    mask = False if args.model == 'egnn' else True
    train_dl = get_data_loader(
        ds_class, args.train_data_root, args.translated_actives,
        batch_size=args.batch_size,
        receptors=train_receptors, radius=args.radius, rot=True, mask=mask,
        mode='train')

    # Is a validation set specified?
    test_dl = None
    if args.test_data_root is not None:
        test_dl = get_data_loader(
            ds_class, args.test_data_root, receptors=test_receptors,
            batch_size=args.batch_size,
            radius=args.radius, rot=False, mode='val')

    wandb_init_kwargs = {
        'project': args.wandb_project, 'allow_val_change': True,
        'config': vars(args)
    }

    if args.wandb_project is not None:
        wandb.init(**wandb_init_kwargs)
        if args.wandb_run is not None:
            wandb.run.name = args.run_name

    if args.model in (
            'lieconv', 'al_lieconv', 'egnn', 'al_egnn'):
        model_kwargs = {
            'act': args.activation,
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

        if args.model.startswith('al_'):
            model_class = BayesianPointNN
            model_kwargs.update({'fc_in_features': args.al_fc_in_features})
            if args.model == 'al_lieconv':
                feature_extractor = LieFeatureExtractor(**model_kwargs)
            elif args.model == 'al_egnn':
                feature_extractor = EnFeatureExtractor(**model_kwargs)
            else:
                raise NotImplementedError('model {} not supported'.format(
                    args.model))
            model_kwargs = {
                'feature_extractor': feature_extractor,
                'fc_in_features': args.al_fc_in_features,
                'fc_out_features': args.al_features,
                'full_cov': False,
                'cov_rank': 2
            }
        elif args.model == 'lieconv':
            model_class = LieResNet
        else:
            model_class = EnResNet

    elif args.model == 'lietransformer':
        model_kwargs = {
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
            'kernel_type': args.kernel_type,
            'kernel_dim': args.kernel_dim,
            'kernel_act': args.activation,
            'mc_samples': 4,
            'fill': 1.0,
            'attention_fn': args.attention_fn,
            'feature_embed_dim': None,
            'max_sample_norm': None,
            'lie_algebra_nonlinearity': None,
        }
        model_class = EquivariantTransformer
    else:
        raise NotImplementedError('Supported models are ' +
                                  ', '.join(allowed_models))

    if args.load_weights is not None:
        if args.model.startswith('al_'):
            raise NotImplementedError(
                'Active learning model loading not yet supported')
        with open(args.load_weights.parents[1] / 'model_kwargs.yaml',
                  'r') as f:
            model_kwargs = yaml.load(
                f, Loader=yaml.FullLoader)

    model = model_class(save_path, args.learning_rate, args.weight_decay,
                        **model_kwargs)

    if args.wandb_project is not None:
        wandb.init(**wandb_init_kwargs)
        if args.wandb_run is not None:
            wandb.run.name = args.wandb_run
    if args.model.startswith('al_'):
        mode = 'control' if args.al_control else 'active'
        active_learning(model, train_dl.dataset, test_dl,
                        args.al_initial_pool_size,
                        args.al_batch_size, mode=mode,
                        wandb_project=args.wandb_project,
                        wandb_run=args.wandb_run,
                        projections=args.al_projections)
    else:
        if args.load_weights is not None:
            checkpoint = torch.load(args.load_weights)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
            model.epoch = checkpoint['epoch']
            model.losses = checkpoint['losses']
            model.bce_loss = checkpoint['bce_loss']
        try:
            wandb.watch(model)
        except ValueError:
            pass
        model.optimise(train_dl, epochs=args.epochs)
        if test_dl is not None:
            model.test(test_dl)
