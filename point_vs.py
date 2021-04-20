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

from point_vs import utils
from point_vs.models.egnn_network import EGNNStack
from point_vs.models.lie_conv import LieResNet
from point_vs.models.lie_transformer import EquivariantTransformer
from point_vs.parse_args import parse_args
from point_vs.preprocessing.data_loaders import multiple_source_dataset, \
    LieConvDataset, get_collate_fn

try:
    import wandb
except ImportError:
    print('Library wandb not available. --wandb and --run flags should not be '
          'used.')
    wandb = None

warnings.filterwarnings("ignore", category=UserWarning)


def get_data_loader(*data_roots, receptors=None, batch_size=32,
                    radius=6, rot=True, feature_dim=12, mode='train'):
    ds_kwargs = {
        'receptors': receptors,
        'radius': radius,
        'rot': rot
    }
    ds = multiple_source_dataset(
        LieConvDataset, *data_roots, balanced=True, **ds_kwargs)
    collate = get_collate_fn(feature_dim)
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

    model_classes = {
        'lieconv': LieResNet,
        'egnn': EGNNStack,
        'lietransformer': EquivariantTransformer
    }

    if args.model not in model_classes.keys():
        raise NotImplementedError(
            'model must be one of ' + ', '.join(model_classes.keys()))

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
        args.train_data_root, args.translated_actives,
        batch_size=args.batch_size,
        receptors=train_receptors, radius=args.radius, rot=True,
        mode='train')

    # Is a validation set specified?
    test_dl = None
    if args.test_data_root is not None:
        test_dl = get_data_loader(
            args.test_data_root, receptors=test_receptors,
            batch_size=args.batch_size, radius=args.radius, rot=False,
            mode='val')

    args_to_record = vars(args)

    model_kwargs = {
        'act': args.activation,
        'bn': True,
        'cache': False,
        'chin': args.channels_in,
        'ds_frac': 1.0,
        'fill': 1.0,
        'group': SE3(0.2),
        'k': args.channels,
        'knn': False,
        'liftsamples': args.liftsamples,
        'mean': True,
        'nbhd': args.nbhd,
        'num_layers': args.layers,
        'pool': True,
        'dropout': args.dropout,
        'dim_input': 12,
        'dim_output': 1,
        'dim_hidden': args.channels,  # == 32
        'num_heads': 8,
        'global_pool': True,
        'global_pool_mean': True,
        'block_norm': "layer_pre",
        'output_norm': "none",
        'kernel_norm': "none",
        'kernel_type': args.kernel_type,
        'kernel_dim': args.kernel_dim,
        'kernel_act': args.activation,
        'mc_samples': 4,
        'attention_fn': args.attention_fn,
        'feature_embed_dim': None,
        'max_sample_norm': None,
        'lie_algebra_nonlinearity': None,
        'pooling_only': args.pooling_only
    }

    args_to_record.update(model_kwargs)

    if args.load_weights is not None:
        with open(args.load_weights.parents[1] / 'model_kwargs.yaml',
                  'r') as f:
            model_kwargs = yaml.load(
                f, Loader=yaml.FullLoader)

    wandb_init_kwargs = {
        'project': args.wandb_project, 'allow_val_change': True,
        'config': args_to_record
    }
    if args.wandb_project is not None:
        wandb.init(**wandb_init_kwargs)
        if args.wandb_run is not None:
            wandb.run.name = args.wandb_run

    model_class = model_classes[args.model]
    model = model_class(
        save_path, args.learning_rate, args.weight_decay, **model_kwargs)

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
