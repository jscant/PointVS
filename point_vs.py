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

from point_vs import utils

try:
    from point_vs.models.egnn_network import EGNN
except (ModuleNotFoundError, OSError):
    EGNNStack = None
from point_vs.models.lie_conv import LieResNet
from point_vs.models.lie_transformer import EquivariantTransformer
from point_vs.parse_args import parse_args
from point_vs.preprocessing.data_loaders import get_data_loader

try:
    import wandb
except ImportError:
    print('Library wandb not available. --wandb and --run flags should not be '
          'used.')
    wandb = None

warnings.filterwarnings("ignore", category=UserWarning)

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
        'egnn': EGNN,
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

    train_dl = get_data_loader(
        args.train_data_root, args.translated_actives,
        batch_size=args.batch_size, compact=args.compact, radius=args.radius,
        use_atomic_numbers=args.use_atomic_numbers, rot=False,
        augmented_actives=args.augmented_actives,
        min_aug_angle=args.min_aug_angle,
        polar_hydrogens=args.hydrogens, receptors=train_receptors, mode='train')

    # Is a validation set specified?
    test_dl = None
    if args.test_data_root is not None:
        test_dl = get_data_loader(
            args.test_data_root, receptors=test_receptors, compact=args.compact,
            use_atomic_numbers=args.use_atomic_numbers, radius=args.radius,
            polar_hydrogens=args.hydrogens, batch_size=args.batch_size,
            rot=False, mode='val')

    args_to_record = vars(args)

    model_kwargs = {
        'act': args.activation,
        'bn': True,
        'cache': False,
        'ds_frac': 1.0,
        'fill': args.fill,
        'group': SE3(0.2),
        'k': args.channels,
        'knn': False,
        'liftsamples': args.liftsamples,
        'mean': True,
        'nbhd': args.nbhd,
        'num_layers': args.layers,
        'pool': True,
        'dropout': args.dropout,
        'dim_input': train_dl.dataset.feature_dim,
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
        'fourier_features': args.fourier_features,
        'norm_coords': args.norm_coords,
        'norm_feats': args.norm_feats,
        'thin_mlps': args.thin_mlps,
    }

    args_to_record.update(model_kwargs)

    if args.load_weights is not None:
        with open(args.load_weights.parents[1] / 'model_kwargs.yaml',
                  'r') as f:
            model_kwargs = yaml.load(f, Loader=yaml.Loader)

    wandb_init_kwargs = {
        'project': args.wandb_project, 'allow_val_change': True,
        'config': args_to_record
    }
    if args.wandb_project is not None:
        wandb.init(**wandb_init_kwargs)
        if args.wandb_run is not None:
            wandb.run.name = args.wandb_run

    model_class = model_classes[args.model]
    if model_class is None:
        print('Required libraries for {} not found. Aborting.'.format(
            args.model))
        exit(1)

    model = model_class(
        save_path, args.learning_rate, args.weight_decay,
        use_1cycle=args.use_1cycle, **model_kwargs)

    if args.load_weights is not None:
        model.load_weights(args.load_weights)

    try:
        wandb.watch(model)
    except ValueError:
        pass

    model.optimise(train_dl, epochs=args.epochs)
    if test_dl is not None:
        model.test(test_dl)
