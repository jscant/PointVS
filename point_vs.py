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
import os
import socket
import warnings
from pathlib import Path

import torch
import yaml
from lie_conv.lieGroups import SE3

from point_vs import utils
from point_vs.utils import load_yaml

try:
    from point_vs.models.egnn_satorras import SartorrasEGNN
    from point_vs.models.egnn_lucid import PygLucidEGNN
except (ModuleNotFoundError, OSError):
    EGNNStack = None
from point_vs.models.lie_conv import LieResNet
from point_vs.models.lie_transformer import EquivariantTransformer
from point_vs.parse_args import parse_args
from point_vs.preprocessing.data_loaders import get_data_loader, \
    PointCloudDataset, PygPointCloudDataset

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
        loaded_args = load_yaml(Path(args.load_args).expanduser())
        for key, value in loaded_args.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # No point even attempting any of this without a GPU
    utils.set_gpu_mode(True)
    if args.wandb_project is None:
        save_path = Path(args.save_path).expanduser()
    elif args.wandb_run is None:
        raise RuntimeError(
            'wandb_run must be specified if wandb_project is specified.')
    else:
        save_path = Path(
            args.save_path, args.wandb_project, args.wandb_run).expanduser()
    save_path.mkdir(parents=True, exist_ok=True)
    args.hostname = socket.gethostname()
    args.slurm_jobid = os.getenv('SLURM_JOBID')

    with open(save_path / 'cmd_args.yaml', 'w') as f:
        yaml.dump(vars(args), f)

    model_classes = {
        'lieconv': LieResNet,
        'egnn': SartorrasEGNN,
        'lucid': PygLucidEGNN,
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

    if args.model in ('lucid', 'egnn'):
        dataset_class = PygPointCloudDataset
    else:
        dataset_class = PointCloudDataset

    train_dl = get_data_loader(
        args.train_data_root, args.translated_actives,
        dataset_class=dataset_class,
        batch_size=args.batch_size, compact=args.compact, radius=args.radius,
        use_atomic_numbers=args.use_atomic_numbers, rot=False,
        augmented_actives=args.augmented_actives,
        min_aug_angle=args.min_aug_angle,
        max_active_rms_distance=args.max_active_rmsd,
        min_inactive_rms_distance=args.min_inactive_rmsd,
        polar_hydrogens=args.hydrogens, receptors=train_receptors, mode='train',
        types_fname=args.train_types, fname_suffix=args.input_suffix,
        edge_radius=args.edge_radius
    )

    # Is a validation set specified?
    test_dl = None
    if args.test_data_root is not None:
        test_dl = get_data_loader(
            args.test_data_root, receptors=test_receptors, compact=args.compact,
            dataset_class=dataset_class,
            use_atomic_numbers=args.use_atomic_numbers, radius=args.radius,
            polar_hydrogens=args.hydrogens, batch_size=args.batch_size,
            types_fname=args.test_types,
            edge_radius=args.edge_radius,
            rot=False, mode='val', fname_suffix=args.input_suffix)

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
        'dim_hidden': args.channels,
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
        'attention': args.egnn_attention,
        'tanh': args.egnn_tanh,
        'normalize': args.egnn_normalise,
        'residual': args.egnn_residual,
    }

    args_to_record.update(model_kwargs)

    if args.load_weights is not None:
        model_kwargs = load_yaml(Path(Path(
            args.load_weights).parents[1].expanduser(), 'model_kwargs.yaml'))
        model_kwargs['group'] = SE3(0.2)

    wandb_init_kwargs = {
        'project': args.wandb_project, 'allow_val_change': True,
        'config': args_to_record, 'dir': save_path
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
        use_1cycle=args.use_1cycle, warm_restarts=args.warm_restarts,
        **model_kwargs)

    if args.load_weights is not None:
        model.load_weights(args.load_weights)

    try:
        wandb.watch(model)
    except ValueError:
        pass

    if args.epochs:
        model.optimise(
            train_dl, epochs=args.epochs,
            epoch_end_validation_set=test_dl if args.val_on_epoch_end else None)
    if test_dl is not None:
        model.test(test_dl)
