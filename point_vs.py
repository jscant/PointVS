"""
PointVS uses the the various group-equivariant layers to perform virtual
screening on protein-ligand complexes. This is the main script, and can be used
like so:

python3 point_vs.py <model> <data_root> <save_path> --train_types_pose <train_types> --<other_options>

for example:
python3 point_vs.py egnn data/small_chembl_test /tmp/test_output --train_types_pose data/small_chembl_test.types
"""
import logging
import os
import socket
import warnings
from pathlib import Path

import torch
import yaml

from point_vs import utils
from point_vs import log

from point_vs.models.geometric.egnn_multitask import MultitaskSatorrasEGNN
from point_vs.models.geometric.egnn_lucid import PygLucidEGNN
from point_vs.models.geometric.egnn_satorras import SartorrasEGNN
from point_vs.parse_args import parse_args
from point_vs.preprocessing.data_loaders import get_data_loader
from point_vs.preprocessing.data_loaders import PygPointCloudDataset
from point_vs.preprocessing.data_loaders import SynthPharmDataset
from point_vs.utils import load_yaml
from point_vs.utils import mkdir

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    args = parse_args()

    logging_levels = {
        'notset': logging.NOTSET,
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    logger = log.create_log_obj(
        'PointVS', level=logging_levels[args.logging_level.lower()])

    try:
        import wandb
        logger.info('Wandb found.')
    except ImportError:
        logger.warning('Library wandb not available. --wandb and --run flags '
                       'should not be used.')
        wandb = None

    if args.model_task == 'both' and args.model != 'multitask':
        raise RuntimeError(
            'Sequential pose -> affinity training is only compatable with the '
            'multitask architecture')

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

    with open(save_path / 'cmd_args.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(vars(args), f)

    if args.model == 'egnn':
        model_class = SartorrasEGNN
    elif args.model == 'lucid':
        model_class = PygLucidEGNN
    elif args.model == 'multitask':
        model_class = MultitaskSatorrasEGNN
    else:
        raise NotImplementedError(
            'model must be one of multitask, egnn or lucid')

    if args.synthpharm:
        dataset_class = SynthPharmDataset
    else:
        dataset_class = PygPointCloudDataset

    regression_task = 'multi_regression' if args.multi_target_affinity else 'regression'
    # For backwards compatibility with older trained models
    if args.model_task == 'multi_regression':
        regression_task = 'multi_regression'

    dl_kwargs = {
        'batch_size': args.batch_size,
        'compact': args.compact,
        'radius': args.radius,
        'use_atomic_numbers': args.use_atomic_numbers,
        'rot': False,
        'polar_hydrogens': args.hydrogens,
        'fname_suffix': args.input_suffix,
        'edge_radius': args.edge_radius,
        'estimate_bonds': args.estimate_bonds,
        'prune': args.prune,
        'extended_atom_types': args.extended_atom_types,
        'include_strain_info': args.include_strain_info
    }

    if args.model_task != 'regression':
        # Either both or classification, need a classification dl
        train_dl_pose = get_data_loader(
            args.train_data_root_pose,
            dataset_class,
            augmented_actives=args.augmented_actives,
            min_aug_angle=args.min_aug_angle,
            max_active_rms_distance=args.max_active_rmsd,
            min_inactive_rms_distance=args.min_inactive_rmsd,
            max_inactive_rms_distance=args.max_inactive_rmsd,
            types_fname=args.train_types_pose,
            mode='train',
            p_noise=args.p_noise,
            p_remove_entity=args.p_remove_entity,
            model_task='classification',
            **dl_kwargs
        )
    else:
        train_dl_pose = None
    if args.model_task in ('both', 'regression'):
        # Need a regression dl
        train_dl_affinity = get_data_loader(
            args.train_data_root_affinity,
            dataset_class,
            augmented_actives=args.augmented_actives,
            min_aug_angle=args.min_aug_angle,
            max_active_rms_distance=args.max_active_rmsd,
            min_inactive_rms_distance=args.min_inactive_rmsd,
            max_inactive_rms_distance=args.max_inactive_rmsd,
            types_fname=args.train_types_affinity,
            mode='train',
            p_noise=args.p_noise,
            p_remove_entity=args.p_remove_entity,
            model_task=regression_task,
            **dl_kwargs
        )
    else:
        train_dl_affinity = None

    try:
        dim_input = train_dl_pose.dataset.feature_dim
    except AttributeError:
        dim_input = train_dl_affinity.dataset.feature_dim

    # Is a validation set specified?
    test_dl_pose, test_dl_affinity = None, None
    if 'regression' not in args.model_task and args.test_data_root_pose is not None:
        test_dl_pose = get_data_loader(
            args.test_data_root_pose,
            dataset_class,
            types_fname=args.test_types_pose,
            mode='val',
            model_task='classification',
            **dl_kwargs)

    if args.model_task != 'classification' and args.test_data_root_affinity is not None:
        test_dl_affinity = get_data_loader(
            args.test_data_root_affinity,
            dataset_class,
            types_fname=args.test_types_affinity,
            mode='val',
            model_task=regression_task,
            **dl_kwargs)

    args_to_record = vars(args)

    model_kwargs = {
        'act': args.activation,
        'bn': True,
        'cache': False,
        'ds_frac': 1.0,
        'k': args.channels,
        'num_layers': args.layers,
        'dropout': args.dropout,
        'dim_input': dim_input,
        'dim_output': 3 if regression_task == 'multi_regression' else 1,
        'norm_coords': args.norm_coords,
        'norm_feats': args.norm_feats,
        'thin_mlps': args.thin_mlps,
        'edge_attention': args.egnn_attention,
        'attention': args.egnn_attention,
        'tanh': args.egnn_tanh,
        'normalize': args.egnn_normalise,
        'residual': args.egnn_residual,
        'edge_residual': args.egnn_edge_residual,
        'linear_gap': args.linear_gap,
        'graphnorm': args.graphnorm,
        'multi_fc': args.multi_fc,
        'update_coords': not args.static_coords,
        'node_final_act': args.lucid_node_final_act,
        'permutation_invariance': args.permutation_invariance,
        'attention_activation_fn': args.attention_activation_function,
        'node_attention': args.node_attention,
        'node_attention_final_only': args.node_attention_final_only,
        'edge_attention_final_only': args.edge_attention_final_only,
        'node_attention_first_only': args.node_attention_first_only,
        'edge_attention_first_only': args.edge_attention_first_only,
        'gated_residual': args.gated_residual,
        'rezero': args.rezero,
        'model_task': args.model_task,
        'include_strain_info': args.include_strain_info,
        'final_softplus': args.final_softplus,
    }

    args_to_record.update(model_kwargs)
    if args.model_task == 'both':
        model_kwargs['model_task'] = 'classification'

    if args.wandb_dir is None:
        wandb_dir = save_path
    else:
        wandb_dir = mkdir(args.wandb_dir)
    wandb_init_kwargs = {
        'project': args.wandb_project, 'allow_val_change': True,
        'config': args_to_record, 'dir': str(wandb_dir)
    }
    if args.wandb_project is not None:
        wandb.init(**wandb_init_kwargs)
        if args.wandb_run is not None:
            wandb.run.name = args.wandb_run

    model = model_class(
        save_path, args.learning_rate, args.weight_decay,
        wandb_project=args.wandb_project, use_1cycle=args.use_1cycle,
        warm_restarts=args.warm_restarts,
        only_save_best_models=args.only_save_best_models,
        optimiser=args.optimiser, **model_kwargs)

    if args.load_weights is not None:
        model.load_weights(args.load_weights)

    try:
        wandb.watch(model)
    except ValueError:
        pass

    if args.epochs_pose and train_dl_pose is not None:
        model.set_task('classification')
        model.train_model(
            train_dl_pose, epochs=args.epochs_pose, top1_on_end=args.top1,
            epoch_end_validation_set=test_dl_pose if args.val_on_epoch_end else None)
    if test_dl_pose is not None:
        model.set_task('classification')
        model.val(test_dl_pose, top1_on_end=args.top1)
    if args.epochs_affinity and train_dl_affinity is not None:
        model.set_task(regression_task)
        model.train_model(
            train_dl_affinity, epochs=args.epochs_affinity, top1_on_end=args.top1,
            epoch_end_validation_set=test_dl_affinity if args.val_on_epoch_end else None)
    if test_dl_affinity is not None:
        model.set_task(regression_task)
        model.val(test_dl_affinity, top1_on_end=args.top1)

    if args.end_flag:
        with open(save_path / '_FINISHED', 'w', encoding='utf-8') as f:
            f.write('')
