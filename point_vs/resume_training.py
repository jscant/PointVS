"""Resume training if a job has crashed."""

import argparse
from pathlib import Path

import wandb

from point_vs.models.load_model import load_model
from point_vs.preprocessing.data_loaders import PygPointCloudDataset, \
    PointCloudDataset, get_data_loader
from point_vs.utils import expand_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_path', type=str,
                        help='Base path from which to resume training.')
    parser.add_argument('--epochs_pose', '-ep', type=int, default=-1,
                        help='Modify the number of epochs for pose prediction '
                             '(multitask only)')
    parser.add_argument('--epochs_affinity', '-ea', type=int, default=-1,
                        help='Modify the number of epochs for affinity '
                             'prediction (multitask only)')

    args = parser.parse_args()
    base_path = expand_path(args.base_path)

    ckpt, model, model_kwargs, cmd_line_args = load_model(
        base_path, init_path=True)

    if cmd_line_args['model'] in ('lucid', 'egnn', 'multitask'):
        dataset_class = PygPointCloudDataset
    else:
        dataset_class = PointCloudDataset

    ea = args.epochs_affinity
    ep = args.epochs_pose

    # Check if model is from before change to pose/affinity distinction, and
    # populate the relevant variables.
    pose_types_train = None
    affinity_types_train = None
    pose_types_test = None
    affinity_types_test = None
    pose_data_root_train = None
    affinity_data_root_train = None
    pose_data_root_test = None
    affinity_data_root_test = None

    pose_train_dl = None
    affinity_train_dl = None
    pose_test_dl = None
    affinity_test_dl = None

    # Older models will have epochs in their recorded commands.
    if cmd_line_args.get('epochs', False):
        if cmd_line_args.get('model_task', 'regression') == 'multi_regression':
            regression_task = 'multi_regression'
        else:
            regression_task = 'regression'
        if cmd_line_args.get('model_task', 'classification') == 'classification':
            epochs_affini = 0
            epochs_classi = cmd_line_args['epochs'] if ep == -1 else ep
            pose_data_root_train = cmd_line_args['train_data_root']
            pose_types_train = cmd_line_args['train_types']
            pose_data_root_test = cmd_line_args['test_data_root']
            pose_types_test = cmd_line_args['test_types']
        else:
            epochs_affini = cmd_line_args['epochs'] if ea == -1 else ea
            epochs_classi = 0
            affinity_data_root_train = cmd_line_args['train_data_root']
            affinity_types_train = cmd_line_args['train_types']
            affinity_data_root_test = cmd_line_args['test_data_root']
            affinity_types_test = cmd_line_args['test_types']
    else:  # Newer models follow this branch.
        if cmd_line_args['multi_target_affinity']:
            regression_task = 'multi_regression'
        else:
            regression_task = 'regression'
        epochs_affini = cmd_line_args['epochs_affinity'] if ea == -1 else ea
        epochs_classi = cmd_line_args['epochs_pose'] if ep == -1 else ep
        affinity_data_root_train = cmd_line_args['train_data_root_affinity']
        affinity_types_train = cmd_line_args['train_types_affinity']
        affinity_data_root_test = cmd_line_args['test_data_root_affinity']
        affinity_types_test = cmd_line_args['test_types_affinity']
        pose_data_root_train = cmd_line_args['train_data_root_pose']
        pose_types_train = cmd_line_args['train_types_pose']
        pose_data_root_test = cmd_line_args['test_data_root_pose']
        pose_types_test = cmd_line_args['test_types_pose']

    if pose_data_root_train is not None:
        pose_train_dl = get_data_loader(
            pose_data_root_train,
            dataset_class=dataset_class,
            batch_size=cmd_line_args['batch_size'],
            compact=cmd_line_args['compact'], radius=cmd_line_args['radius'],
            use_atomic_numbers=cmd_line_args['use_atomic_numbers'], rot=False,
            augmented_actives=cmd_line_args['augmented_actives'],
            min_aug_angle=cmd_line_args['min_aug_angle'],
            max_active_rms_distance=cmd_line_args['max_active_rmsd'],
            min_inactive_rms_distance=cmd_line_args['min_inactive_rmsd'],
            max_inactive_rms_distance=cmd_line_args.get('max_inactive_rmsd', None),
            polar_hydrogens=cmd_line_args['hydrogens'],
            mode='train',
            types_fname=pose_types_train,
            fname_suffix=cmd_line_args['input_suffix'],
            edge_radius=cmd_line_args['edge_radius'],
            estimate_bonds=cmd_line_args.get('estimate_bonds', False),
            prune=cmd_line_args.get('prune', False),
            p_remove_entity=cmd_line_args.get('p_remove_entity', 0),
            extended_atom_types=cmd_line_args.get('extended_atom_types', False),
            p_noise=cmd_line_args.get('p_noise', 0),
            include_strain_info=cmd_line_args.get('include_strain_info', False),
            model_task='classification'
        )

        if pose_data_root_test is not None:
            pose_test_dl = get_data_loader(
                pose_data_root_test,
                compact=cmd_line_args['compact'],
                dataset_class=dataset_class,
                use_atomic_numbers=cmd_line_args['use_atomic_numbers'],
                radius=cmd_line_args['radius'],
                polar_hydrogens=cmd_line_args['hydrogens'],
                batch_size=cmd_line_args['batch_size'],
                types_fname=pose_types_test,
                edge_radius=cmd_line_args['edge_radius'],
                estimate_bonds=cmd_line_args.get('estimate_bonds', False),
                prune=cmd_line_args.get('prune', False),
                rot=False, mode='val', fname_suffix=cmd_line_args['input_suffix'],
                extended_atom_types=cmd_line_args.get('extended_atom_types', False),
                include_strain_info=cmd_line_args.get('include_strain_info', False),
                model_task='classification'
            )

    if affinity_data_root_train is not None:
        affinity_train_dl = get_data_loader(
            affinity_data_root_train,
            dataset_class=dataset_class,
            batch_size=cmd_line_args['batch_size'],
            compact=cmd_line_args['compact'], radius=cmd_line_args['radius'],
            use_atomic_numbers=cmd_line_args['use_atomic_numbers'], rot=False,
            augmented_actives=cmd_line_args['augmented_actives'],
            min_aug_angle=cmd_line_args['min_aug_angle'],
            max_active_rms_distance=cmd_line_args['max_active_rmsd'],
            min_inactive_rms_distance=cmd_line_args['min_inactive_rmsd'],
            max_inactive_rms_distance=cmd_line_args.get('max_inactive_rmsd', None),
            polar_hydrogens=cmd_line_args['hydrogens'],
            mode='train',
            types_fname=affinity_types_train,
            fname_suffix=cmd_line_args['input_suffix'],
            edge_radius=cmd_line_args['edge_radius'],
            estimate_bonds=cmd_line_args.get('estimate_bonds', False),
            prune=cmd_line_args.get('prune', False),
            p_remove_entity=cmd_line_args.get('p_remove_entity', 0),
            extended_atom_types=cmd_line_args.get('extended_atom_types', False),
            p_noise=cmd_line_args.get('p_noise', 0),
            include_strain_info=cmd_line_args.get('include_strain_info', False),
            model_task=regression_task
        )

        if affinity_data_root_test is not None:
            affinity_test_dl = get_data_loader(
                affinity_data_root_test,
                compact=cmd_line_args['compact'],
                dataset_class=dataset_class,
                use_atomic_numbers=cmd_line_args['use_atomic_numbers'],
                radius=cmd_line_args['radius'],
                polar_hydrogens=cmd_line_args['hydrogens'],
                batch_size=cmd_line_args['batch_size'],
                types_fname=affinity_types_test,
                edge_radius=cmd_line_args['edge_radius'],
                estimate_bonds=cmd_line_args.get('estimate_bonds', False),
                prune=cmd_line_args.get('prune', False),
                rot=False, mode='val', fname_suffix=cmd_line_args['input_suffix'],
                extended_atom_types=cmd_line_args.get('extended_atom_types', False),
                include_strain_info=cmd_line_args.get('include_strain_info', False),
                model_task=regression_task
            )

    args_to_record = cmd_line_args
    args_to_record.update(model_kwargs)

    wandb_project = cmd_line_args['wandb_project']
    wandb_run = cmd_line_args['wandb_run']

    save_path = cmd_line_args['save_path']
    if wandb_project is not None and wandb_run is not None:
        save_path = Path(save_path, wandb_project, wandb_run)

    wandb_init_kwargs = {
        'project': wandb_project, 'allow_val_change': True,
        'config': args_to_record, 'dir': save_path
    }
    if wandb_project is not None:
        wandb.init(**wandb_init_kwargs)
        if wandb_run is not None:
            wandb.run.name = wandb_run

    val_on_epoch_end = cmd_line_args.get('val_on_epoch_end', False)
    top1 = cmd_line_args.get('top1', False)
    if pose_train_dl is not None:
        model.train()
        model.set_task('classification')
        model.train_model(
            pose_train_dl, epochs=epochs_classi, top1_on_end=top1,
            epoch_end_validation_set=pose_test_dl if val_on_epoch_end else None)
    if pose_test_dl is not None:
        model.eval()
        model.set_task('classification')
        model.val(pose_test_dl, top1_on_end=top1)
    if affinity_train_dl is not None:
        model.train()
        model.set_task(regression_task)
        model.train_model(
            affinity_train_dl, epochs=epochs_affini, top1_on_end=top1,
            epoch_end_validation_set=affinity_test_dl if val_on_epoch_end else None)
    if affinity_test_dl is not None:
        model.eval()
        model.set_task(regression_task)
        model.val(affinity_test_dl, top1_on_end=top1)
