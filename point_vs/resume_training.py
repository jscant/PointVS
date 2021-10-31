"""Resume training if a job has crashed."""

import argparse
from pathlib import Path

import wandb

from point_vs.models.load_model import load_model
from point_vs.preprocessing.data_loaders import PygPointCloudDataset, \
    PointCloudDataset, get_data_loader
from point_vs.utils import expand_path


def find_latest_checkpoint(root):
    res = None
    max_epoch = -1
    for fname in expand_path(root, 'checkpoints').glob('*.pt'):
        idx = int(fname.with_suffix('').name.split('_')[-1])
        if idx > max_epoch:
            res = fname
            max_epoch = idx
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_path', type=str,
                        help='Base path from which to resume training.')

    base_path = expand_path(parser.parse_args().base_path)
    ckpt = find_latest_checkpoint(base_path)
    print('Found checkpoint:', ckpt)

    model, model_kwargs, cmd_line_args = load_model(ckpt, silent=False)

    if cmd_line_args['model'] in ('lucid', 'egnn'):
        dataset_class = PygPointCloudDataset
    else:
        dataset_class = PointCloudDataset

    train_dl = get_data_loader(
        cmd_line_args['train_data_root'],
        dataset_class=dataset_class,
        batch_size=cmd_line_args['batch_size'],
        compact=cmd_line_args['compact'], radius=cmd_line_args['radius'],
        use_atomic_numbers=cmd_line_args['use_atomic_numbers'], rot=False,
        augmented_actives=cmd_line_args['augmented_actives'],
        min_aug_angle=cmd_line_args['min_aug_angle'],
        max_active_rms_distance=cmd_line_args['max_active_rmsd'],
        min_inactive_rms_distance=cmd_line_args['min_inactive_rmsd'],
        polar_hydrogens=cmd_line_args['hydrogens'],
        mode='train',
        types_fname=cmd_line_args['train_types'],
        fname_suffix=cmd_line_args['input_suffix'],
        edge_radius=cmd_line_args['edge_radius'],
        estimate_bonds=cmd_line_args.get('estimate_bonds', False),
        prune=cmd_line_args.get('prune', False)
    )

    if cmd_line_args['test_data_root'] is not None:
        test_dl = get_data_loader(
            cmd_line_args['test_data_root'],
            compact=cmd_line_args['compact'],
            dataset_class=dataset_class,
            use_atomic_numbers=cmd_line_args['use_atomic_numbers'],
            radius=cmd_line_args['radius'],
            polar_hydrogens=cmd_line_args['hydrogens'],
            batch_size=cmd_line_args['batch_size'],
            types_fname=cmd_line_args['test_types'],
            edge_radius=cmd_line_args['edge_radius'],
            estimate_bonds=cmd_line_args.get('estimate_bonds', False),
            prune=cmd_line_args.get('prune', False),
            rot=False, mode='val', fname_suffix=cmd_line_args['input_suffix'])
    else:
        test_dl = None

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

    if int(cmd_line_args['epochs']):
        model.train_model(
            train_dl, epochs=cmd_line_args['epochs'],
            top1_on_end=cmd_line_args['top1'],
            epoch_end_validation_set=test_dl)

    model = model.eval()
    if test_dl is not None:
        model.val(test_dl, top1_on_end=cmd_line_args['top1'])
