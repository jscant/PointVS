"""Resume training if a job has crashed."""

import argparse

import wandb

from point_vs.models.load_model import load_model
from point_vs.preprocessing.data_loaders import PygPointCloudDataset, \
    PointCloudDataset, get_data_loader
from point_vs.utils import expand_path


def find_latest_checkpoint(root):
    res = ''
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

    model, model_kwargs, cmd_line_args = load_model(ckpt)

    if cmd_line_args['model'] in ('lucid', 'egnn'):
        dataset_class = PygPointCloudDataset
    else:
        dataset_class = PointCloudDataset

    train_dl = get_data_loader(
        cmd_line_args['train_data_root'], receptors=None,
        compact=cmd_line_args['compact'],
        dataset_class=dataset_class,
        use_atomic_numbers=cmd_line_args['use_atomic_numbers'],
        radius=cmd_line_args['radius'],
        polar_hydrogens=cmd_line_args['hydrogens'],
        batch_size=cmd_line_args['batch_size'],
        types_fname=cmd_line_args['train_types'],
        edge_radius=cmd_line_args['edge_radius'],
        rot=False, mode='train', fname_suffix=cmd_line_args['input_suffix'])

    if cmd_line_args['test_data_root'] is not None:
        test_dl = get_data_loader(
            cmd_line_args['test_data_root'], receptors=None,
            compact=cmd_line_args['compact'],
            dataset_class=dataset_class,
            use_atomic_numbers=cmd_line_args['use_atomic_numbers'],
            radius=cmd_line_args['radius'],
            polar_hydrogens=cmd_line_args['hydrogens'],
            batch_size=cmd_line_args['batch_size'],
            types_fname=cmd_line_args['test_types'],
            edge_radius=cmd_line_args['edge_radius'],
            rot=False, mode='val', fname_suffix=cmd_line_args['input_suffix'])
    else:
        test_dl = None

    args_to_record = cmd_line_args
    args_to_record.update(model_kwargs)

    wandb_project = cmd_line_args['wandb_project']
    wandb_run = cmd_line_args['wandb_run']

    wandb_init_kwargs = {
        'project': wandb_project, 'allow_val_change': True,
        'config': args_to_record
    }
    if wandb_project is not None:
        wandb.init(**wandb_init_kwargs)
        if wandb_run is not None:
            wandb.run.name = wandb_run

    if int(cmd_line_args['epochs']):
        model.optimise(
            train_dl, epochs=cmd_line_args['epochs'],
            epoch_end_validation_set=test_dl)

    model = model.eval()
    if test_dl is not None:
        model.test(test_dl)
