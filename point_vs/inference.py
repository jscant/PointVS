"""Perform inference on a validation set with a pretrained model.

usage: inference.py [-h] [--wandb_project WANDB_PROJECT] [--wandb_run WANDB_RUN]
                    model_checkpoint test_types test_data_root

positional arguments:
  model_checkpoint      Location of saved model weights (usually *.pt)
  test_types            Location of types file containing information on
                        location of test data
  test_data_root        Location to which paths in test_types are relative

optional arguments:
  -h, --help            show this help message and exit
  --wandb_project WANDB_PROJECT, -p WANDB_PROJECT
                        Specify a wandb project. If unspecified, wandb will not
                        be used; if set to "SAME", the same project that the
                        original model is in will be used.
  --wandb_run WANDB_RUN, -r WANDB_RUN
                        Specify a wandb run. If unspecified, a random run name
                        will be given. If set to "SAME", the same run name with
                        _VAL-<test_types> appended will be used.
"""

import argparse
from pathlib import Path

import wandb

from point_vs.models.load_model import load_model
from point_vs.preprocessing.data_loaders import get_data_loader, \
    PygPointCloudDataset, PointCloudDataset
from point_vs.utils import expand_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_checkpoint', type=str,
                        help='Location of saved model weights (usually *.pt)')
    parser.add_argument('test_types', type=str,
                        help='Location of types file containing information '
                             'on location of test data')
    parser.add_argument('test_data_root', type=str,
                        help='Location to which paths in test_types are '
                             'relative')
    parser.add_argument('--wandb_project', '-p', type=str,
                        help='Specify a wandb project. If unspecified, wandb '
                             'will not be used; if set to "SAME", the same '
                             'project that the original model is in will be '
                             'used.')
    parser.add_argument('--wandb_run', '-r', type=str,
                        help='Specify a wandb run. If unspecified, a random '
                             'run name will be given. If set to "SAME", '
                             'the same run name with _VAL-<test_types> appended'
                             ' will be used.')
    args = parser.parse_args()

    model, model_kwargs, cmd_line_args = load_model(args.model_checkpoint)

    results_fname = expand_path(Path(
        expand_path(args.model_checkpoint).parents[1],
        'predictions_{0}-{1}.txt'.format(
            Path(args.test_types).with_suffix('').name,
            Path(args.model_checkpoint).with_suffix('').name)))

    # Is a validation set specified?
    if cmd_line_args['model'] in ('lucid', 'egnn'):
        dataset_class = PygPointCloudDataset
    else:
        dataset_class = PointCloudDataset

    test_dl = get_data_loader(
        args.test_data_root, receptors=None,
        compact=cmd_line_args['compact'],
        dataset_class=dataset_class,
        use_atomic_numbers=cmd_line_args['use_atomic_numbers'],
        radius=cmd_line_args['radius'],
        polar_hydrogens=cmd_line_args['hydrogens'],
        batch_size=cmd_line_args['batch_size'],
        types_fname=args.test_types,
        edge_radius=cmd_line_args['edge_radius'],
        rot=False, mode='val', fname_suffix=cmd_line_args['input_suffix'])

    args_to_record = vars(args)

    wandb_project = args.wandb_project
    wandb_run = args.wandb_run
    if wandb_project is not None:
        if wandb_project.lower() == 'same':
            wandb_project = cmd_line_args['wandb_project']
        if wandb_run is not None:
            if wandb_run.lower() == 'same':
                wandb_run = cmd_line_args['wandb_run'] + '_VAL-' + Path(
                    args.test_types).with_suffix('').name

    wandb_init_kwargs = {
        'project': wandb_project, 'allow_val_change': True,
        'config': args_to_record
    }
    if wandb_project is not None:
        wandb.init(**wandb_init_kwargs)
        if wandb_run is not None:
            wandb.run.name = wandb_run
    model = model.eval()
    model.test(test_dl, results_fname)
