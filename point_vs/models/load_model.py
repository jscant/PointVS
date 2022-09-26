"""Load a model from a saved checkpoint."""
from pathlib import Path
from typing import Dict, Tuple, Union

from torch import nn

from point_vs.models.geometric.egnn_lucid import PygLucidEGNN
from point_vs.models.geometric.egnn_satorras import SartorrasEGNN
from point_vs.utils import load_yaml, find_latest_checkpoint

Fname = Union[str, Path]

def load_model(
        model_path: Fname,
        silent: bool = True,
        fetch_args_only: bool = False,
        init_path: bool = False) -> Tuple[Path, nn.Module, Dict, Dict]:
    """Load model weights and arguments used to train model.

    Arguments:
        model_path: Location of the model checkpoint. Either a *.pt file, or
            a directory containing the checkpoints subdirectory in which case
            the latest saved weights will be loaded.
        silent: Do not print messages to stdout.
        fetch_args_only: Do not retrieve model weights.
        init_path: Create output directories.

    Returns:
        Tuple containing the path to the model weights file, the pytorch model
        object, and dictionaries containing the arguments used to train the
        model.
    """
    model_path = Path(model_path).expanduser()
    if model_path.is_dir():
        print(
            'Model specified is directory, searching for latest checkpoint...')
        model_path = find_latest_checkpoint(model_path)
        print('Found checkpoint at', '/'.join(str(model_path).split('/')[-3:]))

    model_kwargs = load_yaml(model_path.parents[1] / 'model_kwargs.yaml')
    #model_kwargs['group'] = SE3(0.2)
    cmd_line_args = load_yaml(model_path.parents[1] / 'cmd_args.yaml')
    if 'node_attention' not in cmd_line_args.keys():
        cmd_line_args['node_attention'] = False
    if 'edge_attention' not in cmd_line_args.keys():
        cmd_line_args['edge_attention'] = cmd_line_args.get(
            'egnn_attention', False)
        model_kwargs['edge_attention'] = cmd_line_args['edge_attention']

    if fetch_args_only:
        return None, None, model_kwargs, cmd_line_args

    model_type = cmd_line_args['model']

    model_class = {
        'egnn': SartorrasEGNN,
        'lucid': PygLucidEGNN,
    }

    if init_path:
        wandb_project = cmd_line_args['wandb_project']
        wandb_run = cmd_line_args['wandb_run']
        save_path = Path(cmd_line_args['save_path'])
        if wandb_project is not None and wandb_run is not None:
            save_path = Path(save_path, wandb_project, wandb_run)
    else:
        save_path = Path()

    model_class = model_class[model_type]
    model = model_class(save_path, learning_rate=cmd_line_args['learning_rate'],
                        weight_decay=cmd_line_args['weight_decay'],
                        use_1cycle=cmd_line_args['use_1cycle'],
                        warm_restarts=cmd_line_args['warm_restarts'],
                        silent=silent, **model_kwargs)

    model.load_weights(model_path, silent=silent)
    model = model.eval()

    return model_path, model, model_kwargs, cmd_line_args
