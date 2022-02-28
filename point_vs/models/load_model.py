from pathlib import Path

from lie_conv.lieGroups import SE3

from point_vs.models.geometric.egnn_lucid import PygLucidEGNN
from point_vs.models.geometric.egnn_satorras import SartorrasEGNN
from point_vs.models.vanilla.lie_conv import LieResNet
from point_vs.models.vanilla.lie_transformer import EquivariantTransformer
from point_vs.utils import load_yaml


def load_model(
        weights_file, silent=True, fetch_args_only=False, init_path=False):
    model_path = Path(weights_file).expanduser()
    model_kwargs = load_yaml(model_path.parents[1] / 'model_kwargs.yaml')
    model_kwargs['group'] = SE3(0.2)
    cmd_line_args = load_yaml(model_path.parents[1] / 'cmd_args.yaml')
    if fetch_args_only:
        return None, model_kwargs, cmd_line_args

    model_type = cmd_line_args['model']

    model_class = {
        'lieconv': LieResNet,
        'egnn': SartorrasEGNN,
        'lucid': PygLucidEGNN,
        'lietransformer': EquivariantTransformer
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

    model.load_weights(model_path)
    model = model.eval()

    return model, model_kwargs, cmd_line_args
