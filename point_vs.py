"""
LieConvVS uses the the LieConv layer to perform virtual screening on
protein-ligand complexes. This is the main script, and can be used like so:

python3 lieconv_vs.py <model> <data_root> <save_path> --batch_size int
    --receptors [str]

for example:
python3 lieconv_vs.py resnet data/small_chembl_test ~/test_output

Specific receptors can be specified as a list for the final argument:
python3 lieconv_vs.py resnet data/small_chembl_test ~/test_output -r 20014 28

<model> can be either of gnina or restnet.
"""

import argparse
import warnings
from pathlib import PosixPath

from session import Session, EvidentialLieResNet, SE3TransformerSigmoid

try:
    import wandb
except ImportError:
    print('Library wandb not available. --wandb and --run flags should not be '
          'used.')
    wandb = None

from settings import LieConvSettings, SessionSettings, SE3TransformerSettings, \
    EvidentialLieConvSettings

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Type of point cloud network to'
                                                ' use: se3trans or lieconv')
    parser.add_argument('train_data_root', type=PosixPath,
                        help='Location of structure training *.parquets files. '
                             'Receptors should be in a directory named '
                             'receptors, with ligands located in their '
                             'specific receptor subdirectory under the '
                             'ligands directory.')
    parser.add_argument('save_path', type=PosixPath,
                        help='Directory in which experiment outputs are '
                             'stored.')
    parser.add_argument('--load', '-l', type=PosixPath, required=False,
                        help='Load a session and model.')
    parser.add_argument('--test_data_root', '-t', type=PosixPath,
                        required=False,
                        help='Location of structure test *.parquets files. '
                             'Receptors should be in a directory named '
                             'receptors, with ligands located in their '
                             'specific receptor subdirectory under the '
                             'ligands directory.')
    parser.add_argument('--translated_actives', type=PosixPath,
                        help='Directory in which translated actives are stored.'
                             ' If unspecified, no translated actives will be '
                             'used. The use of translated actives are is '
                             'discussed in https://pubs.acs.org/doi/10.1021/ac'
                             's.jcim.0c00263')
    parser.add_argument('--n_translated_actives', type=int,
                        help='Maximum number of translated actives to be '
                             'included')
    parser.add_argument('--batch_size', '-b', type=int, required=False,
                        help='Number of examples to include in each batch for '
                             'training.')
    parser.add_argument('--epochs', '-e', type=int, required=False,
                        help='Number of times to iterate through training set.')
    parser.add_argument('--train_receptors', '-r', type=str, nargs='*',
                        help='Names of specific receptors for training. If '
                             'specified, other structures will be ignored.')
    parser.add_argument('--test_receptors', '-q', type=str, nargs='*',
                        help='Names of specific receptors for testing. If '
                             'specified, other structures will be ignored.')
    parser.add_argument('--save_interval', '-s', type=int, default=None,
                        help='Save checkpoints after every <save_interval> '
                             'batches.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=None,
                        help='Learning rate for gradient descent')
    parser.add_argument('--model_conf', '-m', type=PosixPath,
                        help='Config file for model parameters. If unspecified,'
                             'certain defaults are used.')
    parser.add_argument('--session_conf', type=PosixPath,
                        help='Config file for session parameters.')
    parser.add_argument('--wandb', type=str,
                        help='Name of wandb project. If left blank, wandb '
                             'logging will not be used.')
    parser.add_argument('--run', type=str,
                        help='Name of run for wandb logging.')
    args = parser.parse_args()

    save_path = args.save_path.expanduser()
    if args.wandb is not None:
        wandb_path = save_path / 'wandb_{}'.format(args.wandb)
        wandb_path.mkdir(parents=True, exist_ok=True)
        wandb.init(project=args.wandb, allow_val_change=True)
        wandb.config.update(args, allow_val_change=True)

    conf_base_name = args.model + '_conf.yaml'

    if args.model == 'se3trans':
        network_class = SE3TransformerSigmoid
        model_settings_class = SE3TransformerSettings
    elif args.model == 'lieconv':
        network_class = LieConvSettings
        model_settings_class = LieConvSettings
    elif args.model == 'evilieconv':
        network_class = EvidentialLieResNet
        model_settings_class = EvidentialLieConvSettings
    else:
        raise AssertionError('model must be one of se3trans or lieconv.')

    # Load model settings either from custom yaml or defaults. Command line args
    # will take presidence over yaml args
    with model_settings_class(
            args.model_conf, save_path / conf_base_name) as model_settings:
        for arg, value in vars(args).items():
            if value is not None:
                if hasattr(model_settings, arg):
                    setattr(model_settings, arg, value)
        if args.wandb is not None:
            wandb.config.update(model_settings.settings, allow_val_change=True)
        network = network_class(**model_settings.settings)

    # Load session settings either from custom yaml or defaults. Command line
    # args will take presidence over yaml args
    with SessionSettings(
            args.session_conf, save_path / 'session.yaml') as session_settings:
        for arg, value in vars(args).items():
            if value is not None:
                if hasattr(session_settings, arg):
                    setattr(session_settings, arg, value)
        if args.wandb is not None:
            wandb.config.update(
                session_settings.settings, allow_val_change=True)
        sess = Session(network, **session_settings.settings)

    print('Built network with {} params'.format(sess.param_count))

    if args.load is not None:
        sess.load(args.load)
        for arg, value in vars(args).items():
            if value is not None:
                setattr(sess, arg, value)

    if args.wandb is None:
        sess.wandb = None
    if sess.epochs > 0:
        sess.train()
    if sess.test_data_root is not None:
        sess.test()
