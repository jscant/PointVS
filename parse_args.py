"""Set up rather large command line argument list"""

import argparse
from pathlib import PosixPath


def parse_args():
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
    parser.add_argument('--load_weights', '-l', type=PosixPath, required=False,
                        help='Load a model.')
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
    parser.add_argument('--batch_size', '-b', type=int, required=False,
                        default=32,
                        help='Number of examples to include in each batch for '
                             'training.')
    parser.add_argument('--epochs', '-e', type=int, required=False,
                        default=1,
                        help='Number of times to iterate through training set.')
    parser.add_argument('--channels', '-k', type=int, default=32,
                        help='Channels for feature vectors')
    parser.add_argument('--train_receptors', '-r', type=str, nargs='*',
                        help='Names of specific receptors for training. If '
                             'specified, other structures will be ignored.')
    parser.add_argument('--test_receptors', '-q', type=str, nargs='*',
                        help='Names of specific receptors for testing. If '
                             'specified, other structures will be ignored.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.002,
                        help='Learning rate for gradient descent')
    parser.add_argument('--wandb_project', type=str,
                        help='Name of wandb project. If left blank, wandb '
                             'logging will not be used.')
    parser.add_argument('--wandb_run', type=str,
                        help='Name of run for wandb logging.')
    parser.add_argument('--layers', type=int, default=6,
                        help='Number of layers in LieResNet')
    parser.add_argument('--channels_in', '-chin', type=int, default=12,
                        help='Input channels')
    parser.add_argument('--liftsamples', type=int, default=1,
                        help='liftsamples parameter in LieConv')
    parser.add_argument('--radius', type=int, default=6,
                        help='Maximum distance from a ligand atom for a '
                             'receptor atom to be included in input')
    parser.add_argument('--nbhd', type=int, default=25,
                        help='Number of monte carlo samples for integral')
    parser.add_argument('--al_batch_size', '-albs', type=int, default=-1,
                        help='Number of batches to increase training pool size '
                             'at each iteration of active learning. If '
                             'unspecified, active learning will not be used.')
    parser.add_argument('--al_initial_pool_size', '-alips', type=int,
                        default=-1,
                        help='Size of initial pool size for active learning.')
    parser.add_argument('--al_control', action='store_true',
                        help='Active learning with random data selection (used '
                             'as a control).')
    parser.add_argument('--al_projections', type=int, default=64,
                        help='Number of projections for bayes active learning '
                             '(J in the paper)')
    parser.add_argument('--al_features', type=int, default=256,
                        help='Size of finalature embedding for active learning')
    parser.add_argument('--al_fc_in_features', type=int, default=512,
                        help='Size of input to embedding layer for active '
                             'learning')
    parser.add_argument('--load_args', type=PosixPath,
                        help='Load yaml file with command line args. Any args '
                             'specified in the file will overwrite other args '
                             'specified on the command line.')
    parser.add_argument('--double', action='store_true',
                        help='Use 64-bit floating point precision')
    parser.add_argument('--kernel_type', type=str, default='mlp',
                        help='One of 2232, mlp, overrides attention_fn '
                             '(see original repo)')
    parser.add_argument('--attention_fn', type=str, default='dot_product',
                        help='One of norm_exp, softmax, dot_product: '
                             'activation for attention (overridden by '
                             'kernel_type)')
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function')
    return parser.parse_args()
