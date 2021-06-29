"""Set up rather large command line argument list"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='Type of point cloud network to use: '
                             'lietransformer, lieconv or egnn')
    parser.add_argument('train_data_root', type=str,
                        help='Location of structure training *.parquets files. '
                             'Receptors should be in a directory named '
                             'receptors, with ligands located in their '
                             'specific receptor subdirectory under the '
                             'ligands directory.')
    parser.add_argument('save_path', type=str,
                        help='Directory in which experiment outputs are '
                             'stored. If wandb_run and wandb_project are '
                             'specified, save_path/wandb_project/wandb_run '
                             'will be used to store results.')
    parser.add_argument('--load_weights', '-l', type=str, required=False,
                        help='Load a model.')
    parser.add_argument('--test_data_root', '-t', type=str,
                        required=False,
                        help='Location of structure test *.parquets files. '
                             'Receptors should be in a directory named '
                             'receptors, with ligands located in their '
                             'specific receptor subdirectory under the '
                             'ligands directory.')
    parser.add_argument('--translated_actives', type=str,
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
    parser.add_argument('--weight_decay', '-w', type=float, default=1e-4,
                        help='Weight decay for regularisation')
    parser.add_argument('--wandb_project', type=str,
                        help='Name of wandb project. If left blank, wandb '
                             'logging will not be used.')
    parser.add_argument('--wandb_run', type=str,
                        help='Name of run for wandb logging.')
    parser.add_argument('--layers', type=int, default=6,
                        help='Number of layers in LieResNet')
    parser.add_argument('--liftsamples', type=int, default=1,
                        help='liftsamples parameter in LieConv')
    parser.add_argument('--radius', type=int, default=6,
                        help='Maximum distance from a ligand atom for a '
                             'receptor atom to be included in input')
    parser.add_argument('--nbhd', type=int, default=32,
                        help='Number of monte carlo samples for integral')
    parser.add_argument('--load_args', type=str,
                        help='Load yaml file with command line args. Any args '
                             'specified in the file will overwrite other args '
                             'specified on the command line.')
    parser.add_argument('--double', action='store_true',
                        help='Use 64-bit floating point precision')
    parser.add_argument('--kernel_type', type=str, default='mlp',
                        help='One of 2232, mlp, overrides attention_fn '
                             '(see original repo) (LieTransformer)')
    parser.add_argument('--attention_fn', type=str, default='dot_product',
                        help='One of norm_exp, softmax, dot_product: '
                             'activation for attention (overridden by '
                             'kernel_type) (LieTransformer)')
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function')
    parser.add_argument('--kernel_dim', type=int, default=16,
                        help='Size of linear layers in attention kernel '
                             '(LieTransformer)')
    parser.add_argument('--feature_embed_dim', type=int, default=None,
                        help='Feature embedding dimension for attention; '
                             'paper had dv=848 for QM9 (LieTransformer)')
    parser.add_argument('--mc_samples', type=int, default=0,
                        help='Monte carlo samples for attention '
                             '(LieTransformer)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Chance for nodes to be inactivated on each '
                             'trainin batch (EGNN)')
    parser.add_argument('--fill', type=float, default=0.75,
                        help='LieTransformer fill parameter')
    parser.add_argument('--use_1cycle', action='store_true',
                        help='Use 1cycle learning rate scheduling')
    parser.add_argument('--fourier_features', type=int, default=0,
                        help='(EGNN) Number of fourier terms to use when '
                             'encoding distances (default is not to use '
                             'fourier distance encoding)')
    parser.add_argument('--norm_coords', action='store_true',
                        help='(EGNN) Normalise coordinate vectors')
    parser.add_argument('--norm_feats', action='store_true',
                        help='(EGNN) Normalise feature vectors')
    parser.add_argument('--use_atomic_numbers', action='store_true',
                        help='Use atomic numbers rather than smina types')
    parser.add_argument('--compact', action='store_true',
                        help='Use compact rather than true one-hot encodings')
    parser.add_argument('--thin_mlps', action='store_true',
                        help='(EGNN) Use single layer MLPs for edge, node and '
                             'coord updates')
    parser.add_argument('--hydrogens', action='store_true',
                        help='Include polar hydrogens')
    parser.add_argument('--augmented_actives', type=int, default=0,
                        help='Number of randomly rotated actives to be '
                             'included as decoys during training')
    parser.add_argument('--min_aug_angle', type=float, default=30,
                        help='Minimum angle of rotation for augmented actives '
                             'as specified in the augmented_actives argument')
    return parser.parse_args()
