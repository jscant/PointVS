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
import time
from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lie_conv import lieConv
from lie_conv.lieGroups import SE3

import models
from preprocessing import centre_on_ligand, make_box, concat_structs
from utils import format_time, print_with_overwrite, get_eta, \
    plot_with_smoothing


class MolLoader(torch.utils.data.Dataset):
    """Class for feeding structure parquets into network."""

    def __init__(self, base_path, radius=12, receptors=None, **kwargs):
        """Initialise dataset.

        Arguments:
            base_path: path containing the 'receptors' and 'ligands'
                directories, which in turn contain <rec_name>.parquets files
                and folders called <rec_name>_[active|decoy] which in turn
                contain <ligand_name>.parquets files. All parquets files from
                this directory are recursively loaded into the dataset.
            radius: size of the bounding box; all atoms further than <radius>
                Angstroms from the mean ligand atom position are discarded.
            receptors: iterable of strings denoting receptors to include in
                the dataset. if None, all receptors found in base_path are
                included.
            kwargs: keyword arguments passed to the parent class (Dataset).
        """
        super().__init__(**kwargs)
        self.radius = radius
        self.base_path = Path(base_path).expanduser()

        if receptors is None:
            print('Loading all structures in', self.base_path)
            filenames = list((self.base_path / 'ligands').rglob('**/*.parquet'))
        else:
            print('Loading receptors:')
            filenames = []
            for receptor in receptors:
                print(receptor)
                filenames += list((self.base_path / 'ligands').rglob(
                    '{}*/*.parquet'.format(receptor)))

        self.filenames = filenames

        labels = []
        for fname in self.filenames:
            if str(fname).find('active') == -1:
                labels.append(0)
            else:
                labels.append(1)
        labels = np.array(labels)
        class_sample_count = np.array(
            [len(labels) - np.sum(labels), np.sum(labels)])
        weights = 1. / class_sample_count
        self.sample_weights = torch.from_numpy(
            np.array([weights[i] for i in labels])).float()
        self.labels = labels

        self.sampler = torch.utils.data.WeightedRandomSampler(
            self.sample_weights, len(self.sample_weights)
        )

    def __len__(self):
        """Returns the total size of the dataset."""
        return len(self.filenames)

    def __getitem__(self, item):
        """Given an index, locate and preprocess relevant parquets files.

        Arguments:
            item: index in the list of filenames denoting which ligand and
                receptor to fetch

        Returns:
            Tuple containing (a) a tuple with a list of tensors: cartesian
            coordinates, feature vectors and masks for each point, as well as
            the number of points in the structure and (b) the label \in \{0, 1\}
            denoting whether the structure is an active or a decoy.
        """
        lig_fname = self.filenames[item]
        label = self.labels[item]
        rec_name = lig_fname.parent.name.split('_')[0]
        rec_fname = next((self.base_path / 'receptors').glob(
            '{}*.parquet'.format(rec_name)))

        struct = centre_on_ligand(make_box(centre_on_ligand(
            concat_structs(rec_fname, lig_fname)), radius=self.radius))
        p = torch.from_numpy(
            np.expand_dims(struct[struct.columns[:3]].to_numpy(),
                           0)).float()
        v = nn.functional.one_hot(torch.from_numpy(
            np.expand_dims(struct.types.to_numpy(), 0))).float()
        m = torch.from_numpy(np.ones((1, len(struct)))).float()
        return (p, v, m, len(struct)), lig_fname, rec_fname, label


class Session:
    """Handles training of LieConv-based models."""

    def __init__(self, network, train_root, save_path, batch_size,
                 test_root=None, train_receptors=None, test_receptors=None,
                 save_interval=-1, learning_rate=0.01):
        """Initialise session.

        Arguments:
            network: pytorch object inheriting from torch.nn.Module.
            train_root: path containing the 'receptors' and 'ligands' training
                directories, which in turn contain <rec_name>.parquets files
                and folders called <rec_name>_[active|decoy] which in turn
                contain <ligand_name>.parquets files. All parquets files from
                this directory are recursively loaded into the dataset.
            save_path: directory in which experiment outputs are stored.
            batch_size: number of training examples in each batch
            test_root: like train_root, but for the test set.
            train_receptors: iterable of strings denoting receptors to include
                in the dataset. if None, all receptors found in base_path are
                included.
            test_receptors: like train_receptors, but for the test set.
            save_interval: save model checkpoint every <save_interval> batches.
        """
        self.train_data_root = Path(train_root).expanduser()
        self.test_data_root = Path(test_root).expanduser()
        self.save_path = Path(save_path).expanduser()
        self.loss_plot_file = self.save_path / 'loss.png'
        self.network = network
        self.network.apply(self.weights_init)
        self.batch_size = batch_size
        self.losses = []
        self.epoch = 0
        self.batch = 0
        self.save_interval = save_interval
        self.lr = learning_rate

        if isinstance(train_receptors, str):
            train_receptors = tuple([train_receptors])
        if isinstance(test_receptors, str):
            test_receptors = tuple([test_receptors])

        self.train_dataset = MolLoader(
            self.train_data_root, receptors=train_receptors, radius=12)
        self.train_data_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, sampler=self.train_dataset.sampler,
            collate_fn=self.collate
        )
        self.loss_log_file = self.save_path / 'loss.log'

        if self.test_data_root is not None:
            self.test_dataset = MolLoader(
                self.test_data_root, receptors=test_receptors, radius=12)
            self.test_data_loader = torch.utils.data.DataLoader(
                self.test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, collate_fn=self.collate
            )
            self.predictions_file = Path(
                self.save_path, 'predictions_{}.txt'.format(
                    self.test_data_root.name))
        else:
            self.test_dataset = None
            self.test_data_loader = None

        self.last_train_batch_small = int(bool(
            len(self.train_dataset) % self.batch_size))
        self.last_test_batch_small = int(bool(
            len(self.train_dataset) % self.batch_size))
        self.epoch_size_train = self.last_train_batch_small + (
                len(self.train_dataset) // self.batch_size)
        self.epoch_size_test = self.last_test_batch_small + (
                len(self.test_dataset) // self.batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCEWithLogitsLoss().to(device)
        self.optimiser = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.device = device

    def _setup_training_session(self):
        """Puts network on GPU and sets up directory structure."""
        self.network.to(self.device)
        self.save_path.mkdir(parents=True, exist_ok=True)
        for output_file in [self.predictions_file, self.loss_log_file]:
            try:
                output_file.unlink()
            except FileNotFoundError:
                pass
        torch.cuda.empty_cache()
        print('Using device:', self.device, '\n\n')
        return time.time()

    def train(self, epochs=1):
        """Train the network.

        Trains the neural network (in Session.network), displays training
        information and plots the loss. All figures and logs are saved to
        <Session.save_path>.

        Arguments:
            epochs: number of times to iterate through the dataset
        """
        start_time = self._setup_training_session()
        total_iters = epochs * (len(self.train_dataset) // self.batch_size) + \
                      self.last_train_batch_small
        log_interval = 10
        graph_interval = max(1, total_iters // 30)
        global_iter = 0
        self.network.train()
        ax = None
        for self.epoch in range(epochs):
            decoy_mean_pred, active_mean_pred = -1, -1
            for self.batch, ((p, v, m), y_true, _, _) in enumerate(
                    self.train_data_loader):

                p = p.to(self.device)
                v = v.to(self.device)
                m = m.to(self.device)
                y_true = y_true.to(self.device)

                self.optimiser.zero_grad()
                y_pred = self.network((p, v, m))
                loss = self.criterion(y_pred, y_true)

                loss.backward()
                self.optimiser.step()

                eta = get_eta(start_time, global_iter, total_iters)
                time_elapsed = format_time(time.time() - start_time)
                y_true_np = y_true.cpu().detach().numpy()
                y_pred_np = y_pred.cpu().detach().numpy()
                active_idx = np.where(y_true_np > 0.5)
                decoy_idx = np.where(y_true_np < 0.5)

                if len(active_idx[0]):
                    active_mean_pred = np.mean(
                        self.sigmoid(y_pred_np[active_idx]))
                if len(decoy_idx[0]):
                    decoy_mean_pred = np.mean(
                        self.sigmoid(y_pred_np[decoy_idx]))

                print_with_overwrite(
                    ('Epoch:', '{0}/{1}'.format(self.epoch + 1, epochs), '|',
                     'Iteration:', '{0}/{1}'.format(
                        self.batch + 1, self.epoch_size_train)),
                    ('Time elapsed:', time_elapsed, '|',
                     'Time remaining:', eta),
                    ('Loss: {0:.4f}'.format(loss), '|',
                     'Mean active: {0:.4f}'.format(active_mean_pred), '|',
                     'Mean decoy: {0:.4f}'.format(decoy_mean_pred))
                )

                self.losses.append(float(loss))
                if not (self.batch + 1) % log_interval or \
                        self.batch == total_iters - 1:
                    self.save_loss(log_interval)

                if not (self.batch + 1) % graph_interval or \
                        self.batch == total_iters - 1:
                    ax = plot_with_smoothing(
                        self.losses, gap=max(1, self.batch // 15), ax=ax)
                    ax.set_title(
                        'Binary crossentropy training loss for {}'.format(
                            self.train_data_root
                        ))
                    ax.set_xlabel('Batch')
                    ax.set_ylabel('Binary crossentropy loss')
                    ax.set_ylim(bottom=-0.05)
                    plt.savefig(self.loss_plot_file)

                if (self.save_interval > 0 and
                    not global_iter % self.save_interval) or \
                        global_iter == total_iters - 1:
                    self.save()

                global_iter += 1

        ax = plot_with_smoothing(self.losses, gap=max(1, self.batch // 15),
                                 ax=ax)
        ax.set_title('Binary crossentropy training loss for {}'.format(
            self.train_data_root
        ))
        ax.set_xlabel('Batch')
        ax.set_ylabel('Binary crossentropy loss')
        ax.set_ylim(bottom=-0.05)
        plt.savefig(self.loss_plot_file)

    def test(self):
        """Use trained network to perform inference on the test set.

        Uses the neural network (in Session.network), to perform predictions
        on the structures found in <test_data_root>, and saves this output
        to <save_path>/predictions_<test_data_root.name>.txt.
        """
        start_time = time.time()
        log_interval = 10
        decoy_mean_pred, active_mean_pred = -1, -1
        predictions = ''
        self.network.eval()
        with torch.no_grad():
            for self.batch, ((p, v, m), y_true, ligands, receptors) in \
                    enumerate(self.test_data_loader):
                p = p.to(self.device)
                v = v.to(self.device)
                m = m.to(self.device)
                y_true = y_true.to(self.device)

                y_pred = self.network((p, v, m))
                loss = self.criterion(y_pred, y_true)

                eta = get_eta(start_time, self.batch, self.epoch_size_test)
                time_elapsed = format_time(time.time() - start_time)
                y_true_np = y_true.cpu().detach().numpy()
                y_pred_np = y_pred.cpu().detach().numpy()
                active_idx = np.where(y_true_np > 0.5)
                decoy_idx = np.where(y_true_np < 0.5)

                if len(active_idx[0]):
                    active_mean_pred = float(
                        np.mean(self.sigmoid(y_pred_np[active_idx])))
                if len(decoy_idx[0]):
                    decoy_mean_pred = float(
                        np.mean(self.sigmoid(y_pred_np[decoy_idx])))

                print_with_overwrite(
                    ('Inference on: {}'.format(self.test_data_root), '|',
                     'Iteration:', '{0}/{1}'.format(
                        self.batch + 1, self.epoch_size_test)),
                    ('Time elapsed:', time_elapsed, '|',
                     'Time remaining:', eta),
                    ('Loss: {0:.4f}'.format(loss), '|',
                     'Mean active: {0:.4f}'.format(active_mean_pred), '|',
                     'Mean decoy: {0:.4f}'.format(decoy_mean_pred))
                )

                predictions += '\n'.join(['{0} | {1:.7f} {2} {3}'.format(
                    int(y_true_np[i, 1]),
                    self.sigmoid(y_pred_np[i, 1]),
                    receptors[i],
                    ligands[i]) for i in range(len(receptors))]) + '\n'

                # Periodically write predictions to disk
                if not (self.batch + 1) % log_interval or self.batch == len(
                        self.test_data_loader) - 1:
                    with open(self.predictions_file, 'a') as f:
                        f.write(predictions)
                        predictions = ''

    def save(self):
        """Save all session attributes, including internal states."""
        attributes = {}
        accounted_for = set('network')
        for attr in dir(self.network):
            if attr != '__class__' and \
                    'state_dict' in dir(getattr(self.network, attr)):
                attributes.update({
                    attr + '_state_dict': getattr(
                        getattr(self.network, attr), 'state_dict')()})
                accounted_for.add(attr)
        for attr in dir(self):
            if 'state_dict' in dir(getattr(self, attr)):
                attributes.update({
                    attr + '_state_dict': getattr(
                        getattr(self, attr), 'state_dict')()})
                accounted_for.add(attr)
        for var, val in [(varname, getattr(self, varname)) for varname in
                         vars(self) if varname not in accounted_for]:
            attributes.update({var: val})

        Path(self.save_path, 'checkpoints').mkdir(exist_ok=True, parents=True)
        torch.save(attributes, Path(
            self.save_path, 'checkpoints',
            'ckpt_epoch_{}_batch_{}.pt'.format(self.epoch + 1, self.batch + 1)))

    def load(self, checkpoint_path):
        """Fully automatic loading of models saved with self.save.

        Arguments:
            checkpoint_path: directory containing saved model
        """
        checkpoint_path = Path(checkpoint_path).expanduser()
        load_dict = torch.load(checkpoint_path)
        states = {
            key.replace('_state_dict', ''): value for key, value in
            load_dict.items() if key.endswith('_state_dict')}
        attributes = {
            key: value for key, value in
            load_dict.items() if not key.endswith('_state_dict')}

        for attr_name, value in attributes.items():
            setattr(self, attr_name, value)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.network.load_state_dict(states['network'])
        self.network.net.load_state_dict(states['net'])
        self.network.to(device)
        self.optimiser = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.optimiser.load_state_dict(states['optimiser'])

    def save_loss(self, save_interval):
        """Save the loss information to disk.

        Arguments:
            save_interval: how often the loss is being recorded (in batches).
        """
        log_file = self.save_path / 'loss.log'
        start_idx = save_interval * (self.batch // save_interval)
        with open(log_file, 'a') as f:
            f.write('\n'.join(
                [str(idx + start_idx + 1) + ' ' + str(loss) for idx, loss in
                 enumerate(self.losses[-save_interval:])]) + '\n')

    @staticmethod
    def collate(batch):
        """Processing of inputs which takes place after batch is selected.

        LieConv networks take tuples of torch tensors (p, v, m), which are:
            p, (batch_size, n_atoms, 3): coordinates of each atom
            v, (batch_size, n_atoms, n_features): features for each atom
            m, (batch_size, n_atoms): mask for each coordinate slot

        Note that n_atoms is the largest number of atoms in a structure in
        each batch.

        Arguments:
            batch: iterable of individual inputs.

        Returns:
            Tuple of feature vectors ready for input into a LieConv network.
        """
        max_len = max([b[0][-1] for b in batch])
        batch_size = len(batch)
        p_batch = torch.zeros(batch_size, max_len, 3)
        v_batch = torch.zeros(batch_size, max_len, 22)
        m_batch = torch.zeros(batch_size, max_len)
        label_batch = torch.zeros(batch_size, 2)
        ligands, receptors = [], []
        for batch_index, ((p, v, m, _), ligand, receptor, label) in enumerate(
                batch):
            p_batch[batch_index, :p.shape[1], :] = p
            v_batch[batch_index, :v.shape[1], :] = v
            m_batch[batch_index, :m.shape[1]] = m
            label_batch[batch_index, label] = 1
            ligands.append(ligand)
            receptors.append(receptor)
        return (p_batch.float(), v_batch.float(),
                m_batch.bool()), label_batch.float(), ligands, receptors

    @staticmethod
    def weights_init(m):
        """Initialise weights of network using xavier initialisation."""
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

    @staticmethod
    def sigmoid(x):
        """Sigmoid function for reporting."""
        return 1 / (1 + np.exp(-x))

    @property
    def param_count(self):
        return sum([torch.numel(t) for t in self.network.parameters()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='Model architecture; either gnina or resnet.')
    parser.add_argument('train_data_root', type=str,
                        help='Location of structure training *.parquets files. '
                             'Receptors should be in a directory named '
                             'receptors, with ligands located in their '
                             'specific receptor subdirectory under the '
                             'ligands directory.')
    parser.add_argument('save_path', type=str,
                        help='Directory in which experiment outputs are '
                             'stored.')
    parser.add_argument('--load', '-l', type=str, required=False,
                        help='Load a session and model.')
    parser.add_argument('--test_data_root', '-t', type=str, required=False,
                        help='Location of structure test *.parquets files. '
                             'Receptors should be in a directory named '
                             'receptors, with ligands located in their '
                             'specific receptor subdirectory under the '
                             'ligands directory.')
    parser.add_argument('--batch_size', '-b', type=int, required=False,
                        default=8, help='Number of examples to include in '
                                        'each batch for training.')
    parser.add_argument('--epochs', '-e', type=int, required=False,
                        default=1, help='Number of times to iterate through '
                                        'training set.')
    parser.add_argument('--train_receptors', '-r', type=str, nargs='*',
                        help='Names of specific receptors for training. If '
                             'specified, other structures will be ignored.')
    parser.add_argument('--test_receptors', '-q', type=str, nargs='*',
                        help='Names of specific receptors for testing. If '
                             'specified, other structures will be ignored.')
    parser.add_argument('--save_interval', '-s', type=int, default=-1,
                        help='Save checkpoints after every <save_interval> '
                             'batches.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01,
                        help='Learning rate for gradient descent')
    args = parser.parse_args()

    models_dict = {
        'resnet': lieConv.LieResNet,
        'gnina': models.GninaNet
    }
    network = models_dict[args.model](
        22, ds_frac=1., num_outputs=2, k=300,
        nbhd=20, act='swish', bn=True,
        num_layers=6, mean=True, pool=True, liftsamples=1, fill=1.0,
        group=SE3(), knn=False, cache=False
    )

    sess = Session(network, Path(args.train_data_root).expanduser(),
                   Path(args.save_path).expanduser(), args.batch_size,
                   test_root=args.test_data_root,
                   train_receptors=args.train_receptors,
                   test_receptors=args.test_receptors,
                   save_interval=args.save_interval,
                   learning_rate=args.learning_rate)
    print('Built network with {} params'.format(sess.param_count))
    if args.load is not None:
        sess.load(args.load)
    if args.epochs > 0:
        sess.train(args.epochs)
    if args.test_data_root is not None:
        sess.test()
