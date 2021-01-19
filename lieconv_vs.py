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
            print('Loading all receptors from', self.base_path)
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
        return (p, v, m, len(struct)), label


class Session:
    """Handles training of LieConv-based models."""

    def __init__(self, network, data_root, save_path, batch_size,
                 receptors=None):
        """Initialise session.

        Arguments:
            network: pytorch object inheriting from torch.nn.Module.
            data_root: path containing the 'receptors' and 'ligands'
                directories, which in turn contain <rec_name>.parquets files
                and folders called <rec_name>_[active|decoy] which in turn
                contain <ligand_name>.parquets files. All parquets files from
                this directory are recursively loaded into the dataset.
            save_path: directory in which experiment outputs are stored.
            batch_size: number of training examples in each batch
            receptors: iterable of strings denoting receptors to include in
                the dataset. if None, all receptors found in base_path are
                included.
        """
        self.data_root = Path(data_root).expanduser()
        self.save_path = Path(save_path).expanduser()
        self.network = network
        self.batch_size = batch_size
        self.losses = []
        self.batch = 0
        if isinstance(receptors, str):
            receptors = tuple([receptors])

        self.dataset = MolLoader(self.data_root, receptors=receptors)
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False, num_workers=0,
            sampler=self.dataset.sampler, collate_fn=self.collate
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCELoss().to(device)
        self.optimiser = torch.optim.Adam(self.network.parameters(), lr=0.01)
        self.device = device

    def _setup_training_session(self):
        """Puts network on GPU and sets up directory structure."""
        self.network.to(self.device)
        self.network.apply(self.weights_init)
        self.save_path.mkdir(parents=True, exist_ok=True)
        try:
            Path(self.save_path, 'loss.log').unlink()
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
        total_iters = epochs * (len(self.dataset) // self.batch_size)
        log_interval = 10
        for epoch in range(epochs):
            decoy_mean_pred, active_mean_pred = -1, -1
            for self.batch, ((p, v, m), y_true) in enumerate(self.data_loader):
                p = p.to(self.device)
                v = v.to(self.device)
                m = m.to(self.device)
                y_true = y_true.to(self.device)
                self.optimiser.zero_grad()
                y_pred = self.network((p, v, m))
                loss = self.criterion(y_pred, y_true)
                loss.backward()
                self.optimiser.step()
                eta = get_eta(start_time,
                              self.batch + len(self.dataset) * epoch,
                              total_iters + 1)
                time_elapsed = format_time(time.time() - start_time)
                y_true_np = y_true.cpu().detach().numpy()
                y_pred_np = y_pred.cpu().detach().numpy()
                active_idx = np.where(y_true_np > 0.5)
                decoy_idx = np.where(y_true_np < 0.5)
                if len(active_idx[0]):
                    active_mean_pred = np.mean(y_pred_np[active_idx])
                if len(decoy_idx[0]):
                    decoy_mean_pred = np.mean(y_pred_np[decoy_idx])

                print_with_overwrite(
                    ('Epoch:', '{0}/{1}'.format(epoch, epochs),
                     'Iteration:', '{0}/{1}'.format(
                        self.batch, len(self.dataset) // self.batch_size)),
                    ('Time elapsed:', time_elapsed,
                     'Time remaining:', eta),
                    ('Loss: {0:.4f}'.format(loss),
                     'Mean active: {0:.4f}'.format(active_mean_pred),
                     'Mean decoy: {0:.4f}'.format(decoy_mean_pred))
                )

                self.losses.append(float(loss))
                if not (self.batch + 1) % log_interval:
                    self.save_loss(log_interval)
        plot_with_smoothing(self.losses, self.save_path / 'loss.png', gap=50)

    def save_loss(self, save_interval):
        """Save the loss information to disk.

        Arguments:
            save_interval: how often the loss is being recorded (in batches).
        """
        log_file = self.save_path / 'loss.log'
        start_idx = save_interval * (self.batch // save_interval)
        with open(log_file, 'a') as f:
            f.write(
                '\n'.join(
                    [str(idx + start_idx + 1) + ' ' + str(loss)
                     for idx, loss in
                     enumerate(self.losses[-save_interval:])])
                + '\n')

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
        label_batch = torch.zeros(batch_size, 1)
        for batch_index, ((p, v, m, _), label) in enumerate(batch):
            p_batch[batch_index, :p.shape[1], :] = p
            v_batch[batch_index, :v.shape[1], :] = v
            m_batch[batch_index, :m.shape[1]] = m
            label_batch[batch_index] = label
        return (p_batch.float(), v_batch.float(),
                m_batch.bool()), label_batch.float()

    @staticmethod
    def weights_init(m):
        """Initialise weights of network using xavier initialisation."""
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='Model architecture; either gnina or resnet.')
    parser.add_argument('data_root', type=str,
                        help='Location of structure *.parquets files. '
                             'Receptors should be in a directory named '
                             'receptors, with ligands located in their '
                             'specific receptor subdirectory under the '
                             'ligands directory.')
    parser.add_argument('save_path', type=str,
                        help='Directory in which experiment outputs are '
                             'stored.')
    parser.add_argument('--batch_size', '-b', type=int, required=False,
                        default=8, help='Number of examples to include in '
                                        'each batch for training.')
    parser.add_argument('--epochs', '-e', type=int, required=False,
                        default=1, help='Number of times to iterate through '
                                        'training set.')
    parser.add_argument('--receptors', '-r', type=str, nargs='*',
                        help='Names of specific receptors for training. If '
                             'specified, other structures will be ignored.')
    args = parser.parse_args()
    models_dict = {
        'resnet': lieConv.LieResNet,
        'gnina': models.GninaNet
    }
    network = models_dict[args.model](
        22, ds_frac=1., num_outputs=1, k=1536, nbhd=20, act='relu', bn=True,
        num_layers=6, mean=True, pool=True, liftsamples=1,
        fill=1.0, group=SE3(), knn=False, cache=False
    )
    sess = Session(network, Path(args.data_root).expanduser(),
                   Path(args.save_path).expanduser(), args.batch_size,
                   args.receptors)
    sess.train(args.epochs)
