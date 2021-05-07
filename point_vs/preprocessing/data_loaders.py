"""
The dataloader for SE(3)Transformer is heavily edited from
a script developed for similar reasons by Constantin Schneider
github.com/con-schneider

The dataloader for LieConv is my own work.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from point_vs.preprocessing.preprocessing import centre_on_ligand, make_box, \
    concat_structs, make_bit_vector


def random_rotation(x):
    """Apply a random rotation to a set of 3D coordinates."""
    M = np.random.randn(3, 3)
    Q, _ = np.linalg.qr(M)
    return x @ Q


def get_data_loader(*data_roots, receptors=None, batch_size=32, compact=True,
                    use_atomic_numbers=False, radius=6, rot=True,
                    polar_hydrogens=True, feature_dim=12, mode='train'):
    """Give a DataLoader from a list of receptors and data roots."""
    ds_kwargs = {
        'receptors': receptors,
        'radius': radius,
        'rot': rot
    }
    ds = multiple_source_dataset(
        PointCloudDataset, *data_roots, balanced=True, compact=compact,
        polar_hydrogens=polar_hydrogens,
        use_atomic_numbers=use_atomic_numbers, **ds_kwargs)
    collate = get_collate_fn(ds.feature_dim)
    sampler = ds.sampler if mode == 'train' else None
    return DataLoader(
        ds, batch_size, False, sampler=sampler,  # num_workers=mp.cpu_count(),
        collate_fn=collate, drop_last=False, pin_memory=True)


def multiple_source_dataset(
        loader_class, *base_paths, receptors=None, polar_hydrogens=True,
        compact=True, use_atomic_numbers=False, balanced=True, **kwargs):
    """Concatenate mulitple datasets into one, preserving balanced sampling.

    Arguments:
        loader_class: one of either PointCloudDataset or SE3TransformerLoader,
            inheriting from Datset. This class should have in its constructor a
            list of binary labels associated with each item, stored in
            <class>.labels.
        base_paths: locations of parquets files, one for each dataset.
        receptors: receptors to include. If None, all receptors found are used.
        polar_hydrogens:
        compact:
        use_atomic_numbers:
        balanced: whether to sample probabailistically based on class imbalance.
        kwargs: other keyword arguments for loader_class.

    Returns:
        Concatenated dataset including balanced sampler.
    """
    datasets = []
    labels = []
    filenames = []
    base_paths = sorted(
        [Path(bp).expanduser() for bp in base_paths if bp is not None])
    for base_path in base_paths:
        if base_path is not None:
            dataset = loader_class(
                base_path, compact=compact, receptors=receptors,
                polar_hydrogens=polar_hydrogens,
                use_atomic_numbers=use_atomic_numbers, **kwargs)
            labels += list(dataset.labels)
            filenames += dataset.filenames
            datasets.append(dataset)
    labels = np.array(labels)
    class_sample_count = np.array(
        [len(labels) - np.sum(labels), np.sum(labels)])
    if np.sum(labels) == len(labels) or np.sum(labels) == 0 or not balanced:
        sampler = None
    else:
        weights = 1. / class_sample_count
        sample_weights = torch.from_numpy(
            np.array([weights[i] for i in labels]))
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    multi_source_dataset = torch.utils.data.ConcatDataset(datasets)
    multi_source_dataset.sampler = sampler
    multi_source_dataset.class_sample_count = class_sample_count
    multi_source_dataset.labels = labels
    multi_source_dataset.filenames = filenames
    multi_source_dataset.base_path = ', '.join([str(bp) for bp in base_paths])
    multi_source_dataset.feature_dim = datasets[0].feature_dim
    return multi_source_dataset


class PointCloudDataset(torch.utils.data.Dataset):
    """Class for feeding structure parquets into network."""

    def __init__(self, base_path, radius=12, receptors=None,
                 polar_hydrogens=True, use_atomic_numbers=False,
                 compact=True, rot=False, **kwargs):
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
            polar_hydrogens:
            use_atomic_numbers:
            compact_one_hot:
            kwargs: keyword arguments passed to the parent class (Dataset).
        """

        super().__init__(**kwargs)
        self.radius = radius
        self.base_path = Path(base_path).expanduser()
        self.polar_hydrogens = polar_hydrogens
        self.use_atomic_numbers = use_atomic_numbers
        self.compact = compact

        if receptors is None:
            print('Loading all structures in', self.base_path)
            filenames = list((self.base_path / 'ligands').glob('**/*.parquet'))
        else:
            print('Loading receptors:')
            filenames = []
            for receptor in receptors:
                print(receptor)
                filenames += list((self.base_path / 'ligands').glob(
                    '{}*/*.parquet'.format(receptor)))

        self.filenames = sorted(filenames)
        labels = []
        for fname in self.filenames:
            if str(fname.parent.name).find('active') == -1:
                labels.append(0)
            else:
                labels.append(1)
        labels = np.array(labels)
        class_sample_count = np.array(
            [len(labels) - np.sum(labels), np.sum(labels)])
        if np.sum(labels) == len(labels) or np.sum(labels) == 0:
            self.sampler = None
        else:
            weights = 1. / class_sample_count
            self.sample_weights = torch.from_numpy(
                np.array([weights[i] for i in labels]))
            self.sampler = torch.utils.data.WeightedRandomSampler(
                self.sample_weights, len(self.sample_weights)
            )
        self.labels = labels

        # apply random rotations to coordinates?
        self.transformation = random_rotation if rot else lambda x: x

        # H C N O F P S Cl
        recognised_atomic_numbers = (6, 7, 8, 9, 15, 16, 17)
        # various metal ions/halogens which share valence properties
        other_groupings = ((35, 53), (3, 11, 19), (4, 12, 20), (26, 29, 30))
        atomic_number_to_index = {
            num: idx for idx, num in enumerate(recognised_atomic_numbers)
        }
        for grouping in other_groupings:
            atomic_number_to_index.update({elem: max(
                atomic_number_to_index.values()) + 1 for elem in grouping})
        if self.polar_hydrogens:
            atomic_number_to_index.update({
                1: max(atomic_number_to_index.values()) + 1
            })

        # +1 to accommodate for unmapped elements
        self.max_elem_id = max(atomic_number_to_index.values()) + 1

        # Any other elements not accounted for given a category of their own
        self.atomic_number_to_index = defaultdict(lambda: self.max_elem_id)
        self.atomic_number_to_index.update(atomic_number_to_index)

        if compact:
            self.feature_dim = self.max_elem_id + 2
        else:
            self.feature_dim = (self.max_elem_id + 1) * 2

    def __len__(self):
        """Return the total size of the dataset."""
        return len(self.filenames)

    def __getitem__(self, item):
        """Given an index, locate and preprocess relevant parquet file.

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
        try:
            rec_fname = next((self.base_path / 'receptors').glob(
                '{}*.parquet'.format(rec_name)))
        except StopIteration:
            raise RuntimeError(
                'Receptor for ligand {0} not found. Looking for file '
                'named {1}'.format(lig_fname, rec_name + '.parquet'))
        struct = make_box(centre_on_ligand(
            concat_structs(rec_fname, lig_fname)),
            radius=self.radius, relative_to_ligand=False)

        if not self.polar_hydrogens:
            struct = struct[struct['atomic_number'] > 1]

        if self.use_atomic_numbers:
            struct.types = struct['atomic_number'].map(
                self.atomic_number_to_index) + struct.bp * (
                                   self.max_elem_id + 1)
        p = torch.from_numpy(
            np.expand_dims(self.transformation(
                struct[struct.columns[:3]].to_numpy()), 0))

        v = torch.unsqueeze(make_bit_vector(
            struct.types.to_numpy(), self.max_elem_id + 1, self.compact), 0)

        return (p, v, len(struct)), lig_fname, rec_fname, label


def get_collate_fn(dim):
    """Processing of inputs which takes place after batch is selected.

    LieConv networks take tuples of torch tensors (p, v, m), which are:
        p, (batch_size, n_atoms, 3): coordinates of each atom
        v, (batch_size, n_atoms, n_features): features for each atom
        m, (batch_size, n_atoms): mask for each coordinate slot

    Note that n_atoms is the largest number of atoms in a structure in
    each batch.

    Arguments:
        dim: size of input (node) features

    Returns:
        Pytorch collate function
    """

    def collate(batch):
        max_len = max([b[0][-1] for b in batch])
        batch_size = len(batch)
        p_batch = torch.zeros(batch_size, max_len, 3)
        v_batch = torch.zeros(batch_size, max_len, dim)
        m_batch = torch.zeros(batch_size, max_len)
        label_batch = torch.zeros(batch_size, )
        ligands, receptors = [], []
        for batch_index, ((p, v, size), ligand, receptor, label) in enumerate(
                batch):
            p_batch[batch_index, :size, :] = p
            v_batch[batch_index, :size, :] = v
            m_batch[batch_index, :size] = 1
            label_batch[batch_index] = label
            ligands.append(ligand)
            receptors.append(receptor)
        return (p_batch, v_batch,
                m_batch.bool()), label_batch.float(), ligands, receptors

    return collate
