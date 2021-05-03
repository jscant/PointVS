"""
The dataloader for SE(3)Transformer is heavily edited from
a script developed for similar reasons by Constantin Schneider
github.com/con-schneider

The dataloader for LieConv is my own work.
"""
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


def get_data_loader(*data_roots, receptors=None, batch_size=32,
                    radius=6, rot=True, feature_dim=12, mode='train'):
    """Give a DataLoader from a list of receptors and data roots."""
    ds_kwargs = {
        'receptors': receptors,
        'radius': radius,
        'rot': rot
    }
    ds = multiple_source_dataset(
        LieConvDataset, *data_roots, balanced=True, **ds_kwargs)
    collate = get_collate_fn(feature_dim)
    sampler = ds.sampler if mode == 'train' else None
    return DataLoader(
        ds, batch_size, False, sampler=sampler, num_workers=0,
        collate_fn=collate, drop_last=False)


def multiple_source_dataset(loader_class, *base_paths, receptors=None,
                            balanced=True, **kwargs):
    """Concatenate mulitple datasets into one, preserving balanced sampling.

    Arguments:
        loader_class: one of either LieConvLoader or SE3TransformerLoader,
            inheriting from Datset. This class should have in its constructor a
            list of binary labels associated with each item, stored in
            <class>.labels.
        base_paths: locations of parquets files, one for each dataset.
        receptors: receptors to include. If None, all receptors found are used.
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
            dataset = loader_class(base_path, receptors=receptors, **kwargs)
            labels += list(dataset.labels)
            filenames += list(dataset.filenames)
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
    return multi_source_dataset


def one_hot(numerical_category, num_classes):
    """Make one-hot vector from category and total categories."""
    one_hot_array = np.zeros((len(numerical_category), num_classes))

    for i, cat in enumerate(numerical_category):
        one_hot_array[i, int(cat)] = 1

    return one_hot_array


class LieConvDataset(torch.utils.data.Dataset):
    """Class for feeding structure parquets into network."""

    def __init__(self, base_path, radius=12, receptors=None, data_only=False,
                 rot=False, **kwargs):
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
        self.data_only = data_only

        # apply random rotations to coordinates?
        self.transformation = random_rotation if rot else lambda x: x

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
        struct = make_box(centre_on_ligand(
            concat_structs(rec_fname, lig_fname)),
            radius=self.radius, relative_to_ligand=False)

        p = torch.from_numpy(
            np.expand_dims(self.transformation(
                struct[struct.columns[:3]].to_numpy()), 0))

        v = torch.unsqueeze(make_bit_vector(struct.types.to_numpy(), 11), 0)
        m = torch.from_numpy(np.ones((1, len(struct))))

        return (p, v, m, len(struct)), lig_fname, rec_fname, label


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
        for batch_index, ((p, v, m, _), ligand, receptor, label) in enumerate(
                batch):
            p_batch[batch_index, :p.shape[1], :] = p
            v_batch[batch_index, :v.shape[1], :] = v
            m_batch[batch_index, :m.shape[1]] = m
            label_batch[batch_index] = label
            ligands.append(ligand)
            receptors.append(receptor)
        return (p_batch, v_batch,
                m_batch.bool()), label_batch.float(), ligands, receptors

    return collate
