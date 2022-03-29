"""
DataLoaders to take parquet directories and create feature vectors suitable
for use by models found in this project.
"""
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as GeoDataLoader, Data

from point_vs.preprocessing.preprocessing import make_box, \
    concat_structs, make_bit_vector, uniform_random_rotation, generate_edges
from point_vs.utils import load_yaml


class PointCloudDataset(torch.utils.data.Dataset):
    """Class for feeding structure parquets into network."""

    def __init__(
            self, base_path, radius=12,
            polar_hydrogens=True, use_atomic_numbers=False,
            compact=True, rot=False, augmented_active_count=0,
            augmented_active_min_angle=90, max_active_rms_distance=None,
            min_inactive_rms_distance=None, max_inactive_rms_distance=None,
            fname_suffix='parquet',
            types_fname=None, edge_radius=None, estimate_bonds=False,
            prune=False, bp=None, p_remove_entity=0, extended_atom_types=False,
            **kwargs):
        """Initialise dataset.

        Arguments:
            base_path: path containing the 'receptors' and 'ligands'
                directories, which in turn contain <rec_name>.parquets files
                and folders called <rec_name>_[active|decoy] which in turn
                contain <ligand_name>.parquets files. All parquets files from
                this directory are recursively loaded into the dataset.
            radius: size of the bounding box; all atoms further than <radius>
                Angstroms from the mean ligand atom position are discarded.
            polar_hydrogens: include polar hydrogens as input
            use_atomic_numbers: use atomic numbers rather than sminatypes
            compact: compress 1hot vectors by using a single bit to
                signify whether atoms are from the receptor or ligand rather
                than using two input bits per atom type
            rot: random rotation of inputs
            augmented_active_count: number of actives to be rotated randomly
                and used as decoys (per active in the training set)
            augmented_active_min_angle: minimum angle of rotation for each
                augmented active (as specified in augmented_active_count)
            max_active_rms_distance: (pose selection) maximum rmsd between
                relaxed crystal structure and crystal structure
            min_inactive_rms_distance: (pose selection) minimum rmsd between
                redocked and relaxed crystal structure
            include_relaxed: (pose selection) include the relaxed crystal
                structure as an active
            types_fname:
            edge_radius:
            estimate_bonds:
            kwargs: keyword arguments passed to the parent class (Dataset).
        """

        assert not ((max_active_rms_distance is None) != (
                min_inactive_rms_distance is None))
        super().__init__()
        self.radius = radius
        self.estimate_bonds = estimate_bonds
        self.base_path = Path(base_path).expanduser()
        self.prune = prune
        self.bp = bp
        self.edge_radius = edge_radius
        self.p_remove_entity = p_remove_entity

        self.fname_suffix = fname_suffix
        if not self.base_path.exists():
            raise FileNotFoundError(
                'Dataset {} does not exist.'.format(self.base_path))
        self.polar_hydrogens = polar_hydrogens
        self.use_atomic_numbers = use_atomic_numbers
        self.compact = compact

        labels = []
        self.use_types = False if types_fname is None else True
        if max_active_rms_distance is not None \
                or min_inactive_rms_distance is not None \
                or max_inactive_rms_distance is not None:
            label_by_rmsd = True
            if max_active_rms_distance is None:
                max_active_rms_distance = np.inf
            if max_inactive_rms_distance is None:
                max_inactive_rms_distance = np.inf
            if min_inactive_rms_distance is None:
                min_inactive_rms_distance = 0
        else:
            label_by_rmsd = False

        aug_recs, aug_ligs = [], []
        confirmed_ligs = []
        confirmed_recs = []
        if self.use_types:
            _labels, rmsds, receptor_fnames, ligand_fnames = \
                types_to_list(types_fname)

            # Do we use provided labels or do we generate our own using rmsds?
            labels = [] if label_by_rmsd else _labels
            for path_idx, (receptor_fname, ligand_fname) in enumerate(
                    zip(receptor_fnames, ligand_fnames)):
                if label_by_rmsd:
                    # Pose selection, filter by max/min active/inactive rmsd
                    # from xtal poses
                    rmsd = rmsds[path_idx]
                    if rmsd < 0:
                        continue
                    elif rmsd < max_active_rms_distance:
                        labels.append(1)
                        aug_ligs += [ligand_fname] * augmented_active_count
                        aug_recs += [receptor_fname] * augmented_active_count
                    elif rmsd >= max_inactive_rms_distance:
                        continue
                    elif rmsd >= min_inactive_rms_distance:
                        labels.append(0)
                    else:  # discard this entry (do not add to confirmed_ligs)
                        continue
                elif labels[path_idx]:
                    aug_ligs += [ligand_fname] * augmented_active_count
                    aug_recs += [receptor_fname] * augmented_active_count
                confirmed_ligs.append(ligand_fname)
                confirmed_recs.append(receptor_fname)
            self.receptor_fnames = confirmed_recs + aug_recs
        else:
            print('Loading all structures in', self.base_path)
            ligand_fnames = list(
                (self.base_path / 'ligands').glob('**/*.' + fname_suffix))
            if label_by_rmsd:
                rmsd_info_fname = Path(self.base_path, 'rmsd_info.yaml')
                rmsd_info = load_yaml(rmsd_info_fname)

            for path_idx, ligand_fname in enumerate(ligand_fnames):
                if label_by_rmsd:
                    if str(ligand_fname.parent.name).find('active') != -1:
                        continue
                    pdbid = ligand_fname.parent.name.split('_')[0]
                    idx = int(Path(ligand_fname.name).stem.split('_')[-1])
                    try:
                        rmsd = rmsd_info[pdbid]['docked_wrt_crystal'][idx]
                    except KeyError:
                        continue
                    if rmsd < 0:
                        continue
                    if rmsd < max_active_rms_distance:
                        labels.append(1)
                        aug_ligs += [ligand_fname] * augmented_active_count
                    elif rmsd >= min_inactive_rms_distance:
                        labels.append(0)
                    else:  # discard this entry (do not add to confirmed_ligs)
                        continue
                else:
                    if str(ligand_fname.parent.name).find('active') == -1:
                        labels.append(0)
                    else:
                        labels.append(1)
                        aug_ligs += [ligand_fname] * augmented_active_count
                confirmed_ligs.append(ligand_fname)
                self.receptor_fnames = None

        self.pre_aug_ds_len = len(ligand_fnames)
        self.ligand_fnames = confirmed_ligs + aug_ligs

        labels += [0] * len(aug_ligs)
        labels = np.array(labels)
        active_count = np.sum(labels)
        class_sample_count = np.array(
            [len(labels) - active_count, active_count])
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
        print('There are', len(labels), 'training points in', base_path)

        # apply random rotations to ALL coordinates?
        self.transformation = uniform_random_rotation if rot else lambda x: x

        if use_atomic_numbers:
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
            self.max_feature_id = max(atomic_number_to_index.values()) + 1

            # Any other elements not accounted for given a category of their own
            self.atomic_number_to_index = defaultdict(
                lambda: self.max_feature_id)
            self.atomic_number_to_index.update(atomic_number_to_index)

        elif polar_hydrogens:
            self.max_feature_id = 11 + 8 * extended_atom_types
        else:
            self.max_feature_id = 10 + 8 * extended_atom_types  # No polar Hs

        if compact:
            self.feature_dim = self.max_feature_id + 2
        else:
            self.feature_dim = (self.max_feature_id + 1) * 2

        self.augmented_active_min_angle = augmented_active_min_angle

    def __len__(self):
        """Return the total size of the dataset."""
        return len(self.ligand_fnames)

    def index_to_parquets(self, item):
        label = self.labels[item]
        if self.use_types:
            lig_fname = Path(self.ligand_fnames[item])
            rec_fname = Path(self.receptor_fnames[item])
        else:
            lig_fname = self.ligand_fnames[item]
            rec_name = lig_fname.parent.name.split('_')[0]
            try:
                rec_fname = next((self.base_path / 'receptors').glob(
                    '{0}*.{1}'.format(rec_name, self.fname_suffix)))
            except StopIteration:
                raise RuntimeError(
                    'Receptor for ligand {0} not found. Looking for file '
                    'named {1}'.format(
                        lig_fname, rec_name + '.' + self.fname_suffix))
        return lig_fname, rec_fname, label

    def parquets_to_inputs(self, lig_fname, rec_fname, item=None):

        # Are we using an active and labelling it as a decoy through random
        # rotation? This determination is made using the index, made possible
        # due to the construction of the filenames class variable in the
        # constructor: all actives labelled as decoys are found at the end.
        if item is None or item < self.pre_aug_ds_len:
            aug_angle = 0
        else:
            aug_angle = self.augmented_active_min_angle

        if self.use_types:
            rec_fname = self.base_path / rec_fname
            lig_fname = self.base_path / lig_fname
        if not lig_fname.is_file() * rec_fname.is_file():
            print(lig_fname, rec_fname, lig_fname.is_file(), rec_fname.is_file())
            raise

        struct = make_box(concat_structs(
            rec_fname, lig_fname, min_lig_rotation=aug_angle),
            radius=self.radius, relative_to_ligand=True)

        if not self.polar_hydrogens:
            struct = struct[struct['atomic_number'] > 1]

        if self.use_atomic_numbers:
            struct.types = struct['atomic_number'].map(
                self.atomic_number_to_index) + struct.bp * (
                                   self.max_feature_id + 1)
        force_zero_label = False
        if self.p_remove_entity > 0 and random.random() < self.p_remove_entity:
            force_zero_label = True
            if random.random() < 0.5:
                struct = struct[struct['bp'] == 0]
            else:
                struct = struct[struct['bp'] == 1]

        p = torch.from_numpy(
            self.transformation(
                np.vstack([struct['x'], struct['y'], struct['z']]).T))

        v = make_bit_vector(
            struct.types.to_numpy(), self.max_feature_id + 1, self.compact)

        return p, v, struct, force_zero_label

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
        lig_fname, rec_fname, label = self.index_to_parquets(item)
        p, v, struct, force_zero_label = self.parquets_to_inputs(
            lig_fname, rec_fname, item=item)
        if force_zero_label:
            label = 0

        return (p, v, len(struct)), lig_fname, rec_fname, label


class PygPointCloudDataset(PointCloudDataset):
    """Class for feeding structure parquets into network."""

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
        lig_fname, rec_fname, label = self.index_to_parquets(item)

        p, v, struct, force_zero_label = self.parquets_to_inputs(
            lig_fname, rec_fname, item=item)
        if force_zero_label:
            label = 0
        edge_radius = self.edge_radius if self.edge_radius > 0 else 4
        intra_radius = 2.0 if self.estimate_bonds else edge_radius

        if self.bp is not None:
            struct = struct[struct.bp == self.bp]

        if self.edge_radius >= 0:
            struct, edge_indices, edge_attrs = generate_edges(
                struct, inter_radius=edge_radius, intra_radius=intra_radius,
                prune=self.prune)
            edge_indices = torch.from_numpy(np.vstack(edge_indices)).long()
            edge_attrs = one_hot(torch.from_numpy(edge_attrs).long(), 3)

        else:

            edge_indices, edge_attrs = torch.ones(1), torch.ones(1)

        return Data(
            x=v,
            edge_index=edge_indices,
            edge_attr=edge_attrs,
            pos=p,
            y=torch.from_numpy(np.array(label)).long(),
            rec_fname=rec_fname,
            lig_fname=lig_fname,
        )


def get_data_loader(
        data_root, dataset_class, receptors=None, batch_size=32, compact=True,
        use_atomic_numbers=False, radius=6, rot=True,
        augmented_actives=0, min_aug_angle=30,
        polar_hydrogens=True, mode='train',
        max_active_rms_distance=None, fname_suffix='parquet',
        min_inactive_rms_distance=None, types_fname=None, edge_radius=None,
        prune=False, estimate_bonds=False, bp=None, **kwargs):
    """Give a DataLoader from a list of receptors and data roots."""
    ds = dataset_class(
        data_root, compact=compact, receptors=receptors,
        augmented_active_count=augmented_actives,
        augmented_active_min_angle=min_aug_angle,
        polar_hydrogens=polar_hydrogens,
        max_active_rms_distance=max_active_rms_distance,
        min_inactive_rms_distance=min_inactive_rms_distance,
        use_atomic_numbers=use_atomic_numbers,
        fname_suffix=fname_suffix,
        types_fname=types_fname,
        edge_radius=edge_radius,
        estimate_bonds=estimate_bonds,
        prune=prune, bp=bp, radius=radius, rot=rot,
        **kwargs)
    sampler = ds.sampler if mode == 'train' else None
    if dataset_class == PointCloudDataset:
        collate = get_collate_fn(ds.feature_dim)
        return DataLoader(
            ds, batch_size, False, sampler=sampler,
            collate_fn=collate, drop_last=False, pin_memory=True,
            num_workers=min(4, psutil.cpu_count()))
    else:
        return GeoDataLoader(
            ds, batch_size, False, sampler=sampler,
            drop_last=False, pin_memory=True,
            num_workers=min(4, psutil.cpu_count()))


def types_to_list(types_fname):
    """Take a types file and returns four lists containing paths and labels.

    Types files should be of the format:
        <label> <...> <rmsd> <receptor_filename> <ligand_filename> <...>
    where:
        <label> is a label \in \{0, 1\}
        <...> is 0 or more other fields
        <rmsd> is the root mean squared distance of the pose from the crystal
            pose
        <receptor_fname> is the location of the receptor file
        <ligand_fname> is the location of the ligand file

    Arguments:
        types_fname: location of types file

    Returns:
        Four lists containing the labels, rmsds, receptor paths and ligand
        paths.
    """

    def find_paths(types_line):
        recpath, ligpath = None, None
        chunks = types_line.strip().split()
        if not len(chunks):
            return None, None, None, None
        label = int(chunks[0])
        for idx, chunk in enumerate(chunks):
            if chunk.startswith('#'):
                continue
            try:
                float(chunk)
            except ValueError:
                if recpath is None:
                    recpath = chunk
                    rmsd = float(chunks[idx - 1])
                else:
                    ligpath = chunk
                    break
        return label, rmsd, recpath, ligpath

    labels, rmsds, recs, ligs = [], [], [], []
    with open(types_fname, 'r') as f:
        for line in f.readlines():
            label, rmsd, rec, lig = find_paths(line)
            if rec is not None and lig is not None:
                labels.append(label)
                rmsds.append(rmsd)
                recs.append(rec)
                ligs.append(lig)
    return labels, rmsds, recs, ligs


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
