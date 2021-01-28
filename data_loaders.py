"""
The dataloader for SE(3)Transformer is heavily edited from
a script developed for similar reasons by Constantin Schneider
github.com/con-schneider

The dataloader for LieConv is my own work.
"""

from pathlib import Path

import dgl
import numpy as np
import torch
import torch as th
from scipy.spatial.distance import cdist

from preprocessing import centre_on_ligand, make_box, concat_structs, \
    make_bit_vector


class LieConvLoader(torch.utils.data.Dataset):
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
            concat_structs(rec_fname, lig_fname)), radius=10))

        p = torch.from_numpy(
            np.expand_dims(struct[struct.columns[:3]].to_numpy(),
                           0)).float()
        v = torch.unsqueeze(make_bit_vector(struct.types.to_numpy(), 11), 0)
        # v = F.one_hot(torch.from_numpy(
        #    np.expand_dims(struct.types.to_numpy(), 0)), num_classes=22).float()
        m = torch.from_numpy(np.ones((1, len(struct)))).float()
        return (p, v, m, len(struct)), lig_fname, rec_fname, label

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
        v_batch = torch.zeros(batch_size, max_len, 12)
        m_batch = torch.zeros(batch_size, max_len)
        label_batch = torch.zeros(batch_size, 1)
        ligands, receptors = [], []
        for batch_index, ((p, v, m, _), ligand, receptor, label) in enumerate(
                batch):
            p_batch[batch_index, :p.shape[1], :] = p
            v_batch[batch_index, :v.shape[1], :] = v
            m_batch[batch_index, :m.shape[1]] = m
            label_batch[batch_index] = label
            ligands.append(ligand)
            receptors.append(receptor)
        return (p_batch.float(), v_batch.float(),
                m_batch.bool()), label_batch.long(), ligands, receptors


def one_hot(numerical_category, num_classes):
    one_hot_array = np.zeros((len(numerical_category), num_classes))

    for i, cat in enumerate(numerical_category):
        one_hot_array[i, int(cat)] = 1

    return one_hot_array


class SE3TransformerLoader(torch.utils.data.Dataset):

    def __init__(
            self,
            base_path,
            radius: float = 12.0,
            interaction_dist: float = 4,
            receptors=None,
            mode: str = 'all',
            transform=None):

        self.base_path = Path(base_path).expanduser()
        self.filenames, self.labels, self.sampler = self.get_files_and_labels(
            self.base_path, receptors)
        self.radius = radius
        self.atom_feature_size = 11
        self.edge_dim = 2
        self.interaction_dist = interaction_dist
        self.transform = transform

        if mode == 'all':
            self.interface_neighbourhoods_only = False
            self.only_interacting_atoms = False
            self.only_interacting_edges = False

        elif mode == 'interaction_neighbourhoods':
            self.interface_neighbourhoods_only = True
            self.only_interacting_atoms = False
            self.only_interacting_edges = False

        elif mode == 'interaction_atoms':
            self.interface_neighbourhoods_only = True
            self.only_interacting_atoms = True
            self.only_interacting_edges = False

        elif mode == 'interaction_edges':
            self.interface_neighbourhoods_only = True
            self.only_interacting_atoms = True
            self.only_interacting_edges = True
            self.edge_dim = 1
        else:
            raise AttributeError('mode {} not recognised'.format(mode))

    @staticmethod
    def get_files_and_labels(base_path, receptors=None):
        if receptors is None:
            print('Loading all structures in', base_path)
            filenames = list((base_path / 'ligands').rglob('*/*.parquet'))
        else:
            print('Loading receptors:')
            filenames = []
            for receptor in receptors:
                print(receptor)
                filenames += list((base_path / 'ligands').rglob(
                    '{}*/*.parquet'.format(receptor)))

        labels = []
        for fname in filenames:
            if str(fname).find('active') == -1:
                labels.append(0)
            else:
                labels.append(1)
        labels = np.array(labels)
        class_sample_count = np.array(
            [len(labels) - np.sum(labels), np.sum(labels)])
        weights = 1. / class_sample_count
        sample_weights = torch.from_numpy(
            np.array([weights[i] for i in labels])).float()
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights)
        )
        return filenames, labels, sampler

    def populate_graph(self, rec_fname, lig_fname):
        interaction_dist = self.interaction_dist

        df = make_box(centre_on_ligand(
            concat_structs(rec_fname, lig_fname)), radius=self.radius)

        lig_coords = df[df.bp == 0].loc[:, ['x', 'y', 'z']].to_numpy()
        rec_coords = df[df.bp == 1].loc[:, ['x', 'y', 'z']].to_numpy()

        lig_types = df[df.bp == 0]['types'].to_numpy()
        rec_types = df[df.bp == 1]['types'].to_numpy()

        lig_dist_internal = cdist(lig_coords, lig_coords)
        rec_dist_internal = cdist(rec_coords, rec_coords)

        # remove solitary atoms
        lig_connected = (lig_dist_internal < interaction_dist).sum(
            axis=1) > 1  # diagonal always true
        rec_connected = (rec_dist_internal < interaction_dist).sum(axis=1) > 1

        lig_coords = lig_coords[lig_connected, :]
        rec_coords = rec_coords[rec_connected, :]

        lig_types = lig_types[lig_connected]
        rec_types = rec_types[rec_connected]

        # also remove from the dist_map, to avoid numbering issues
        lig_dist_internal = lig_dist_internal[
                            lig_connected, :][:, lig_connected]
        rec_dist_internal = rec_dist_internal[
                            rec_connected, :][:, rec_connected]

        dist_between = cdist(lig_coords, rec_coords)

        if not self.interface_neighbourhoods_only:
            np.fill_diagonal(lig_dist_internal, 100)  # higher than the cutoff
            lig_src, lig_dst = np.where(lig_dist_internal < interaction_dist)

            rec_offset = len(lig_coords)
            np.fill_diagonal(rec_dist_internal, 100)
            rec_src, rec_dst = np.where(rec_dist_internal < interaction_dist)
            rec_src += rec_offset
            rec_dst += rec_offset

            src = np.concatenate([lig_src, rec_src])
            dst = np.concatenate([lig_dst, rec_dst])

            edge_types = np.zeros(len(src))

            lig_src, rec_dst = np.where(dist_between < interaction_dist)
            rec_dst += rec_offset
            rec_src = rec_dst
            lig_dst = lig_src

            src = np.concatenate([src, lig_src, rec_src])
            dst = np.concatenate([dst, rec_dst, lig_dst])

            # edge_types = np.ones(len(src))
            edge_types = np.concatenate(
                [edge_types, np.ones(len(lig_src) + len(rec_src))])
            edge_types = one_hot(edge_types, self.edge_dim)

            src = np.array(src).astype(np.uint32)
            dst = np.array(dst).astype(np.uint32)

            lig_features = make_bit_vector(lig_types,
                                           self.atom_feature_size)
            rec_features = make_bit_vector(rec_types,
                                           self.atom_feature_size)

            x = np.concatenate([lig_coords, rec_coords], axis=0)

            features = np.concatenate([lig_features, rec_features], axis=0)[
                ..., None]
        else:
            # get interface nodes
            lig_inter_src, rec_inter_dst = np.where(
                dist_between < interaction_dist)

            lig_inter_nodes = sorted(np.unique(lig_inter_src))
            rec_inter_nodes = sorted(np.unique(rec_inter_dst))

            # get intra nodes and edges
            lig_intra_src, lig_intra_dst = np.where(
                lig_dist_internal < interaction_dist)
            rec_intra_src, rec_intra_dst = np.where(
                rec_dist_internal < interaction_dist)

            # quick implementation
            # [n in lig_inter_nodes for n in lig_intra_src]
            lig_edge_selection = np.in1d(lig_intra_src, lig_inter_nodes)
            # [n in rec_inter_nodes for n in rec_intra_src]
            rec_edge_selection = np.in1d(rec_intra_src, rec_inter_nodes)

            # the destination node also has to be within the interacting nodes
            if self.only_interacting_atoms:
                lig_edge_selection_dst = np.in1d(lig_intra_dst, lig_inter_nodes)
                lig_edge_selection = np.logical_and(
                    lig_edge_selection, lig_edge_selection_dst)

                rec_edge_selection_dst = np.in1d(rec_intra_dst, rec_inter_nodes)
                rec_edge_selection = np.logical_and(
                    rec_edge_selection, rec_edge_selection_dst)

            lig_intra_src = lig_intra_src[lig_edge_selection]
            lig_intra_dst = lig_intra_dst[lig_edge_selection]

            rec_intra_src = rec_intra_src[rec_edge_selection]
            rec_intra_dst = rec_intra_dst[rec_edge_selection]

            # select coords using intra edges
            lig_nodes = sorted(
                np.unique(np.concatenate([lig_intra_src, lig_intra_dst])))
            lig_coords = lig_coords[lig_nodes, :]
            lig_types = lig_types[lig_nodes]

            rec_nodes = sorted(
                np.unique(np.concatenate([rec_intra_src, rec_intra_dst])))
            rec_coords = rec_coords[rec_nodes, :]
            rec_types = rec_types[rec_nodes]

            lig_node_map = dict(zip(
                lig_nodes, list(range(len(lig_nodes)))
            ))
            rec_node_map = dict(zip(
                rec_nodes,
                [i + len(lig_nodes) for i in list(range(len(rec_nodes)))]
            ))

            # adjust node numbering
            lig_inter_src = [lig_node_map[n] for n in lig_inter_src]
            lig_intra_src = [lig_node_map[n] for n in lig_intra_src]
            lig_intra_dst = [lig_node_map[n] for n in lig_intra_dst]
            rec_inter_dst = [rec_node_map[n] for n in rec_inter_dst]
            rec_intra_src = [rec_node_map[n] for n in rec_intra_src]
            rec_intra_dst = [rec_node_map[n] for n in rec_intra_dst]

            if self.only_interacting_edges:
                src = np.concatenate([
                    lig_inter_src,
                    rec_inter_dst
                ])
                dst = np.concatenate([
                    rec_inter_dst,
                    lig_inter_src
                ])
                edge_types = np.ones((len(src), 1))
            else:
                # full src and dst list (each edge doubled)
                src = np.concatenate([
                    lig_inter_src,
                    lig_intra_src,
                    rec_intra_src,
                    rec_inter_dst,
                    lig_intra_dst,
                    rec_intra_dst
                ])
                dst = np.concatenate([
                    rec_inter_dst,
                    lig_intra_dst,
                    rec_intra_dst,
                    lig_inter_src,
                    lig_intra_src,
                    rec_intra_src,
                ])

                edge_types = np.concatenate([
                    np.ones(len(lig_inter_src)),
                    np.zeros(len(lig_intra_src)),
                    np.zeros(len(rec_intra_src)),
                    np.ones(len(lig_inter_src)),
                    np.zeros(len(lig_intra_src)),
                    np.zeros(len(rec_intra_src)),
                ])
                edge_types = one_hot(edge_types, self.edge_dim)

            src = np.array(src).astype(np.uint32)
            dst = np.array(dst).astype(np.uint32)

            lig_features = make_bit_vector(lig_types, 11)
            rec_features = make_bit_vector(rec_types, 11)

            lig_features = lig_features
            rec_features = rec_features

            x = np.concatenate([lig_coords, rec_coords], axis=0)

            features = np.concatenate([lig_features, rec_features], axis=0)[
                ..., None]

        if self.transform:
            x = self.transform(x).astype(np.float32)
        try:
            G = dgl.graph((src.astype(np.int32), dst.astype(np.int32)))
            G.ndata['x'] = torch.from_numpy(x.astype(np.float32))
            G.ndata['f'] = torch.from_numpy(features.astype(np.float32))
            G.edata['d'] = torch.from_numpy(
                (x[dst] - x[src]).astype(np.float32))
            G.edata['w'] = torch.from_numpy(
                np.array(edge_types).astype(np.float32))
        except Exception:
            raise AssertionError('Graph creation for failed')

        return G

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        lig_fname = self.filenames[item]
        label = self.labels[item]
        rec_name = lig_fname.parent.name.split('_')[0]
        rec_fname = next((self.base_path / 'receptors').glob(
            '{}*.parquet'.format(rec_name)))
        graph = self.populate_graph(rec_fname, lig_fname)
        return graph, label, lig_fname, rec_fname

    @staticmethod
    def collate(samples):
        ligands = [s[2] for s in samples]
        receptors = [s[3] for s in samples]
        samples = [(s[0], s[1]) for s in samples]
        graphs, y = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return (batched_graph,), th.tensor(y), ligands, receptors
