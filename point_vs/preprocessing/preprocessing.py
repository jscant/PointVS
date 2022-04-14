"""Some helper functions for cutting inputs down to size."""
import argparse
import sys
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

from point_vs.utils import expand_path

# For graph walking
sys.setrecursionlimit(10000)


def generate_random_z_axis_rotation():
    """Generate random rotation matrix about the z axis (NOT UNIFORM)."""
    R = np.eye(3)
    x1 = np.random.rand()
    R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
    R[0, 1] = -np.sin(2 * np.pi * x1)
    R[1, 0] = np.sin(2 * np.pi * x1)
    return R


def uniform_random_rotation(x):
    """Apply a random rotation in 3D, with a distribution uniform over the
    sphere.

    Algorithm taken from "Fast Random Rotation Matrices" (James Avro, 1992):
    https://doi.org/10.1016/B978-0-08-050755-2.50034-8
    """

    # There are two random variables in [0, 1) here (naming is same as paper)
    x2 = 2 * np.pi * np.random.rand()
    x3 = np.random.rand()

    # Rotation of all points around x axis using matrix
    R = generate_random_z_axis_rotation()
    v = np.array([
        np.cos(x2) * np.sqrt(x3),
        np.sin(x2) * np.sqrt(x3),
        np.sqrt(1 - x3)
    ])
    H = np.eye(3) - (2 * np.outer(v, v))
    M = -(H @ R)
    x = x.reshape((-1, 3))
    mean_coord = np.mean(x, axis=0)
    return ((x - mean_coord) @ M) + mean_coord @ M


def angle_3d(v1, v2):
    """Calculate the angle between two 3d vectors"""
    v1, v2 = v1.reshape((-1, 3)), v2.reshape((-1, 3))
    dot = np.einsum('ij, ij -> i', v1, v2)[0]

    # Angle is undefined if either vector is zero, avoid div0 errors
    denom = max(1e-7, np.linalg.norm(v1) * np.linalg.norm(v2))
    cos = dot / denom
    angle = np.arccos(np.clip(cos, -1.0, 1.0))
    return angle


def generate_edges(struct, inter_radius=4.0, intra_radius=2.0, prune=True,
                   synthpharm=False):
    """Generate edges of graph with specified distance cutoff.

    Arguments:
        struct: DataFrame containing x, y, z, types, bp series.
        inter_radius: maximum distance between two atoms different molecules
            for there to be an edge
        intra_radius: maximum distance between two atoms in the same molecule
            for there to be an edge
        prune: remove receptor atoms that are not connected (have no edges to)
            the main graph containing ligand atoms
    Returns:
        Tuple containing the edge indices and edge attributes. Edge attributes
        are 0 for ligand-ligand edges, 1 for ligand-receptor edges, and 2 for
        receptor-receptor edges.
    """

    def bfs(starting_node, node_list):
        def _bfs(visited, to_visit):
            while len(to_visit):
                s = to_visit.pop()
                visited.append(s)
                for child in node_list[s]:
                    if child not in visited:
                        to_visit.append(child)
                return _bfs(visited, to_visit)
            return list(set(visited))

        visited_nodes = []
        to_visit = [starting_node]
        return np.array(_bfs(visited_nodes, to_visit))

    struct.reset_index(inplace=True, drop=True)
    coords = extract_coords(struct)

    if synthpharm:
        struct['bp'] = struct['atom_id'].apply(lambda x: int(x <= 2))

    lig_or_rec = struct.bp.to_numpy()
    distances = cdist(coords, coords, 'euclidean')

    adj_inter = (distances < inter_radius) & (distances > 1e-7)
    edge_indices_inter = np.where(adj_inter)

    inter_mask = abs(
        lig_or_rec[edge_indices_inter[0]] - lig_or_rec[edge_indices_inter[1]])
    edge_indices_inter = (edge_indices_inter[0][np.where(inter_mask)],
                          edge_indices_inter[1][np.where(inter_mask)])
    n_edges_inter = sum(inter_mask)

    adj_intra = (distances < intra_radius) & (distances > 1e-7)
    n_edges_intra = np.sum(adj_intra)
    edge_indices_intra = np.where(adj_intra)

    bp_0_inter = lig_or_rec[edge_indices_inter[0]]
    bp_1_inter = lig_or_rec[edge_indices_inter[1]]

    bp_0_intra = lig_or_rec[edge_indices_intra[0]]
    bp_1_intra = lig_or_rec[edge_indices_intra[1]]

    edge_attrs_inter = np.zeros((n_edges_inter,), dtype='int32')
    edge_attrs_intra = np.zeros((n_edges_intra,), dtype='int32')

    edge_attrs_inter[np.where((bp_0_inter == 0) & (bp_1_inter == 1))] = 1
    edge_attrs_inter[np.where((bp_0_inter == 1) & (bp_1_inter == 0))] = 1

    edge_attrs_intra[np.where((bp_0_intra == 1) & (bp_1_intra == 1))] = 2

    edge_attrs = np.concatenate([edge_attrs_inter, edge_attrs_intra])

    edge_indices = (
        np.concatenate([edge_indices_inter[0], edge_indices_intra[0]]),
        np.concatenate([edge_indices_inter[1], edge_indices_intra[1]])
    )

    if prune and n_edges_inter:
        node_list = defaultdict(list)
        for idx in range(len(edge_indices[0])):
            node_list[edge_indices[0][idx]].append(edge_indices[1][idx])
            node_list[edge_indices[1][idx]].append(edge_indices[0][idx])
        starting_node = edge_indices[0][0]
        nodes_to_keep = bfs(starting_node, node_list)
        nodes_to_drop = np.setdiff1d(struct.index, nodes_to_keep)
        struct.drop(nodes_to_drop, inplace=True)
        return generate_edges(struct.copy(), inter_radius, intra_radius, False)

    return struct, edge_indices, edge_attrs


def extract_coords(struct, bp=None):
    """Get numpy coordinates from pd.DataFrame."""
    entity = struct[(struct.bp == bp)] if bp is not None else struct
    return np.vstack(
        [entity.x.to_numpy(), entity.y.to_numpy(), entity.z.to_numpy()]).T


def make_box(struct, radius=4, relative_to_ligand=True):
    """Truncate receptor atoms which are too far away from the ligand.

    Arguments:
        struct: DataFrame containing x, y, z, types, bp series.
        radius: maximum distance from a ligand atom a receptor atom can be
            to avoid being discarded.
        relative_to_ligand: if True, radius means minimum distance to closest
            ligand atom; if False, radius means distance to centre of ligand

    Returns:
        DataFrame of the same format as the input <struct>, with all ligand
        atoms and receptor atoms that are within <radius> angstroms of any
        ligand atom.
    """

    ligand_np = extract_coords(struct, 0)
    receptor_np = extract_coords(struct, 1)

    if relative_to_ligand:
        result = struct[struct.bp == 0].copy()
        rec_struct = struct[struct.bp == 1].copy()
        rec_struct.reset_index(inplace=True)
        distances = cdist(ligand_np, receptor_np, 'euclidean')
        mask = distances < radius
        keep = np.where(np.sum(mask, axis=0))[0]
        result = result.append(rec_struct[rec_struct.index.isin(keep)],
                               ignore_index=True)
        result.reset_index(drop=True, inplace=True)
        del result['index']
        return result

    ligand_centre = np.mean(ligand_np, axis=0)

    struct['sq_dist'] = ((struct.x - ligand_centre[0]) ** 2 +
                         (struct.y - ligand_centre[1]) ** 2 +
                         (struct.z - ligand_centre[2]) ** 2)

    struct = struct[
        (struct.sq_dist < radius ** 2) | (struct.bp == 0)].copy()
    struct.reset_index(drop=True, inplace=True)
    del struct['sq_dist']
    try:
        del struct['index']
    except KeyError:
        pass
    return struct


def make_bit_vector(atom_types, n_atom_types, compact=True):
    """Make one-hot bit vector from indices, with switch for structure type.

    Arguments:
        atom_types: ids for each type of atom
        n_atom_types: number of different atom types in dataset
        compact: instead of a true one hot encoding (with two bits for each
            element: rec and lig), use one-hot encoding for each element then
            use a final bit to denote whether the atom is part of the ligand or
            the receptor

    Returns:
        One-hot bit vector (torch tensor) of atom ids.
    """
    # n_atom_types = 10 + 1 == 11
    # there are 11 possible atom types, +1 for a compact bit
    if compact:
        # atom_types: 0 -> 0 0, 11 -> 0 1
        indices = torch.from_numpy(atom_types % n_atom_types).long()
        one_hot = F.one_hot(indices, num_classes=n_atom_types + 1)
        type_bit = torch.from_numpy((atom_types // n_atom_types)).int()
        one_hot[:, -1] = type_bit
    else:
        one_hot = F.one_hot(
            torch.from_numpy(atom_types), num_classes=n_atom_types * 2)
    return one_hot


def centre_on_ligand(struct):
    """Move all coordinates in dataframe to be centred on the ligand.

    Arguments:
        struct: panda dataframe including x, y, and z columns, as well as
        a bp column denoting whether the rows are ligand atoms (0) or
        receptor atoms (1).

    Returns:
        Dataframe centred on geometric centre of all ligand atoms.
    """
    ligand_atoms = struct[struct.bp == 0]
    ligand_xyz = ligand_atoms[ligand_atoms.columns[:3]].to_numpy()
    mean_x, mean_y, mean_z = np.mean(ligand_xyz, axis=0)
    struct.x -= mean_x
    struct.y -= mean_y
    struct.z -= mean_z
    return struct


def concat_structs(rec, lig, n_features, min_lig_rotation=0, parsers=None,
                   extended=False, synth_pharm=False):
    """Concatenate the receptor and ligand parquet structures."""
    min_lig_rotation_rads = np.pi * min_lig_rotation / 180

    if parsers is None:
        lig_struct = pd.read_parquet(lig)
        rec_struct = pd.read_parquet(rec)
    else:
        lig_struct = parsers[0].file_to_parquets(lig, add_polar_hydrogens=True)
        rec_struct = parsers[1].file_to_parquets(rec, add_polar_hydrogens=True)

    if not synth_pharm:
        rec_struct.types += n_features + extended * 8

        if min_lig_rotation:
            lig_coords_init = np.vstack(
                [lig_struct.x, lig_struct.y, lig_struct.z]).T
            orig_vector = lig_coords_init[0, :]
            candidate_vector = orig_vector
            candidate_coords = lig_coords_init
            while angle_3d(orig_vector, candidate_vector) < min_lig_rotation_rads:
                candidate_coords = uniform_random_rotation(lig_coords_init)
                candidate_vector = candidate_coords[0, :]
            lig_struct.x = candidate_coords[:, 0]
            lig_struct.y = candidate_coords[:, 1]
            lig_struct.z = candidate_coords[:, 2]

        concatted_structs = lig_struct.append(rec_struct, ignore_index=True)
    else:
        atomic_nums = (6, 7, 8, 9, 15, 16, 17, 35, 53)
        lig_struct['atom_id'] = lig_struct['type'].map({
            num: (idx + 3) for idx, num in enumerate(atomic_nums)
        })
        rec_struct['atom_id'] = rec_struct['type']
        concatted_structs = lig_struct.append(rec_struct, ignore_index=True)
    return concatted_structs


def plot_struct(struct, edges=None):
    """Helper function for plotting inputs."""

    def set_axes_equal(ax):
        """Make axes of 3D plot have equal scale so that spheres appear as
        spheres, cubes as cubes, etc.

        Arguments:
          ax: a matplotlib axis
        """

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    s = struct
    xyz = np.vstack([s['x'], s['y'], s['z']]).T
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    atoms = s.bp.to_numpy()
    colours = ['black', 'red']
    ax.scatter(x, y, z, c=atoms, cmap=matplotlib.colors.ListedColormap(colours),
               marker='o', s=80)

    if edges is not None:
        cols = {
            0: 'g-',
            1: 'r-',
            2: 'b-'
        }
        for idx, (i, j) in enumerate(zip(*edges[0])):
            col = cols[edges[1][idx]]
            ax.plot(
                ([xyz[i, 0], xyz[j, 0]]),
                ([xyz[i, 1], xyz[j, 1]]),
                ([xyz[i, 2], xyz[j, 2]]),
                col
            )

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    set_axes_equal(ax)
    plt.savefig('/home/scantleb-admin/Desktop/point_cloud.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('max_dist_from_lig', type=float,
                        help='Maximum distance from a ligand atom that a '
                             'receptor atom can be to be included')
    parser.add_argument('inter_radius', type=float,
                        help='Maximum inter-atomic distance for an edge '
                             'between atoms in different molecules')
    parser.add_argument('intra_radius', type=float,
                        help='Maximum inter-atomic distance for an edge '
                             'between atoms in the same molecule')
    parser.add_argument('--prune', '-p', action='store_true',
                        help='Prune graphs which are disconnected from '
                             'main protein-ligand graph')
    args = parser.parse_args()
    bp = expand_path('data/small_chembl_test')
    struct = make_box(concat_structs(
        bp / 'receptors/12968.parquet',
        bp / 'ligands/12968_actives/mol25_7.parquet',
        min_lig_rotation=0),
        radius=args.max_dist_from_lig, relative_to_ligand=True)

    struct, edge_indices, edge_attrs = generate_edges(
        struct, inter_radius=args.inter_radius,
        intra_radius=args.intra_radius, prune=args.prune)
    plot_struct(struct, (edge_indices, edge_attrs))
