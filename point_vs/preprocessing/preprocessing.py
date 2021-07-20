"""Some helper functions for cutting inputs down to size."""

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


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
    citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.53.1357&rep=rep1&type=pdf
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


def make_box(struct, radius=4, relative_to_ligand=True):
    """Truncates receptor atoms which are too far away from the ligand.

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

    def extract_coords(bp):
        entity = struct[(struct.bp == bp) & (struct.atomic_number > 1)]
        return np.vstack(
            [entity.x.to_numpy(), entity.y.to_numpy(), entity.z.to_numpy()]).T

    ligand_np = extract_coords(0)
    receptor_np = extract_coords(1)
    ligand_centre = np.mean(ligand_np, axis=0)

    struct['sq_dist'] = ((struct.x - ligand_centre[0]) ** 2 +
                         (struct.y - ligand_centre[1]) ** 2 +
                         (struct.z - ligand_centre[2]) ** 2)

    if not relative_to_ligand:
        struct = struct[(struct.sq_dist < radius ** 2) | (struct.bp == 0)].copy()
        del struct['sq_dist']
        return struct

    struct = struct[struct.sq_dist < 12].copy()
    del struct['sq_dist']
    ligand = struct[struct.bp == 0].copy()
    receptor = struct[struct.bp == 1].copy()
    ligand_np = extract_coords(0)
    r_squared = radius ** 2
    include = np.zeros((len(receptor, )))
    for rec_idx in range(len(include)):
        for lig_idx in range(ligand_np.shape[0]):
            sq_dist = sum(
                np.square(receptor_np[rec_idx, :] - ligand_np[lig_idx, :]))
            if sq_dist <= r_squared:
                include[rec_idx] = 1
                break
    receptor['include'] = include
    receptor = receptor[receptor.include == 1]
    result = ligand.append(receptor, ignore_index=True)
    del result['include']
    return result


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
    if compact:
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


def concat_structs(rec, lig, min_lig_rotation=0):
    """Concatenate the receptor and ligand parquet structures."""
    min_lig_rotation = np.pi * min_lig_rotation / 180
    lig_struct = pd.read_parquet(lig)
    rec_struct = pd.read_parquet(rec)
    ############################################################################
    # Uncommenting the following would result in excluding 'other' atoms:

    # lig_struct = lig_struct[lig_struct.types != 10]
    # lig_struct.types = lig_struct.types.clip(upper=10)
    # rec_struct = rec_struct[rec_struct.types != 10]
    # lig_struct.types = rec_struct.types.clip(upper=10)
    # rec.struct.types += 11 # (replace the analogous line below)
    ############################################################################
    rec_struct.types += 12

    if min_lig_rotation:
        lig_coords_init = np.vstack(
            [lig_struct.x, lig_struct.y, lig_struct.z]).T
        orig_vector = lig_coords_init[0, :]
        candidate_vector = orig_vector
        candidate_coords = lig_coords_init
        while angle_3d(orig_vector, candidate_vector) < min_lig_rotation:
            candidate_coords = uniform_random_rotation(lig_coords_init)
            candidate_vector = candidate_coords[0, :]
        lig_struct.x = candidate_coords[:, 0]
        lig_struct.y = candidate_coords[:, 1]
        lig_struct.z = candidate_coords[:, 2]

    concatted_structs = lig_struct.append(rec_struct, ignore_index=True)
    return concatted_structs


"""Everything below is for debugging (ignore)"""


def plot_struct(struct):
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
    xyz = s[s.columns[:3]].to_numpy()
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    atoms = s.bp.to_numpy()
    colours = ['black', 'red']
    ax.scatter(x, y, z, c=atoms, cmap=matplotlib.colors.ListedColormap(colours),
               marker='o', s=80)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    set_axes_equal(ax)
    plt.savefig('/home/scantleb-admin/Desktop/point_cloud.png')
    plt.show()


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Arguments:
        ax: a matplotlib axis, e.g., as output from plt.gca().

    (From stackoverflow)
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

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


if __name__ == '__main__':
    random_coords = np.random.rand(30, 3)
    vecs = []
    angles = []
    for i in range(300):
        rot = uniform_random_rotation(random_coords)
        vecs.append(rot[0, :])
        angles.append(180 * angle_3d(random_coords[0, :], rot[0, :]) / np.pi)

    vecs = np.array(vecs)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    ax.scatter(x, y, z, marker='o', s=20)
    ax.scatter(*random_coords[0, :], c='r', s=20, marker='o')
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    set_axes_equal(ax)

    plt.savefig('sphere.png')
    plt.show()
    plt.clf()
    plt.hist(angles, bins=50)
    plt.savefig('angles.png')
