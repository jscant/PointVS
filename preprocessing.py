import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def make_box(struct, radius=12):
    """Truncate spatial coordinates such that all entires have x,y,z < radius"""
    struct = struct[abs(struct.x) < radius]
    struct = struct[abs(struct.y) < radius]
    struct = struct[abs(struct.z) < radius]
    return struct


def make_bit_vector(atom_types, n_atom_types):
    """Make one-hot bit vector from indices, with switch for structure type.

    Arguments:
        atom_types: ids for each type of atom
        n_atom_types: number of different atom types in dataset

    Returns:
        One-hot bit vector (torch tensor) of atom ids, including leftmost bit
        which indicates the structure (ligand == 0, receptor == 1).
    """
    indices = torch.from_numpy(atom_types % n_atom_types).long()
    one_hot = F.one_hot(indices).float()
    rows, cols = one_hot.shape
    result = torch.zeros(rows, cols + 1)
    result[:, 1:] = one_hot
    type_bit = torch.from_numpy(
        (atom_types // n_atom_types).astype('bool').astype('int')).float()
    result[:, 0] = type_bit
    return result


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


def concat_structs(rec, lig):
    """Concatenate the receptor and ligand parquet structures."""
    lig_struct = pd.read_parquet(lig)
    rec_struct = pd.read_parquet(rec)
    return lig_struct.append(rec_struct, ignore_index=True)
