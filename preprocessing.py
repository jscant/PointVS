import numpy as np
import pandas as pd


def make_box(struct, radius=12):
    struct = struct[abs(struct.x) < radius]
    struct = struct[abs(struct.y) < radius]
    struct = struct[abs(struct.z) < radius]
    return struct


def centre_on_ligand(struct):
    ligand_atoms = struct[struct.bp == 1]
    ligand_xyz = ligand_atoms[ligand_atoms.columns[:3]].to_numpy()
    mean_x, mean_y, mean_z = np.mean(ligand_xyz, axis=0)
    struct.x -= mean_x
    struct.y -= mean_y
    struct.z -= mean_z
    return struct


def concat_structs(rec, lig):
    lig_struct = pd.read_parquet(lig)
    rec_struct = pd.read_parquet(rec)
    return lig_struct.append(rec_struct, ignore_index=True)
