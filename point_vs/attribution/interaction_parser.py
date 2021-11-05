"""
Convert pdb, sdf and mol2 coordinates files into pandas-readable parquet files,
which can be used by point cloud/GNN models in this repo. Usage:

pdb_to_parquet.py <base_path> <output_path>

<base_path> should be structured like so:

<base_path>
├── ligands
│   ├── receptor_a
│   │   └── ligands.sdf
│   └── receptor_b
│       └── ligands.sdf
└── receptors
    ├── receptor_a
    │   └── receptor.pdb
    └── receptor_a
        └── receptor.pdb
"""
import itertools
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pandas as pd

from point_vs.dataset_generation.types_to_parquet import StructuralFileParser
from point_vs.utils import coords_to_string, PositionDict, PositionSet

try:
    from openbabel import pybel
except (ModuleNotFoundError, ImportError):
    import pybel

RESIDUE_IDS = {'MET', 'ARG', 'SER', 'TRP', 'HIS', 'CYS', 'LYS', 'GLU', 'THR',
               'LEU', 'TYR', 'PRO', 'ASN', 'ASP', 'PHE', 'GLY', 'VAL', 'ALA',
               'ILE', 'GLN'}


def fetch_pdb(pdbid):
    """Modified plip function."""
    pdbid = pdbid.lower()
    pdburl = f'https://files.rcsb.org/download/{pdbid}.pdb'
    try:
        pdbfile = urlopen(pdburl).read().decode()
        if 'sorry' in pdbfile:
            print('No file in PDB format available from wwPDB for', pdbid)
            return None, None
    except HTTPError:
        print('No file in PDB format available from wwPDB for', pdbid)
        return None, None
    return [pdbfile, pdbid]


class StructuralInteractionParser(StructuralFileParser):
    """Python reimplementation of the gninatyper function,
    as per https://pubs.acs.org/doi/10.1021/acs.jcim.6b00740
    (some code modified from Constantin Schneider, OPIG)
    """

    def mol_calculate_interactions(self, mol, pl_interaction):
        """Return dataframe with interactions from plip mol object"""

        interaction_info = {}

        # Process interactions
        hbonds_lig_donors = pl_interaction.hbonds_ldon
        hbonds_rec_donors = pl_interaction.hbonds_pdon
        interaction_info['rec_acceptors'] = {
            coords_to_string(hbond.a.coords): 1
            for hbond in hbonds_lig_donors}
        interaction_info['lig_donors'] = {
            coords_to_string(hbond.d.coords): 1
            for hbond in hbonds_lig_donors}
        interaction_info['rec_donors'] = {
            coords_to_string(hbond.d.coords): 1
            for hbond in hbonds_rec_donors}
        interaction_info['lig_acceptors'] = {
            coords_to_string(hbond.a.coords): 1
            for hbond in hbonds_rec_donors}
        pi_stacking_atoms = [interaction.proteinring.atoms for interaction
                             in pl_interaction.pistacking]
        pi_stacking_atoms += [interaction.ligandring.atoms for interaction
                              in pl_interaction.pistacking]
        pi_stacking_atoms = [
            atom for ring in pi_stacking_atoms for atom in ring]
        interaction_info['pi_stacking'] = {
            coords_to_string(atom.coords): 1 for atom in pi_stacking_atoms}

        # Book-keeping to track ligand atoms by coordinates
        ligand_mols = [ligand.mol for ligand in mol.ligands]
        ligand_atoms = [mol.atoms for mol in ligand_mols]

        all_ligand_coords = PositionSet({
            coords_to_string(atom.coords) for atom in pl_interaction.ligand.all_atoms
        })
        
        return self.featurise_interaction(
            mol, interaction_info, all_ligand_coords)

    def featurise_interaction(self, mol, interaction_dict, all_ligand_coords,
                              include_noncovalent_bonds=True):
        """Return dataframe with interactions from one particular plip site."""

        xs, ys, zs, types, atomic_nums, bp = self.get_coords_and_types_info(
            mol.atoms.values(), all_ligand_coords, add_polar_hydrogens=True)

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        zs = np.array(zs, dtype=float)
        types = np.array(types, dtype=int)
        atomic_nums = np.array(atomic_nums, dtype=int)

        pistacking = np.zeros((len(types),), dtype=np.int32)
        hba = np.zeros_like(pistacking)
        hbd = np.zeros_like(pistacking)

        for i in range(len(xs)):
            coords = coords_to_string((xs[i], ys[i], zs[i]))
            hba[i] = interaction_dict['lig_acceptors'].get(
                coords, interaction_dict['rec_acceptors'].get(coords, 0))
            hbd[i] = interaction_dict['lig_donors'].get(
                coords, interaction_dict['rec_donors'].get(coords, 0))
            pistacking[i] = interaction_dict['pi_stacking'].get(
                coords, 0)

        df = pd.DataFrame()

        df['x'] = xs
        df['y'] = ys
        df['z'] = zs
        df['atomic_number'] = atomic_nums
        df['types'] = types
        df['bp'] = bp
        if include_noncovalent_bonds:
            df['pistacking'] = pistacking
            df['hba'] = hba
            df['hbd'] = hbd

        return df
