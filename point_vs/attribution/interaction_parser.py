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
from collections import defaultdict
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pandas as pd

from point_vs.preprocessing.pdb_to_parquet import PDBFileParser

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


class PDBInteractionParser(PDBFileParser):
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
            '{0:.3f} {1:.3f} {2:.3f}'.format(*hbond.a.coords): 1
            for hbond in hbonds_lig_donors}
        interaction_info['lig_donors'] = {
            '{0:.3f} {1:.3f} {2:.3f}'.format(*hbond.d.coords): 1
            for hbond in hbonds_lig_donors}
        interaction_info['rec_donors'] = {
            '{0:.3f} {1:.3f} {2:.3f}'.format(*hbond.d.coords): 1
            for hbond in hbonds_rec_donors}
        interaction_info['lig_acceptors'] = {
            '{0:.3f} {1:.3f} {2:.3f}'.format(*hbond.a.coords): 1
            for hbond in hbonds_rec_donors}
        pi_stacking_atoms = [interaction.proteinring.atoms for interaction
                             in pl_interaction.pistacking]
        pi_stacking_atoms += [interaction.ligandring.atoms for interaction
                              in pl_interaction.pistacking]
        pi_stacking_atoms = [
            atom for ring in pi_stacking_atoms for atom in ring]
        interaction_info['pi_stacking'] = {
            '{0:.3f} {1:.3f} {2:.3f}'.format(
                *atom.coords): 1 for atom in pi_stacking_atoms
        }

        all_ligand_indices = [list(ligand.can_to_pdb.values()) for ligand in
                              mol.ligands]
        all_ligand_indices = [idx for idx_list in all_ligand_indices for idx in
                              idx_list]

        return self.featurise_interaction(
            mol, interaction_info, all_ligand_indices)

    def featurise_interaction(self, mol, interaction_dict, all_ligand_indices,
                              include_noncovalent_bonds=True):
        """Return dataframe with interactions from one particular plip site."""
        xs, ys, zs, types, atomic_nums, atomids = [], [], [], [], [], []
        keep_atoms = []
        types_addition = []
        obabel_to_sequential = defaultdict(lambda: len(obabel_to_sequential))
        max_types_value = max(self.type_map.values()) + 1
        for atomid, atom in mol.atoms.items():
            resname = atom.OBAtom.GetResidue().GetName().upper()
            if atom.atomicnum > 1:
                if resname in RESIDUE_IDS or atomid in all_ligand_indices:
                    keep_atoms.append(atomid)
                    # Book keeping for DSSP
                    chain = atom.OBAtom.GetResidue().GetChain()
                    residue_id = str(atom.OBAtom.GetResidue().GetNum())
                    residue_identifier = ':'.join([chain, residue_id])

            atomids.append(atomid)

            smina_type = self.obatom_to_smina_type(atom)
            if smina_type == "NumTypes":
                smina_type_int = len(self.atom_type_data)
            else:
                smina_type_int = self.atom_types.index(smina_type)
            type_int = self.type_map[smina_type_int]
            if resname in RESIDUE_IDS and atomid not in all_ligand_indices:
                types_addition.append(max_types_value)
            else:
                types_addition.append(0)

            x, y, z = [float('{:.3f}'.format(i)) for i in atom.coords]
            xs.append(x)
            ys.append(y)
            zs.append(z)
            types.append(type_int)
            atomic_nums.append(atom.atomicnum)

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        zs = np.array(zs, dtype=float)
        types = np.array(types, dtype=int)
        atomids = np.array(atomids, dtype=int)
        atomic_nums = np.array(atomic_nums, dtype=int)
        types_addition = np.array(types_addition, dtype=int)
        types += types_addition

        pistacking = np.zeros((len(types),), dtype=np.int32)
        hba = np.zeros_like(pistacking)
        hbd = np.zeros_like(pistacking)

        for i in range(len(xs)):
            coords = '{0:.3f} {1:.3f} {2:.3f}'.format(xs[i], ys[i], zs[i])
            hba[i] = interaction_dict['lig_acceptors'].get(
                coords, interaction_dict['rec_acceptors'].get(coords, 0))
            hbd[i] = interaction_dict['lig_donors'].get(
                coords, interaction_dict['rec_donors'].get(coords, 0))
            pistacking[i] = interaction_dict['pi_stacking'].get(
                coords, 0)

        mask = np.zeros_like(pistacking)
        mask[keep_atoms] = 1

        pistacking = pistacking[np.where(keep_atoms)]
        hba = hba[np.where(keep_atoms)]
        hbd = hbd[np.where(keep_atoms)]
        xs = xs[np.where(keep_atoms)]
        ys = ys[np.where(keep_atoms)]
        zs = zs[np.where(keep_atoms)]
        types = types[np.where(keep_atoms)]
        atomic_nums = atomic_nums[np.where(keep_atoms)]
        atomids = atomids[np.where(keep_atoms)]

        df = pd.DataFrame()
        if include_noncovalent_bonds:
            df['atom_id'] = atomids
        df['x'] = xs
        df['y'] = ys
        df['z'] = zs
        df['atomic_number'] = atomic_nums
        df['types'] = types
        if include_noncovalent_bonds:
            df['pistacking'] = pistacking
            df['hba'] = hba
            df['hbd'] = hbd
        return df
