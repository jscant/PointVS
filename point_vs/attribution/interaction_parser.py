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
from collections import defaultdict
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pandas as pd

from point_vs.preprocessing.pdb_to_parquet import PDBFileParser
from point_vs.utils import coords_to_string, truncate_float

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
        ligand_coords = [coords_to_string(atom.coords) for atom in
                         list(itertools.chain(*ligand_atoms))]

        return self.featurise_interaction(
            mol, interaction_info, ligand_coords)

    def featurise_interaction(self, mol, interaction_dict, all_ligand_coords,
                              include_noncovalent_bonds=True):
        """Return dataframe with interactions from one particular plip site."""

        def keep_atom(atom):
            coords_str = coords_to_string(atom.coords)
            if atom.OBAtom.IsNonPolarHydrogen():
                return False
            if atom.OBAtom.GetResidue().GetName().upper() in RESIDUE_IDS:
                return True
            if coords_str in all_ligand_coords:
                return True
            return False

        xs, ys, zs, types, atomic_nums, atomids = [], [], [], [], [], []
        keep_atoms = []
        types_addition = []
        obabel_to_sequential = defaultdict(lambda: len(obabel_to_sequential))
        max_types_value = max(self.type_map.values()) + 1
        for atomid, atom in mol.atoms.items():
            if keep_atom(atom):
                keep_atoms.append(atomid)

            atomids.append(atomid)

            smina_type = self.obatom_to_smina_type(atom)
            if smina_type == "NumTypes":
                smina_type_int = len(self.atom_type_data)
            else:
                smina_type_int = self.atom_types.index(smina_type)
            type_int = self.type_map[smina_type_int]
            if coords_to_string(atom.coords) in all_ligand_coords:
                types_addition.append(0)
            else:
                types_addition.append(max_types_value)

            x, y, z = [truncate_float(i) for i in atom.coords]
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
        bp = (types_addition > 0).astype('int')
        types += types_addition

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

        mask = np.zeros_like(pistacking)
        mask[keep_atoms] = 1

        pistacking = pistacking[np.where(keep_atoms)]
        hba = hba[np.where(mask)]
        hbd = hbd[np.where(mask)]
        xs = xs[np.where(mask)]
        ys = ys[np.where(mask)]
        zs = zs[np.where(mask)]
        types = types[np.where(mask)]
        atomic_nums = atomic_nums[np.where(mask)]
        atomids = atomids[np.where(mask)]
        bp = bp[np.where(mask)]

        df = pd.DataFrame()
        if include_noncovalent_bonds:
            df['atom_id'] = atomids

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
