import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from plip.basic.remote import VisualizerData
from plip.visualization.pymol import PyMOLVisualizer
from pymol import cmd, stored

from point_vs.models.point_neural_network import to_numpy


class VisualizerDataWithMolecularInfo(VisualizerData):
    """VisualizerData but with the mol, ligand and pli objects stored."""

    def __init__(self, mol, site):
        pli = mol.interaction_sets[site]
        self.ligand = pli.ligand
        self.pli = pli
        self.mol = mol
        super().__init__(mol, site)


class PyMOLVisualizerWithBFactorColouring(PyMOLVisualizer):

    def colour_b_factors(self, model, pdb_file, dt, input_dim, attribution_fn,
                         chain='', quiet=False, radius=12, bs=16):

        def atom_data_extract(chain, atom_to_bfactor_map):
            bdat = {}

            for idx, b_factor in atom_to_bfactor_map.items():
                bdat.setdefault(chain, {})[idx] = (b_factor, '')

            return bdat

        df = dt.mol_calculate_interactions(self.plcomplex.mol)

        all_indices = df['atom_id'].to_numpy()

        centre_coords = find_ligand_centre(self.plcomplex.ligand)
        print('Attributing scores to site:', self.plcomplex.uid)
        mean_x, mean_y, mean_z = centre_coords

        df['x'] -= mean_x
        df['y'] -= mean_y
        df['z'] -= mean_z
        df['sq_dist'] = (df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2)
        df = df[df.sq_dist < radius ** 2].copy()
        df['x'] += mean_x
        df['y'] += mean_y
        df['z'] += mean_z

        labelled_indices = df['atom_id'].to_numpy()
        unlabelled_indices = np.setdiff1d(all_indices, labelled_indices)

        p = torch.from_numpy(
            np.expand_dims(df[df.columns[1:4]].to_numpy(), 0).astype('float32'))
        m = torch.from_numpy(np.ones((1, len(df)))).bool()

        v = torch.unsqueeze(F.one_hot(torch.as_tensor(
            df.types.to_numpy()), input_dim), 0).float()

        model = model.eval().cuda()
        model_labels = attribution_fn(
            model, p.cuda(), v.cuda(), m.cuda(), bs=bs)

        df['attribution'] = model_labels
        with pd.option_context('display.max_colwidth', None):
            with pd.option_context('display.max_rows', None):
                with pd.option_context('display.max_columns', None):
                    print(df)

        atom_to_bfactor_map = {
            labelled_indices[i]: model_labels[i] for i in range(len(df))}
        atom_to_bfactor_map.update({
            idx: 0 for idx in unlabelled_indices})

        # change self.protname to ''
        b_factor_labels = atom_data_extract(
            chain, atom_to_bfactor_map)

        def b_lookup(chain, resi, name, atom_id, b):
            def _lookup(chain, resi, name, atom_id):
                if resi in b_factor_labels[chain] and isinstance(
                        b_factor_labels[chain][resi], dict):
                    return b_factor_labels[chain][resi][name][0]
                else:
                    # find data by ID
                    return b_factor_labels[chain][int(atom_id)][0]

            try:
                if chain not in b_factor_labels:
                    chain = ''
                b = _lookup(chain, resi, name, atom_id)
                if not quiet:
                    print('///%s/%s/%s new: %f' % (chain, resi, name, b))
            except KeyError:
                if not quiet:
                    print('///%s/%s/%s keeping: %f' % (chain, resi, name, b))
            return b

        stored.b = b_lookup

        cmd.alter('all', "b=0")
        cmd.alter(
            'all', '%s=stored.b(chain, resi, name, ID, %s)' % ('b', 'b'))
        print(self.ligname)
        cmd.spectrum('b', 'white_red')
        cmd.show('sticks', 'b > 0')
        cmd.rebuild()


def find_ligand_centre(ligand):
    positions = []
    for atom in ligand.molecule.atoms:
        positions.append(atom.coords)
    return np.mean(np.array(positions), axis=0)
