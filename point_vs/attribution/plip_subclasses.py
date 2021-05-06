from collections import defaultdict

import numpy as np
import torch
from plip.basic.remote import VisualizerData
from plip.visualization.pymol import PyMOLVisualizer
from pymol import cmd

from point_vs.models.point_neural_network import to_numpy
from point_vs.preprocessing.preprocessing import make_bit_vector


class VisualizerDataWithMolecularInfo(VisualizerData):
    """VisualizerData but with the mol, ligand and pli objects stored."""

    def __init__(self, mol, site):
        pli = mol.interaction_sets[site]
        self.ligand = pli.ligand
        self.pli = pli
        self.mol = mol
        super().__init__(mol, site)


class PyMOLVisualizerWithBFactorColouring(PyMOLVisualizer):

    def colour_b_factors_pdb(
            self, model, dt, input_dim, attribution_fn, results_fname,
            radius=12, bs=16):

        def change_bfactors(bfactors):
            """Modify bfactors based on spatial location.

            Due to inconsistencies in the indexing of atoms, residues and chains
            between openbabel, plip and pymol, we must use coordinates to
            identify atoms.

            Arguments:
                bfactors: dict of dict of dicts with the mapping:
                    x: y: z: value
                where value is the number we wish to assign to the atom (as the
                b-factor) for labelling. The x, y and z coordinates should be
                in the format of strings, with 1 decimal place to avoid
                problems with comparing floats (use '{:.1f}.format(<coord>).
            """

            def modify_bfactor(x, y, z):
                """Return b factor given the x, y, z coordinates."""
                x, y, z = ['{:.3f}'.format(coord) for coord in (x, y, z)]
                bfactor = bfactors[x][y][z]
                return bfactor

            space = {'modify_bfactor': modify_bfactor}
            cmd.alter_state(
                0, '(all)', 'b=modify_bfactor(x, y, z)', space=space,
                quiet=True)

        def find_ligand_centre(ligand):
            """Calculate the mean coordinates of all ligand atoms."""
            positions = []
            for atom in ligand.molecule.atoms:
                positions.append(atom.coords)
            return np.mean(np.array(positions), axis=0)

        df = dt.mol_calculate_interactions(
            self.plcomplex.mol, self.plcomplex.pli)

        centre_coords = find_ligand_centre(self.plcomplex.ligand)
        print('Attributing scores to site:', self.plcomplex.uid)
        mean_x, mean_y, mean_z = centre_coords

        df['x'] -= mean_x
        df['y'] -= mean_y
        df['z'] -= mean_z
        radius = radius
        df['sq_dist'] = df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2
        df = df[df.sq_dist < radius ** 2].copy()
        df['x'] += mean_x
        df['y'] += mean_y
        df['z'] += mean_z
        del df['sq_dist']

        x, y, z = df['x'].to_numpy(), df['y'].to_numpy(), df['z'].to_numpy()
        p = torch.from_numpy(
            np.expand_dims(df[df.columns[1:4]].to_numpy(), 0).astype('float32'))

        m = torch.from_numpy(np.ones((1, len(df)))).bool()

        v = make_bit_vector(df.types.to_numpy(), input_dim - 1)[
            None, ...].float()

        model = model.eval().cuda()
        print('Original score:', float(to_numpy(
            torch.sigmoid(model((p.cuda(), v.cuda(), m.cuda()))))))

        model_labels = attribution_fn(
            model, p.cuda(), v.cuda(), m.cuda(), bs=bs)

        atom_to_bfactor_map = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
        all_bfactors = []
        for i in range(len(df)):
            bfactor = float(model_labels[i])
            all_bfactors.append(bfactor)
            atom_to_bfactor_map[
                '{:.3f}'.format(x[i])][
                '{:.3f}'.format(y[i])][
                '{:.3f}'.format(z[i])] = bfactor

        min_bfactor = min(min(all_bfactors), 0)
        max_bfactor = max(max(all_bfactors), 0)
        max_absolute_bfactor = max(abs(max_bfactor), abs(min_bfactor))
        if isinstance(max_absolute_bfactor, (list, np.ndarray)):
            max_absolute_bfactor = max_absolute_bfactor[0]
        print('Colouring b-factors in range ({0:0.3f}, {1:0.3f})'.format(
            -max_absolute_bfactor, max_absolute_bfactor))

        df['attribution'] = model_labels
        df['any_interaction'] = df['hba'] | df['hbd'] | df['pistacking']
        df.sort_values(by='attribution', ascending=False, inplace=True)
        df.to_csv(results_fname)

        cmd.alter('all', "b=0")
        change_bfactors(atom_to_bfactor_map)
        cmd.spectrum(
            'b', 'red_white_green', minimum=-max_absolute_bfactor,
            maximum=max_absolute_bfactor)
        cmd.show('sticks', 'b > 0')
        cmd.rebuild()
