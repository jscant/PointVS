from collections import defaultdict

import numpy as np
import torch
from plip.basic.remote import VisualizerData
from plip.visualization.pymol import PyMOLVisualizer
from pymol import cmd

from point_vs.models.point_neural_network import to_numpy
from point_vs.preprocessing.preprocessing import make_bit_vector
from point_vs.utils import coords_to_string, truncate_float, PositionDict


class VisualizerDataWithMolecularInfo(VisualizerData):
    """VisualizerData but with the mol, ligand and pli objects stored."""

    def __init__(self, mol, site):
        pli = mol.interaction_sets[site]
        self.ligand = pli.ligand
        self.pli = pli
        self.mol = mol
        super().__init__(mol, site)


class PyMOLVisualizerWithBFactorColouring(PyMOLVisualizer):

    def score_atoms(
            self, parser, only_process, model, attribution_fn, model_args,
            quiet=False):

        def find_ligand_centre(ligand):
            """Calculate the mean coordinates of all ligand atoms."""
            positions = []
            for atom in ligand.molecule.atoms:
                positions.append(atom.coords)
            mean = np.mean(np.array(positions), axis=0)
            return np.array([truncate_float(i) for i in mean])

        bs = model_args['batch_size']
        radius = model_args['radius']
        polar_hydrogens = model_args['hydrogens']
        compact = model_args['compact']
        use_atomic_numbers = model_args['use_atomic_numbers']

        df = parser.mol_calculate_interactions(
            self.plcomplex.mol, self.plcomplex.pli)

        # H C N O F P S Cl
        recognised_atomic_numbers = (6, 7, 8, 9, 15, 16, 17)
        # various metal ions/halogens which share valence properties
        other_groupings = ((35, 53), (3, 11, 19), (4, 12, 20), (26, 29, 30))
        atomic_number_to_index = {
            num: idx for idx, num in enumerate(recognised_atomic_numbers)
        }
        for grouping in other_groupings:
            atomic_number_to_index.update({elem: max(
                atomic_number_to_index.values()) + 1 for elem in grouping})
        if polar_hydrogens:
            atomic_number_to_index.update({
                1: max(atomic_number_to_index.values()) + 1
            })

        # +1 to accommodate for unmapped elements
        max_elem_id = max(atomic_number_to_index.values()) + 1

        # Any other elements not accounted for given a category of their own
        atomic_number_to_index = defaultdict(lambda: max_elem_id)
        atomic_number_to_index.update(atomic_number_to_index)

        if compact:
            feature_dim = max_elem_id + 2
        else:
            feature_dim = (max_elem_id + 1) * 2

        triplet_code = self.plcomplex.uid.split(':')[0]
        if len(only_process) and triplet_code not in only_process:
            return

        if not quiet:
            print('Attributing scores to site:', self.plcomplex.uid)
        mean_x, mean_y, mean_z = find_ligand_centre(self.plcomplex.ligand)

        df['x'] -= mean_x
        df['y'] -= mean_y
        df['z'] -= mean_z
        df['sq_dist'] = df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2
        df = df[df.sq_dist < radius ** 2].copy()
        df['x'] += mean_x
        df['y'] += mean_y
        df['z'] += mean_z
        del df['sq_dist']

        if not polar_hydrogens:
            df = df[df['atomic_number'] > 1]
        if use_atomic_numbers:
            df.types = df['atomic_number'].map(
                atomic_number_to_index) + df.bp * (max_elem_id + 1)

        p = torch.from_numpy(
            np.expand_dims(df[df.columns[1:4]].to_numpy(), 0).astype('float32'))

        m = torch.from_numpy(np.ones((1, len(df)))).bool()

        v = torch.unsqueeze(make_bit_vector(
            df.types.to_numpy(), max_elem_id + 1, compact), 0).float()

        model = model.eval().cuda()
        if not quiet or 1:
            print('Original score:', float(to_numpy(
                torch.sigmoid(model((p.cuda(), v.cuda(), m.cuda()))))))

        model_labels = attribution_fn(
            model, p.cuda(), v.cuda(), m.cuda(), bs=bs)
        df['attribution'] = model_labels
        df['any_interaction'] = df['hba'] | df['hbd'] | df['pistacking']
        df.sort_values(by='attribution', ascending=False, inplace=True)
        return df

    def colour_b_factors_pdb(
            self, model, parser, attribution_fn, results_fname, model_args,
            only_process=None):

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
                in the format of strings, with 3 decimal places to avoid
                problems with comparing floats (use coords_to_string(<coord>)).
            """

            def modify_bfactor(x, y, z):
                """Return b factor given the x, y, z coordinates."""
                coords = coords_to_string((x, y, z))
                bfactor = bfactors.get(coords, 0)
                return bfactor

            space = {'modify_bfactor': modify_bfactor}
            cmd.alter_state(
                0, '(all)', 'b=modify_bfactor(x, y, z)', space=space,
                quiet=True)

        df = self.score_atoms(
            parser, only_process, model, attribution_fn, model_args)

        model_labels = df['attribution'].to_numpy()
        x, y, z = df['x'].to_numpy(), df['y'].to_numpy(), df['z'].to_numpy()

        atom_to_bfactor_map = PositionDict(eps=0.01)
        all_bfactors = []
        for i in range(len(df)):
            bfactor = float(model_labels[i])
            all_bfactors.append(bfactor)
            atom_to_bfactor_map[coords_to_string((x[i], y[i], z[i]))] = bfactor

        min_bfactor = min(min(all_bfactors), 0)
        max_bfactor = max(max(all_bfactors), 0)
        max_absolute_bfactor = max(abs(max_bfactor), abs(min_bfactor))
        if isinstance(max_absolute_bfactor, (list, np.ndarray)):
            max_absolute_bfactor = max_absolute_bfactor[0]
        print('Colouring b-factors in range ({0:0.3f}, {1:0.3f})'.format(
            -max_absolute_bfactor, max_absolute_bfactor))

        df.to_csv(results_fname, index=False, float_format='%.3f')

        cmd.alter('all', "b=0")
        print('Changing bfactors...')
        change_bfactors(atom_to_bfactor_map)
        print('Done!')
        cmd.spectrum(
            'b', 'red_white_green', minimum=-max_absolute_bfactor,
            maximum=max_absolute_bfactor)
        cmd.show('sticks', 'b > 0')
        cmd.rebuild()

        return df
