from collections import defaultdict

import numpy as np
import torch
from einops import repeat
from pandas import DataFrame
from plip.basic.remote import VisualizerData
from plip.visualization.pymol import PyMOLVisualizer
from pymol import cmd

from point_vs.models.point_neural_network import to_numpy
from point_vs.preprocessing.preprocessing import make_bit_vector, make_box
from point_vs.utils import coords_to_string, PositionDict


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

        bs = model_args['batch_size']
        radius = model_args['radius']
        polar_hydrogens = model_args['hydrogens']
        compact = model_args['compact']
        use_atomic_numbers = model_args['use_atomic_numbers']

        triplet_code = self.plcomplex.uid.split(':')[0]
        if len(only_process) and triplet_code not in only_process:
            return None, None

        df = parser.mol_calculate_interactions(
            self.plcomplex.mol, self.plcomplex.pli)

        if not quiet:
            print('Attributing scores to site:', self.plcomplex.uid)

        df = make_box(df, radius=radius, relative_to_ligand=False)

        if use_atomic_numbers:
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
            max_feature_id = max(atomic_number_to_index.values()) + 1

            # Any other elements not accounted for given a category of their own
            atomic_number_to_index = defaultdict(lambda: max_feature_id)
            atomic_number_to_index.update(atomic_number_to_index)
            df.types = df['atomic_number'].map(
                atomic_number_to_index) + df.bp * (max_feature_id + 1)

        elif polar_hydrogens:
            max_feature_id = 11
        else:
            max_feature_id = 10
        if not polar_hydrogens:
            df = df[df['atomic_number'] > 1]

        coords = np.vstack([df.x.to_numpy(), df.y.to_numpy(), df.z.to_numpy()]).T

        p = torch.from_numpy(coords).float()
        p = repeat(p, 'n d -> b n d', b=1)

        m = torch.from_numpy(np.ones((len(df),))).bool()
        m = repeat(m, 'n -> b n', b=1)

        v = make_bit_vector(
            df.types.to_numpy(), max_feature_id + 1, compact).float()
        v = repeat(v, 'n d -> b n d', b=1)

        model = model.eval().cuda()
        score = float(to_numpy(
            torch.sigmoid(model((p.cuda(), v.cuda(), m.cuda()))[0, ...])))
        if not quiet:
            print('Original score: {:.4}'.format(score))

        model_labels = attribution_fn(
            model, p.cuda(), v.cuda(), m.cuda(), bs=bs)
        df['attribution'] = model_labels
        df['any_interaction'] = df['hba'] | df['hbd'] | df['pistacking']

        return score, df

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

        score, df = self.score_atoms(
            parser, only_process, model, attribution_fn, model_args)
        if not isinstance(df, DataFrame):
            return None, None

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
        cmd.rebuild()
        return score, df
