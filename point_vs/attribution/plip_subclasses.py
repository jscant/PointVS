from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from einops import repeat
from pandas import DataFrame
from plip.basic.remote import VisualizerData
from plip.visualization.pymol import PyMOLVisualizer
from pymol import cmd
from pymol.cgo import CYLINDER
from torch.nn.functional import one_hot
from torch_geometric.data import Data

from point_vs.attribution.attribution_fns import edge_attention, \
    edge_embedding_attribution
from point_vs.models.geometric.pnn_geometric_base import PNNGeometricBase
from point_vs.models.point_neural_network_base import to_numpy
from point_vs.preprocessing.preprocessing import make_bit_vector, make_box, \
    generate_edges
from point_vs.preprocessing.pyg_single_item_dataset import \
    get_pyg_single_graph_for_inference
from point_vs.utils import coords_to_string, PositionDict, \
    get_colour_interpolation_fn, expand_path, print_df


class VisualizerDataWithMolecularInfo(VisualizerData):
    """VisualizerData but with the mol, ligand and pli objects stored."""

    def __init__(self, mol, site):
        pli = mol.interaction_sets[site]
        self.ligand = pli.ligand
        self.pli = pli
        self.mol = mol
        super().__init__(mol, site)


class PyMOLVisualizerWithBFactorColouring(PyMOLVisualizer):

    def show_hbonds(
            self, bonding_strs=None, atom_blind=False, inverse_colour=False):
        """Visualizes hydrogen bonds."""

        """iterate_state with x y z could also be used if the names do not 
        match"""

        bonding_strs_holder = []

        def print_info(resi, resn, name):
            # Returning anything from a pymol fn is incredibly difficult,
            # use placeholder for residue identification instead
            nonlocal bonding_strs_holder
            bonding_strs_holder.append(
                '{0}:{1}:{2}'.format(resi, resn, name).replace("'", ''))

        hbonds = self.plcomplex.hbonds
        if isinstance(bonding_strs, dict) and len(bonding_strs):
            max_bond_score = max(bonding_strs.values())
            min_bond_score = min(bonding_strs.values())
            rgb_interp = get_colour_interpolation_fn(
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                min_bond_score,
                max_bond_score
            )
        for group in [['HBondDonor-P', hbonds.prot_don_id],
                      ['HBondAccept-P', hbonds.prot_acc_id]]:
            if not len(group[1]) == 0:
                self.select_by_ids(group[0], group[1], restrict=self.protname)
        for group in [['HBondDonor-L', hbonds.lig_don_id],
                      ['HBondAccept-L', hbonds.lig_acc_id]]:
            if not len(group[1]) == 0:
                self.select_by_ids(group[0], group[1], restrict=self.ligname)

        if atom_blind:
            resis = []
            if isinstance(bonding_strs, dict):
                for idx, (identifier, bond_score) in enumerate(
                        bonding_strs.items()):
                    atom_1_id, atom_2_id = identifier.split('-')

                    atom_1_chunks = atom_1_id.split(':')
                    atom_2_chunks = atom_2_id.split(':')
                    if len(atom_1_chunks) == 3:
                        resi_1, resn_1, name_1 = atom_1_chunks
                        resi_2, resn_2, name_2 = atom_2_chunks
                        chain_1, chain_2 = None, None
                    elif not len(atom_1_chunks[0].strip()):
                        _, resi_1, resn_1, name_1 = atom_1_chunks
                        _, resi_2, resn_2, name_2 = atom_2_chunks
                        chain_1, chain_2 = None, None
                    else:
                        chain_1, resi_1, resn_1, name_1 = atom_1_chunks
                        chain_2, resi_2, resn_2, name_2 = atom_2_chunks

                    atom_1_id = atom_1_id.replace(':', 'Z')
                    atom_2_id = atom_2_id.replace(':', 'Z')

                    resis += [resi_1, resi_2]

                    atom_1_sele = 'resi {0} & resn {1} & name {2}'.format(
                            resi_1, resn_1, name_1)
                    atom_2_sele = 'resi {0} & resn {1} & name {2}'.format(
                            resi_2, resn_2, name_2)
                    if chain_1 is not None and chain_2 is not None:
                        atom_1_sele += ' & chain ' + chain_1
                        atom_2_sele += ' & chain ' + chain_2

                    cmd.select(atom_1_id, atom_1_sele)
                    cmd.select(atom_2_id, atom_2_sele)

                    x1, y1, z1 = cmd.get_model(atom_1_id).get_coord_list()[0]
                    x2, y2, z2 = cmd.get_model(atom_2_id).get_coord_list()[0]

                    if inverse_colour:
                        interp_score = bond_score
                    else:
                        interp_score = (max_bond_score +
                                        min_bond_score -
                                        bond_score)

                    col = rgb_interp(interp_score)

                    ps_name = 'PS' + str(idx)
                    obj = [
                        CYLINDER, x1, y1, z1, x2, y2, z2, 0.08, *col, *col]
                    cmd.load_cgo(obj, 'bond' + str(idx))
                    cmd.pseudoatom(ps_name, label='{:.2g}'.format(bond_score))
                    cmd.set('label_color', 'black', ps_name)
                    cmd.set('label_size', 20, ps_name)

            elif isinstance(bonding_strs, (list, tuple)):
                for identifier in bonding_strs:
                    atom_1_id, atom_2_id = identifier.split('-')
                    resi_1, resn_1, name_1 = atom_1_id.split(':')
                    resi_2, resn_2, name_2 = atom_2_id.split(':')
                    resis += [resi_1, resi_2]

                    cmd.select(
                        'atom1', 'resi {0} & resn {1} & name {2}'.format(
                            resi_1, resn_1, name_1))
                    cmd.select(
                        'atom2', 'resi {0} & resn {1} & name {2}'.format(
                            resi_2, resn_2, name_2))
                    cmd.distance(identifier, 'atom1', 'atom2')
                    cmd.set('dash_color', [0, 0, 255], identifier)
            else:
                return None
            return resis

        for i in hbonds.ldon_id:
            cmd.select('tmp_bs', 'id %i & %s' % (i[0], self.protname))
            cmd.select('tmp_lig', 'id %i & %s' % (i[1], self.ligname))
            cmd.iterate('tmp_lig', 'print_info(resi, resn, name)',
                        space={'print_info': print_info})
            cmd.iterate('tmp_bs', 'print_info(resi, resn, name)',
                        space={'print_info': print_info})
            bond_str = '-'.join(bonding_strs_holder)
            if bonding_strs is None or (isinstance(bonding_strs, (list, tuple))
                                        and bond_str in bonding_strs):
                cmd.distance('HBonds', 'tmp_bs', 'tmp_lig')
            elif isinstance(bonding_strs, dict):
                bond_score = bonding_strs.get(bond_str)
                if bond_score is not None:
                    atom_1_id, atom_2_id = bond_str.split('-')

                    resi_1, resn_1, name_1 = atom_1_id.split(':')
                    resi_2, resn_2, name_2 = atom_2_id.split(':')

                    atom_1_id = atom_1_id.replace(':', 'Z')
                    atom_2_id = atom_2_id.replace(':', 'Z')

                    cmd.select(
                        atom_1_id, 'resi {0} & resn {1} & name {2}'.format(
                            resi_1, resn_1, name_1))
                    cmd.select(
                        atom_2_id, 'resi {0} & resn {1} & name {2}'.format(
                            resi_2, resn_2, name_2))

                    x1, y1, z1 = cmd.get_model(atom_1_id).get_coord_list()[0]
                    x2, y2, z2 = cmd.get_model(atom_2_id).get_coord_list()[0]
                    if inverse_colour:
                        interp_score = bond_score
                        suffix = ' A'
                    else:
                        interp_score = (max_bond_score +
                                        min_bond_score -
                                        bond_score)
                        suffix = ''

                    col = rgb_interp(interp_score)
                    obj = [
                        CYLINDER, x1, y1, z1, x2, y2, z2, 0.05, *col, *col]
                    cmd.load_cgo(obj, bond_str)
                    cmd.pseudoatom(
                        bond_str + '_label', label='{0:.2g}{1}'.format(
                            bond_score, suffix))
                    cmd.set('label_color', 'black', bond_str + '_label')
            bonding_strs_holder = []
        for i in hbonds.pdon_id:
            cmd.select('tmp_bs', 'id %i & %s' % (i[1], self.protname))
            cmd.select('tmp_lig', 'id %i & %s' % (i[0], self.ligname))
            cmd.iterate('tmp_lig', 'print_info(resi, resn, name)',
                        space={'print_info': print_info})
            cmd.iterate('tmp_bs', 'print_info(resi, resn, name)',
                        space={'print_info': print_info})
            bond_str = '-'.join(bonding_strs_holder)
            if bonding_strs is None or (isinstance(bonding_strs, (list, tuple))
                                        and bond_str in bonding_strs):
                cmd.distance('HBonds', 'tmp_bs', 'tmp_lig')
            elif isinstance(bonding_strs, dict):
                bond_score = bonding_strs.get(bond_str)
                if bond_score is not None:
                    atom_1_id, atom_2_id = bond_str.split('-')

                    resi_1, resn_1, name_1 = atom_1_id.split(':')
                    resi_2, resn_2, name_2 = atom_2_id.split(':')

                    atom_1_id = atom_1_id.replace(':', 'Z')
                    atom_2_id = atom_2_id.replace(':', 'Z')

                    cmd.select(
                        atom_1_id, 'resi {0} & resn {1} & name {2}'.format(
                            resi_1, resn_1, name_1))
                    cmd.select(
                        atom_2_id, 'resi {0} & resn {1} & name {2}'.format(
                            resi_2, resn_2, name_2))

                    x1, y1, z1 = cmd.get_model(atom_1_id).get_coord_list()[0]
                    x2, y2, z2 = cmd.get_model(atom_2_id).get_coord_list()[0]
                    if inverse_colour:
                        interp_score = bond_score
                    else:
                        interp_score = (max_bond_score +
                                        min_bond_score -
                                        bond_score)

                    col = rgb_interp(interp_score)
                    obj = [
                        CYLINDER, x1, y1, z1, x2, y2, z2, 0.05, *col, *col]
                    cmd.load_cgo(obj, bond_str)
                    cmd.pseudoatom(
                        bond_str + '_label', label='{:.2g}'.format(bond_score))
                    cmd.set('label_color', 'black', bond_str + '_label')

            bonding_strs_holder = []
        if self.object_exists('HBonds'):
            cmd.set('dash_color', 'blue', 'HBonds')
        return None

    def score_atoms(
            self, parser, only_process, model, attribution_fn, model_args,
            quiet=False, gnn_layer=None, pdb_file=None,
            coords_to_identifier=None):

        def calc_xtal_dist(s):
            coords = s.split(';')
            p1 = np.array([float(i) for i in coords[0].split(':')])
            p2 = np.array([float(i) for i in coords[1].split(':')])
            return np.linalg.norm(p2 - p1)

        def find_identifier(coords):
            try:
                return coords_to_identifier[coords]
            except KeyError:
                x, y, z = coords.split(':')
                x_int, x_dec = x.split('.')
                y_int, y_dec = y.split('.')
                z_int, z_dec = z.split('.')
                max_dec = max(len(x_dec), len(y_dec), len(z_dec))
                x = x_int + '.' + x_dec + '0' * (max_dec - len(x_dec))
                y = y_int + '.' + y_dec + '0' * (max_dec - len(y_dec))
                z = z_int + '.' + z_dec + '0' * (max_dec - len(z_dec))
                for i in range(3):
                    try:
                        return coords_to_identifier[':'.join((x, y, z))]
                    except KeyError:
                        x += '0'
                        y += '0'
                        z += '0'
                raise KeyError('Cannot find coords ', coords)

        def find_bond(both_coords):
            coords = both_coords.split(';')
            try:
                atom_1 = find_identifier(coords[0])
                atom_2 = find_identifier(coords[1])
            except KeyError:
                return '-'
            bond = [atom_1, atom_2]
            bond.sort()
            return '-'.join(bond)

        bs = model_args['batch_size']
        radius = model_args['radius']
        polar_hydrogens = model_args['hydrogens']
        compact = model_args['compact']
        use_atomic_numbers = model_args['use_atomic_numbers']
        prune = model_args.get('prune', False)

        triplet_code = self.plcomplex.uid.split(':')[0]
        if len(only_process) and triplet_code not in only_process:
            return None, None, None, None

        df = parser.mol_calculate_interactions(
            self.plcomplex.mol, self.plcomplex.pli)

        if not quiet:
            print('Attributing scores to site:', self.plcomplex.uid)

        # CHANGE RELATIVE_TO_LIGAND TO CORRECT ARGUMENT
        df = make_box(df, radius=radius, relative_to_ligand=True)
        if not polar_hydrogens:
            df = df[df['atomic_number'] > 1]

        if isinstance(model, PNNGeometricBase):
            edge_radius = model_args.get('edge_radius', 4)
            if model_args.get('estimate_bonds', False):
                intra_radius = 2.0
            else:
                intra_radius = edge_radius
            df, edge_indices, edge_attrs = generate_edges(
                df, inter_radius=edge_radius, intra_radius=intra_radius,
                prune=prune)
        else:
            edge_indices, edge_attrs = None, None

        if model is None:
            return -1, df, edge_indices, None

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

        coords = np.vstack([df.x, df.y, df.z]).T
        p = torch.from_numpy(coords).float()
        p = repeat(p, 'n d -> b n d', b=1)

        m = torch.from_numpy(np.ones((len(df),))).bool()
        m = repeat(m, 'n -> b n', b=1)

        v = make_bit_vector(
            df.types.to_numpy(), max_feature_id + 1, compact).float()
        v = repeat(v, 'n d -> b n d', b=1)

        model = model.eval().cuda()
        if isinstance(model, PNNGeometricBase):

            edge_indices = torch.from_numpy(np.vstack(edge_indices)).long()
            edge_attrs = one_hot(torch.from_numpy(edge_attrs).long(), 3)

            pre_activation = model(get_pyg_single_graph_for_inference(Data(
                x=v.squeeze(),
                edge_index=edge_indices,
                edge_attr=edge_attrs,
                pos=p.squeeze()
            )))

        else:
            pre_activation = model((p.cuda(), v.cuda(), m.cuda()))[0, ...]
            edge_indices = None

        score = float(to_numpy(torch.sigmoid(pre_activation)))

        if not quiet or 1:
            print('Original score: {:.4}'.format(score))

        model_labels = attribution_fn(
            model, p.cuda(), v.cuda(), m.cuda(), edge_attrs=edge_attrs,
            edge_indices=edge_indices, bs=bs, gnn_layer=gnn_layer)

        df['any_interaction'] = df['hba'] | df['hbd'] | df['pistacking']
        if attribution_fn in (edge_attention, edge_embedding_attribution):
            edge_scores = model_labels
        else:
            df['attribution'] = model_labels
            edge_scores = None
        if coords_to_identifier is not None:
            if edge_indices is not None and edge_scores is not None:
                edge_indices = to_numpy(edge_indices)
                lig_rec_indices = np.where(edge_attrs[:, 1])
                edge_indices = edge_indices[:, lig_rec_indices].squeeze()
                edge_scores = edge_scores[lig_rec_indices]

                x, y, z = df['x'].to_numpy(), df['y'].to_numpy(), df[
                    'z'].to_numpy()
                coord_strs = {}
                for idx, (x_, y_, z_) in enumerate(zip(x, y, z)):
                    coord_strs[idx] = ':'.join([str(i) for i in (x_, y_, z_)])
                df = pd.DataFrame({
                    'edge_idx_0': edge_indices[0, :],
                    'edge_idx_1': edge_indices[1, :],
                })
                df['coords_0'] = df['edge_idx_0'].map(coord_strs)
                df['coords_1'] = df['edge_idx_1'].map(coord_strs)
                df['both_coords'] = df['coords_0'] + ';' + df[
                    'coords_1']
                df['bond_identifier'] = df['both_coords'].apply(
                    find_bond)
                df['bond_score'] = edge_scores
                #mean_bond_scores = defaultdict(list)
                mean_bond_scores = defaultdict(int)
                for bond_id, bond_score in zip(
                        df['bond_identifier'], df['bond_score']):
                    mean_bond_scores[bond_id] = max(
                        mean_bond_scores[bond_id], bond_score)
                    #mean_bond_scores[bond_id].append(bond_score)
                #for key, value in mean_bond_scores.items():
                #    mean_bond_scores[key] = np.mean(value)
                df['bond_score'] = df['bond_identifier'].map(mean_bond_scores)
                df.drop_duplicates(
                    subset='bond_identifier', keep='first', inplace=True)
                df.drop(
                    index=df[df['bond_identifier'] == '-'].index, inplace=True)

                df['xtal_distance'] = df['both_coords'].apply(
                    calc_xtal_dist)
                df['identifier_0'] = df['coords_0'].apply(
                    find_identifier)
                df['identifier_1'] = df['coords_1'].apply(
                    find_identifier)

                for col in ['coords_0', 'coords_1', 'both_coords', 'edge_idx_0',
                            'edge_idx_1']:
                    del df[col]
                df.sort_values(
                    by='xtal_distance', ascending=True, inplace=True)
                df['bond_length_rank'] = np.arange(len(df))
                df.sort_values(
                    by='bond_score', ascending=False, inplace=True)
                df['gnn_rank'] = np.arange(len(df))
            else:
                df['coords'] = df['x'].apply(str) + ':' + df['y'].apply(str) + ':' + df['z'].apply(str)
                df['atom_id'] = df['coords'].apply(find_identifier)
                del df['coords']
                df.sort_values(by='attribution', ascending=False, inplace=True)
                df['gnn_rank'] = np.arange(len(df))

        return score, df, edge_indices, edge_scores

    def colour_b_factors_pdb(
            self, model, parser, attribution_fn, results_fname, model_args,
            gnn_layer=None, only_process=None, pdb_file=None,
            coords_to_identifier=None):

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

        score, df, edge_indices, edge_scores = self.score_atoms(
            parser, only_process, model, attribution_fn, model_args,
            gnn_layer=gnn_layer, pdb_file=pdb_file,
            coords_to_identifier=coords_to_identifier)

        if edge_scores is not None:
            return score, df, edge_indices, edge_scores
        elif not isinstance(df, DataFrame):
            return None, None, None, None
        if model is None:
            return score, df, None, None

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
        return score, df, None, None
