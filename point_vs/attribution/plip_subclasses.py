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

from point_vs import logging
from point_vs.global_objects import DEVICE
from point_vs.attribution.attribution_fns import edge_attention, \
    track_bond_lengths, node_attention, \
    attention_wrapper, cam_wrapper, atom_masking, bond_masking, masking_wrapper
from point_vs.attribution.interaction_parser import StructuralInteractionParser
from point_vs.models.geometric.pnn_geometric_base import PNNGeometricBase
from point_vs.models.point_neural_network_base import to_numpy
from point_vs.preprocessing.preprocessing import make_bit_vector, make_box, \
    generate_edges
from point_vs.preprocessing.pyg_single_item_dataset import \
    get_pyg_single_graph_for_inference
from point_vs.utils import coords_to_string, PositionDict, \
    get_colour_interpolation_fn


LOG = logging.get_logger('PointVS')


class VisualizerDataWithMolecularInfo(VisualizerData):
    """VisualizerData but with the mol, ligand and pli objects stored."""

    def __init__(self, mol, site):
        pli = mol.interaction_sets[site]
        self.ligand = pli.ligand
        self.pli = pli
        self.mol = mol
        super().__init__(mol, site)


class PyMOLVisualizerWithBFactorColouring(PyMOLVisualizer):

    def show_hbonds_original(self):
        """Visualizes hydrogen bonds."""
        hbonds = self.plcomplex.hbonds
        for group in [['HBondDonor-P', hbonds.prot_don_id],
                      ['HBondAccept-P', hbonds.prot_acc_id]]:
            if not len(group[1]) == 0:
                self.select_by_ids(group[0], group[1], restrict=self.protname)
        for group in [['HBondDonor-L', hbonds.lig_don_id],
                      ['HBondAccept-L', hbonds.lig_acc_id]]:
            if not len(group[1]) == 0:
                self.select_by_ids(group[0], group[1], restrict=self.ligname)
        for i in hbonds.ldon_id:
            cmd.select('tmp_bs', 'id %i & %s' % (i[0], self.protname))
            cmd.select('tmp_lig', 'id %i & %s' % (i[1], self.ligname))
            cmd.distance('HBonds', 'tmp_bs', 'tmp_lig')
        for i in hbonds.pdon_id:
            cmd.select('tmp_bs', 'id %i & %s' % (i[1], self.protname))
            cmd.select('tmp_lig', 'id %i & %s' % (i[0], self.ligname))
            cmd.distance('HBonds', 'tmp_bs', 'tmp_lig')
        if self.object_exists('HBonds'):
            cmd.set('dash_color', 'blue', 'HBonds')

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

                    atom_1_id = atom_1_id.replace(':', '_')
                    atom_2_id = atom_2_id.replace(':', '_')

                    resis += [resi_1, resi_2]

                    alphabet = [''] + [
                        char for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']

                    for char in alphabet:
                        atom_1_sele = 'resi {0} & resn {1} & name {2}'.format(
                            resi_1 + char, resn_1, name_1)
                        if chain_1 is not None:
                            atom_1_sele += ' & chain ' + chain_1

                        cmd.select(atom_1_id, atom_1_sele)
                        try:
                            x1, y1, z1 = cmd.get_model(
                                atom_1_id).get_coord_list()[0]
                        except IndexError:
                            pass
                        else:
                            break

                    for char in alphabet:
                        atom_2_sele = 'resi {0} & resn {1} & name {2}'.format(
                            resi_2 + char, resn_2, name_2)
                        if chain_2 is not None:
                            atom_2_sele += ' & chain ' + chain_2

                        cmd.select(atom_2_id, atom_2_sele)
                        try:
                            x2, y2, z2 = cmd.get_model(
                                atom_2_id).get_coord_list()[0]
                        except IndexError:
                            pass
                        else:
                            break

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
            self, only_process, model, model_args, attribution_fn,
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
        extended = model_args.get('extended_atom_types', False)

        triplet_code = self.plcomplex.uid.split(':')[0]
        if len(only_process) and triplet_code not in only_process:
            return None, None, None, None

        edge_radius = model_args.get('edge_radius', 4)
        if model_args.get('estimate_bonds', False):
            intra_radius = 2.0
        else:
            intra_radius = edge_radius

        parser = StructuralInteractionParser()

        df = parser.mol_calculate_interactions(
            self.plcomplex.mol, self.plcomplex.pli)

        if not polar_hydrogens:
            df = df[df['atomic_number'] > 1]

        # CHANGE RELATIVE_TO_LIGAND TO CORRECT ARGUMENT
        df = make_box(df, radius=radius, relative_to_ligand=True)

        if attribution_fn == 1:
            df_, edge_indices, edge_attrs = generate_edges(
                df, inter_radius=edge_radius, intra_radius=intra_radius,
                prune=prune)
            dfs, edge_indices_, edge_attrs_, resis = [df], [edge_indices], [edge_attrs], [-2]
            for resi in set(df['resi']):
                if resi < 0:
                    continue
                df_ = df[df['resi'] != resi].copy()
                df_, edge_indices, edge_attrs = generate_edges(
                    df_, inter_radius=edge_radius, intra_radius=intra_radius,
                    prune=prune)
                dfs.append(df_)
                edge_indices_.append(edge_indices)
                edge_attrs_.append(edge_attrs)
                resis.append(resi)

        LOG.info(f'Attributing scores to site: {self.plcomplex.uid}')

        if not polar_hydrogens:
            df = df[df['atomic_number'] > 1]

        if attribution_fn != 1:
            # CHANGE RELATIVE_TO_LIGAND TO CORRECT ARGUMENT
            df = make_box(df, radius=radius, relative_to_ligand=True)

            if isinstance(model, PNNGeometricBase):
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
                raise NotImplementedError('Hydrogens temporarily disabled.')
            else:
                max_feature_id = 10 + 8 * extended

            coords = np.vstack([df.x, df.y, df.z]).T

            p = torch.from_numpy(coords).float()
            p = repeat(p, 'n d -> b n d', b=1)

            m = torch.from_numpy(np.ones((len(df),))).bool()
            m = repeat(m, 'n -> b n', b=1)

            v = make_bit_vector(
                df.types.to_numpy(), max_feature_id + 1, compact).float()

            v = repeat(v, 'n d -> b n d', b=1)

            model = model.eval().to(DEVICE)

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
                pre_activation = model((p.to(DEVICE), v.to(DEVICE), m.to(DEVICE)))[0, ...]
                edge_indices = None

            if model.model_task == 'classification':
                score = float(to_numpy(torch.sigmoid(pre_activation)))
                if not quiet:
                    LOG.info('Original score: {:.4f}'.format(score))
            elif model.model_task == 'multi_regression':
                score = to_numpy(pre_activation).squeeze()
                if not quiet:
                    LOG.info(
                        'Original scores: pKi={0:.4f} ({3:.1f} nm), pKd={1:.4f} '
                        '({4:.1f} nm), pIC50={2:.4f} ({5:.1f} nm)'.format(
                        *score, *[10 ** (-x) * 10 ** 9 for x in score]))
                score = score[0]
            else:
                score = float(to_numpy(pre_activation))
                if not quiet:
                    LOG.info('Original score: {:.4f}'.format(score))
            model_labels = attribution_fn(
                model, p.to(DEVICE), v.to(DEVICE), edge_attrs=edge_attrs,
                edge_indices=edge_indices, bs=bs, gnn_layer=gnn_layer,
                resis=df['resi'].to_numpy())

        else:
            residue_scores = {}
            for idx, (df_, edge_indices, edge_attrs, resi) in enumerate(zip(
                    dfs, edge_indices_, edge_attrs_, resis)):
                if use_atomic_numbers:
                    # H C N O F P S Cl
                    recognised_atomic_numbers = (6, 7, 8, 9, 15, 16, 17)
                    # various metal ions/halogens which share valence properties
                    other_groupings = (
                    (35, 53), (3, 11, 19), (4, 12, 20), (26, 29, 30))
                    atomic_number_to_index = {
                        num: idx for idx, num in
                        enumerate(recognised_atomic_numbers)
                    }
                    for grouping in other_groupings:
                        atomic_number_to_index.update({elem: max(
                            atomic_number_to_index.values()) + 1 for elem in
                                                       grouping})
                    if polar_hydrogens:
                        atomic_number_to_index.update({
                            1: max(atomic_number_to_index.values()) + 1
                        })

                    # +1 to accommodate for unmapped elements
                    max_feature_id = max(atomic_number_to_index.values()) + 1

                    # Any other elements not accounted for given a category of
                    # their own
                    atomic_number_to_index = defaultdict(lambda: max_feature_id)
                    atomic_number_to_index.update(atomic_number_to_index)
                    df_.types = df_['atomic_number'].map(
                        atomic_number_to_index) + df_.bp * (max_feature_id + 1)

                elif polar_hydrogens:
                    raise NotImplementedError('Hydrogens temporarily disabled.')
                else:
                    max_feature_id = 10 + 8 * extended

                coords = np.vstack([df_.x, df_.y, df_.z]).T

                p = torch.from_numpy(coords).float()
                p = repeat(p, 'n d -> b n d', b=1)

                m = torch.from_numpy(np.ones((len(df_),))).bool()
                m = repeat(m, 'n -> b n', b=1)

                v = make_bit_vector(
                    df_.types.to_numpy(), max_feature_id + 1, compact).float()

                v = repeat(v, 'n d -> b n d', b=1)

                model = model.eval().to(DEVICE)

                edge_indices = torch.from_numpy(np.vstack(edge_indices)).long()
                edge_attrs = one_hot(torch.from_numpy(edge_attrs).long(), 3)

                pre_activation = model(get_pyg_single_graph_for_inference(Data(
                    x=v.squeeze(),
                    edge_index=edge_indices,
                    edge_attr=edge_attrs,
                    pos=p.squeeze()
                )))

                if model.model_task == 'classification':
                    score = float(to_numpy(torch.sigmoid(pre_activation)))
                    if not quiet:
                        LOG.info('Original score: {:.4f}'.format(score))
                elif model.model_task == 'multi_regression':
                    score = to_numpy(pre_activation).squeeze()
                    if not quiet:
                        LOG.info(
                            'Original scores: pKi={0:.4f} ({3:.1f} nm), '
                            'pKd={1:.4f} '
                            '({4:.1f} nm), pIC50={2:.4f} ({5:.1f} nm)'.format(
                                *score, *[10 ** (-x) * 10 ** 9 for x in score]))
                    score = score[0]
                else:
                    score = float(to_numpy(pre_activation))
                    if not quiet:
                        LOG.info('Original score: {:.4f}'.format(score))

                if not idx:
                    original_score = score
                    atomic_model_labels = attribution_fn(
                        model, p.to(DEVICE), v.to(DEVICE), edge_attrs=edge_attrs,
                        edge_indices=edge_indices, bs=bs, gnn_layer=gnn_layer)
                else:
                    residue_scores[resi] = original_score - score

            model_labels = []
            for resi, ml in zip(df['resi'], atomic_model_labels):
                if resi < 0:
                    model_labels.append(ml)
                else:
                    model_labels.append(ml / 2 + residue_scores[resi] / 2)


        df['any_interaction'] = df['hba'] | df['hbd'] | df['pistacking']
        if attribution_fn in (edge_attention, track_bond_lengths, bond_masking):
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
                mean_bond_scores = defaultdict(list)
                for bond_id, bond_score in zip(
                        df['bond_identifier'], df['bond_score']):
                    mean_bond_scores[bond_id].append(bond_score)
                for key, value in mean_bond_scores.items():
                    mean_bond_scores[key] = np.mean(value)
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
                    # if col in ('edge_idx_0', 'edge_idx_1', 'both_coords'):
                    del df[col]

                df.sort_values(
                    by='xtal_distance', ascending=True, inplace=True)
                df['bond_length_rank'] = np.arange(len(df))
                df.sort_values(
                    by='bond_score', ascending=False, inplace=True)
                df['gnn_rank'] = np.arange(len(df))
            else:
                df['coords'] = df['x'].apply(str) + ':' + df['y'].apply(
                    str) + ':' + df['z'].apply(str)
                df['atom_id'] = df['coords'].apply(find_identifier)
                del df['coords']
                LOG.info(df)
                df.sort_values(by='attribution', ascending=False, inplace=True)
                df['gnn_rank'] = np.arange(len(df))

        return score, df, edge_indices, edge_scores

    def colour_b_factors_pdb(
            self, model, parser, attribution_fn, results_fname, model_args,
            gnn_layer=None, only_process=None, pdb_file=None,
            coords_to_identifier=None, quiet=False, split_by_mol=True):

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

        scoring_args = (only_process, model, model_args)
        scoring_kwargs = {
            'gnn_layer': gnn_layer,
            'pdb_file': pdb_file,
            'coords_to_identifier': coords_to_identifier,
            'quiet': quiet,
        }

        if attribution_fn == attention_wrapper:
            score, edge_df, edge_edge_indices, edge_edge_scores = \
                self.score_atoms(
                    *scoring_args, **scoring_kwargs,
                    attribution_fn=edge_attention)
            score, df, node_edge_indices, node_edge_scores = \
                self.score_atoms(
                    *scoring_args, **scoring_kwargs,
                    attribution_fn=node_attention)
        elif attribution_fn == cam_wrapper:
            score, edge_df, edge_edge_indices, edge_edge_scores = \
                self.score_atoms(
                    *scoring_args, **scoring_kwargs,
                    attribution_fn=edge_attention)
            score, df, node_edge_indices, node_edge_scores = \
                self.score_atoms(
                    *scoring_args, **scoring_kwargs,
                    attribution_fn=atom_masking)
        elif attribution_fn == masking_wrapper:

            _, atom_mask_df, _, _ = self.score_atoms(
                *scoring_args, **scoring_kwargs,
                attribution_fn=atom_masking)
            score, edge_df, edge_indices, edge_scores = \
                self.score_atoms(
                    *scoring_args, **scoring_kwargs,
                    attribution_fn=bond_masking)

            LOG.info(atom_mask_df.sort_values(by='attribution'))
            LOG.info(edge_df.sort_values(by='bond_score', ascending=False))

            atom_id_to_score = {aid: score for aid, score in zip(
                atom_mask_df['atom_id'], atom_mask_df['attribution'])}

            edge_df['atom_1_bs'] = edge_df['identifier_0'].map(
                atom_id_to_score)
            edge_df['atom_2_bs'] = edge_df['identifier_1'].map(
                atom_id_to_score)
            edge_df[
                'bond_score'] -= (edge_df['atom_1_bs'] + edge_df['atom_2_bs'])
            del edge_df['atom_1_bs'], edge_df['atom_2_bs']
            edge_df = edge_df.sort_values(by='bond_score', ascending=False)
            edge_df['gnn_rank'] = range(len(edge_df))

            return score, edge_df, edge_indices, edge_scores
        else:
            edge_df = None
            score, df, edge_indices, edge_scores = self.score_atoms(
                *scoring_args, **scoring_kwargs,
                attribution_fn=attribution_fn)

            if edge_scores is not None:
                return score, df, edge_indices, edge_scores
            if not isinstance(df, DataFrame):
                return None, None, None, None
            if model is None:
                return score, df, None, None

        model_labels = df['attribution'].to_numpy()
        bps = df['bp'].to_numpy()
        x, y, z = df['x'].to_numpy(), df['y'].to_numpy(), df['z'].to_numpy()

        atom_to_bfactor_map = PositionDict(eps=0.01)
        all_bfactors = []
        lig_bfactors = []
        rec_bfactors = []
        for i in range(len(df)):
            bfactor = float(model_labels[i])
            all_bfactors.append(bfactor)
            if bps[i]:
                rec_bfactors.append(bfactor)
            else:
                lig_bfactors.append(bfactor)
            atom_to_bfactor_map[coords_to_string((x[i], y[i], z[i]))] = bfactor

        if attribution_fn in (attention_wrapper, node_attention):
            left_bfactor_limit = min(all_bfactors)
            right_bfactor_limit = max(all_bfactors)
            left_bfactor_limit_lig = min(lig_bfactors)
            right_bfactor_limit_lig = max(lig_bfactors)
            left_bfactor_limit_rec = min(rec_bfactors)
            right_bfactor_limit_rec = max(rec_bfactors)
            # left_bfactor_limit = 0
            # right_bfactor_limit = 1
            # colour_scheme = 'red_white_green'
            colour_scheme = 'white_green'
        else:
            min_bfactor = min(min(all_bfactors), 0)
            max_bfactor = max(max(all_bfactors), 0)
            min_bfactor_lig = min(min(lig_bfactors), 0)
            max_bfactor_lig = max(max(lig_bfactors), 0)
            min_bfactor_rec = min(min(rec_bfactors), 0)
            max_bfactor_rec = max(max(rec_bfactors), 0)
            colour_scheme = 'red_white_green'
            max_absolute_bfactor = max(abs(max_bfactor), abs(min_bfactor))
            max_absolute_bfactor_lig = max(
                abs(max_bfactor_lig), abs(min_bfactor_lig))
            max_absolute_bfactor_rec = max(
                abs(max_bfactor_rec), abs(min_bfactor_rec))

            if isinstance(max_absolute_bfactor, (list, np.ndarray)):
                max_absolute_bfactor = max_absolute_bfactor[0]
                max_absolute_bfactor_lig = max_absolute_bfactor_lig[0]
                max_absolute_bfactor_rec = max_absolute_bfactor_rec[0]

            left_bfactor_limit = -max_absolute_bfactor
            right_bfactor_limit = max_absolute_bfactor
            left_bfactor_limit_lig = -max_absolute_bfactor_lig
            right_bfactor_limit_lig = max_absolute_bfactor_lig
            left_bfactor_limit_rec = -max_absolute_bfactor_rec
            right_bfactor_limit_rec = max_absolute_bfactor_rec

        if not quiet:
            if split_by_mol:
                LOG.info('Colouring ligand b-factors in range ({0:0.3f}, '
                      '{1:0.3f})'.format(
                    left_bfactor_limit_lig, right_bfactor_limit_lig))
                LOG.info('Colouring receptor b-factors in range ({0:0.3f}, '
                      '{1:0.3f})'.format(
                    left_bfactor_limit_rec, right_bfactor_limit_rec))
            else:
                LOG.info(
                    'Colouring b-factors in range ({0:0.3f}, {1:0.3f})'.format(
                        left_bfactor_limit, right_bfactor_limit))

        df.to_csv(results_fname, index=False, float_format='%.3f')

        cmd.alter('all', "b=0")
        if not quiet:
            LOG.info('Changing bfactors...')
        change_bfactors(atom_to_bfactor_map)
        if not quiet:
            LOG.info('Done!')
        if split_by_mol:
            cmd.spectrum('b', colour_scheme, selection='name is LIG',
                         minimum=left_bfactor_limit_lig,
                         maximum=right_bfactor_limit_lig)
            cmd.spectrum('b', colour_scheme, selection='name is not LIG',
                         minimum=left_bfactor_limit_rec,
                         maximum=right_bfactor_limit_rec)
        else:
            cmd.spectrum(
                'b', colour_scheme, minimum=left_bfactor_limit,
                maximum=right_bfactor_limit)
        cmd.rebuild()
        if edge_df is not None:
            df = edge_df
        return score, df, None, None
