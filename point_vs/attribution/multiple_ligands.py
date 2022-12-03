import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pymol import cmd

from point_vs import logging
from point_vs.attribution.attribution import attribute
from point_vs.attribution.attribution import pdb_coords_to_identifier
from point_vs.dataset_generation.types_to_parquet import StructuralFileParser
from point_vs.models.load_model import load_model
from point_vs.utils import expand_path
from point_vs.utils import print_df
from point_vs.utils import mkdir


LOG = logging.get_logger('PointVS')
AA_TRIPLET_CODES = {'ILE', 'GLU', 'CYS', 'TRP', 'ALA', 'PRO', 'PHE', 'ASN',
                    'GLY', 'THR', 'ARG', 'MET', 'HIS', 'VAL', 'GLN', 'TYR',
                    'LYS', 'LEU', 'SER', 'ASP'}
VDW_RADII = {1: 1.1, 2: 1.4, 3: 1.82, 4: 1.53, 5: 1.92, 6: 1.7, 7: 1.55,
             8: 1.52, 9: 1.47, 10: 1.54, 11: 2.27, 12: 1.73, 13: 1.84, 14: 2.1,
             15: 1.8, 16: 1.8, 17: 1.75, 18: 1.88, 19: 2.75, 20: 2.31, 21: 2.15,
             22: 2.11, 23: 2.07, 24: 2.06, 25: 2.05, 26: 2.04, 27: 2.0,
             28: 1.97, 29: 1.96, 30: 2.01, 31: 1.87, 32: 2.11, 33: 1.85,
             34: 1.9, 35: 1.85, 36: 2.02, 37: 3.03, 38: 2.49, 39: 2.32,
             40: 2.23, 41: 2.18, 42: 2.17, 43: 2.16, 44: 2.13, 45: 2.1, 46: 2.1,
             47: 2.11, 48: 2.18, 49: 1.93, 50: 2.17, 51: 2.06, 52: 2.06,
             53: 1.98, 54: 2.16, 55: 3.43, 56: 2.68, 57: 2.43, 58: 2.42,
             59: 2.4, 60: 2.39, 61: 2.38, 62: 2.36, 63: 2.35, 64: 2.34,
             65: 2.33, 66: 2.31, 67: 2.3, 68: 2.29, 69: 2.27, 70: 2.26,
             71: 2.24, 72: 2.23, 73: 2.22, 74: 2.18, 75: 2.16, 76: 2.16,
             77: 2.13, 78: 2.13, 79: 2.14, 80: 2.23, 81: 1.96, 82: 2.02,
             83: 2.07, 84: 1.97, 85: 2.02, 86: 2.2, 87: 3.48, 88: 2.83,
             89: 2.47, 90: 2.45, 91: 2.43, 92: 2.41, 93: 2.39, 94: 2.4}


def get_mol_to_hbond_map(pdb, lig_name=None):
    coords_to_atom_id = pdb_coords_to_identifier(pdb)
    pdb_parser = StructuralFileParser('receptor')
    mol = pdb_parser.read_file(pdb)[0]
    res = {}
    for atom in mol:
        if atom.atomicnum == 1 or atom.OBAtom.GetResidue() is None:
            continue
        if lig_name is None:
            if atom.OBAtom.GetResidue().GetName() not in AA_TRIPLET_CODES:
                continue
        elif atom.OBAtom.GetResidue().GetName() != lig_name:
            continue

        # (x, y, z) -> A:XXXX:YYY:NAME
        identifier = find_identifier(coords_to_atom_id, atom.coords)
        if atom.OBAtom.IsHbondAcceptor() and atom.OBAtom.IsHbondDonor():
            res[identifier] = 'hbda'
        elif atom.OBAtom.IsHbondAcceptor():
            res[identifier] = 'hba'
        elif atom.OBAtom.IsHbondDonor():
            res[identifier] = 'hbd'
        else:
            res[identifier] = 'none'
    return res


def find_identifier(coords_to_identifier, coords):
    """Find the unique PDB atom identifier given the atom's coordinates.

    The need for this function arises from the difficulty when using floating
    points as dictionary keys.

    Arguments:
        coords_to_identifier: dictionary mapping from (x, y, z) tuples of
            coordinates to the identifying PDB string, of the format
            CHAIN:RESIDUE_NUMBER:RESIDUE_NAME:ATOM_NAME, for example
            A:1021:PHE:O
        coords: tuple of three floats with atomic coordinates of desired atom

    Returns:
         Unique PDB identifier for atom with input coords.
    """
    if not isinstance(coords, str):
        coords = ':'.join([str(coord) for coord in coords])
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


def binding_events_to_ranked_protein_atoms(
        fnames, model_path, output_dir, ligand_name, layer=1):
    """Use multiple protein-ligand structures to score protein atoms.

    The importance of each protein atom is assumed to be related to its mean
    importance across multiple binding events with different ligands. Each
    atom's importance per structure is taken to the maximum GNN attention
    score across all its connected edges in that structure.

    Arguments:
        fnames: return-delimited text file containing filenames of PDB
            structures, which in turn should contain only ligands in the binding
            site of interest. All ligands should have the same residue name. The
            same protein should be present in each structure, although the
            conformation may vary; the residue numbering system should be
            identical within the binding site across sturctures where possible.
        model_path: location of pytorch saved GNN weights file
        output_dir: directory in which results should be stored
        ligand_name: residue name of the ligand (same across all inputs)
        layer: which layer from the GNN to take attention weights from for
            scoring

    Returns:
        pd.DataFrame with columns:
            atom_name: name of protein atom, with format
                CHAIN:RESIDUE_NUMBER:RESIDUE_NAME:ATOM_NAME, for example
                A:1021:PHE:O
            mean_score: mean GNN score of protein atom across all structures
            gnn_rank: rank from 0 to n of atom importance according to mean
                GNN attention score.
    """

    def find_lig_pharm(x):
        if x < 0:
            return 'hbd'
        elif x > 0:
            return 'hba'
        return 'none'

    def find_protein_atom(bond_identifier):
        id_1, id_2 = bond_identifier.split('-')
        if id_1.split(':')[-2] == ligand_name:
            return id_2
        elif id_2.split(':')[-2] == ligand_name:
            return id_1
        LOG.warning(
            f'Protein triplet code not found in either atom {bond_identifier}.')
        return 'none'

    def find_ligand_atom(bond_identifier):
        id_1, id_2 = bond_identifier.split('-')
        if id_1.split(':')[-2] == ligand_name:
            return id_1
        elif id_2.split(':')[-2] == ligand_name:
            return id_2
        LOG.warning(
            f'Ligand triplet code not found in either atom {bond_identifier}.')
        return 'none'

    _, model, _, _ = load_model(expand_path(model_path))

    processed_dfs = []
    prot_atom_to_max_lig_atom = defaultdict(list)
    for fname in fnames:
        lig_to_hbond_map = get_mol_to_hbond_map(fname, lig_name=ligand_name)
        pro_to_hbond_map = get_mol_to_hbond_map(fname, lig_name=None)
        dfs = attribute(
            'edge_attention', model_path, expand_path(output_dir),
            input_file=fname, only_process=ligand_name, write_stats=False,
            gnn_layer=layer, write_pse=False, atom_blind=True,
            loaded_model=model,
            quiet=True
        )
        scores = []
        for site_code, (score, df, _, _) in dfs.items():
            # if site_code.split(':')[1] != 'A':
            #    continue
            scores.append(score)
            df['protein_atom'] = df['bond_identifier'].apply(find_protein_atom)
            df['ligand_atom'] = df['bond_identifier'].apply(find_ligand_atom)
            df = df[df['protein_atom'] != 'none']
            df = df[df['ligand_atom'] != 'none']
            protein_atom_dfs = []
            for protein_atom in list(set(df['protein_atom'])):
                sub_df = df[df['protein_atom'] == protein_atom].reset_index(
                    drop=True)

                max_idx = np.argmax(sub_df['protein_atom'].to_numpy())
                max_row = sub_df.iloc[[max_idx]].copy()

                max_score = max_row['bond_score'].to_numpy()[0]
                lig_pharm = lig_to_hbond_map[find_ligand_atom(
                    max_row['bond_identifier'].to_list()[0])]
                pro_pharm = pro_to_hbond_map[find_protein_atom(
                    max_row['bond_identifier'].to_list()[0])]

                if lig_pharm == 'hba':
                    lig_pharm_score = [max_score]
                elif lig_pharm == 'hbd':
                    lig_pharm_score = [-max_score]
                elif lig_pharm == 'hbda':
                    lig_pharm_score = [max_score, -max_score]
                else:
                    lig_pharm_score = [0]

                prot_atom_to_max_lig_atom[protein_atom] += lig_pharm_score
                max_row['pro_pharm'] = [pro_pharm]
                max_row['lig_pharm'] = [lig_pharm]
                protein_atom_dfs.append(max_row)

            f_index = len(processed_dfs)
            df = pd.concat(protein_atom_dfs)
            df.sort_values(by='bond_score', inplace=True, ascending=False)
            df = df.rename(
                columns={'bond_score': 'bond_score_{}'.format(f_index),
                         'xtal_distance': 'xtal_distance_{}'.format(f_index),
                         'ligand_atom': 'ligand_atom_{}'.format(f_index),
                         'lig_pharm': 'lig_pharm_{}'.format(f_index),
                         'pro_pharm': 'pro_pharm_{}'.format(f_index)})
            df.reset_index(inplace=True, drop=True)

            for col in ('bond_length_rank', 'bond_identifier', 'identifier_0',
                        'identifier_1', 'gnn_rank'):
                del df[col]
            df['gnn_rank_{}'.format(f_index)] = range(len(df))
            df.reset_index(drop=True, inplace=True)
            processed_dfs.append(df)

        scores_str = ', '.join(['{:.3f}'.format(score) for score in scores])
        LOG.info(
            f'Completed file {fname.name}, with scores: {scores_str}')

    f_index = len(processed_dfs)
    processed_dfs = [df.set_index('protein_atom') for df in processed_dfs]
    concat_df = processed_dfs[0].join(processed_dfs[1:])
    concat_df['mean_gnn_rank'] = concat_df[
        ['gnn_rank_{}'.format(x) for x in range(f_index)]].mean(axis=1)
    concat_df['mean_bond_score'] = concat_df[
        ['bond_score_{}'.format(x) for x in range(f_index)]].mean(axis=1)
    concat_df.sort_values(by='mean_gnn_rank', ascending=True, inplace=True)
    
    return concat_df


def bond_rank_correlation(df, cutoff=3.2):
    n_structures = len(df.columns) // 6
    atom_dict = defaultdict(lambda: defaultdict(list))
    for n in range(n_structures):
        subdf = pd.DataFrame({
            s: df[s] for s in df.columns if s.find('_{}'.format(n)) != -1})
        subdf = subdf[subdf['xtal_distance_{}'.format(n)] < cutoff]
        atoms = subdf.index
        rank_scores = subdf['gnn_rank_{}'.format(n)]
        bond_scores = subdf['bond_score_{}'.format(n)]
        for atom, rank_score, bond_score in zip(
                atoms, rank_scores, bond_scores):
            atom_dict[atom]['bond_scores'].append(bond_score)
            atom_dict[atom]['gnn_ranks'].append(rank_score)
    res = []
    for atom, d in atom_dict.items():
        res.append((atom,
                    len(d['bond_scores']),
                    np.mean(d['bond_scores']),
                    np.var(d['bond_scores']),
                    np.mean(d['gnn_ranks']),
                    np.var(d['gnn_ranks'])))
    res = pd.DataFrame(
        res, columns=['atom', 'occurences', 'mean_score', 'var_score',
                      'mean_rank', 'var_rank'])
    res = res.sort_values(by='occurences', ascending=False)
    res = res.reset_index(drop=True)
    return atom_dict, res


def rename_lig(fname, output_fname=None, ligname='LIG'):
    fname = str(expand_path(fname))
    cmd.load(fname)
    cmd.remove('solvent')
    cmd.select('ligand', 'HETATM')
    cmd.alter('ligand', 'resn="{}"'.format(ligname))
    cmd.save(fname if output_fname is None else str(expand_path(output_fname)))
    cmd.remove('all')
    cmd.delete('all')
    cmd.reset()


def read_inputs(input_fnames, output_dir, ligand_name='LIG', clean_pdbs=False):
    input_fnames = expand_path(input_fnames)
    if input_fnames.is_file():
        with open(expand_path(input_fnames), 'r') as f:
            fnames = [expand_path(fname.strip()) for fname in
                      f.readlines() if len(fname.strip())]
    else:
        fnames = list(input_fnames.glob('*.pdb'))
    if clean_pdbs:
        output_dir = mkdir(Path(output_dir) / 'cleaned_pdbs')
        clean_fnames = []
        for fname in fnames:
            new_fname = Path(
                output_dir, fname.with_suffix('').name + '_fixed.pdb')
            # new_fname = Path(str(fname.with_suffix('')) + '_fixed.pdb')
            rename_lig(fname, new_fname, ligand_name)
            clean_fnames.append(new_fname)
        fnames = clean_fnames
    return fnames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', type=str,
                        help='Return-delimited text file containing PDB '
                             'filenames')
    parser.add_argument('apo_protein', type=str,
                        help='Structure of protein for which pharmacophores '
                             'are generated')
    parser.add_argument('model', type=str,
                        help='Filename of pytorch saved model checkpoint')
    parser.add_argument('ligand_residue_code', type=str,
                        help='PDB residue triplet code for ligand')
    parser.add_argument('output_dir', type=str,
                        help='Location in which to store results')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='Which GNN layer to take attention weights from')
    parser.add_argument('--clean_pdbs', '-c', action='store_true',
                        help='Rename ligands to ligand_residue_code and '
                             'renumber residues')
    parser.add_argument('--distance_threshold', '-d', type=float, default=3.2,
                        help='Include interactions within --distance_threshold '
                             'Angstroms of the protein atom')

    args = parser.parse_args()
    pd.set_option('display.float_format', lambda x: '%.3g' % x)

    fnames = read_inputs(
        args.filenames, args.output_dir,
        ligand_name=args.ligand_residue_code.upper(),
        clean_pdbs=args.clean_pdbs)
    rank_df = binding_events_to_ranked_protein_atoms(
        fnames, args.model, args.output_dir,
        args.ligand_residue_code.upper(), layer=args.layer)

    brc, df = bond_rank_correlation(rank_df, cutoff=args.distance_threshold)

    print_df(df)
    df.to_csv(expand_path(args.output_dir) / 'results_summary.csv')
    rank_df.to_csv(expand_path(args.output_dir) / 'results.csv')
