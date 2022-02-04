import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pymol
from plip.basic import config

from point_vs.attribution.attribution import attribute
from point_vs.attribution.interaction_parser import get_str_repr
from point_vs.utils import execute_cmd, expand_path, mkdir, get_directory_state

# Let's use the MD to decide which HBA-HBD paris are in fact interacting
config.HBOND_DON_ANGLE_MIN = 0
config.HBOND_DIST_MAX = np.inf


# config.PLUGIN_MODE = True


def remove_solvent(input_pdb):
    """Split a PDB file into the receptor and the ligand.

    Arguments:
        input_pdb: combined input structure
    """
    input_pdb = expand_path(input_pdb)
    pymol.cmd.load(input_pdb, 'combined')
    pymol.cmd.remove('resn hoh')
    pymol.cmd.remove('solvent')
    pymol.cmd.remove('metals')
    pymol.cmd.save(input_pdb)
    pymol.cmd.delete('all')


def gromacs_to_pdb(input_file, output_file):
    """Convert a gromacs format file into a pdb file using the command line."""
    input_file = expand_path(input_file)
    output_file = expand_path(output_file)

    # gromacs throws an exception even when it sucessfully converts to pdb (?)
    execute_cmd(
        'editconf -f {0} -o {1}'.format(input_file, output_file),
        raise_exceptions=False)


def attribute_gromacs_file(
        attribution_type, model_file, pdb_file, output_dir,
        only_process='MOL', gnn_layer=None, atom_blind=False):
    if isinstance(only_process, str):
        only_process = [only_process]
    output_dir = mkdir(output_dir)
    protonated_pdb = Path(str(pdb_file).replace('.pdb', '_protonated.pdb')).name
    dir = protonated_pdb.replace('_protonated.pdb', '')
    d = attribute(
        attribution_type=attribution_type, model_file=model_file,
        output_dir=output_dir, input_file=pdb_file,
        only_process=only_process, write_stats=False, gnn_layer=gnn_layer,
        write_pse=False, atom_blind=atom_blind, inverse_colour=False,
        pdb_file=output_dir / dir / protonated_pdb)
    if len(d) != 1:
        raise RuntimeError(
            'Only works with single site - {} sites detected'.format(len(d)))
    for key, value in d.items():
        spl = key.split(':')
        identifier = spl[2] + ':' + spl[0]
        score = value[0]
        df = value[1]
        edge_indices = value[2]
        edge_scores = value[3]
        return identifier, score, df, edge_indices, edge_scores


def parse_gromacs_file(gromacs_file):
    """Read .gro file and return map from x,y,z -> resi:resn:name"""

    def parse_line(s):
        chunks = s.split()
        res_id = chunks[0][:-3]
        res_name = chunks[0][-3:]
        atom_name = chunks[1]
        x, y, z = [10 * float(i) for i in chunks[-3:]]
        return res_id, res_name, atom_name, (x, y, z)

    gromacs_file = expand_path(gromacs_file)
    result = defaultdict(lambda: defaultdict(dict))
    unique_check = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    with open(gromacs_file, 'r') as f:
        for line in f.readlines():
            if len(line.split()) < 6:
                continue
            info = parse_line(line)
            if info[1].lower() != 'hoh':
                x, y, z = info[-1]
                result[get_str_repr(x)][get_str_repr(y)][
                    get_str_repr(z)] = ':'.join(info[:-1])
                unique_check[info[0]][info[1]][info[2]] += 1
                if unique_check[info[0]][info[1]][info[2]] > 1:
                    raise RuntimeError(
                        'Cannot determine unique mapping for {}'.format(
                            gromacs_file))
    return result


def get_identifier_to_attribution_map(
        attribution_type, model_file, pdb_file, gromacs_file,
        output_dir, gnn_layer=None,
        only_process='MOL', atom_blind=False):
    if gnn_layer is None and attribution_type == 'edge_attention':
        raise RuntimeError(
            'gnn_layer must be specified when using edge_attention')
    lig_id, score, df, edge_indices, edge_scores = attribute_gromacs_file(
        attribution_type, model_file, pdb_file, output_dir,
        only_process=only_process, gnn_layer=gnn_layer, atom_blind=atom_blind)
    position_to_identifiers = parse_gromacs_file(gromacs_file)
    id_to_score_map = {}
    if edge_scores is None:
        positions = np.vstack([df.x, df.y, df.z]).T
        attributions = df['attribution']
        for i in range(len(df)):
            identifier = position_to_identifiers[tuple(positions[i, :])]
            id_to_score_map[identifier] = attributions[i]
    else:
        id_to_score_map = {
            identifier: score for identifier, score in zip(df.bond_identifier, df.bond_score)
        }
        d_ = {}
        for key in id_to_score_map.keys():
            key_ = key
            if key_.startswith(':'):
                key_ = key_[1:]
            key_ = key_.replace('-:', '-')
            d_[key_] = id_to_score_map[key]
            d_['-'.join(key_.split('-')[::-1])] = id_to_score_map[key]
        id_to_score_map = d_
        """
        for i in range(len(edge_scores)):
            edge_score = edge_scores[i]
            node_x = edge_indices[0, i]
            node_y = edge_indices[1, i]

            pos_x = [get_str_repr(coord) for coord in positions[node_x, :]]
            pos_y = [get_str_repr(coord) for coord in positions[node_y, :]]

            identifier_x = position_to_identifiers[pos_x[0]][pos_x[1]][pos_x[2]]
            identifier_y = position_to_identifiers[pos_y[0]][pos_y[1]][pos_y[2]]

            id_to_score_map[identifier_x + '-' + identifier_y] = edge_score
            id_to_score_map[identifier_y + '-' + identifier_x] = edge_score
        """
    return id_to_score_map, edge_scores is not None, score


def make_gromacs_df(trajectories_fname, gromacs_fname, only_process='MOL'):
    trajectory_df = pd.read_csv(expand_path(trajectories_fname))
    position_to_identifiers = parse_gromacs_file(gromacs_fname)
    lig_strs = set()
    for x, ydict in position_to_identifiers.items():
        for y, zdict in ydict.items():
            for z, atom_id in zdict.items():
                if atom_id.split(':')[1] == only_process:
                    lig_strs.add(':'.join(atom_id.split(':')[:-1]))
    assert len(lig_strs) == 1, 'Found multiple ligand string prefixes'
    lig_str = list(lig_strs)[0]

    for to_delete in ['Fragment', 'Unnamed: 0']:
        if to_delete in trajectory_df.columns:
            del trajectory_df[to_delete]

    cols = trajectory_df.columns
    time_steps = list(cols)[4:]
    distances = trajectory_df.copy()
    del distances['bs resnumber']
    del distances['bs resname']
    del distances['bs atom type']
    del distances['ligand atom type']
    distances = distances.to_numpy()
    variances = np.var(distances, axis=1)
    mean_dist = np.mean(distances, axis=1)
    trajectory_df['md_mean_distance'] = mean_dist
    trajectory_df['md_var_distance'] = variances

    trajectory_df['rec_identifier'] = (
            trajectory_df['bs resnumber'].astype(str) + ':' +
            trajectory_df['bs resname'] + ':' +
            trajectory_df['bs atom type'])
    trajectory_df['lig_identifier'] = lig_str + ':' + trajectory_df[
        'ligand atom type']
    trajectory_df['bond_identifier'] = trajectory_df['lig_identifier'] + '-' + \
                                       trajectory_df['rec_identifier']

    del trajectory_df['bs resnumber']
    del trajectory_df['bs resname']
    del trajectory_df['bs atom type']
    del trajectory_df['ligand atom type']

    trajectory_df.dropna(inplace=True)
    trajectory_df.head()
    for time_step in time_steps:
        del trajectory_df[str(time_step)]

    trajectory_df.sort_values('md_mean_distance', ascending=True, inplace=True)
    trajectory_df.reset_index(inplace=True, drop=True)

    return lig_str, trajectory_df


def make_vis_md(df, model_file, max_dist, max_var, output_dir, gromacs_file,
                only_process='MOL', atom_blind=False):
    keep_df = df.drop(df[(df.md_mean_distance > max_dist) |
                         (df.md_var_distance > max_var)].index)
    keep_df.reset_index(inplace=True)
    keep_df.drop(index=keep_df[keep_df.index > 4].index, inplace=True)
    del keep_df['index']
    bond_strs = {b_id: dist for b_id, dist in zip(
        keep_df.bond_identifier, keep_df.md_mean_distance)}

    output_dir = mkdir(output_dir)
    pdb_file = output_dir / Path(gromacs_file).name.replace('.gro', '.pdb')
    if pdb_file.exists() and pdb_file.is_file():
        pdb_file.unlink()
    gromacs_to_pdb(gromacs_file, pdb_file)
    remove_solvent(pdb_file)
    protonated_pdb = Path(str(pdb_file).replace('.pdb', '_protonated.pdb')).name
    dir = protonated_pdb.replace('_protonated.pdb', '')

    dfs = list(attribute(
        attribution_type='None',
        model_file=model_file,
        output_dir=output_dir,
        input_file=pdb_file,
        bonding_strs=bond_strs,
        write_stats=False,
        only_process=only_process,
        atom_blind=atom_blind,
        inverse_colour=True,
        pdb_file=output_dir / dir / protonated_pdb).values())

    atom_to_pos_map = {
        atom_id: (x, y, z) for atom_id, x, y, z in zip(
            dfs[0][1].atom_identifier, dfs[0][1].x, dfs[0][1].y, dfs[0][1].z)
    }
    return pdb_file, atom_to_pos_map, dfs[0][1]


def marry_trajectories_with_scores(
        attribution_type, df, model_file, output_dir, pdb_file, lig_str,
        id_to_score_map, atom_to_pos_map, is_edges, only_process='MOL',
        atom_blind=False):
    def get_xtal_dist(bond_id):
        atom_1, atom_2 = bond_id.split('-')
        try:
            atom_1_pos = atom_to_pos_map[atom_1]
            atom_2_pos = atom_to_pos_map[atom_2]
            return np.linalg.norm(np.array(atom_1_pos) - np.array(atom_2_pos))
        except KeyError:
            return -1

    if is_edges:
        df['xtal_distance'] = df['bond_identifier'].apply(get_xtal_dist)
        df['bond_score'] = df['bond_identifier'].map(
            id_to_score_map)
        df.dropna(inplace=True)
        df.sort_values('bond_score', ascending=False, inplace=True)
        df['gnn_bond_rank'] = np.arange(len(df))

        max_bond_score = max(df['bond_score'])
        keep_df = df.drop(df[(df.bond_score < 0.2 * max_bond_score)].index)
        keep_df.reset_index(inplace=True)
        keep_df.drop(index=keep_df[keep_df.index > 4].index, inplace=True)
        if 'index' in keep_df.columns:
            del keep_df['index']
        bond_identifiers = list(keep_df.bond_identifier)
        bond_scores = list(keep_df.bond_score)
        bond_strs = {bid: bs for bid, bs in zip(bond_identifiers, bond_scores)}

        protonated_pdb = Path(
            str(pdb_file).replace('.pdb', '_protonated.pdb')).name
        dir = protonated_pdb.replace('_protonated.pdb', '')
        attribute(
            attribution_type='None', model_file=model_file,
            output_dir=output_dir,
            input_file=pdb_file,
            bonding_strs=bond_strs,
            write_stats=False,
            only_process=only_process,
            override_attribution_name=attribution_type,
            atom_blind=atom_blind,
            inverse_colour=False,
            pdb_file=Path(output_dir) / dir / protonated_pdb
        )
    else:
        id_to_score_list_rec = sorted(
            [(identifier, score) for identifier, score in
             id_to_score_map.items()
             if not identifier.startswith(lig_str)],
            key=lambda x: x[1], reverse=True)
        id_to_score_list_lig = sorted(
            [(identifier, score) for identifier, score in
             id_to_score_map.items()
             if identifier.startswith(lig_str)],
            key=lambda x: x[1], reverse=True)
        id_to_rank_rec = {identifier[0]: rank for rank, identifier in
                          enumerate(id_to_score_list_rec)}
        id_to_rank_lig = {identifier[0]: rank for rank, identifier in
                          enumerate(id_to_score_list_lig)}

        df['rec_score'] = df['rec_identifier'].map(
            id_to_score_map)
        df['lig_score'] = df['lig_identifier'].map(
            id_to_score_map)
        df['lig_rank'] = df['lig_identifier'].map(
            id_to_rank_lig)
        df['rec_rank'] = df['rec_identifier'].map(
            id_to_rank_rec)
        df.dropna(inplace=True)

    cols = list(df.columns.values)
    df = df[
        ['xtal_distance'] + [col for col in cols if col != 'xtal_distance']]

    df.sort_values('md_mean_distance', ascending=True, inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def master(trajectories_file, gromacs_file, model_file, output_dir, gnn_layer,
           only_process='MOL', atom_blind=False):
    lig_str, md_df = make_gromacs_df(trajectories_file, gromacs_file,
                                     only_process=only_process)

    pdb_file, atom_to_pos_map, pos_df = make_vis_md(
        md_df, model_file, 5, 0.5, output_dir, gromacs_file, only_process,
        atom_blind=atom_blind)

    id_to_score_map, is_edges, score = get_identifier_to_attribution_map(
        'edge_attention',
        model_file, pdb_file, gromacs_file,
        output_dir, gnn_layer=gnn_layer, only_process=only_process,
        atom_blind=atom_blind)

    gnn_df = marry_trajectories_with_scores(
        'edge_attention',
        md_df,
        model_file,
        output_dir,
        pdb_file,
        lig_str,
        id_to_score_map,
        atom_to_pos_map,
        True,
        only_process=only_process,
        atom_blind=atom_blind)

    return gnn_df, score


def parse_plip(fname):
    def coords_from_atom_idx(indices):
        indices = [int(i) for i in indices.split(',')]
        s = ''
        if not len(indices):
            return s
        with open(expand_path(fname), 'r') as f:
            for line in f.readlines():
                if not line.startswith('HETATM') and not line.startswith(
                        'ATOM'):
                    continue
                atomid = int(line[6:11])
                if atomid in indices:
                    s += (line[30:38].strip() + ':'
                          + line[38:46].strip() + ':'
                          + line[46:54].strip() + ';')
                    del indices[indices.index(atomid)]
                    if not len(indices):
                        return s[:-1]
        raise RuntimeError(
            'Could not find atoms with indices ' +
            ', '.join([str(i) for i in indices]))

    plip_output = execute_cmd('plip -f {} -O -t'.format(expand_path(fname)),
                              raise_exceptions=False, silent=True)
    plip_output = plip_output.replace('=', '').replace('+', '').replace('|',
                                                                        ' ').replace(
        '--', '')
    plip_output = [line for line in plip_output.split('\n')
                   if not line.startswith('-') and len(line.split())]
    colon_lines = [idx for idx, line in enumerate(plip_output)
                   if line.find(':') != -1]
    if len(colon_lines) > 4:
        plip_output = plip_output[:colon_lines[4]]

    star_indices = []
    for idx, line in enumerate(plip_output):
        if line.startswith('**'):
            star_indices.append(idx)
    star_indices.append(len(plip_output))

    res = {}
    for i in range(len(star_indices) - 1):
        bond_type = plip_output[star_indices[i]].replace(
            '**', '').lower().replace(' ', '_')
        if bond_type == 'water_bridges':
            continue
        data = [line.replace(', ', ',').split() for line in
                plip_output[star_indices[i] + 2:star_indices[i + 1]]]
        cols = plip_output[star_indices[i] + 1]
        if 'LIGCOO' in cols.split():
            lig_idx = cols.split().index('LIGCOO')
            data = [chunks[:lig_idx] + chunks[lig_idx].split(',') +
                    chunks[lig_idx + 1:] for chunks in data]
            rec_idx = cols.split().index('PROTCOO') + 2
            data = [chunks[:rec_idx] + chunks[rec_idx].split(',') +
                    chunks[rec_idx + 1:] for chunks in data]

        cols = cols.replace('LIGCOO', 'LIGX LIGY LIGZ').replace(
            'PROTCOO', 'RECX RECY RECZ')
        cols = cols.split()

        df = pd.DataFrame.from_records(
            [r for r in data], columns=cols)
        if 'PROT_IDX_LIST' in cols:
            df['PROT_IDX_LIST'] = df['PROT_IDX_LIST'].map(coords_from_atom_idx)
            df['LIG_IDX_LIST'] = df['LIG_IDX_LIST'].map(coords_from_atom_idx)

        res[bond_type] = df

    return df


if __name__ == '__main__':
    # fname = 'jacks_distances/data/pdbs/F400-PHIPA-x20910.pdb'
    # parse_plip(fname)
    # raise

    parser = argparse.ArgumentParser()
    parser.add_argument('attribution_type', type=str,
                        help='Type of attribution - one of cam, masking, '
                             'edges or attention')
    parser.add_argument('gromacs_file', type=str,
                        help='Gromacs structural data file')
    parser.add_argument('model_file', type=str,
                        help='Location of trained pytorch neural network '
                             'weights')
    parser.add_argument('trajectories_file', type=str,
                        help='CSV file containing bond lengths as a function '
                             'of time')
    parser.add_argument('output_dir', type=str,
                        help='Directory in which to store results')
    parser.add_argument('--gnn_layer', '-l', type=str, default='-1',
                        help='Index of layer to take attention weights from  '
                             '(starting at 1)')
    parser.add_argument('--pymol', '-p', action='store_true',
                        help='Generate pymol visualisation')
    parser.add_argument('--only_process', '-o', type=str, default='MOL',
                        help='Only process ligands with a given three letter '
                             'code')
    parser.add_argument('--atom_blind', '-a', action='store_true',
                        help='Bonds can be drawn between any atoms, no matter '
                             'their identities, as long as the relevant '
                             'scoring function ranks them highly enough.')
    args = parser.parse_args()
    original_dir_state = get_directory_state(args.output_dir)
    try:
        args.gnn_layer = int(args.gnn_layer)
    except ValueError:
        pass

    gnn_df, score = master(args.trajectories_file, args.gromacs_file,
                           args.model_file,
                           args.output_dir, args.gnn_layer,
                           only_process=args.only_process,
                           atom_blind=args.atom_blind)
    print(gnn_df.sort_values(by='bond_score', ascending=False)[:10])

    gnn_df.to_csv(Path(mkdir(
        args.output_dir, Path(args.gromacs_file).with_suffix('').name),
        '{}_distances_vs_scores.csv'.format(
            Path(args.model_file).parents[1].name)))
    # wipe_new_pdbs(args.output_dir, exempt=original_dir_state)
