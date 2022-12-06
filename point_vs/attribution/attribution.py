"""Driver script for various graph attribution methods."""
import argparse
import logging
import urllib
from pathlib import Path
from point_vs import logging

import matplotlib
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from plip.basic import config
from plip.basic.supplemental import extract_pdbid
from plip.exchange.webservices import fetch_pdb

from point_vs.attribution.attribution_fns import atom_masking, cam, \
    edge_attention, node_attention, \
    replace_coords, track_position_changes, track_bond_lengths, cam_wrapper, \
    attention_wrapper, bond_masking, masking_wrapper
from point_vs.attribution.process_pdb import score_and_colour_pdb
from point_vs.models.load_model import load_model
from point_vs.models.geometric.egnn_multitask import MultitaskSatorrasEGNN
from point_vs.utils import ensure_writable, expand_path, mkdir, rename_lig, \
    find_latest_checkpoint

matplotlib.use('agg')

from matplotlib import pyplot as plt

ALLOWED_METHODS = ('atom_masking', 'cam')
LOG = logging.get_logger('PointVS')


def download_pdb_file(pdbid, output_dir):
    """Given a PDB ID, downloads the corresponding PDB structure.
    Checks for validity of ID and handles error while downloading.
    Returns the path of the downloaded file (From PLIP)"""
    pdbid = pdbid.lower()
    output_dir = Path(output_dir).expanduser()
    pdbpath = output_dir / '{}.pdb'.format(pdbid)
    if pdbpath.is_file():
        LOG.warning(f'{pdbpath} already exists.')
        return pdbpath
    if len(pdbid) != 4 or extract_pdbid(
            pdbid.lower()) == 'UnknownProtein':
        raise RuntimeError('Unknown protein ' + pdbid)
    while True:
        try:
            pdbfile, pdbid = fetch_pdb(pdbid.lower())
        except urllib.error.URLError:
            LOG.warning(f'Fetching pdb {pdbid} failed, retrying...'.format)
        else:
            break
    if pdbfile is None:
        return 'none'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(pdbpath, 'w', encoding='utf-8') as g:
        g.write(pdbfile)
    LOG.info(f'File downloaded as {pdbpath}')
    return pdbpath


def precision_recall(df, save_path=None):
    attributions = df['attribution'].to_numpy()
    attribution_range = max(attributions) - min(attributions)
    normalised_attributions = (attributions -
                               min(attributions)) / attribution_range
    interactions = df['any_interaction'].to_numpy()
    # interactions = (df['hbd'] | df['hba']).to_numpy()
    random_average_precision = sum(interactions) / len(interactions)
    average_precision = average_precision_score(
        interactions, normalised_attributions)
    LOG.info('Average precision (random classifier): {:.3f}'.format(
        random_average_precision))
    LOG.info('Average precision (neural network)   : {:.3f}'.format(
        average_precision))
    precision, recall, thresholds = precision_recall_curve(
        interactions, normalised_attributions)
    _, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(recall, precision, 'k-')
    ax.set_title('Precision-recall plot')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    if save_path is not None:
        ensure_writable(save_path)
        plt.savefig(str(Path(save_path).expanduser()), dpi=100)
    return random_average_precision, average_precision


def pdb_coords_to_identifier(pdb_file, include_chain=True):
    """Read .pdb file and return map from x,y,z -> chain:resi:resn:name"""

    res = {}
    with open(expand_path(pdb_file), 'r') as f:
        for line in f.readlines():
            if not line.startswith('HETATM') and not line.startswith(
                    'ATOM'):
                continue
            x = line[30:38].strip()
            y = line[38:46].strip()
            z = line[46:54].strip()
            chain = line[21].strip()
            resn = line[17:20].strip()
            resi = line[22:26].strip()
            name = line[12:16].strip()
            if include_chain:
                identifiers = [chain, resi, resn, name]
            else:
                identifiers = [resi, resn, name]
            if resn.lower() != 'hoh':
                res[':'.join([x, y, z])] = ':'.join(identifiers)
    return res


def has_multiple_conformations(pdb_file):
    conf_lines = []
    with open(expand_path(pdb_file), 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if not line.startswith('HETATM') and not line.startswith('ATOM'):
                continue
            if len(line) < 60:
                continue
            if line[56:60].strip() != '1.00':
                conf_lines.append(idx + 1)
    return conf_lines


def attribute(attribution_type, model_file, output_dir, pdbid=None,
              input_file=None, only_process=None, write_stats=True,
              gnn_layer=None, bonding_strs=None, write_pse=True,
              override_attribution_name=None, atom_blind=False,
              inverse_colour=False, pdb_file=None, loaded_model=None,
              quiet=False, track_atom_positions=False, check_multiconf=True,
              prediction_mode='classification',
              rename=False, only_first=False, split_by_mol=True):
    config.NOFIX = True
    if attribution_type in ('edge_attention', 'node_attention'):
        dir_name = '{0}_{1}'.format(attribution_type, gnn_layer)
    else:
        dir_name = attribution_type
    if pdbid is None:
        leaf_dir = Path(Path(input_file).name).stem
        output_dir = mkdir(Path(output_dir, leaf_dir, dir_name))
        pdbpath = Path(input_file).expanduser()
    else:
        output_dir = mkdir(Path(output_dir, pdbid, dir_name))
        pdbpath = download_pdb_file(pdbid, output_dir)

    if check_multiconf:
        conf_lines = has_multiple_conformations(pdbpath)
        if len(conf_lines):
            LOG.warning('WARNING: {0} contains multiple conformations! Multiconf line indices: {1}'.format(
                pdbpath, ' '.join([str(s) for s in conf_lines])))

    if rename:
        only_process = 'LIG'
        new_pdb_path = Path(
            output_dir, pdbpath.with_suffix('').name + '_fixed.pdb')
        rename_lig(pdbpath, new_pdb_path.parent, handle_conformers='discard',
                   atom_count_threshold=6)
        pdbpath = new_pdb_path

    coords_to_identifier = pdb_coords_to_identifier(pdbpath)
    attribution_fn = {'atom_masking': atom_masking,
                      'cam': cam,
                      'node_attention': node_attention,
                      'edge_attention': edge_attention,
                      'displacement': track_position_changes,
                      'bond_lengths': track_bond_lengths,
                      'attention': attention_wrapper,
                      'class_activation': cam_wrapper,
                      'bond_masking': bond_masking,
                      'masking': masking_wrapper
                      }.get(attribution_type, None)

    if expand_path(model_file).is_dir():
        model_file = find_latest_checkpoint(model_file)

    _, model, _, cmd_line_args = load_model(
        model_file, silent=False,
        fetch_args_only=attribution_fn is None or loaded_model is not None)
    if loaded_model is not None:
        model = loaded_model
    if isinstance(model, MultitaskSatorrasEGNN):
        model.set_task(prediction_mode)

    dfs = score_and_colour_pdb(
        model=model, attribution_fn=attribution_fn,
        pdbfile=str(pdbpath), outpath=str(output_dir),
        model_args=cmd_line_args, gnn_layer=gnn_layer,
        only_process=only_process if only_process is not None else [],
        bonding_strs=bonding_strs, write_pse=write_pse,
        override_attribution_name=override_attribution_name,
        atom_blind=atom_blind, inverse_colour=inverse_colour,
        pdb_file=pdb_file, coords_to_identifier=coords_to_identifier,
        split_by_mol=split_by_mol,
        quiet=quiet, only_first=only_first, extended=cmd_line_args.get(
            'extended_atom_types', False))

    if track_atom_positions:
        positions_path = mkdir(output_dir / 'atom_positions')
        original_coords = model.layers[0].intermediate_coords
        for layer in range(1, cmd_line_args['layers'] + 1):
            imc = model.layers[layer].intermediate_coords
            replace_coords(
                pdbpath, Path(positions_path, 'layer_{}.pdb'.format(layer)),
                original_coords, imc)

    precision_str = 'pdbid,lig:chain:res,rnd_avg_precision_lig,rnd_agv_precision_rec' \
                    'model_avg_precision_lig,model_avg_precision_rec,score\n'
    if write_stats:
        if attribution_type in ('edge_attention', 'bond_masking', 'masking'):
            for lig_id, (score, df, edge_indices, edge_scores) in dfs.items():
                df.to_csv(output_dir / '{}_results.csv'.format(
                    lig_id.replace(':', '_').replace(
                        ' ', '').replace('__', '_')
                ))
        else:
            try:
                for lig_id, (
                        score, df, edge_indices, edge_scores) in dfs.items():
                    r_aps, aps = [], []
                    for bp in (0, 1):
                        r_ap, ap = precision_recall(
                            df[df['bp'] == bp],
                            save_path=Path(
                                output_dir, 'precision_recall_{}.png'.format(
                                    lig_id.replace(':', '_').replace(
                                        ' ', '').replace('__', '_'))))
                        r_aps.append(r_ap)
                        aps.append(ap)
                    precision_str += '{0},{1},{2:.4f},{3:.4f},{4:.4f},{5:.4f},{6:.4f}\n'.format(
                        pdbid, lig_id, r_aps[0], r_aps[1], aps[0], aps[1], score)
                with open(output_dir / 'average_precisions.txt',
                          'w', encoding='utf-8') as f:
                    f.write(precision_str)
            except (KeyError, ValueError):
                pass

    return dfs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('attribution_type', type=str,
                        help='Method of graph attribution; just '
                        f'{",".join(ALLOWED_METHODS)} for now.')
    parser.add_argument('model', type=str, help='Saved pytorch model')
    parser.add_argument('output_dir', type=str,
                        help='Directory in which to store results')
    parser.add_argument('--pdbid', '-p', type=str,
                        help='PDB ID for structure to analyse')
    parser.add_argument('--input_file', '-i', type=str,
                        help='Input PDB file')
    parser.add_argument('--only_process', '-o', type=str, nargs='?', default=[],
                        help='Only process ligands with the given 3 letter '
                             'residue codes (UNK, for example)')
    parser.add_argument('--gnn_layer', '-l', type=int, default=1)
    parser.add_argument('--track_atom_positions', '-t', action='store_true',
                        help='Record atom positions as information flows '
                             'through GNN, in PDB format')
    parser.add_argument('--rename', '-r', action='store_true',
                        help='Rename ligands to LIG, and only attribute to '
                             'the first one found')
    parser.add_argument('--only_first', '-f', action='store_true',
                        help='Only process first instance of ligand in PDB '
                             'file')
    parser.add_argument('--split_by_mol', '-s', action='store_true',
                        help='Treat receptor and ligand as separate when '
                             'colouring and calculating precision-recall')
    parser.add_argument('--prediction_mode', type=str,
                        default='classsification',
                        help='(For multitask models only) Mode for attribution.'
                        ' Either classification or regression')
    args = parser.parse_args()

    config.NOFIX = True
    if isinstance(args.pdbid, str) + isinstance(args.input_file, str) != 1:
        raise RuntimeError(
            'Specify exactly one of either --pdbid or --input_file.')
    pd.set_option('display.float_format', lambda x: f'{x:.3f}')

    if not (args.only_first and args.split_by_mol):
        for res in attribute(
            args.attribution_type, args.model, args.output_dir, pdbid=args.pdbid,
            input_file=args.input_file, only_process=args.only_process,
            atom_blind=True, gnn_layer=args.gnn_layer,
            track_atom_positions=args.track_atom_positions,
            split_by_mol=args.split_by_mol, rename=args.rename,
            prediction_mode=args.prediction_mode,
            only_first=args.only_first).values():
            df = res[1]
            LOG.info(df[:10])
    else:
        df = list(attribute(
            args.attribution_type, args.model, args.output_dir, pdbid=args.pdbid,
            input_file=args.input_file, only_process=args.only_process,
            atom_blind=True, gnn_layer=args.gnn_layer,
            track_atom_positions=args.track_atom_positions,
            split_by_mol=args.split_by_mol, rename=args.rename,
            prediction_mode=args.prediction_mode,
            only_first=args.only_first).values())[0][1]
        try:
            LOG.info(df)
            LOG.info('Ligand:')
            LOG.info(df[df['bp'] == 0][:10])
            LOG.info('Receptor:')
            LOG.info(df[df['bp'] == 1][:10])
        except KeyError:
            LOG.info(df[:10])
