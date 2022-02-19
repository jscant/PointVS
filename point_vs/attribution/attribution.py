"""Driver script for various graph attribution methods."""
import argparse
import urllib
from pathlib import Path

import matplotlib
import pandas as pd
from plip.basic import config
from plip.basic.supplemental import extract_pdbid
from plip.exchange.webservices import fetch_pdb
from sklearn.metrics import average_precision_score, \
    precision_recall_curve

from point_vs.attribution.attribution_fns import masking, cam, \
    edge_embedding_attribution, edge_attention, node_attention
from point_vs.attribution.process_pdb import score_and_colour_pdb
from point_vs.models.load_model import load_model
from point_vs.utils import ensure_writable, expand_path
from point_vs.utils import mkdir

matplotlib.use('agg')

from matplotlib import pyplot as plt

ALLOWED_METHODS = ('masking', 'cam')


def download_pdb_file(pdbid, output_dir):
    """Given a PDB ID, downloads the corresponding PDB structure.
    Checks for validity of ID and handles error while downloading.
    Returns the path of the downloaded file (From PLIP)"""
    pdbid = pdbid.lower()
    output_dir = Path(output_dir).expanduser()
    pdbpath = output_dir / '{}.pdb'.format(pdbid)
    if pdbpath.is_file():
        print(pdbpath, 'already exists.')
        return pdbpath
    if len(pdbid) != 4 or extract_pdbid(
            pdbid.lower()) == 'UnknownProtein':
        raise RuntimeError('Unknown protein ' + pdbid)
    while True:
        try:
            pdbfile, pdbid = fetch_pdb(pdbid.lower())
        except urllib.error.URLError:
            print('Fetching pdb {} failed, retrying...'.format(
                pdbid))
        else:
            break
    if pdbfile is None:
        return 'none'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(pdbpath, 'w') as g:
        g.write(pdbfile)
    print('File downloaded as', pdbpath)
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
    print('Average precision (random classifier): {:.3f}'.format(
        random_average_precision))
    print('Average precision (neural network)   : {:.3f}'.format(
        average_precision))
    precision, recall, thresholds = precision_recall_curve(
        interactions, normalised_attributions)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
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
              quiet=False):
    if pdbid is None:
        leaf_dir = Path(Path(input_file).name).stem
        output_dir = mkdir(
            Path(output_dir, leaf_dir))
        pdbpath = Path(input_file).expanduser()
    else:
        output_dir = mkdir(Path(output_dir, attribution_type, pdbid))
        pdbpath = download_pdb_file(pdbid, output_dir)

    conf_lines = has_multiple_conformations(pdbpath)
    if len(conf_lines):
        print()
        print('WARNING:', pdbpath, 'contains multiple conformations!',
              'Multiconf line indices:')
        print(*conf_lines)
        print()
    coords_to_identifier = pdb_coords_to_identifier(pdbpath)
    attribution_fn = {'masking': masking,
                      'cam': cam,
                      'node_attention': node_attention,
                      'edge_attention': edge_attention,
                      'edges': edge_embedding_attribution
                      }.get(attribution_type, None)

    model, model_kwargs, cmd_line_args = load_model(
        model_file,
        fetch_args_only=attribution_fn is None or loaded_model is not None)
    if loaded_model is not None:
        model = loaded_model

    dfs = score_and_colour_pdb(
        model=model, attribution_fn=attribution_fn,
        pdbfile=str(pdbpath), outpath=str(output_dir),
        model_args=cmd_line_args, gnn_layer=gnn_layer,
        only_process=only_process if only_process is not None else [],
        bonding_strs=bonding_strs, write_pse=write_pse,
        override_attribution_name=override_attribution_name,
        atom_blind=atom_blind, inverse_colour=inverse_colour,
        pdb_file=pdb_file, coords_to_identifier=coords_to_identifier,
        quiet=quiet)
    precision_str = 'pdbid,lig:chain:res,rnd_avg_precision,' \
                    'model_avg_precision,score\n'
    if write_stats:
        if attribution_type != 'edge_attention':
            try:
                for lig_id, (
                        score, df, edge_indices, edge_scores) in dfs.items():
                    r_ap, ap = precision_recall(
                        df,
                        save_path=Path(output_dir,
                                       'precision_recall_{}.png'.format(
                                           lig_id.replace(':', '_'))))
                    precision_str += '{0},{1},{2:.4f},{3:.4f},{4:.4f}\n'.format(
                        pdbid, lig_id, r_ap, ap, score)
                    print()
                with open(output_dir / 'average_precisions.txt', 'w') as f:
                    f.write(precision_str)
            except KeyError:
                pass
        else:
            for lig_id, (score, df, edge_indices, edge_scores) in dfs.items():
                df.to_csv(output_dir / '{}_edge_attributions.csv'.format(
                    lig_id.replace(':', '-')
                ))
    return dfs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('attribution_type', type=str,
                        help='Method of graph attribution; just {} for '
                             'now.'.format(', '.join(ALLOWED_METHODS)))
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
    args = parser.parse_args()
    config.NOFIX = True
    if isinstance(args.pdbid, str) + isinstance(args.input_file, str) != 1:
        raise RuntimeError(
            'Specify exactly one of either --pdbid or --input_file.')
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    print(*[item[1][:10] for item in attribute(
        args.attribution_type, args.model, args.output_dir, pdbid=args.pdbid,
        input_file=args.input_file, only_process=args.only_process,
        atom_blind=True, gnn_layer=args.gnn_layer).values()], sep='\n\n')
