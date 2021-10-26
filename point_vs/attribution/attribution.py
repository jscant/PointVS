"""Driver script for various graph attribution methods."""
import argparse
import urllib
from pathlib import Path

import matplotlib
from plip.basic.supplemental import extract_pdbid
from plip.exchange.webservices import fetch_pdb
from sklearn.metrics import average_precision_score, \
    precision_recall_curve

from point_vs.attribution.attribution_fns import masking, cam, \
    attention_attribution, edge_embedding_attribution
from point_vs.attribution.process_pdb import score_and_colour_pdb
from point_vs.models.load_model import load_model
from point_vs.utils import mkdir, ensure_writable

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


def attribute(args):
    if args.pdbid is not None:
        output_dir = mkdir(
            Path(args.output_dir, args.attribution_type, args.pdbid))
        pdbpath = download_pdb_file(args.pdbid, output_dir)
    else:
        leaf_dir = Path(Path(args.input_file).name).stem
        output_dir = mkdir(
            Path(args.output_dir, args.attribution_type, leaf_dir))
        pdbpath = Path(args.input_file).expanduser()

    model, model_kwargs, cmd_line_args = load_model(args.model)

    attribution_fn = {'masking': masking,
                      'cam': cam,
                      'attention': attention_attribution,
                      'edges': edge_embedding_attribution,
                      }[args.attribution_type]

    dfs = score_and_colour_pdb(model, attribution_fn,
                               str(pdbpath), str(output_dir),
                               model_args=cmd_line_args,
                               only_process=args.only_process)
    precision_str = 'pdbid,lig:chain:res,rnd_avg_precision,' \
                    'model_avg_precision,score\n'
    for lig_id, (score, df) in dfs.items():
        r_ap, ap = precision_recall(
            df, save_path=Path(output_dir, 'precision_recall_{}.png'.format(
                lig_id.replace(':', '_'))))
        precision_str += '{0},{1},{2:.4f},{3:.4f},{4:.4f}\n'.format(
            args.pdbid, lig_id, r_ap, ap, score)
        print()
    with open(output_dir / 'average_precisions.txt', 'w') as f:
        f.write(precision_str)


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
    args = parser.parse_args()
    if isinstance(args.pdbid, str) + isinstance(args.input_file, str) != 1:
        raise RuntimeError(
            'Specify exactly one of either --pdbid or --input_file.')
    attribute(args)
