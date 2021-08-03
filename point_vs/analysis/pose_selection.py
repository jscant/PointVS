"""Gather stats on pose selection performance on a validation set."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from point_vs.analysis.ranking import Ranking
from point_vs.utils import load_yaml


def parse_results(
        predictions_fname_or_sdf_root, rmsd_info=None, rmsd_info_fname=None):
    """Parse results stored in text format.

    Arguments:
        predictions_fname_or_sdf_root: location of the results (either a text
            file containing predictions or the root of a directory containing
            docking results in sdf format)
        rmsd_info: dicts which map from pdbids to dicts which map from indices
            to RMSD values
        rmsd_info_fname: file containing yaml dict in the format specified for
            rmsd_info
    Returns:
        Ranking object containing various statistics.
    """

    def extract_energies(sdf):
        """Return {index: minimizedAffinity} for each docked item in an sdf."""
        energies = {}
        record_next = False
        with open(Path(sdf).expanduser(), 'r') as f:
            for line in f.readlines():
                if line.startswith('> <minimizedAffinity>'):
                    record_next = True
                    continue
                if record_next:
                    energies[len(energies)] = float(line.strip())
                    record_next = False
        return energies

    assert not (rmsd_info is None and rmsd_info_fname is None)

    if rmsd_info_fname is not None:
        with open(Path(rmsd_info_fname).expanduser(), 'r') as f:
            rmsd_info = yaml.load(f, Loader=yaml.FullLoader)

    predictions_fname_or_sdf_root = Path(
        predictions_fname_or_sdf_root).expanduser()
    if predictions_fname_or_sdf_root.is_file():
        df = pd.read_csv(predictions_fname_or_sdf_root, sep=' ',
                         names=['y_true', '|', 'y_pred', 'rec', 'lig'])
        y_true = list(df.y_true)
        y_pred = list(df.y_pred)
        recs = list(df.rec)
        ligs = [Path(ligname).name.split('.')[0] for ligname in list(df.lig)]

        pdbid_to_scores_and_rmsds = defaultdict(list)
        top_ranked_rmsds = []
        for i in range(len(df)):
            pdbid = Path(recs[i]).name.split('.')[0]
            rmsd_info_record = rmsd_info[pdbid]
            if ligs[i].startswith('minimised'):
                continue
            rmsd = rmsd_info_record['docked_wrt_crystal'][int(
                ligs[i].split('_')[-1])]
            pdbid_to_scores_and_rmsds[recs[i]].append(
                (y_true[i], y_pred[i], rmsd))

        sorted_scores_and_rmsds_lst = []
        for rec, lst in pdbid_to_scores_and_rmsds.items():
            sorted_scores_and_rmsds = sorted(
                lst, key=lambda x: x[1], reverse=True)
            sorted_scores_and_rmsds = np.array(sorted_scores_and_rmsds)
            sorted_scores_and_rmsds_lst.append(sorted_scores_and_rmsds)

            top_ranked_rmsd = sorted_scores_and_rmsds[0][2]
            top_ranked_rmsds.append(top_ranked_rmsd)
    elif predictions_fname_or_sdf_root.is_dir():
        sorted_scores_and_rmsds_lst = []
        for docked_sdf in Path(predictions_fname_or_sdf_root).expanduser().glob(
                '**/docked_poses.sdf'):
            try:
                rmsds = rmsd_info[docked_sdf.parent.name]['docked_wrt_crystal']
            except KeyError:
                continue
            docked_energies = extract_energies(docked_sdf)
            combined = np.array(sorted([(
                0, docked_energies[key], rmsds[key]) for key in
                docked_energies.keys()], key=lambda x: x[1]))
            combined[:, 0] = combined[:, 2] < 2
            sorted_scores_and_rmsds_lst.append(np.array(combined))
    else:
        raise FileNotFoundError(
            str(predictions_fname_or_sdf_root) + ' does not exist.')

    return Ranking(predictions_fname_or_sdf_root, sorted_scores_and_rmsds_lst)


def plot_top_n(label_to_ranking, max_n=10, threshold_rmsd=2.0):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel('RSMD threshold (Angstroms)')
    ax.set_ylabel(
        'Fraction of top-ranked poses within RMSD\nthreshold of relaxed xtal pose')
    ax.set_title(
        'Fraction of top-ranked poses within given cutoff of relaxed xtal pose')
    x_rng = range(1, max_n + 1)
    for label, ranking in label_to_ranking.items():
        top_n = []
        for n in x_rng:
            top_n.append(ranking.get_top_n(n, threshold_rmsd))
        ax.plot(x_rng, top_n, '-x', label=label)
    ax.set_xlabel('N')
    ax.set_ylabel('TopN')
    ax.set_title(
        'Fraction of top-ranked poses within {} A of relaxed xtal pose'.format(
            threshold_rmsd))
    ax.set_ylim([0, 1])
    ax.set_xlim([1, max_n])
    ax.grid()
    ax.legend()
    return fig, ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rmsd_info', type=str,
                        help='Yaml file containing RMSD information')
    parser.add_argument('results', type=str, nargs='+',
                        help='Results of PointVS model inference in text '
                             'format, or directory containing Smina docking '
                             'results.')
    parser.add_argument('--threshold_rmsd', '-t', type=float, default=2.0,
                        help='Threshold for the definition of a "good" dock')
    parser.add_argument('--n', '-n', type=int, default=10,
                        help='Maximum TopN to plot')

    args = parser.parse_args()
    rmsd_info = load_yaml(args.rmsd_info)
    label_to_ranking = {}
    for fname in args.results:
        label_to_ranking[Path(fname).parent.name] = parse_results(
            fname, rmsd_info=rmsd_info)
    _, ax = plot_top_n(label_to_ranking, args.n, args.threshold_rmsd)
    plt.show()
    plt.savefig('topn.png')
