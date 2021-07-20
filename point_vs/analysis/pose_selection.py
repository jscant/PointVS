"""Gather stats on pose selection performance on a validation set."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from point_vs.analysis.ranking import Ranking


def parse_results(fname, rmsd_info=None, rmsd_info_fname=None):
    """Parse results stored in text format.

    Arguments:
        fname: location of the results

    Returns:
        Ranking object containing various statistics.
    """
    assert not (rmsd_info is None and rmsd_info_fname is None)
    if rmsd_info_fname is not None:
        with open(Path(rmsd_info_fname).expanduser(), 'r') as f:
            rmsd_info = yaml.load(f, Loader=yaml.FullLoader)

    df = pd.read_csv(Path(fname).expanduser(), sep=' ',
                     names=['y_true', '|', 'y_pred', 'rec', 'lig'])
    y_true = list(df.y_true)
    y_pred = list(df.y_pred)
    recs = list(df.rec)
    ligs = [Path(ligname).name.split('.')[0] for ligname in list(df.lig)]

    res = defaultdict(list)
    placements = []
    positive_scores = []
    top_ranked_rmsds = []
    for i in range(len(df)):
        pdbid = Path(recs[i]).name.split('.')[0]
        rmsd_info_record = rmsd_info[pdbid]
        if ligs[i].startswith('minimised'):
            rmsd = rmsd_info_record['relaxed_rmsd']
        else:
            rmsd = rmsd_info_record['docked_rmsd'][int(ligs[i].split('_')[-1])]
        res[recs[i]].append((y_true[i], y_pred[i], rmsd))
    for rec, lst in res.items():
        pdbid = Path(rec).parent.name
        res[rec] = np.array(sorted(lst, key=lambda x: x[1], reverse=True))
        placement = int(np.where(res[rec][:, 0] > 0)[0])
        top_ranked_rmsd = res[rec][0, 2]
        top_ranked_rmsds.append(top_ranked_rmsd)
        positive_score = res[rec][placement, 1]
        placements.append(placement)
        positive_scores.append(positive_score)

    return Ranking(fname,
                   np.array(placements),
                   scores=np.array(positive_scores),
                   top_ranked_rmsds=np.array(top_ranked_rmsds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=str,
                        help='Results of PointVS model inference')
    parser.add_argument('rmsd_info', type=str,
                        help='Yaml file containing RMSD information')
    args = parser.parse_args()
    print(parse_results(args.results, rmsd_info_fname=args.rmsd_info))
