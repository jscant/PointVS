"""Gather stats on pose selection performance on a validation set."""

from pathlib import Path
import pandas as pd
from collections import defaultdict
import numpy as np
import argparse


class Ranking:
    """Simple struct and print class for ranking results."""

    def __init__(self, fname, placements, scores):
        self.placements = placements
        self.scores = scores
        self.fname = fname

    def get_frac_correct(self):
        """Calculate the fraction of targets for which the correct pose is
        identified."""
        return len(np.where(self.placements == 0)[0]) / len(self.placements)

    def __str__(self):
        return ('Mean placement of relaxed xtal structure: {0:0.5f}\n + ' \
                'Mean score of relaxed xtal structure: {1:0.5f}\n + ' \
                'Fraction of correctly identified poses: {2:0.5f}').format(
            np.mean(self.placements + 1),
            np.mean(self.scores),
            self.get_frac_correct())

    def __repr__(self):
        return 'Ranking object obtained from {} containing stats:\n'.format(
            self.fname) + self.__str__()


def parse_results(fname):
    """Parse results stored in text format.

    Arguments:
        fname: location of the results

    Returns:
        Ranking object containing various statistics.
    """
    df = pd.read_csv(Path(fname).expanduser(), sep=' ',
                     names=['y_true', '|', 'y_pred', 'rec', 'lig'])
    y_true = list(df.y_true)
    y_pred = list(df.y_pred)
    recs = list(df.rec)

    res = defaultdict(list)
    placements = []
    positive_scores = []
    for i in range(len(df)):
        res[recs[i]].append((y_true[i], y_pred[i]))
    for rec, lst in res.items():
        res[rec] = np.array(sorted(lst, key=lambda x: x[1], reverse=True))
        placement = int(np.where(res[rec][:, 0] > 0)[0])
        positive_score = res[rec][placement, 1]
        placements.append(placement)
        positive_scores.append(positive_score)

    return Ranking(fname, np.array(placements), np.array(positive_scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=str,
                        help='Results of PointVS model inference')
    args = parser.parse_args()
    print(parse_results(args.results))
