import numpy as np


class Ranking:
    """Simple struct and print class for ranking results."""

    def __init__(self, fname, sorted_scores_and_rmsds):
        self.fname = fname
        self.sorted_scores_and_rmsds = sorted_scores_and_rmsds

    def get_top_n(self, n, threshold=2.0):
        in_top_n = 0
        for idx, info in enumerate(self.sorted_scores_and_rmsds):
            top_n_rmsd = info[:n, -1]
            if len(np.where(top_n_rmsd <= threshold)[0]):
                in_top_n += 1
        return in_top_n / len(self.sorted_scores_and_rmsds)

    def __str__(self):
        res = 'Mean RMSD of top ranked structure: {0:0.5f}\n'.format(
            self.get_mean_top_ranked_rmsd())
        res += 'Top1 at 2.0 A: {0:0.5f}\n'.format(
            self.get_top_n(1, 2.0))
        return res

    def get_mean_top_ranked_rmsd(self):
        return np.mean([item[0, -1] for item in self.sorted_scores_and_rmsds])

    def __repr__(self):
        return 'Ranking object obtained from {} containing stats:\n'.format(
            self.fname) + self.__str__()
