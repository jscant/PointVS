import numpy as np


class Ranking:
    """Simple struct and print class for ranking results."""

    def __init__(self, fname, placements, scores=None, top_ranked_rmsds=None):
        self.placements = np.array(placements)
        self.scores = None if scores is None else np.array(scores)
        self.top_ranked_rmsds = None if top_ranked_rmsds is None else np.array(
            top_ranked_rmsds)
        self.fname = fname

    def get_frac_correct(self):
        """Calculate the fraction of targets for which the correct pose is
        identified."""
        return len(np.where(self.placements == 0)[0]) / len(self.placements)

    def __str__(self):
        res = 'Mean placement of relaxed xtal structure: {0:0.5f}\n'.format(
            np.mean(self.placements) + 1)
        if self.scores is not None:
            res += 'Mean score of relaxed xtal structure: {0:0.5f}\n'.format(
                np.mean(self.scores))
        if self.top_ranked_rmsds is not None:
            'Mean score of relaxed xtal structure: {0:0.5f}\n'.format(
                np.mean(self.scores))
        res += 'Fraction of correctly identified poses: {0:0.5f}'.format(
            self.get_frac_correct())    
        return res

    def get_mean_placement(self):
        return np.mean(self.placements) + 1

    def get_mean_xtal_score(self):
        return np.mean(self.scores) if self.scores is not None else None

    def get_mean_top_ranked_rmsd(self):
        return np.mean(self.top_ranked_rmsds)

    def __repr__(self):
        return 'Ranking object obtained from {} containing stats:\n'.format(
            self.fname) + self.__str__()
