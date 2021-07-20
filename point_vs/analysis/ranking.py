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
        if self.scores is not None and self.top_ranked_rmsds is None:
            return ('Mean placement of relaxed xtal structure: {0:0.5f}\n' + \
                    'Mean score of relaxed xtal structure: {1:0.5f}\n' + \
                    'Fraction of correctly identified poses: {2:0.5f}').format(
                np.mean(self.placements) + 1,
                np.mean(self.scores),
                self.get_frac_correct())
        if self.scores is None and self.top_ranked_rmsds is not None:
            return ('Mean placement of relaxed xtal structure: {0:0.5f}\n' + \
                    'Mean RMSD of top ranked structure: {1:0.5f}\n' + \
                    'Fraction of correctly identified poses: {2:0.5f}').format(
                np.mean(self.placements) + 1,
                np.mean(self.top_ranked_rmsds),
                self.get_frac_correct())
        elif self.scores is not None and self.top_ranked_rmsds is not None:
            return ('Mean placement of relaxed xtal structure: {0:0.5f}\n' + \
                    'Mean score of relaxed xtal structure: {1:0.5f}\n' + \
                    'Mean RMSD of top ranked structure: {2:0.5f}\n' + \
                    'Fraction of correctly identified poses: {3:0.5f}').format(
                np.mean(self.placements) + 1,
                np.mean(self.scores),
                np.mean(self.top_ranked_rmsds),
                self.get_frac_correct())

    def get_mean_placement(self):
        return np.mean(self.placements) + 1

    def get_mean_xtal_score(self):
        return np.mean(self.scores) if self.scores is not None else None

    def get_mean_top_ranked_rmsd(self):
        return np.mean(self.top_ranked_rmsds)

    def __repr__(self):
        return 'Ranking object obtained from {} containing stats:\n'.format(
            self.fname) + self.__str__()
