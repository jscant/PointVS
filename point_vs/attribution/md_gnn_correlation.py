from pathlib import Path

import matplotlib

from point_vs.utils import mkdir

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from point_vs.attribution.gromacs import master

import numpy as np


def plot_gnn_score_vs_bond_length(df, output_dir, name, score):
    mean_distances = df.md_mean_distance.to_numpy()
    mean_distances = mean_distances[np.where(mean_distances < 5)]
    var_distances = df.md_var_distance.to_numpy()[:len(mean_distances)]
    gnn_scores_md = df.bond_score.to_numpy()[:len(mean_distances)]

    df.sort_values(by='xtal_distance', inplace=True)
    xtal_distances = df.xtal_distance.to_numpy()
    xtal_distances = xtal_distances[np.where(xtal_distances < 5)]
    gnn_scores_xtal = df.bond_score.to_numpy()[:len(xtal_distances)]
    length_spr, _ = spearmanr(mean_distances, gnn_scores_md)
    spr_xtal, _ = spearmanr(xtal_distances, gnn_scores_xtal)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(name + '    y = {0:.3f}'.format(score))
    for idx, (dists, scores, ax) in enumerate(zip(
            (mean_distances, xtal_distances),
            (gnn_scores_md, gnn_scores_xtal),
            (ax1, ax2))):
        c = [var_distances, 'b'][idx]
        spr = [length_spr, spr_xtal][idx]
        ax.scatter(
            dists, scores, c=c, s=100 * scores,
            cmap='autumn')
        xlabel = ['Mean bond length (Angstroms)',
                  'Crystal bond length (Angstroms)'][idx]

        ax.set_xlabel(xlabel)
        ax.set_ylabel('GNN attention bond score')
        ax.set_title('r={0:.3f}'.format(spr))
        ax.set_yscale('log')

        try:
            m, b = list(np.polyfit(dists, np.exp(scores), 1))
        except TypeError:
            return -1
        #ax.plot(np.arange(2, 4.9, 0.1), np.log(m * np.arange(2, 4.9, 0.1) + b),
        #        'b-')
        ax.set_xlim([2, 4.9])
        ax.set_ylim([0.0001, 0.4])
        ax.grid(linewidth=1.0)
        #ax.axvline(x=3.0, color='k', linestyle='--')
    plt.savefig(Path(output_dir) / '{}.png'.format(name))
    plt.cla()

    var_spr = None
    return length_spr, var_spr


if __name__ == '__main__':
    frag = 'F760'
    base = Path('jacks_distances/data')
    model_file = 'good_attribution_models/egnn_0' \
                 '.0_dropout_feats_atn_48L_estb_1e-4wd_e10_static_PI/checkpoints' \
                 '/ckpt_epoch_10.pt'
    # model_file = 'newer/egnn_0.0_dropout_atn_48L_estb_1e-5wd_e10/' \
    #             'checkpoints/ckpt_epoch_4.pt'
    for model_file in [
        #'good_attribution_models/egnn_0' \
        #'.0_dropout_feats_atn_48L_estb_1e-4wd_e10_static_PI/checkpoints' \
        #'/ckpt_epoch_10.pt',
        'good_attribution_models/egnn_0' \
        '.0_dropout_feats_atn_48L_estb_1e-4wd_e10_PAPER/checkpoints' \
        '/ckpt_epoch_10.pt'
    ]:
        output_dir = mkdir(
            'diagnostics/gromacs/correlations/both/{}'.format(
                Path(model_file).parents[1].name))

        atom_blind = True
        if Path(output_dir / 'correlation_coefficients.txt').exists():
            Path(output_dir / 'correlation_coefficients.txt').unlink()
        for trajectories_file in Path(base / 'csvs').glob('*.csv'.format(frag)):
            gromacs_file = str(trajectories_file).replace(
                '_distances.csv', '.gro').replace('/csvs/', '/gromacs/')
            print(trajectories_file)
            print(gromacs_file)
            frag = trajectories_file.with_suffix('').name.split('-')[0]
            try:
                gnn_df, score = master(
                    trajectories_file,
                    gromacs_file,
                    model_file,
                    output_dir,
                    1,
                    only_process='MOL',
                    atom_blind=atom_blind
                )
            except UnicodeDecodeError:
                print('Failed to decode file for', frag)
                continue
            p = plot_gnn_score_vs_bond_length(gnn_df, output_dir, frag, score)
            if p != -1:
                with open(output_dir / 'correlation_coefficients.txt',
                          'a') as f:
                    f.write('{0} {1:.3f} {2:.3f}\n'.format(
                        frag, p[0], p[1] if p[1] is not None else -1))

                gnn_df.to_csv(Path(mkdir(
                    output_dir, Path(gromacs_file).with_suffix('').name),
                    '{}_distances_vs_scores.csv'.format(frag)))
            #raise

    trajectories_file = next(Path(base / 'csvs').glob('{}*'.format(frag)))
    gromacs_file = next(Path(base / 'gromacs').glob('{}*'.format(frag)))
    # trajectories_file = 'jacks_distances/data/csvs/F760-PHIPA' \
    #                    '-x20045_distances.csv'
    # gromacs_file = 'jacks_distances/data/gromacs/F709'
