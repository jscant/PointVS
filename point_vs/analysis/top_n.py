from collections import defaultdict
from pathlib import Path

import pandas as pd

from point_vs.utils import expand_path


def _extract_scores(types_file, exclude_lig_substr=None):

    def extract_dock_sdf_name(pth):
        return '_'.join(str(Path(pth).with_suffix('')).split('_')[:-1]) + '.sdf'

    def extract_dock_pdb_name(pth):
        return str(Path(pth).with_suffix('.pdb'))

    def extract_vina_rank(lig):
        return int(Path(Path(lig).name).stem.split('_')[-1])

    df = pd.read_csv(expand_path(types_file), sep=' ',
                     names=['y_true', '|', 'y_pred', 'rec', 'lig'])
    df['vina_rank'] = df['lig'].map(extract_vina_rank)
    del df['|']
    if exclude_lig_substr is not None:
        df = df[~df['lig'].str.contains(exclude_lig_substr)]
    df.reset_index(inplace=True, drop=True)
    df['lig_sdf'] = df['lig'].map(extract_dock_sdf_name)
    df['rec_pdb'] = df['rec'].map(extract_dock_pdb_name)
    df['reclig'] = df['rec'] + '__' + df['lig_sdf']
    return df


def _gnn_score(types_file, exclude_lig_substr=None):
    scores = defaultdict(list)
    df = _extract_scores(types_file, exclude_lig_substr=exclude_lig_substr)
    y_trues = df['y_true'].to_numpy()
    y_preds = df['y_pred'].to_numpy()
    recligs = df['rec'].to_numpy()
    for reclig, y_true, y_pred in zip(recligs, y_trues, y_preds):
        scores[reclig].append((y_pred, int(y_true)))
    for reclig, values in scores.items():
        scores[reclig] = sorted(values, key=lambda x: x[0], reverse=True)
    return scores


def top_n(types_file, n=1, exclude_lig_substr=None):
    scores = _gnn_score(types_file, exclude_lig_substr=exclude_lig_substr   )
    s = [[j[1] for j in i] for i in scores.values()]
    return sum([1 for i in s if sum(i[:n])]) / len(scores)
