from collections import defaultdict
from pathlib import Path

import pandas as pd

from point_vs.constants import GNINA_TEST_PDBIDS as VAL_PDBIDS
from point_vs.utils import expand_path


def _extract_scores(types_file, pdbid_whitelist=False):
    def drop_record(rec):
        pdbid = rec.split('/')[-1].split('_')[0].lower()
        return False if not pdbid_whitelist else pdbid not in VAL_PDBIDS

    def extract_dock_sdf_name(pth):
        return '_'.join(str(Path(pth).with_suffix('')).split('_')[:-1]) + '.sdf'

    def extract_dock_pdb_name(pth):
        return str(Path(pth).with_suffix('.pdb'))

    def extract_vina_rank(lig):
        return int(Path(Path(lig).name).stem.split('_')[-1])

    df = pd.read_csv(expand_path(types_file), sep=' ',
                     names=['y_true', '|', 'y_pred', 'rec', 'lig'])
    df['vina_rank'] = df['lig'].map(extract_vina_rank)
    df['remove'] = df.rec.map(drop_record)
    df.drop(df[df.remove].index, inplace=True)
    del df['remove']
    del df['|']
    df.reset_index(inplace=True, drop=True)
    df['lig_sdf'] = df['lig'].map(extract_dock_sdf_name)
    df['rec_pdb'] = df['rec'].map(extract_dock_pdb_name)
    df['reclig'] = df['rec'] + '__' + df['lig_sdf']
    return df


def _gnn_score(types_file, pdbid_whitelist=False):
    scores = defaultdict(list)
    df = _extract_scores(types_file, pdbid_whitelist)
    y_trues = df['y_true'].to_numpy()
    y_preds = df['y_pred'].to_numpy()
    recligs = df['reclig'].to_numpy()
    for reclig, y_true, y_pred in zip(recligs, y_trues, y_preds):
        scores[reclig].append((y_pred, y_true))
    for reclig, values in scores.items():
        scores[reclig] = sorted(values, key=lambda x: x[0], reverse=True)
    return scores


def top_n(types_file, n=1, pdbid_whitelist=True):
    scores = _gnn_score(types_file, pdbid_whitelist=pdbid_whitelist)
    s = [[j[1] for j in i] for i in scores.values()]
    return sum([1 for i in s if sum(i[:n])]) / len(scores)
