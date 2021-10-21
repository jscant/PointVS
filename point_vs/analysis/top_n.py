from collections import defaultdict
from pathlib import Path

import pandas as pd

from point_vs.utils import expand_path

try:
    with open('data/val_pdbids.txt', 'r') as f:
        VAL_PDBIDS = set([line.lower().strip() for line in f.readlines()])
except FileNotFoundError:  # File is not in repo as it is not ready
    VAL_PDBIDS = []


def _extract_scores(types_file, pdbid_whitelist=None):
    print(VAL_PDBIDS)
    def extract_pdbid(rec):
        pdbid = rec.split('/')[-1].split('_')[0].lower()
        return pdbid in VAL_PDBIDS if pdbid_whitelist else True

    def extract_dock_sdf_name(pth):
        return '_'.join(str(Path(pth).with_suffix('')).split('_')[:-1]) + '.sdf'

    def extract_dock_pdb_name(pth):
        return str(Path(pth).with_suffix('.pdb'))

    def extract_vina_rank(lig):
        return int(Path(Path(lig).name).stem.split('_')[-1])

    df = pd.read_csv(expand_path(types_file), sep=' ',
                     names=['y_true', '|', 'y_pred', 'rec', 'lig'])
    df['vina_rank'] = df['lig'].map(extract_vina_rank)
    df['keep'] = df.rec.map(extract_pdbid)
    df.drop(df[df.keep].index, inplace=True)
    del df['keep']
    del df['|']
    df.drop((df['rec'] == '3FZY/3FZY_PRO.parquet').index)
    df.drop((df['rec'] == '3EEB/3EEB_PRO.parquet').index)
    df.reset_index(inplace=True, drop=True)
    df['lig_sdf'] = df['lig'].map(extract_dock_sdf_name)
    df['rec_pdb'] = df['rec'].map(extract_dock_pdb_name)
    df['reclig'] = df['rec'] + '__' + df['lig_sdf']
    return df


def _gnn_score(types_file, pdbid_whitelist=None):
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