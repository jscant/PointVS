import argparse
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from point_vs import logging
from point_vs.utils import expand_path, find_delta_E, save_yaml


LOG = logging.get_logger('PointVS')


def find_sdfs(types_file, data_root):
    def get_sdf(parquet):
        chunks = str(parquet).split('_')
        return str(Path(data_root, '_'.join(chunks[:-1]) + '.sdf'))

    data_root = str(data_root)
    types_file = expand_path(types_file)
    with open(types_file, 'r') as f:
        for line in f:
            n_fields = len(line.strip().split())
            break
    cols = ['label', 'vinascore', 'rmsd', 'rec', 'lig']
    while len(cols) < n_fields:
        cols.append('field_{}'.format(len(cols)))
    df = pd.read_csv(types_file, sep='\s+', names=cols)
    df['sdf'] = df['lig'].apply(get_sdf)
    sdfs = list(set(df['sdf'].to_list()))
    return sdfs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str, help='Location of sdf files')
    parser.add_argument('types_file', type=str,
                        help='Types file containing parquets')

    args = parser.parse_args()

    RDLogger.DisableLog('rdApp.*')

    data_root = expand_path(args.data_root)
    types_file = expand_path(args.types_file)

    n_processes = mp.cpu_count()

    sdfs = [[] for _ in range(n_processes)]
    ligands = sorted(find_sdfs(types_file, data_root))

    for idx, sdf in enumerate(ligands):
        sdfs[idx % n_processes].append(sdf)


    def worker(func, sdfs, multiple_structures, return_dict):
        """worker function"""
        for sdf in sdfs:
            base = Path(sdf)
            base = str(Path(base.parent.name, base.with_suffix('').name))
            energies = func(sdf, multiple_structures)
            for idx, info in energies.items():
                structure = base + '_{}.parquet'.format(idx)
                if isinstance(info, tuple):
                    dE = info[0]
                    rmsd = info[1]
                    return_dict[structure] = {'dE': dE, 'rmsd': rmsd}
                else:
                    return_dict[structure] = {'dE': info, 'rmsd': info}


    manager = mp.Manager()
    strain_energies = manager.dict()
    jobs = []
    for i in range(n_processes):
        p = mp.Process(
            target=worker, args=(find_delta_E, sdfs[i], True, strain_energies))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    LOG.info('Saving..')
    save_yaml(dict(strain_energies), data_root / 'strain_energies.yaml')
    saved_pth = data_root / 'strain_energies.yaml'
    LOG.info(f'Saved to {saved_pth}')
