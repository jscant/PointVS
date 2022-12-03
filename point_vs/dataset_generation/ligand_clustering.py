"""Cluster ligands by tanimoto similarity of morgan fingerprints."""
import argparse
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import AllChem, MolFromMol2File, SDMolSupplier
from rdkit.DataStructs import TanimotoSimilarity, UIntSparseIntVect
from rich.progress import Progress

from point_vs import logging
from point_vs.utils import expand_path, py_mollify, get_n_cols, execute_cmd, \
    mkdir


LOG = logging.get_logger('PointVS')


def is_similar(mol1, mol2, cutoff):
    if not isinstance(mol1, UIntSparseIntVect):
        mol1_fp = AllChem.GetMorganFingerprint(mol1, 3)
    else:
        mol1_fp = mol1
    if not isinstance(mol2, UIntSparseIntVect):
        mol2_fp = AllChem.GetMorganFingerprint(mol2, 3)
    else:
        mol2_fp = mol2

    return TanimotoSimilarity(mol1_fp, mol2_fp) >= cutoff


def get_mol(sdf, directory, pdbids=None, ligs=None):
    if ligs is not None:
        leaf = str(sdf)[
               max(0, str(sdf).find(directory) + len(str(directory))):]
        if leaf.startswith('/'):
            leaf = leaf[1:]
        if leaf not in ligs:
            raise RuntimeError('leaf not in ligs')
    elif sdf.parent.name not in pdbids:
        raise RuntimeError('sdf not in pdbids')

    mol = next(SDMolSupplier(str(sdf)))
    if mol is None:
        pym_sdf = py_mollify(sdf)
        mol = next(SDMolSupplier(str(pym_sdf)))
        if mol is None:
            mol = MolFromMol2File(str(pym_sdf).replace('.sdf', '.mol2'))

    if mol is None:
        obabel_mol = str(sdf).replace('.sdf', '_obabel.sdf')
        execute_cmd('obabel {0} -O{1} -d'.format(
            sdf, obabel_mol), raise_exceptions=False)
        execute_cmd('obabel {0} -O{1} -h'.format(
            obabel_mol, obabel_mol), raise_exceptions=False)
        try:
            mol = next(SDMolSupplier(str(obabel_mol)))
        except OSError:
            mol = None
        if mol is None:
            pym_sdf = py_mollify(obabel_mol)
            try:
                mol = next(SDMolSupplier(str(pym_sdf)))
            except OSError:
                mol = None

    if mol is not None:
        return AllChem.GetMorganFingerprint(mol, 3)

    raise RuntimeError('Molecule could not be read')


def get_mols(directory, pdbid_file=None, types_file=None):
    assert not (pdbid_file is None and types_file is None)
    if types_file is not None:
        n_cols = get_n_cols(types_file)
        df = pd.read_csv(expand_path(types_file), sep='\s+', names=(
            'x', 'y', 'z', 'rec', 'lig', *[str(i) for i in range(n_cols - 5)]))
        ligs = [str(s.replace('.parquet', '.sdf')) for s in df['lig']]
    else:
        ligs = None
    mols = {}

    if pdbid_file is not None:
        with open(expand_path(pdbid_file), 'r') as f:
            pdbids = set([s.strip() for s in f.readlines()])

    sdfs_not_present = []
    for idx, sdf in enumerate(expand_path(directory).glob('*/*_ligand.sdf')):
        try:
            mol = get_mol(sdf, directory, pdbids=pdbids, ligs=ligs)
        except RuntimeError:
            continue
        except OSError:
            sdfs_not_present.append(sdf)
        else:
            mols[str(sdf)] = mol
    return mols, sdfs_not_present


def types_to_sdfs(sdf_base_dir, types_file):
    sdf_base_dir = expand_path(sdf_base_dir)
    res = {}
    with open(expand_path(types_file), 'r') as f:
        for line in f.readlines():
            chunks = line.split()
            if len(chunks) >= 5:
                res[str(sdf_base_dir / chunks[4].replace(
                    '.parquet', '.sdf'))] = line.strip()
    return res


if __name__ == '__main__':

    def worker_types(_progress, return_dict, sdfs, directory, ligs, proc):
        for idx, sdf in enumerate(sdfs):
            _progress[proc] = {'progress': idx + 1, 'total': len(sdfs)}
            try:
                mol = get_mol(sdf, directory, ligs=ligs)
            except RuntimeError:
                continue
            except OSError:
                return_dict[str(sdf)] = None
            else:
                return_dict[str(sdf)] = mol


    def worker_tanimoto(
            _progress, return_lines, train_mols, test_mols, line_dict, proc):
        keep = []
        discard = []
        types_lines = []
        for idx, (sdf, mol) in enumerate(train_mols.items()):
            _progress[proc] = {'progress': idx + 1, 'total': len(train_mols)}
            keep_sdf = True
            if mol is not None:
                for test_pdbid, test_mol in test_mols.items():
                    if is_similar(mol, test_mol, cutoff=args.threshold):
                        discard.append(sdf)
                        keep_sdf = False
                        break
                if keep_sdf:
                    keep.append(sdf)
                    types_lines.append(line_dict[str(sdf)])
            else:
                discard.append(sdf)
        return_lines += types_lines


    def remove_index(s):
        suffix = '.' + s.split('.')[-1]
        return '_'.join(s.replace(suffix, '').split('_')[:-1]) + suffix


    parser = argparse.ArgumentParser()
    parser.add_argument('--train_structure_dir', '-train', type=str,
                        help='Location of raw train sdf files')
    parser.add_argument('--test_structure_dir', '-test', type=str,
                        help='Location of raw test sdf files')
    parser.add_argument('--train_types', type=str, help='Original (baised) '
                                                        'training set types '
                                                        'file')
    parser.add_argument('--test_types', type=str,
                        help='Original test set types '
                             'file')
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Where to store results')
    parser.add_argument('--threshold', '-t', type=float, default=0.9,
                        help='Sequence similarity threshold')

    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')

    line_to_sdf_map_test = types_to_sdfs(
        args.test_structure_dir, args.test_types)

    n_cols_test = get_n_cols(args.test_types)
    df_test = pd.read_csv(expand_path(args.test_types), sep='\s+', names=(
        'x', 'y', 'z', 'rec', 'lig', *[str(i) for i in range(
            n_cols_test - 5)]))
    ligs_test = list(set([str(s.replace('.parquet', '_0.sdf')) for s in
                          df_test['lig'].apply(remove_index)]))

    manager = mp.Manager()
    test_mols = manager.dict()
    jobs = []
    n_processes = mp.cpu_count()
    sdfs = [[] for _ in range(n_processes)]

    for idx, sdf in enumerate(ligs_test):
        sdfs[idx % n_processes].append(
            str(expand_path(args.test_structure_dir) / sdf))

    A = manager.list()


    def callback(obj):
        global A
        A.append(1)


    with Progress(refresh_per_second=1) as progress:
        p = mp.Pool()
        _progress = manager.dict()
        for i in range(n_processes):
            task_id = progress.add_task(
                f'[red]Test types CPU {i}', visible=False)
            p.apply_async(
                worker_types, args=(
                    _progress, test_mols, sdfs[i], args.test_structure_dir,
                    ligs_test, i), callback=callback)
        while sum(A) < n_processes:
            for task_id, update_data in _progress.items():
                latest = update_data['progress']
                total = update_data['total']
                progress.update(task_id, completed=latest, total=total,
                                visible=True)
        p.close()
        p.join()

    sdfs_not_present_test = list(
        {sdf for sdf, mol in test_mols.items() if mol is None})
    test_mols = {sdf: mol for sdf, mol in test_mols.items() if mol is not None}
    LOG.info(f'There are {len(test_mols)} test molecules')
    if len(sdfs_not_present_test):
        LOG.warning(
            f'There are {len(sdfs_not_present_test)} missing test sdfs')

    with open(mkdir(args.output_dir) / 'missing_sdfs_test.txt', 'w') as f:
        f.write('\n'.join(sdfs_not_present_test) + '\n')

    line_to_sdf_map_train = types_to_sdfs(
        args.train_structure_dir, args.train_types)

    n_cols_train = get_n_cols(args.train_types)
    df_train = pd.read_csv(expand_path(args.train_types), sep='\s+', names=(
        'x', 'y', 'z', 'rec', 'lig', *[str(i) for i in range(
            n_cols_train - 5)]))
    ligs_train = [str(s.replace('.parquet', '.sdf')) for s in df_train['lig']]
    train_mols = manager.dict()
    jobs = []
    sdfs = [[] for _ in range(n_processes)]

    for idx, sdf in enumerate(ligs_train):
        sdfs[idx % n_processes].append(
            str(expand_path(args.train_structure_dir) / sdf))

    A = manager.list()

    with Progress(refresh_per_second=1) as progress:
        p = mp.Pool()
        _progress = manager.dict()
        for i in range(n_processes):
            task_id = progress.add_task(
                f'[blue]Train types CPU {i}', visible=False)
            p.apply_async(
                worker_types, args=(
                    _progress, train_mols, sdfs[i], args.train_structure_dir,
                    ligs_train, i), callback=callback)
        while sum(A) < n_processes:
            for task_id, update_data in _progress.items():
                latest = update_data['progress']
                total = update_data['total']
                progress.update(task_id, completed=latest, total=total,
                                visible=True)
        p.close()
        p.join()

    sdfs_not_present_train = list(
        {sdf for sdf, mol in train_mols.items() if mol is None})
    train_mols = {sdf: mol for sdf, mol in train_mols.items() if
                  mol is not None}
    LOG.info(f'There are {len(train_mols)} train molecules')
    if len(sdfs_not_present_train):
        LOG.warning(
            f'There are {len(sdfs_not_present_train)} missing train sdfs')
    with open(mkdir(args.output_dir) / 'missing_sdfs_train.txt', 'w') as f:
        f.write('\n'.join(sdfs_not_present_train) + '\n')

    train_mols_split = [{} for _ in range(n_processes)]
    return_lines = manager.list()
    jobs = []

    for idx, (key, mol) in enumerate(train_mols.items()):
        train_mols_split[idx % n_processes][key] = mol

    A = manager.list()
    with Progress(refresh_per_second=1) as progress:
        p = mp.Pool()
        _progress = manager.dict()
        for i in range(n_processes):
            task_id = progress.add_task(
                f'[green]Tanimoto CPU {i}', visible=False)
            p.apply_async(
                worker_tanimoto, args=(
                    _progress, return_lines, train_mols_split[i], test_mols,
                    line_to_sdf_map_train, i), callback=callback)

        while sum(A) < n_processes:
            for task_id, update_data in _progress.items():
                latest = update_data['progress']
                total = update_data['total']
                progress.update(task_id, completed=latest, total=total,
                                visible=True)
        p.close()
        p.join()

    for proc in jobs:
        proc.join()

    return_lines = list(return_lines)
    return_lines = '\n'.join(
        [line.strip() for line in return_lines if len(line.strip())])

    output_types_file = mkdir(
        args.output_dir) / '{}_ligand_filtered.types'.format(
        Path(args.train_types).with_suffix('').name
    )
    LOG.info(f'Types file written to {str(output_types_file)}.')
    with open(output_types_file, 'w') as f:
        f.write(return_lines + '\n')
