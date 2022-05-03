import argparse
import multiprocessing as mp
import time
from pathlib import Path

import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import AllChem, MolFromMol2File, SDMolSupplier
from rdkit.DataStructs import TanimotoSimilarity, UIntSparseIntVect

from point_vs.utils import expand_path, py_mollify, get_n_cols, execute_cmd, \
    mkdir


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
        leaf = str(sdf)[str(sdf).find(directory) + len(str(directory)) + 1:]
        if leaf not in ligs:
            raise RuntimeError('leaf not in ligs')
    elif sdf.parent.name.lower() not in pdbids:
        raise RuntimeError('sdf not in pdbids')

    mol = None
    try:
        mol = next(SDMolSupplier(str(sdf)))
    except OSError:
        pass
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
    # print('Error:', sdf)
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

    for idx, sdf in enumerate(expand_path(directory).glob('*/*_ligand.sdf')):
        try:
            mol = get_mol(sdf, directory, pdbids=pdbids, ligs=ligs)
        except RuntimeError:
            continue
        else:
            mols[str(sdf)] = mol
    return mols


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

    def worker_types(return_dict, sdfs, directory, ligs, proc):
        start_time = time.time()
        for idx, sdf in enumerate(sdfs):
            try:
                mol = get_mol(sdf, directory, ligs=ligs)
            except:
                continue
            else:
                return_dict[str(sdf)] = mol
            if not proc and not idx % 1000:
                elasped = time.time() - start_time
                print('Time: {0:.3f}'.format(elasped),
                      idx * n_processes,
                      'Time remaining: {0:.3f} s'.format(
                          780000 * (elasped / (
                                      (idx * n_processes) + 1)) - elasped))


    def worker_tanimoto(return_lines, train_mols, test_mols, line_dict, proc):
        keep = []
        discard = []
        types_lines = ''
        start_time = time.time()
        for idx, (sdf, mol) in enumerate(train_mols.items()):
            keep_sdf = True
            if mol is not None:
                for test_pdbid, test_mol in test_mols.items():
                    if is_similar(mol, test_mol, cutoff=args.threshold):
                        discard.append(sdf)
                        keep_sdf = False
                        break
                if keep_sdf:
                    keep.append(sdf)
                    types_lines += line_dict[str(sdf)] + '\n'
            else:
                discard.append(sdf)
            if not proc and not idx % 1000:
                elasped = time.time() - start_time
                print(elasped,
                      idx * n_processes,
                      'Time remaining: {0:.3f} s'.format(
                          780000 * (elasped / (
                                  (idx * n_processes) + 1)) - elasped))
        return_lines.append(types_lines)


    parser = argparse.ArgumentParser()
    parser.add_argument('train_structure_dir', type=str,
                        help='Location of raw train sdf files')
    parser.add_argument('test_structure_dir', type=str,
                        help='Location of raw test sdf files')
    parser.add_argument('test_pdbids', type=str, help='Test set PDBIDs file')
    parser.add_argument('train_types', type=str, help='Original (baised) set '
                                                      'types file')
    parser.add_argument('output_dir', type=str, help='Where to store results')
    parser.add_argument('--threshold', '-t', type=float, default=0.9,
                        help='Sequence similarity threshold')

    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')

    line_to_sdf_map = types_to_sdfs(args.train_structure_dir, args.train_types)
    test_mols = get_mols(args.test_structure_dir, args.test_pdbids)
    print('There are', len(test_mols), 'test set molecules')

    n_cols = get_n_cols(args.train_types)
    df = pd.read_csv(expand_path(args.train_types), sep='\s+', names=(
        'x', 'y', 'z', 'rec', 'lig', *[str(i) for i in range(n_cols - 5)]))
    ligs = [str(s.replace('.parquet', '.sdf')) for s in df['lig']]

    manager = mp.Manager()
    train_mols = manager.dict()
    jobs = []
    n_processes = mp.cpu_count()
    sdfs = [[] for _ in range(n_processes)]

    for idx, sdf in enumerate(ligs):
        sdfs[idx % n_processes].append(
            str(expand_path(args.train_structure_dir) / sdf))

    for i in range(n_processes):
        p = mp.Process(
            target=worker_types, args=(
                train_mols, sdfs[i], args.train_structure_dir, ligs, i))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    train_mols = dict(train_mols)
    print('There are', len(train_mols), 'training molecules')

    train_mols_split = [{} for _ in range(n_processes)]
    return_lines = manager.list()
    jobs = []

    for idx, (key, mol) in enumerate(train_mols.items()):
        train_mols_split[idx % n_processes][key] = mol


    for i in range(n_processes):
        print(i)
        p = mp.Process(target=worker_tanimoto, args=(
            return_lines, train_mols_split[i], test_mols, line_to_sdf_map, i))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    return_lines = list(return_lines)
    return_lines = '\n'.join(
        [line.strip() for line in return_lines if len(line.strip())])

    output_types_file = mkdir(
        args.output_dir) / '{}_ligand_filtered.types'.format(
        Path(args.train_types).with_suffix('').name
    )
    print('Types file written to', str(output_types_file))
    with open(output_types_file, 'w') as f:
        f.write(return_lines + '\n')
