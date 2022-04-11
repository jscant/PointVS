"""
Generate a types file including RMSD calculations if specified from PDB, SDF
and MOL2 files.

usage: generate_types_file.py [-h] [--receptor_pattern RECEPTOR_PATTERN]
                              [--crystal_pose_pattern CRYSTAL_POSE_PATTERN]
                              [--docked_pose_pattern DOCKED_POSE_PATTERN]
                              [--active_pattern ACTIVE_PATTERN]
                              [--inactive_pattern INACTIVE_PATTERN]
                              base_path output_path

positional arguments:
  base_path             Root directory in which pdb, sdf and mol2 files to be
                        converted to parquets are stored
  output_path           Directory in which to store output parquets

optional arguments:
  -h, --help            show this help message and exit
  --receptor_pattern RECEPTOR_PATTERN, -r RECEPTOR_PATTERN
                        Regex pattern for receptor structure
  --crystal_pose_pattern CRYSTAL_POSE_PATTERN, -x CRYSTAL_POSE_PATTERN
                        Regex pattern for crystal poses
  --docked_pose_pattern DOCKED_POSE_PATTERN, -d DOCKED_POSE_PATTERN
                        Regex pattern for docked poses
  --active_pattern ACTIVE_PATTERN, -a ACTIVE_PATTERN
                        Regex pattern for active poses
  --inactive_pattern INACTIVE_PATTERN, -i INACTIVE_PATTERN
                        Regex pattern for inactive structures

The directory structure must be set up as follows:
In the base_path, there should be multiple folders each containing 1 or 0
PDB receptor files, alongside sdf/mol2 ligand files. There are two modes,
which are selected by specifying EITHER:

    ====== --crystal_pose_pattern AND --docked_pose_pattern ======
    Each pose found in sdf files using the --docked_pose_pattern expression will
    be compared to the structure found in the sdf matching the
    --crystal_pose_pattern, so the RMSD will be found and labels will be
    assigned based on whether the RMSD is above or below 2A.

    If there are multiple docked and crystal poses in each subdirectory, an
    attempt will be made to match each docked sdf with a crystal sdf, based on
    the largest common substring. If this cannot be done in a 1-to-1 manner,
    an exception will be thrown.

                        ==== OR ====

    ====== --active_pattern AND --inactive_pattern ======
    No RMSD will be calculated; structures matching the active and inactive
    patterns will be assigned labels 1 and 0 respectively.
"""
import argparse
import io
import re
import subprocess
from difflib import SequenceMatcher
from itertools import product
from pathlib import Path

import pandas as pd
from openbabel import pybel
from pandas._libs.lib import NoDefault

from point_vs.utils import expand_path, pretify_dict, mkdir


def extract_pdbbind_affinities(csv):
    def extract_affinity_metric(affinity):
        if '<' in affinity:
            split_char = '<'
        elif '>' in affinity:
            split_char = '>'
        elif '=' in affinity:
            split_char = '='
        elif '~' in affinity:
            split_char = '~'
        else:
            return None
        return 'p' + affinity.split(split_char)[0].lower()

    year = 2020
    with open(expand_path(csv), 'r') as f:
        lines = []
        for idx, line in enumerate(f.readlines()):
            if line.startswith('#'):
                lines.append(line.strip())
            elif idx:
                break
            else:
                if line.startswith('ID'):
                    names = ('ID', 'PDB code', 'Subset', 'Affinity Data',
                             'pKd pKi pIC50', 'Ligand Name')
                else:
                    names = NoDefault.no_default
                year = 2016
    if year == 2020:
        names = (lines[-2][2:].split(', ')[:5])
        with open(expand_path(csv), 'r') as f:
            csv_as_str = '\n'.join(
                [' '.join(line.split()[:5]) for line in f.readlines()])
        pdbbind_csv = pd.read_csv(
            io.StringIO(csv_as_str), sep='\s+', header=idx, names=names)
        affinity_field = 'Kd/Ki'
        pk_field = '-logKd/Ki'
    else:
        pdbbind_csv = pd.read_csv(expand_path(csv), sep=',', names=names)
        affinity_field = 'Affinity Data'
        pk_field = 'pKd pKi pIC50'

    pdbbind_csv['metric'] = pdbbind_csv[affinity_field].map(
        extract_affinity_metric)
    return pd.DataFrame({
        'pdbid': pdbbind_csv['PDB code'],
        'affinity': pdbbind_csv[pk_field],
        'metric': pdbbind_csv['metric']
    })


def execute_cmd(cmd, raise_exceptions=True, silent=False):
    """Helper for executing obrms commands an capturing the output."""

    class res:
        def __init__(self, stdout, stderr, returncode):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    proc_result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True)
    if proc_result.stderr and raise_exceptions:
        raise subprocess.CalledProcessError(
            returncode=proc_result.returncode,
            cmd=proc_result.args,
            stderr=proc_result.stderr)
    if proc_result.stdout and not silent:
        print(proc_result.stdout.decode('utf-8'))

    return res(proc_result.stdout.decode('utf-8'),
               proc_result.stderr.decode('utf-8'),
               proc_result.returncode)


def get_rmsd(reference_fname, docked_fname):
    """Return the rmsds between a reference sdf and structures in another sdf"""
    reference_fname = expand_path(reference_fname)
    docked_fname = expand_path(docked_fname)
    cmd = 'obrms {0} {1}'.format(docked_fname, reference_fname)
    output = execute_cmd(cmd, raise_exceptions=False, silent=True).stdout
    rmsds = []
    for line in output.split('\n'):
        if len(line.split()) and line.split()[0] == 'RMSD':
            rmsds.append(float(line.split()[-1]))
    return rmsds


def generate_types_str(directory, pdb_exp, crystal_exp=None, docked_exp=None,
                       active_exp=None, inactive_exp=None,
                       include_crystal_structure=True, separated_files=True,
                       affinity_dict=None):
    """Generate a portion of a types file."""

    def re_glob(exp):
        return [f for f in directory.glob('*') if f.is_file() and
                re.match(exp, str(f.name))]

    def types_line_regression(receptor_pdb, ligand_sdf, affinity, metric):
        affinities = [-1, -1, -1]
        try:
            affinities[['pki', 'pkd', 'pic50'].index(metric)] = affinity
        except IndexError:
            print('Could not find affinity data for', receptor_pdb)
            return None
        affinity_str = '{0} {1} {2}'.format(*affinities)
        rec_path = Path(
            directory.name, receptor_pdb.with_suffix('.parquet').name)
        lig_path = Path(
            directory.name,
            ligand_sdf.with_suffix('').name + '_0.parquet')
        print('{0} {1} {2}\n'.format(affinity_str, rec_path, lig_path))
        return '{0} {1} {2}\n'.format(affinity_str, rec_path, lig_path)

    def types_line_classification(
            receptor_pdb, ref_sdf=None, query_sdf=None, label=None, ics=True):
        dir_name = directory.name
        template = '{0} -1 {1} {2} {3}\n'

        if label is None:
            rmsds = get_rmsd(ref_sdf, query_sdf)
        else:
            rmsds = [-1 for _ in pybel.readfile(query_sdf.suffix[1:],
                                                str(query_sdf))]

        if include_crystal_structure and ics:
            res = template.format(
                1, '0.00000',
                Path(
                    dir_name,
                    receptor_pdb.with_suffix('.parquet').name
                ),
                Path(
                    dir_name,
                    ref_sdf.with_suffix('').name + '_0.parquet'
                )
            )
        else:
            res = ''

        for idx, rmsd in enumerate(rmsds):
            label_ = int(rmsd < 2.0) if label is None else label
            res += template.format(
                label_,
                rmsd,
                Path(
                    dir_name,
                    receptor_pdb.with_suffix('.parquet').name
                ),
                Path(
                    dir_name,
                    query_sdf.with_suffix('').name + '_{}.parquet'.format(idx)
                )
            )
        return res

    directory = expand_path(directory)
    pdbs = [f for f in directory.glob('*') if f.is_file() and
            re.match(pdb_exp, str(f.name))]
    if len(pdbs) == 0:
        return -1
    s = ''
    for r_idx, receptor_pdb in enumerate(pdbs):
        # print(receptor_pdb)
        if crystal_exp is not None and docked_exp is not None:
            xtal_matches = re_glob(crystal_exp)
            docked_matches = re_glob(docked_exp)
            # print(receptor_pdb, len(xtal_matches), len(docked_matches))
            if len(xtal_matches) * len(docked_matches):
                types_str = None
                if len(xtal_matches) * len(docked_matches) == 1:
                    crystal_sdf = xtal_matches[0]
                    docked_sdf = docked_matches[0]
                elif not separated_files:
                    longest_match = 0
                    receptor_name = receptor_pdb.with_suffix('').name
                    for xtal_match_path in xtal_matches:
                        xtal_match = xtal_match_path.with_suffix('').name
                        match = SequenceMatcher(
                            None, xtal_match, receptor_name
                        ).find_longest_match(
                            0, len(xtal_match), 0, len(receptor_name))
                        if match.size > longest_match:
                            crystal_sdf = xtal_match_path
                            longest_match = match.size
                    longest_match = 0
                    for docked_match_path in docked_matches:
                        docked_match = docked_match_path.with_suffix('').name
                        match = SequenceMatcher(
                            None, docked_match, receptor_name
                        ).find_longest_match(
                            0, len(docked_match), 0, len(receptor_name))
                        if match.size > longest_match:
                            docked_sdf = docked_match_path
                            longest_match = match.size
                else:
                    types_str = ''
                    for idx, (xtal_match, docked_match) in enumerate(product(
                            xtal_matches, docked_matches)):
                        types_str += types_line_classification(
                            receptor_pdb, xtal_match, docked_match, None,
                            ics=not idx)

                if types_str is None:
                    types_str = types_line_classification(
                        receptor_pdb, crystal_sdf, docked_sdf, None)
            else:
                xtal_to_docked_map = {}
                types_str = ''
                for xtal_match_path in xtal_matches:
                    xtal_match = xtal_match_path.with_suffix('').name
                    longest_match = 0
                    for docked_match_path in docked_matches:
                        docked_match = docked_match_path.with_suffix('').name
                        match = SequenceMatcher(None, xtal_match,
                                                docked_match) \
                            .find_longest_match(0, len(xtal_match), 0,
                                                len(docked_match))
                        if match.size > longest_match:
                            longest_match = match.size
                            xtal_to_docked_map[
                                xtal_match_path] = docked_match_path
                if len(set(xtal_to_docked_map.values())) != len(xtal_matches):
                    print(pretify_dict(xtal_to_docked_map))
                    raise RuntimeError('Could not determine matching pattern '
                                       'for {}'.format(directory))
                for crystal_sdf, docked_sdf in xtal_to_docked_map.items():
                    types_str += types_line_classification(
                        receptor_pdb, crystal_sdf, docked_sdf)
        elif active_exp is not None and inactive_exp is not None:
            types_str = ''
            active_matches = re_glob(active_exp)
            inactive_matches = re_glob(inactive_exp)
            for active_match in active_matches:
                types_str += types_line_classification(
                    receptor_pdb, query_sdf=active_match, label=1)
            for inactive_match in inactive_matches:
                types_str += types_line_classification(
                    receptor_pdb, query_sdf=inactive_match, label=0)
        elif crystal_exp is not None and affinity_dict:
            types_str = ''
            crystal_match = list(re_glob(crystal_exp))[0]
            recstem = receptor_pdb.with_suffix('').name
            pdbid, affinity, metric = None, None, None
            for i in range(len(recstem) - 4):
                candidate_pdbid = recstem[i:i + 4]
                try:
                    affinity, metric = affinity_dict[candidate_pdbid]
                except:
                    pass
                else:
                    pdbid = candidate_pdbid
                    break
            if pdbid is None:
                print(
                    'Could not find affinity data for pdb' + str(receptor_pdb))
                continue
            types_line = types_line_regression(
                receptor_pdb, crystal_match, affinity, metric)
            if types_line is not None:
                types_str += types_line
        else:
            raise RuntimeError('Either specify both crystal_exp and docked_exp '
                               'or active_exp and inactive_exp')
        s += types_str + '\n'
    return s[:-1]


def get_intra_rmsd(docked_fname):
    """Get the cross-RMSD using obrms for all structures in an sdf file."""
    docked_fname = expand_path(docked_fname)
    cmd = 'obrms {0} -x'.format(docked_fname)
    output = execute_cmd(cmd, silent=True).stdout
    lines = output.split('\n')[:-1]
    idx_pair_to_rmsd = {}
    for line_idx, line in enumerate(lines):
        rmsds = line.split(', ')[1:][line_idx + 1:]
        for chunks_idx, rmsd in enumerate(rmsds):
            idx_pair_to_rmsd[(line_idx, line_idx + chunks_idx + 1)] = rmsd
    return idx_pair_to_rmsd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_path', type=str,
                        help='Root directory in which pdb, sdf and mol2 files '
                             'to be converted to parquets are stored')
    parser.add_argument('output_path', type=str,
                        help='Directory in which to store output parquets')
    parser.add_argument('--receptor_pattern', '-r', type=str,
                        help='Regex pattern for receptor structure')
    parser.add_argument('--crystal_pose_pattern', '-x', type=str,
                        help='Regex pattern for crystal poses')
    parser.add_argument('--docked_pose_pattern', '-d', type=str,
                        help='Regex pattern for docked poses')
    parser.add_argument('--active_pattern', '-a', type=str,
                        help='Regex pattern for active poses')
    parser.add_argument('--inactive_pattern', '-i', type=str,
                        help='Regex pattern for inactive structures')
    parser.add_argument('--split_sdfs', '-s', action='store_true',
                        help='Input ligands are split up so that there is only '
                             'one strucure per sdf')
    parser.add_argument('--affinity', '-p', type=str, default=None,
                        help='CSV file from PDBBind website with affinity data '
                             'for regression (only crystal_pose_pattern must be'
                             ' supplied with this')
    args = parser.parse_args()

    base_path = expand_path(args.base_path)
    output_path = mkdir(args.output_path)

    pdb_exp = args.receptor_pattern
    xtal_exp = args.crystal_pose_pattern
    docked_exp = args.docked_pose_pattern
    active_exp = args.active_pattern
    inactive_exp = args.inactive_pattern

    s = ''
    n_pdbs = len([p for p in base_path.glob('*') if p.is_dir()])

    affinity_dict = None
    if args.affinity is not None:
        affinity_df = extract_pdbbind_affinities(args.affinity)
        print(affinity_df)
        affinity_dict = {
            pdbid: (affinity, metric) for
            pdbid, affinity, metric in zip(
                affinity_df['pdbid'],
                affinity_df['affinity'],
                affinity_df['metric'])
        }
    for idx, path in enumerate(base_path.glob('*')):
        if path.is_dir():
            s_ = generate_types_str(
                path, pdb_exp, xtal_exp, docked_exp, active_exp, inactive_exp,
                separated_files=args.split_sdfs, affinity_dict=affinity_dict)
            if s_ != -1:
                s += s_.strip()
                if args.split_sdfs:
                    s += '\n'
            print('Completed {0}/{1} targets'.format(idx, n_pdbs))

    s = '\n'.join([line for line in s.split('\n') if len(line.split()) > 1])
    with open(expand_path(
            output_path / (output_path.parent.name + '.types')), 'w') as f:
        f.write(s)
